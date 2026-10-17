"""Episode-scoped temperature sampling for online collection."""

import math
from collections.abc import Sequence

import numpy as np
import torch


def check_temperature_range(low: float, high: float) -> None:
    if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high:
        raise ValueError("online temperature bounds must be finite and satisfy 0 <= min <= max")


class EpisodeTemperature:
    """Uniformly sample once per episode per stream, including across inner episodes."""

    def __init__(self, low: float, high: float, rng: np.random.Generator):
        check_temperature_range(low, high)
        self.low = low
        self.high = high
        self.rng = rng
        self.episodes: dict[int, tuple[int, float]] = {}

    def for_episode(self, stream: int, episode: int) -> float:
        current = self.episodes.get(stream)
        if current is None or current[0] != episode:
            temperature = float(self.rng.uniform(self.low, self.high))
            self.episodes[stream] = (episode, temperature)
        return self.episodes[stream][1]

    def actions(
        self, scores: torch.Tensor, episode_ids: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperatures = [self.for_episode(i, episode) for i, episode in enumerate(episode_ids)]
        return sample_tempered_actions(scores, temperatures, self.rng)


def last_action_scores(scores: torch.Tensor, valid: torch.Tensor | None, num_actions: int) -> torch.Tensor:
    """Last valid readout for each decode stream, restricted to legal actions."""
    if valid is not None and scores.ndim == 3:
        if valid.dtype != torch.bool or valid.shape != scores.shape[:2]:
            raise ValueError("valid must be bool [batch, steps] matching scores")
    if scores.ndim == 3:
        if valid is None:
            scores = scores[:, -1]
        else:
            indices = torch.arange(scores.shape[1], device=scores.device).expand_as(valid)
            last = indices.masked_fill(~valid, -1).amax(dim=-1)
            if bool((last < 0).any()):
                raise ValueError("every stream must have a valid action readout")
            scores = scores[torch.arange(scores.shape[0], device=scores.device), last]
    if scores.ndim != 2 or not 1 <= num_actions <= scores.shape[-1]:
        raise ValueError("expected [batch, actions] scores and a valid action count")
    return scores[:, :num_actions]


def sample_tempered_actions(
    scores: torch.Tensor, temperatures: Sequence[float], rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample softmax(scores / temperature), returning exact behavior log-probs.

    Zero is greedy (log probability 0). Center before dividing to keep very
    small positive temperatures stable. Sampling uses the collector's own RNG.
    """
    if scores.ndim != 2 or len(temperatures) != scores.shape[0] or scores.shape[1] == 0:
        raise ValueError("one temperature is required per row of action scores")
    if any(not math.isfinite(t) or t < 0 for t in temperatures):
        raise ValueError("temperatures must be finite and nonnegative")
    # Float64 keeps tiny positive temperatures representable and avoids overflow
    # in centering otherwise finite fp32 scores.
    values = scores.detach().to(dtype=torch.float64)
    temp = torch.tensor(temperatures, device=scores.device, dtype=torch.float64)
    greedy = values.argmax(dim=-1)
    logits = (values - values.amax(dim=-1, keepdim=True)) / temp.clamp_min(torch.finfo(temp.dtype).tiny)[:, None]
    log_probs = logits.log_softmax(dim=-1)
    probabilities = log_probs.exp().cpu().numpy()
    actions = greedy.cpu().numpy().copy()
    for i, temperature in enumerate(temperatures):
        if temperature > 0:
            actions[i] = rng.choice(scores.shape[1], p=probabilities[i])
    taken = torch.as_tensor(actions, device=scores.device, dtype=torch.long)
    behavior = log_probs.gather(1, taken[:, None]).squeeze(1)
    behavior = torch.where(temp == 0, torch.zeros_like(behavior), behavior)
    return taken, behavior.to(dtype=torch.float32)




def tempered_policy_logits(scores: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """Differentiable logits for the same episode policy used during collection.

    Zero-temperature rows are point masses and have zero policy gradient.
    Finite saturation keeps entropy's 0 * log(0) term well-defined, including
    at tiny temperatures. Float64 agrees with the collection distribution.
    """
    if scores.ndim != 2 or temperatures.shape != scores.shape[:1]:
        raise ValueError("expected [steps, actions] scores and [steps] temperatures")
    if not bool(torch.isfinite(temperatures).all()) or bool((temperatures < 0).any()):
        raise ValueError("temperatures must be finite and nonnegative")
    values = scores.to(torch.float64)
    temp = temperatures.to(device=scores.device, dtype=torch.float64)
    centered = values - values.amax(-1, keepdim=True)
    logits = (centered / temp.clamp_min(torch.finfo(temp.dtype).tiny)[:, None]).clamp_min(-1e30)
    greedy = torch.full_like(logits, -1e30).scatter(1, values.argmax(-1, keepdim=True), 0.)
    return torch.where((temp == 0)[:, None], greedy, logits)
