"""Optional discrete action sampling for application-owned Agent.get_action methods."""
from __future__ import annotations
from tensordict import TensorDict

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from mouse_core.sampling import check_temperature_range


def action_readouts(
    scores: torch.Tensor, inputs: TensorDict, *, sequence_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Select the last marked token of each requested sequence for environment use.

    This is token selection only. The caller decides which environments need an
    action. Packed scores contain all marked tokens; cached scores left-pad them.
    Step mappings and objective labels are neither needed nor inspected.
    """
    selected = np.asarray(list(range(inputs["num_sequences"])) if sequence_ids is None else sequence_ids)
    if selected.ndim != 1 or (selected.size and selected.dtype.kind not in "iu"):
        raise ValueError("sequence_ids must be a sequence of integer IDs")
    selected = selected.astype(np.int64)
    if selected.size and (selected.min() < 0 or selected.max() >= inputs["num_sequences"]):
        raise ValueError("sequence_ids must identify sequences in the input batch")
    head_sequences = inputs["sequence_ids"][inputs["head_mask"]]
    counts = torch.bincount(head_sequences, minlength=inputs["num_sequences"])
    selected = torch.as_tensor(selected, device=counts.device)
    if bool((counts[selected] == 0).any()):
        raise ValueError("Action requested for a sequence without a marked head token")
    if scores.ndim == 2:
        if scores.shape[0] != head_sequences.numel():
            raise ValueError("Packed scores must have one row per marked head token")
        positions = counts.cumsum(0)[selected] - 1
        return scores[positions.to(scores.device)]
    if scores.ndim == 3:
        width = int(counts.max()) if counts.numel() else 0
        if scores.shape[:2] != (inputs["num_sequences"], width):
            raise ValueError("Cached scores must match the input batch's left-padded readouts")
        return scores[selected.to(scores.device), width - 1] if selected.numel() else scores.new_empty((0, scores.shape[-1]))
    raise ValueError("Action scores must be packed [P, A] or cached [B, P_max, A]")


def select_discrete_action(
    *, scores: torch.Tensor, temperature: float, valid: torch.Tensor | None = None,
    num_actions: int | None = None,
) -> torch.Tensor:
    """Choose one action per prediction, greedy at zero temperature.

    Preserve every leading score dimension. An optional boolean mask over those
    dimensions selects real predictions from padded cached scores, returning a
    flat action vector. This sampler never selects a step or sequence.
    """
    check_temperature_range(temperature, temperature)
    if valid is not None:
        if valid.dtype != torch.bool or valid.shape != scores.shape[:-1]:
            raise ValueError("head_valid_mask must be boolean and match score prediction dimensions")
        scores = scores[valid]
    if num_actions is not None:
        if type(num_actions) is not int or not 1 <= num_actions <= scores.shape[-1]:
            raise ValueError("num_actions must select at least one available action")
        scores = scores[..., :num_actions]
    if temperature == 0.0:
        return scores.argmax(dim=-1)
    scores = scores.to(torch.float64)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = F.softmax(scores / temperature, dim=-1)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(scores.shape[:-1])
