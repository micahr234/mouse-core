"""REINFORCE for sampled reasoning traces, with detached outcome rewards."""
import torch


def reinforce_loss(*, log_prob_sums: torch.Tensor, rewards: torch.Tensor,
                   baseline: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Negative advantage times summed token log probabilities (including the stopping token).

    The baseline must be independent of the sampled trace. Rewards, baseline,
    and participation weights receive no gradient. A zero-weight DQN row does
    not train reasoning. Summing rather than averaging token log probabilities
    preserves the sequence-level REINFORCE estimator for variable-length traces.
    """
    if log_prob_sums.ndim != 1 or any(x.shape != log_prob_sums.shape for x in (rewards, baseline, weights)):
        raise ValueError("REINFORCE inputs must all be vectors of the same shape")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("REINFORCE weights must be finite and nonnegative")
    weight = weights.detach()
    active = weight > 0
    advantage = torch.where(active, (rewards - baseline).detach(), 0)
    scores = torch.where(active, log_prob_sums, 0)
    terms = -advantage * scores
    return (weight * terms).sum() / weight.sum().clamp_min(1e-12)
