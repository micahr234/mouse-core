"""Optional discrete action sampling for application-owned Agent.get_action methods."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def select_discrete_action(
    *, scores: torch.Tensor, temperature: float, valid: torch.Tensor | None = None,
    num_actions: int | None = None,
) -> torch.Tensor:
    """Select IDs from the final valid readout: greedy at zero, sampled otherwise.

    Scores are ``[B, S, A]`` for cached decode or ``[1, A]`` for one packed
    step. ``valid`` marks real cached readouts so padding is never selected.
    The agent chooses which prediction supplies these scores.
    """
    scores = _last_action_scores(scores, name="action", valid=valid)
    if num_actions is not None:
        scores = scores[:, :num_actions]
    if temperature == 0.0:
        return scores.argmax(dim=-1)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = F.softmax(scores / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _last_valid_step(valid: torch.Tensor) -> torch.Tensor:
    """Last True index along dim=-1. Every row must have a valid head-output."""
    if valid.ndim != 2:
        raise ValueError(
            f"head_valid_mask must have shape [B, S], got {tuple(valid.shape)}"
        )
    has = valid.any(dim=-1)
    if not bool(has.all()):
        empty = (~has).nonzero(as_tuple=True)[0]
        raise ValueError(
            "get_action needs a valid head-output token in every row; "
            f"empty rows: {empty.tolist()}."
        )
    idx = valid.long() * torch.arange(valid.shape[1], device=valid.device)
    return idx.max(dim=-1).values


def _last_action_scores(
    raw: torch.Tensor,
    *,
    name: str,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Action scores at the last valid head-output of each row.

    Q / logit heads: ``[B, S, A]`` / ``[N, A]``. ``valid`` is the decode
    ``[B, S]`` mask; omitted, the last step column is used (left-padded
    decode).
    """
    if valid is not None and raw.ndim >= 3 and valid.shape != raw.shape[:2]:
        raise ValueError(
            f"{name} scores {tuple(raw.shape)} do not match "
            f"head_valid_mask {tuple(valid.shape)}"
        )
    step = None if valid is None else _last_valid_step(valid)
    batch = None if step is None else torch.arange(raw.shape[0], device=raw.device)
    if raw.ndim == 3:
        if step is None:
            return raw[:, -1]
        return raw[batch, step]
    if raw.ndim == 2:
        if raw.shape[0] != 1:
            raise ValueError(
                f"{name} has shape {tuple(raw.shape)}; get_action on flat "
                "[N, A] training outputs needs N=1 (one step). Use "
                "cached-decode [B, S, A] outputs for a batch."
            )
        return raw[-1].unsqueeze(0)
    raise ValueError(f"{name} expects [B, S, A] or [N, A], got {tuple(raw.shape)}")


