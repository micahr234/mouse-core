"""Supervised policy objective: hard CE onto action ids at head-output positions."""

from __future__ import annotations

from typing import overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction


def best_action(q_targets: torch.Tensor) -> torch.Tensor:
    """Index of a uniformly random finite maximizer per row.

    Callers that distill from a Q vector (e.g. ``info_q_star``) run this
    outside ``SpObjective`` and pass the resulting action ids as
    ``targets=``.

    ``-inf`` padding is never selected. Rows with no finite entry fall through
    to ``argmax`` of an all-``-inf`` mask (index 0), matching ``torch.argmax``.
    """
    finite = torch.isfinite(q_targets)
    q = q_targets.masked_fill(~finite, -torch.inf)
    is_max = finite & (q == q.max(dim=-1, keepdim=True).values)
    scores = torch.rand(q.shape, device=q.device, dtype=torch.float32)
    return scores.masked_fill(~is_max, -torch.inf).argmax(dim=-1)


def sp_ce(
    target_actions: torch.Tensor,
    logits: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    invalid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard CE onto integer action labels (aligned rows).

    ``invalid`` is an optional ``[N, A]`` bool mask of action slots that do
    not exist. Those slots are excluded from the student softmax and from
    label smoothing.

    Args:
        target_actions: ``[N]`` integer action ids.
        logits: ``[N, A]`` student action logits.
        label_smoothing: Mixes uniform mass over the *valid* actions into the
            hard label.
        invalid: Optional ``[N, A]`` bool; ``True`` marks a padded action.
    """
    if target_actions.ndim != 1:
        raise ValueError(
            f"sp_ce target_actions must be 1-D [N], got shape {tuple(target_actions.shape)}."
        )
    if logits.ndim != 2:
        raise ValueError(
            f"sp_ce logits must be 2-D [N, A], got shape {tuple(logits.shape)}."
        )
    if target_actions.shape[0] != logits.shape[0]:
        raise ValueError(
            f"sp_ce row count mismatch: target_actions {target_actions.shape[0]} "
            f"vs logits {logits.shape[0]}."
        )
    if invalid is not None:
        if invalid.shape != logits.shape:
            raise ValueError(
                f"sp_ce invalid shape {tuple(invalid.shape)} must match "
                f"logits {tuple(logits.shape)}."
            )
        if invalid.dtype != torch.bool:
            raise TypeError(
                f"sp_ce invalid must be bool, got {invalid.dtype}."
            )
    fill = torch.finfo(logits.dtype).min / 4
    if invalid is None:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target_actions.unsqueeze(-1)).squeeze(-1)
        if label_smoothing > 0.0:
            smooth = -log_probs.mean(dim=-1)
            nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
        return nll.mean()
    log_probs = F.log_softmax(logits.masked_fill(invalid, fill), dim=-1)
    nll = -log_probs.gather(dim=-1, index=target_actions.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0.0:
        valid = (~invalid).to(dtype=log_probs.dtype)
        num_valid = valid.sum(dim=-1).clamp(min=1.0)
        smooth = -(log_probs * valid).sum(dim=-1) / num_valid
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    return nll.mean()


def _skip_mask(mask: torch.Tensor, n_rows: int) -> torch.Tensor:
    """True where the mask is nonzero (bool True counts as skip)."""
    values = mask.reshape(-1)
    if values.shape[0] != n_rows:
        raise ValueError(
            f"mask length ({values.shape[0]}) must match flattened target rows ({n_rows})."
        )
    if values.dtype == torch.bool:
        return values
    return values != 0


class SpObjective(Objective):
    """Hard CE from integer action ids onto action logits at head-output positions.

    Reads ``predictions`` (shape ``[B, S, A]``) and compares against
    ``targets`` integer actions shaped ``[B, S]`` (or ``[B, S, 1]``). Filter
    Q* outside with ``best_action`` (uniform among tied maxima) and pass
    those ids, or pass dataset / behavior actions. Callers pass the tensor
    at call time (same pattern as DQN ``predictions=`` /
    ``delayed_predictions=``); there is no ``targets_key`` batch lookup.

    Rows where ``mask_key`` is True or any nonzero number are dropped. Pass
    ``mask_key="episode_done"`` to skip terminated and truncated steps.

    Args:
        label_smoothing: Mixes uniform mass over all action slots into the
            hard label.
        mask_key: Key in ``objective_data`` for a per-row skip mask (bool True
            or any nonzero number). ``None`` disables the skip.
    """

    def __init__(
        self,
        *,
        label_smoothing: float = 0.0,
        mask_key: str | None = "episode_done",
    ) -> None:
        self.label_smoothing = label_smoothing
        self.mask_key = mask_key

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        delayed_predictions: None = None,
        value_predictions: None = None,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "SpObjective",
            delayed_predictions=delayed_predictions,
            value_predictions=value_predictions,
        )
        target_tensor = _require_prediction(
            targets, owner="SpObjective", name="targets"
        )
        logits: torch.Tensor = predictions
        A = logits.shape[-1]
        leading = logits.shape[:-1]
        logits_flat = logits.reshape(-1, A)
        n_rows = logits_flat.shape[0]

        if n_rows == 0:
            raise ValueError("SpObjective: batch is empty (no tokens).")

        if self.mask_key is not None:
            valid_rows = ~_skip_mask(objective_data[self.mask_key], n_rows)
        else:
            valid_rows = torch.ones(n_rows, dtype=torch.bool, device=logits.device)

        if target_tensor.shape != leading and target_tensor.shape != (*leading, 1):
            raise ValueError(
                f"SpObjective: targets must have shape "
                f"{tuple(leading)} or {(*leading, 1)}, got {tuple(target_tensor.shape)}."
            )
        target_actions = target_tensor.reshape(-1).to(dtype=torch.long)
        if not valid_rows.any():
            raise ValueError(
                "SpObjective: no rows left after applying the skip mask."
            )
        logits_flat = logits_flat[valid_rows]
        target_actions = target_actions[valid_rows]
        if (target_actions < 0).any() or (target_actions >= A).any():
            raise ValueError(
                f"SpObjective: targets action ids must be in "
                f"[0, {A}), got min={int(target_actions.min())} "
                f"max={int(target_actions.max())}."
            )
        loss = sp_ce(
            target_actions=target_actions,
            logits=logits_flat,
            label_smoothing=self.label_smoothing,
        )
        metrics: dict[str, float | torch.Tensor] = {"action": float(loss.detach().item())}
        return loss, metrics
