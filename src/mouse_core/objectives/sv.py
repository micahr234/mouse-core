"""Supervised value objective on PREDICTION positions."""

from __future__ import annotations

from typing import Literal, overload

import torch

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction


class SvObjective(Objective[...]):
    """Supervised value regression objective on per-action Q targets.

    By default, reads ``predictions`` ``[..., A]`` and regresses toward
    same-shaped Q targets, returning a scalar. With ``batched_heads=True``,
    reads independent heads ``[P, K, A]``; targets may be ``[P, A]`` shared
    across heads or ``[P, K, A]`` per head. Batched heads return a loss and
    metric shaped ``[K]``; callers choose the head reduction.
    Callers pass the Q target tensor at call
    time (same pattern as DQN ``predictions=`` / ``backup_value_predictions=``);
    there is no ``targets_key`` batch lookup. Every finite target entry
    participates, including terminal / truncated rows — unlike
    :class:`~mouse_core.objectives.sp.SpObjective`, there is no
    ``mask_key``. ``-inf`` sentinels used for padded or invalid actions
    are excluded.

    Args:
        loss_type: ``"mse"`` (L2) or ``"mae"`` (L1) regression loss.
        batched_heads: Treat ``[P, K, A]`` as independent heads and return
            one loss per head. Defaults to the original scalar reduction.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["mse", "mae"] = "mse",
        batched_heads: bool = False,
    ) -> None:
        self.loss_type = loss_type
        self.batched_heads = batched_heads

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        backup_value_predictions: None = None,
        backup_action_predictions: None = None,
        value_predictions: None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        backup_value_predictions: torch.Tensor | None = None,
        backup_action_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "SvObjective",
            backup_value_predictions=backup_value_predictions,
            backup_action_predictions=backup_action_predictions,
            value_predictions=value_predictions,
        )
        q_targets_raw = _require_prediction(
            targets, owner="SvObjective", name="targets"
        )
        logits: torch.Tensor = predictions
        batched = self.batched_heads
        if batched and logits.ndim != 3:
            raise ValueError("SvObjective: batched heads require predictions [P, K, A].")
        if logits.ndim < 1:
            raise ValueError("SvObjective: predictions must have shape [..., A].")
        A = logits.shape[-1]
        if logits.numel() == 0 or A == 0:
            raise ValueError("SvObjective: batch is empty (no tokens or actions).")
        if batched:
            predictions_heads = logits
            P, K, _ = predictions_heads.shape
        else:
            P, K = logits.numel() // A, 1
            predictions_heads = logits.reshape(P, K, A)
        if not batched and q_targets_raw.shape == logits.shape:
            q_targets = q_targets_raw.reshape(P, 1, A)
        elif batched and q_targets_raw.shape == (P, A):
            q_targets = q_targets_raw[:, None, :].expand(P, K, A)
        elif batched and q_targets_raw.shape == logits.shape:
            q_targets = q_targets_raw
        else:
            raise ValueError(
                "SvObjective: predictions and targets must have the same shape "
                "or targets must have shared shape [P, A]."
            )

        compute_dtype = torch.promote_types(torch.float32, logits.dtype)
        predictions_heads = predictions_heads.to(dtype=compute_dtype)
        q_targets = q_targets.detach().to(dtype=compute_dtype)
        finite_mask = torch.isfinite(q_targets)
        if not bool(finite_mask.any(dim=(0, 2)).all()):
            raise ValueError(
                "SvObjective: targets contains no finite values for at least one head "
                "(all NaN or -inf)."
            )
        # Substitute zeros before subtraction so invalid targets (and predictions
        # at those positions) cannot produce 0 * inf or NaN gradients.
        difference = (
            torch.where(finite_mask, predictions_heads, 0)
            - torch.where(finite_mask, q_targets, 0)
        )
        if self.loss_type == "mse":
            per_action = difference.square()
        elif self.loss_type == "mae":
            per_action = difference.abs()
        else:
            raise ValueError(
                f"Invalid SvObjective loss_type: {self.loss_type!r} (expected 'mse' or 'mae')."
            )
        loss_by_head = per_action.sum(dim=(0, 2)) / finite_mask.sum(dim=(0, 2))
        loss = loss_by_head if batched else loss_by_head.squeeze(0)
        metric = loss.detach() if batched else float(loss.detach().item())
        metrics: dict[str, float | torch.Tensor] = {"value": metric}
        return {"value": loss}, metrics
