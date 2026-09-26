"""Supervised value objective on PREDICTION positions."""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction


class SvObjective(Objective):
    """Supervised value regression objective on per-action Q targets.

    Reads ``predictions`` (shape ``[B, S, A]``) and regresses toward
    ``targets`` (same shape). Callers pass the Q target tensor at call
    time (same pattern as DQN ``predictions=`` / ``delayed_predictions=``);
    there is no ``targets_key`` batch lookup. Every finite target entry
    participates, including terminal / truncated rows — unlike
    :class:`~mouse_core.objectives.sp.SpObjective`, there is no
    ``mask_key``. ``-inf`` sentinels used for padded or invalid actions
    are excluded.

    Args:
        loss_type: ``"mse"`` (L2) or ``"mae"`` (L1) regression loss.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["mse", "mae"] = "mse",
    ) -> None:
        self.loss_type = loss_type

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
            "SvObjective",
            delayed_predictions=delayed_predictions,
            value_predictions=value_predictions,
        )
        q_targets_raw = _require_prediction(
            targets, owner="SvObjective", name="targets"
        )
        logits: torch.Tensor = predictions

        A = logits.shape[-1]
        logits = logits.reshape(-1, A)
        q_targets = q_targets_raw.reshape(-1, A).to(dtype=logits.dtype)

        if q_targets.shape[0] == 0:
            raise ValueError("SvObjective: batch is empty (no tokens).")

        finite_mask = torch.isfinite(q_targets)
        if not finite_mask.any():
            raise ValueError(
                "SvObjective: targets contains no finite values (all NaN or -inf)."
            )

        if self.loss_type == "mse":
            loss = F.mse_loss(logits[finite_mask], q_targets[finite_mask])
        elif self.loss_type == "mae":
            loss = F.l1_loss(logits[finite_mask], q_targets[finite_mask])
        else:
            raise ValueError(
                f"Invalid SvObjective loss_type: {self.loss_type!r} (expected 'mse' or 'mae')."
            )

        metrics: dict[str, float | torch.Tensor] = {"value": float(loss.detach().item())}
        return loss, metrics
