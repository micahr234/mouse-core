"""Supervised value objective on PREDICTION positions."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from mouse_core.models.heads.base import BaseHead
from mouse_core.objectives.base import Objective, predictions_for, require_head


class SvObjective(Objective):
    """Supervised value regression objective on per-action Q targets.

    Reads the tensor for ``head`` (shape ``[B, S, A]``) and regresses toward
    ``objective_data[targets_key]``. Every finite target entry participates,
    including terminal / truncated rows — unlike :class:`~mouse_core.objectives.sp.SpObjective`,
    there is no ``mask_key``. ``-inf`` sentinels used for padded or invalid
    actions are excluded.

    Args:
        loss_type: ``"mse"`` (L2) or ``"mae"`` (L1) regression loss.
        head: Value head this objective trains. Must be the same
            instance passed to ``Model(heads=)``.
        targets_key: Key in ``objective_data`` that holds ``[B, S, A]`` Q targets
            (default ``"info_q_star"``).
    """

    def __init__(
        self,
        *,
        loss_type: Literal["mse", "mae"] = "mse",
        head: BaseHead,
        targets_key: str = "info_q_star",
    ) -> None:
        self.loss_type = loss_type
        self.head = require_head(head=head, what="head")
        self.targets_key = targets_key

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        delayed_predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions_for(head=self.head, predictions=predictions, who="SvObjective")

        A = logits.shape[-1]
        logits = logits.reshape(-1, A)
        q_targets = objective_data[self.targets_key].reshape(-1, A).to(dtype=logits.dtype)

        if q_targets.shape[0] == 0:
            raise ValueError("SvObjective: batch is empty (no tokens).")

        finite_mask = torch.isfinite(q_targets)
        if not finite_mask.any():
            raise ValueError(
                f"SvObjective: {self.targets_key!r} contains no finite values (all NaN or -inf)."
            )

        if self.loss_type == "mse":
            loss = F.mse_loss(logits[finite_mask], q_targets[finite_mask])
        elif self.loss_type == "mae":
            loss = F.l1_loss(logits[finite_mask], q_targets[finite_mask])
        else:
            raise ValueError(
                f"Invalid SvObjective loss_type: {self.loss_type!r} (expected 'mse' or 'mae')."
            )

        metrics: dict[str, float] = {"value": float(loss.detach().item())}
        return loss, metrics
