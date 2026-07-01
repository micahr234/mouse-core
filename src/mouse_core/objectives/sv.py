"""Supervised value objective on PREDICTION positions."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


class SvObjective(Objective):
    """Supervised value regression objective on per-action Q targets.

    Reads ``predictions[predictions_key]`` (shape ``[B, S, A]``) and regresses toward
    ``objective_data[targets_key]``. Only finite target entries participate; ``-inf``
    sentinels used for padded or invalid actions are automatically excluded.

    Args:
        loss_type: ``"mse"`` (L2) or ``"mae"`` (L1) regression loss.
        predictions_key: Key in ``predictions`` that holds the ``[B, S, A]`` value logits.
        targets_key: Key in ``objective_data`` that holds ``[B, S, A]`` Q targets
            (default ``"info_q_star"``).
    """

    def __init__(
        self,
        *,
        loss_type: Literal["mse", "mae"] = "mse",
        predictions_key: str = "value",
        targets_key: str = "info_q_star",
    ) -> None:
        self.loss_type = loss_type
        self.predictions_key = predictions_key
        self.targets_key = targets_key

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions[self.predictions_key]

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
