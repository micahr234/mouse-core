"""Supervised value objective on PREDICTION positions."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


class SvObjective(Objective):
    """Supervised Q-star value regression objective.

    Reads ``predictions["value"]`` (shape ``[B, S, A]``) from the model's value head.
    Only finite entries of ``info_q_star`` participate in the loss; ``-inf``
    sentinels used for padded or invalid actions are automatically excluded.

    Args:
        loss_type: ``"mse"`` (L2) or ``"mae"`` (L1) regression loss.
        predictions_key: Key in ``predictions`` that holds the ``[B, S, A]`` value logits.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["mse", "mae"] = "mse",
        predictions_key: str = "value",
    ) -> None:
        self.loss_type = loss_type
        self.predictions_key = predictions_key

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions[self.predictions_key]

        A = logits.shape[-1]
        logits = logits.reshape(-1, A)
        q_targets = objective_data["info_q_star"].reshape(-1, A).to(dtype=logits.dtype)  # [B*S, A]

        if q_targets.shape[0] == 0:
            raise ValueError("SvObjective: batch is empty (no tokens).")

        finite_mask = torch.isfinite(q_targets)  # [N, A]
        if not finite_mask.any():
            raise ValueError("SvObjective: info_q_star contains no finite values (all NaN or -inf).")

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
