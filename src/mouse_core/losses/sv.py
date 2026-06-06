"""Supervised value loss on PREDICTION positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mouse_core.losses.base import LossConfig

import torch
import torch.nn.functional as F
from tensordict import TensorDict


@dataclass(frozen=True)
class SvLossConfig(LossConfig):
    """Supervised q_star loss at PREDICTION (see ``sv_loss``)."""

    weight: float = 0.0  # omit ``loop.sv.weight`` or set 0 = do not compute SV loss (YAML default)
    loss_type: Literal["mse", "mae"] = "mse"


def sv_loss(
    step_stream: TensorDict,
    logits: torch.Tensor,
    cfg: SvLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised q_star loss over all ``[B, S]`` step positions, restricted to finite action slots.

    ``q_star_tok`` uses ``-inf`` as a sentinel for padded/invalid actions; only finite entries
    participate in the loss so padding never contributes gradients.

    Returns:
        Scalar loss and scalar metrics for logging (e.g. W&B).
    """
    A = logits.shape[-1]
    logits = logits.reshape(-1, A)
    q_targets = step_stream["q_star"].reshape(-1, A).to(dtype=logits.dtype)  # [B*S, A]

    if q_targets.shape[0] == 0:
        raise ValueError("sv_loss: batch is empty (no tokens).")

    finite_mask = torch.isfinite(q_targets)  # [N, A]
    if not finite_mask.any():
        raise ValueError("sv_loss: q_star contains no finite values (all NaN or -inf).")

    if cfg.loss_type == "mse":
        loss = F.mse_loss(logits[finite_mask], q_targets[finite_mask])
    elif cfg.loss_type == "mae":
        loss = F.l1_loss(logits[finite_mask], q_targets[finite_mask])
    else:
        raise ValueError(
            f"Invalid SV loss loss_type: {cfg.loss_type!r} (expected 'mse' or 'mae')."
        )

    metrics: dict[str, float] = {}
    metrics["sv_loss"] = float(loss.detach().item())

    return loss, metrics
