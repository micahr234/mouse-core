"""Trainable scalar for reward centering via mean TD error.

Not a head. A one-parameter module the train loop owns and steps with
its own AdamW learning rate. Learns the mean TD error by MSE; the DQN
objective subtracts the detached constant from the TD residual so Q
sees a centered backup error.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RewardCentering(nn.Module):
    """Scalar ``center`` that tracks mean TD error with real gradients.

    ``loss_scale`` multiplies the centering MSE (finite ``>= 0``).
    ``init`` is the starting value of ``center``. Neither Polyak nor
    EMA updates this parameter — the optimizer steps it from the MSE
    against detached TD errors.
    """

    def __init__(self, *, init: float, loss_scale: float) -> None:
        super().__init__()
        init_f = float(init)
        if not math.isfinite(init_f):
            raise ValueError(f"init must be a finite float, got {init!r}")
        scale = float(loss_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"loss_scale must be a finite float >= 0, got {loss_scale!r}"
            )
        self.center = nn.Parameter(torch.tensor(init_f, dtype=torch.float32))
        self.loss_scale = scale

    def centering_mse(
        self,
        *,
        td_error: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """``loss_scale * mean_w (center - td_error.detach())²``.

        Only ``center`` receives a gradient. ``td_error`` is detached so
        this term never trains the Q head.
        """
        if td_error.shape != weight.shape:
            raise ValueError(
                f"td_error shape {tuple(td_error.shape)} must match "
                f"weight shape {tuple(weight.shape)}."
            )
        err = (self.center - td_error.detach()) ** 2
        w = weight.to(dtype=err.dtype)
        return self.loss_scale * (w * err).sum() / w.sum().clamp(min=1)
