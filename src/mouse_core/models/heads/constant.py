"""Scalar head that does not read the backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from mouse_core.models.heads.base import BaseHead


class ConstantHead(BaseHead):
    """One scalar parameter. ``forward`` ignores the backbone state.

    The output is ``scale * value``. ``value`` starts at 0, so the
    output starts at 0. ``scale`` sets how far one step of ``value``
    moves the output. Pass the online head as ``reward_center=`` and
    the delayed copy as ``delayed_reward_center=`` on
    ``DqnObjective``. The backup uses the delayed center. The
    ``reward_center`` loss trains this one. No path reaches the
    backbone: the output does not depend on ``h``.
    """

    def __init__(self, *, scale: float) -> None:
        super().__init__()
        sc = float(scale)
        if sc < 0.0:
            raise ValueError(f"scale must be >= 0, got {scale!r}.")
        self.scale = sc
        self.value = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"h must be a Tensor, got {type(h).__name__}.")
        return self.value * self.scale
