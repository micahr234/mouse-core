"""DQNHead: SwiGLUHead with EMA target and Polyak averaging."""

from __future__ import annotations

import torch

from mouse_core.models.heads.base import BaseHeadWithTarget
from mouse_core.models.heads.swiglu import SwiGLUHead


class DQNHead(BaseHeadWithTarget):
    """SwiGLUHead paired with an EMA target copy and Polyak averaging.

    ``forward`` runs the online head. ``target_forward`` runs the target head
    (no gradient tracking). Call ``polyak_update(tau)`` after each optimizer
    step to soft-update the target:  θ_target ← τ·θ_online + (1−τ)·θ_target.
    Initialize with ``tau=1.0`` to copy online weights into the target.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int,
        scale: float = 1.0,
        use_norm: bool = True,
    ):
        super().__init__()
        self.online = SwiGLUHead(
            in_features=in_features,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            scale=scale,
            use_norm=use_norm,
        )
        self._init_target(self.online)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Run the online head; returns Q-value logits ``[B, S, A]``."""
        return self.online(h)
