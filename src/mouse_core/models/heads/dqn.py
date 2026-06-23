"""DiscreteActionValueHead: per-discrete-action values (e.g. Q(s,a)) with target network."""

from __future__ import annotations

import torch

from mouse_core.models.heads.base import BaseHeadWithTarget
from mouse_core.models.heads.swiglu import SwiGLUHead


class DiscreteActionValueHead(BaseHeadWithTarget):
    """Head that outputs a value for each discrete action.

    Wraps a SwiGLU MLP and maintains an EMA target copy (for TD-style objectives).
    ``forward`` runs the online network; ``target_forward`` and ``polyak_update``
    are available for the target.
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
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.scale = float(scale)
        self.use_norm = bool(use_norm)
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
        """Returns per-action values ``[B, S, A]``."""
        return self.online(h)
