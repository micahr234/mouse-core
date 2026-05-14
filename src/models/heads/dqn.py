"""DQNHead: SwiGLUHead with EMA target and Polyak averaging."""

from __future__ import annotations

import torch
import torch.nn as nn

from mouse.models.heads.swiglu import SwiGLUHead


class DQNHead(nn.Module):
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
        head_kwargs = dict(
            in_features=in_features,
            out_features=out_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            scale=scale,
            use_norm=use_norm,
        )
        self.online = SwiGLUHead(**head_kwargs)
        self.target = SwiGLUHead(**head_kwargs)
        self.target.requires_grad_(False)
        self.polyak_update(tau=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the online head.

        Args:
            x: Step representations ``[B, S, D]``.

        Returns:
            Q-value logits ``[B, S, A]``.
        """
        return self.online(x)

    def target_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the target head (no gradient tracking).

        Args:
            x: Step representations ``[B, S, D]``.

        Returns:
            Q-value logits ``[B, S, A]`` from the EMA target network.
        """
        return self.target(x)

    def polyak_update(self, tau: float) -> None:
        """Soft-update target toward online: θ_target ← τ·θ_online + (1−τ)·θ_target."""
        if tau <= 0.0:
            return
        for online_p, target_p in zip(self.online.parameters(), self.target.parameters()):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)
