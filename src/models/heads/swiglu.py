"""SwiGLU activation and SwiGLUHead MLP building block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mouse.models.embedding.linear import ScaledLinear
from mouse.models.heads.base import BaseHead


class SwiGLU(nn.Module):
    """Gated linear unit with SiLU on the gate: ``silu(x @ W1) * (x @ W2)`` via one fused ``Linear`` to ``2 * dim``."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.linear(x).chunk(2, dim=-1)
        return F.silu(a) * b


class SwiGLUHead(BaseHead):
    """MLP head built from stacked ``SwiGLU`` blocks with a scaled output projection.

    Architecture::

        [RMSNorm →] SwiGLU(D→hidden) × (num_layers−1) → ScaledLinear(hidden→out)

    The optional ``RMSNorm`` (``use_norm=True``) is applied to the input before
    the first SwiGLU block. ``scale`` controls the output weight initialisation
    magnitude — set small (e.g. ``0.01``) for a near-zero initial output.

    Args:
        in_features: Input dimension ``D``.
        out_features: Output dimension (number of actions ``A``, or ``A * vec_dim``).
        hidden_dim: Width of the SwiGLU hidden layers.
        num_layers: Total depth including the final linear; must be ``>= 1``.
        scale: ``ScaledLinear`` weight init multiplier for the output projection.
        use_norm: Whether to prepend an ``RMSNorm`` layer.
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
        if use_norm:
            self.norm = nn.RMSNorm(in_features, elementwise_affine=True, eps=1e-5)
        else:
            self.norm = None
        dims = [in_features] + [hidden_dim] * (num_layers - 1) + [out_features]
        self.layers = nn.Sequential(
            *[SwiGLU(in_features=dims[i], hidden_dim=dims[i+1]) for i in range(num_layers - 1)],
            ScaledLinear(in_features=dims[-2], out_features=dims[-1], scale=scale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        return self.layers(x)
