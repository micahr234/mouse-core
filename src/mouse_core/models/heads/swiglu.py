"""Prenormalized MLP body for prediction heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mouse_core.models.embedding.linear import ScaledLinear
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.mlp import swiglu_blocks


class _MlpHead(BaseHead):
    """Stacked SwiGLU blocks with a scaled output projection.

    Architecture::

        RMSNorm → SwiGLU(D→hidden) × (num_layers−1) → ScaledLinear(hidden→out)

    Every head owns an input ``RMSNorm`` before its MLP, independent of
    the backbone's final norm and the projection's input norm.
    ``propagate_gradient`` in ``[0, 1]`` scales how much of this head's
    loss gradient reaches the projection (when present) and backbone
    through the head input.
    ``1`` is full gradient flow. ``0`` detaches the pooled input so the
    projection and backbone get none. A value in between keeps the forward value as
    ``h`` and scales the upstream gradient by ``propagate_gradient``.
    The head's own parameters still receive the full loss gradient.
    ``scale`` multiplies only the final linear layer's initial weight and
    bias. Set it small (e.g. ``0.01``) for a near-zero initial output.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int,
        propagate_gradient: float,
        scale: float = 1.0,
    ):
        super().__init__()
        if (
            not isinstance(propagate_gradient, (int, float))
            or isinstance(propagate_gradient, bool)
            or not math.isfinite(float(propagate_gradient))
            or not 0.0 <= float(propagate_gradient) <= 1.0
        ):
            raise ValueError(
                "propagate_gradient must be a finite number in [0, 1], "
                f"got {propagate_gradient!r}."
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.scale = float(scale)
        self.propagate_gradient = float(propagate_gradient)
        self.norm = nn.RMSNorm(in_features, elementwise_affine=True, eps=1e-5)
        self.layers = nn.Sequential(
            *swiglu_blocks(in_features=in_features, hidden_dim=hidden_dim, num_blocks=num_layers - 1),
            ScaledLinear(
                in_features=hidden_dim if num_layers > 1 else in_features,
                out_features=out_features,
                scale=scale,
            ),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.propagate_gradient == 0.0:
            h = h.detach()
        elif self.propagate_gradient != 1.0:
            h = h + (1.0 - self.propagate_gradient) * (h.detach() - h)
        return self.layers(self.norm(h))
