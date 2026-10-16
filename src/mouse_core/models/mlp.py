"""One prenormalized MLP for feature projections, values, and policy logits."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from mouse_core.models.embedding.linear import ScaledLinear


class SwiGLU(nn.Module):
    """Apply ``silu(gate) * value`` using one fused linear projection."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.linear(x).chunk(2, dim=-1)
        return F.silu(gate) * value


class MLP(nn.Module):
    """RMSNorm → SwiGLU × (num_layers − 1) → linear output, in float32.

    Instantiate separately for each role: ``out_features`` sets the projection
    width, number of logits, or number of regression values (one for a scalar).
    Each instance always normalizes its input and returns raw linear outputs.

    ``scale`` optionally scales only the final linear layer's initial weight
    and bias. Omit it for feature projections; set it on prediction instances
    when a small initial output is useful.

    ``propagate_gradient`` scales gradients into upstream modules, from zero
    (detach the input) to one (full flow). This MLP's own parameters always
    receive the full loss gradient.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int = 2,
        propagate_gradient: float = 1.0,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        for name, value in (
            ("in_features", in_features), ("out_features", out_features),
            ("hidden_dim", hidden_dim), ("num_layers", num_layers),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if (
            not isinstance(propagate_gradient, (int, float))
            or isinstance(propagate_gradient, bool)
            or not math.isfinite(propagate_gradient)
            or not 0.0 <= propagate_gradient <= 1.0
        ):
            raise ValueError(
                "propagate_gradient must be a finite number in [0, 1], "
                f"got {propagate_gradient!r}."
            )
        if scale is not None and (
            not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or not math.isfinite(scale) or scale < 0
        ):
            raise ValueError(f"scale must be a finite number >= 0, got {scale!r}.")
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.propagate_gradient = float(propagate_gradient)
        self.scale = float(scale) if scale is not None else None
        self.norm = nn.RMSNorm(in_features, eps=1e-5)
        blocks = [
            SwiGLU(in_features if index == 0 else hidden_dim, hidden_dim)
            for index in range(num_layers - 1)
        ]
        output_width = hidden_dim if num_layers > 1 else in_features
        output = (
            nn.Linear(output_width, out_features) if scale is None
            else ScaledLinear(output_width, out_features, scale=scale)
        )
        self.layers = nn.Sequential(*blocks, output)
        self.to(device="cpu", dtype=torch.float32)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.propagate_gradient == 0.0:
            h = h.detach()
        elif self.propagate_gradient != 1.0:
            h = h + (1.0 - self.propagate_gradient) * (h.detach() - h)
        return self.layers(self.norm(h.float()))
