"""Residual MLPs with Linear or CosineLinear outputs."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from mouse_core.models.embedding.linear import ScaledLinear
from mouse_core.models.cosine_linear import CosineLinear


class SwiGLU(nn.Module):
    """Split a fused gate/value input and return ``silu(gate) * value``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = x.chunk(2, dim=-1)
        return F.silu(gate) * value


class ResidualBlock(nn.Module):
    """Pre-normalized Linear → SwiGLU → Linear with an identity skip."""

    def __init__(self, *, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        for name, value in (("in_features", in_features), ("hidden_dim", hidden_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        self.in_features = self.out_features = in_features
        self.hidden_dim = hidden_dim
        self.norm = nn.RMSNorm(in_features, eps=1e-5)
        self.input_linear = nn.Linear(in_features, 2 * hidden_dim)
        self.activation = SwiGLU()
        self.output_linear = nn.Linear(hidden_dim, in_features)
        self.to(device="cpu", dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x + self.output_linear(self.activation(self.input_linear(self.norm(x))))


class MLP(nn.Module):
    """N−1 residual blocks → final readout, with optional fixed RMSNorm at each end.

    ``num_layers`` counts each complete Linear → SwiGLU → Linear residual
    block as one layer and includes the final output layer as one layer.
    Every residual block
    retains ``in_features`` width; ``hidden_dim`` is its inner SwiGLU width.
    The final linear maps to ``out_features`` without an extra norm between
    the residual stack and linear. ``normalize_input`` normalizes before the
    residual stack; ``normalize_output`` normalizes after the final linear.
    Both use RMSNorm without affine parameters. Residual-block norms remain learned.

    ``scale`` optionally scales only the final linear layer's initial weight
    and bias. Omit it for feature projections; set it on prediction instances
    when a small initial output is useful. With ``output_layer="cosine_linear"``,
    the final layer normalizes its features and weight rows on every forward;
    its per-output alpha starts at one and beta at zero. Both learn.
    CosineLinear does not accept a separate ``scale``.

    ``propagate_gradient`` scales gradients into upstream modules, from zero
    (detach the input) to one (full flow), including through the skips.
    This MLP's own parameters always receive the full loss gradient.
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
        normalize_input: bool = True,
        normalize_output: bool = False,
        output_layer: Literal["linear", "cosine_linear"] = "linear",
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
        for name, value in (("normalize_input", normalize_input), ("normalize_output", normalize_output)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool, got {value!r}.")
        if output_layer not in ("linear", "cosine_linear"):
            raise ValueError("output_layer must be linear or cosine_linear.")
        self.output_layer = output_layer
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.input_norm = (
            nn.RMSNorm(in_features, eps=1e-5, elementwise_affine=False)
            if normalize_input else nn.Identity()
        )
        blocks = [
            ResidualBlock(in_features=in_features, hidden_dim=hidden_dim)
            for _ in range(num_layers - 1)
        ]
        if output_layer == "cosine_linear":
            if scale is not None:
                raise ValueError("scale is only supported for output_layer=linear.")
            output = CosineLinear(in_features, out_features)
        else:
            output = (
                nn.Linear(in_features, out_features) if scale is None
                else ScaledLinear(in_features, out_features, scale=scale)
            )
        self.layers = nn.Sequential(*blocks, output)
        self.output_norm = (
            nn.RMSNorm(out_features, eps=1e-5, elementwise_affine=False)
            if normalize_output else nn.Identity()
        )
        self.to(device="cpu", dtype=torch.float32)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.propagate_gradient == 0.0:
            h = h.detach()
        elif self.propagate_gradient != 1.0:
            h = h + (1.0 - self.propagate_gradient) * (h.detach() - h)
        return self.output_norm(self.layers(self.input_norm(h.float())))
