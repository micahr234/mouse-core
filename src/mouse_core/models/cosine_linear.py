"""Cosine readout with learned global or per-output affine parameters."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CosineLinear(nn.Module):
    """Compute ``(RMSNorm(x) @ RMSNorm(W).T / D) * alpha + beta``.

    Both RMS normalizations are non-affine, over the last dimension. Ignoring
    epsilon, the normalized dot product equals cosine similarity. Trainable
    alpha starts at one and beta starts at zero. ``global_affine=True`` shares
    scalar alpha and beta across all outputs; otherwise each output has its
    own pair. ``bias=False`` omits beta. Raw weights start uniformly distributed in
    [-sqrt(3), sqrt(3)], keeping their RMS near one independently of width.

    Gradients flow through both normalizations. No post-update normalization
    or separate initialization scale is needed. Alpha and beta can learn the
    scale and offset of scalar values trained against ordinary TD targets.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        global_affine: bool = False,
    ) -> None:
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
        if (isinstance(eps, bool) or not isinstance(eps, (int, float))
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError("eps must be finite and positive.")
        if type(bias) is not bool:
            raise ValueError("bias must be a bool.")
        if type(global_affine) is not bool:
            raise ValueError("global_affine must be a bool.")
        self.global_affine = global_affine
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        affine_shape = () if global_affine else (out_features,)
        self.alpha = nn.Parameter(torch.empty(affine_shape, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.empty(affine_shape, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -math.sqrt(3), math.sqrt(3))
        nn.init.ones_(self.alpha)
        if self.beta is not None:
            nn.init.zeros_(self.beta)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = F.rms_norm(input, (self.in_features,), eps=self.eps)
        weights = F.rms_norm(self.weight, (self.in_features,), eps=self.eps)
        values = self.alpha * F.linear(features, weights) / self.in_features
        return values if self.beta is None else values + self.beta

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.beta is not None}, eps={self.eps}, global_affine={self.global_affine}"
