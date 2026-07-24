"""Linear layers and weight-initialisation helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaledEmbedding(nn.Embedding):
    """``nn.Embedding`` with default ``Normal(0, 1)`` init multiplied by ``scale``."""

    def __init__(self, num_embeddings: int, embedding_dim: int, scale: float = 1.0, **kwargs) -> None:
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.weight.data.mul_(scale)


class ScaledLinear(nn.Linear):
    """Linear layer with Kaiming-uniform init multiplied by ``scale``.

    Kaiming-uniform is applied first (``a=sqrt(5)``, same as ``nn.Linear`` default),
    then both weights and bias are multiplied by ``scale`` in-place.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        sc = float(scale)
        if sc < 0.0:
            raise ValueError(f"scale must be >= 0, got {scale!r}.")
        with torch.no_grad():
            self.weight.mul_(sc)
            if self.bias is not None:
                self.bias.mul_(sc)
