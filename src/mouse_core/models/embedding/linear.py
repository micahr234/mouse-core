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


class PosLinear(nn.Module):
    """Position-conditioned linear projection.

    Stores one independent ``(weight, bias)`` pair per position index.  At
    forward time, each element selects its projection via an integer position
    index, enabling per-dimension embeddings without separate ``nn.Linear``
    instances.

    Args:
        num_positions: Number of distinct positions (embedding table size).
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        device: Tensor device for parameters.
        dtype: Tensor dtype for parameters.
    """

    def __init__(
        self,
        num_positions: int,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(num_positions, out_features, in_features, **factory))
        self.bias = nn.Parameter(torch.empty(num_positions, out_features, **factory))
        nn.init.kaiming_uniform_(self.weight.view(-1, in_features), a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Apply the position-specific projection.

        Args:
            x: Input tensor ``[*batch, in_features]``.
            pos: Integer position indices ``[*batch]``; same leading shape as ``x``.

        Returns:
            Output tensor ``[*batch, out_features]``.
        """
        x_flat = x.reshape(-1, x.shape[-1])
        pos_flat = pos.reshape(-1)
        w = self.weight[pos_flat]
        b = self.bias[pos_flat]
        out = torch.bmm(w, x_flat.unsqueeze(-1)).squeeze(-1) + b
        out_shape: tuple[int, ...] = tuple(int(s) for s in pos.shape) + (self.out_features,)
        return out.view(out_shape)


class ScaledPosLinear(PosLinear):
    """``PosLinear`` with Kaiming-uniform weights multiplied by ``scale``.

    Use ``scale = output_std / input_std`` to match a desired output standard deviation.
    """

    def __init__(
        self,
        num_positions: int,
        in_features: int,
        out_features: int,
        scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            num_positions=num_positions,
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale!r}.")
        with torch.no_grad():
            self.weight.mul_(scale)
            if self.bias is not None:
                self.bias.mul_(scale)
