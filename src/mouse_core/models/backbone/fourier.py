"""Fourier add for ``numeric`` token embeddings."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


def check_num_frequencies(num_frequencies: int | None) -> int | None:
    """Return ``num_frequencies`` when it can build a Fourier bank."""
    if num_frequencies is None:
        return None
    if isinstance(num_frequencies, bool) or not isinstance(num_frequencies, int):
        raise TypeError(
            f"num_frequencies must be an int, got {type(num_frequencies).__name__}"
        )
    if num_frequencies < 1:
        raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}")
    return num_frequencies


class FourierFeatures(nn.Module):
    """Map one scalar onto a ``hidden_dim`` vector.

    Frequencies are ``pi * 2^k`` for ``k = 0 .. num_frequencies - 1``.
    Sine and cosine of ``value * frequency`` go through a bias-free
    linear map. That map starts at 0, so the add is 0 until it trains
    and a pretrained token embedding is unchanged at step 0.
    """

    def __init__(self, *, num_frequencies: int, hidden_dim: int) -> None:
        super().__init__()
        checked = check_num_frequencies(num_frequencies)
        if checked is None:
            raise TypeError("FourierFeatures requires num_frequencies=")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
            raise TypeError(
                f"hidden_dim must be an int, got {type(hidden_dim).__name__}"
            )
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        bands = torch.pi * torch.pow(
            torch.tensor(2.0),
            torch.arange(checked, dtype=torch.float32),
        )
        self.num_frequencies = checked
        self.register_buffer("bands", bands, persistent=False)
        self.proj = nn.Linear(2 * checked, hidden_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 1:
            raise ValueError(
                "FourierFeatures expected a 1-D values tensor, "
                f"got {tuple(values.shape)}"
            )
        weight = self.proj.weight
        scaled = values.to(dtype=weight.dtype)
        bands = cast(torch.Tensor, self.bands).to(device=scaled.device, dtype=scaled.dtype)
        angles = scaled.unsqueeze(-1) * bands
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.proj(features)
