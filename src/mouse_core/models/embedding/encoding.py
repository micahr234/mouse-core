"""Positional and feature encoding modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class StaticFourierFeatures(nn.Module):
    """Deterministic log-spaced Fourier features (no learnable parameters).

    Each feature is ``output_scale * cos(ω x + b)`` where ω is fixed log-spaced
    over ``[1/in_max, 1/in_min]`` and phases ``b`` are fixed evenly in
    ``[0, 2π)``. Sign-sensitive via nonzero phases; no ``nn.Parameter``.

    ``num_freq_sets`` independent banks support one bank per vector coordinate
    (selected via ``freq_idx``).
    """

    def __init__(
        self,
        num_features: int,
        in_min: float = 1e-2,
        in_max: float = 1e2,
        num_freq_sets: int = 1,
        output_scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        if in_min <= 0 or in_max <= 0:
            raise ValueError("in_min and in_max must be > 0")
        if in_min >= in_max:
            raise ValueError("in_min must be < in_max")
        if num_freq_sets < 1:
            raise ValueError("num_freq_sets must be >= 1")

        log_w_min = math.log(1.0 / in_max)
        log_w_max = math.log(1.0 / in_min)
        # Fixed log-spaced frequencies (not random).
        t = torch.linspace(0.0, 1.0, num_features, dtype=dtype)
        log_w = log_w_min + t * (log_w_max - log_w_min)
        freqs = log_w.exp().unsqueeze(0).expand(num_freq_sets, -1).contiguous()
        # Fixed phases: offset each freq-set so banks differ.
        phase_base = torch.linspace(0.0, 2.0 * math.pi, num_features + 1, dtype=dtype)[:-1]
        phases = torch.stack(
            [(phase_base + (2.0 * math.pi * s / num_freq_sets)) % (2.0 * math.pi) for s in range(num_freq_sets)],
            dim=0,
        )
        self.register_buffer("freqs", freqs, persistent=True)
        self.register_buffer("phases", phases, persistent=True)
        self.register_buffer(
            "output_scale",
            torch.tensor(float(output_scale), dtype=dtype),
            persistent=True,
        )

    def forward(self, x: torch.Tensor, freq_idx: torch.Tensor | int = 0) -> torch.Tensor:
        """Map scalar inputs to static Fourier embeddings.

        Args:
            x: Scalar inputs ``(*batch,)``.
            freq_idx: Frequency bank index (int or same shape as ``x``).

        Returns:
            ``(*batch, num_features)``.
        """
        if isinstance(freq_idx, torch.Tensor) and freq_idx.shape != x.shape:
            raise ValueError(
                f"x and freq_idx must have the same shape, got {tuple(x.shape)} and "
                f"{tuple(freq_idx.shape)}"
            )
        freqs = self.get_buffer("freqs")
        phases = self.get_buffer("phases")
        scale = self.get_buffer("output_scale")
        x = x.to(dtype=freqs.dtype)
        w = freqs[freq_idx]
        b = phases[freq_idx]
        return scale * (x.unsqueeze(-1) * w + b).cos()


# Alias kept only for older checkpoints that expect the name in state dict docs;
# new code must use StaticFourierFeatures (no learnable weight).
RandomFourierFeatures = StaticFourierFeatures


class NormalizedPixel(nn.Module):
    """Maps integer pixel values (0-255) to [-1, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / 255.0 * 2.0) - 1.0
