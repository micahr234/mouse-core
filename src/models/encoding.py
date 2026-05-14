"""Positional and feature encoding modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn



class RandomFourierFeatures(nn.Module):
    """Random Fourier Features (Rahimi & Recht, 2007) with log-uniform frequencies.

    Each feature is ``cos(ωx + b)`` where ω is sampled log-uniformly over
    [1/in_max, 1/in_min] and b ~ Uniform(0, 2π). The random phase breaks the
    even-function symmetry of plain cos, making the encoding sign-sensitive
    (x ≠ −x) while halving the output size vs a sin+cos encoding.

    ``in_min`` and ``in_max`` are expressed in input-space units:
    - the lowest frequency (ω = 1/in_max) completes one cycle across ``in_max`` units;
    - the highest frequency (ω = 1/in_min) completes one cycle across ``in_min`` units.

    ``num_freq_sets`` independent (ω, b) pairs are sampled, forming banks of
    shape ``(num_freq_sets, num_features)``. ``forward`` requires a ``freq_idx``
    integer tensor of the same shape as ``x`` to select which set to use per
    element. Both buffers are persistent so they are saved with the checkpoint.

    Output has per-dim std ≈ output_scale / √2 before the affine.

    After the cosine, a learnable per-(freq set, feature) weight is applied
    (shape ``(num_freq_sets, num_features)``; initialised to ``output_scale``).
    """

    def __init__(
        self,
        num_features: int,
        in_min: float = 1e-2,
        in_max: float = 1e2,
        num_freq_sets: int = 1,
        output_scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0")
        if in_min <= 0 or in_max <= 0:
            raise ValueError("in_min and in_max must be > 0")
        if in_min >= in_max:
            raise ValueError("in_min must be < in_max")
        if num_freq_sets < 1:
            raise ValueError("num_freq_sets must be >= 1")
        # ω = 1/x so that one cycle spans x units of input: ω_min = 1/in_max, ω_max = 1/in_min
        log_w_min = math.log(1.0 / in_max)
        log_w_max = math.log(1.0 / in_min)
        freqs = torch.empty(num_freq_sets, num_features, dtype=dtype).uniform_(log_w_min, log_w_max).exp()
        phases = torch.empty(num_freq_sets, num_features, dtype=dtype).uniform_(0.0, 2.0 * math.pi)
        self.register_buffer("freqs", freqs, persistent=True)
        self.register_buffer("phases", phases, persistent=True)
        self.weight = nn.Parameter(torch.full((num_freq_sets, num_features), float(output_scale), dtype=dtype))

    def forward(self, x: torch.Tensor, freq_idx: torch.Tensor | int) -> torch.Tensor:
        """Map scalar inputs to RFF embeddings.

        Args:
            x:        Scalar inputs, shape ``(*batch,)``.
            freq_idx: Which frequency set(s) to use. Either:
                      - an ``int`` constant — same set broadcast over all elements, or
                      - an integer tensor of the same shape as ``x`` — one set per element.
        Returns:
            Tensor of shape ``(*batch, num_features)``.
        """
        if isinstance(freq_idx, torch.Tensor):
            assert x.shape == freq_idx.shape, (
                f"x and freq_idx must have the same shape, got {x.shape} and {freq_idx.shape}"
            )
        freqs = self.get_buffer("freqs")    # (num_freq_sets, num_features)
        phases = self.get_buffer("phases")  # (num_freq_sets, num_features)
        x = x.to(dtype=freqs.dtype)
        w = freqs[freq_idx]   # (num_features,) or (*batch, num_features)
        b = phases[freq_idx]  # (num_features,) or (*batch, num_features)
        aw = self.weight[freq_idx]
        return aw * (x.unsqueeze(-1) * w + b).cos()


class NormalizedPixel(nn.Module):
    """Maps integer pixel values (0-255) to [-1, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x / 255.0 * 2.0) - 1.0
