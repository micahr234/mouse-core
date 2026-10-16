"""Parameter interpolation for explicit delayed-agent updates."""

import torch
from torch import nn


@torch.no_grad()
def polyak_update(*, target: nn.Module, source: nn.Module, tau: float) -> None:
    """Interpolate matching parameters in place; leave buffers unchanged.

    The modules must have the same parameter order and shapes.
    """
    if not 0 <= tau <= 1:
        raise ValueError("Polyak rate must be in [0, 1].")
    for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.lerp_(source_param, tau)
