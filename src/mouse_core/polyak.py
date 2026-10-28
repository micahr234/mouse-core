"""Parameter interpolation for explicit delayed-agent updates."""

import torch
from torch import nn


@torch.no_grad()
def polyak_update(*, target: nn.Module, source: nn.Module, tau: float) -> None:
    """Interpolate matching parameters in place; leave buffers unchanged.

    Parameters are matched by name. Names, shapes, devices, and dtypes must
    agree; validate the entire pair before changing any target parameter.
    """
    if not 0 <= tau <= 1:
        raise ValueError("Polyak rate must be in [0, 1].")
    targets = dict(target.named_parameters())
    sources = dict(source.named_parameters())
    if targets.keys() != sources.keys():
        raise ValueError("Polyak modules must have matching parameter names.")
    for name, target_param in targets.items():
        source_param = sources[name]
        if (not (target_param.is_floating_point() or target_param.is_complex()) or
                target_param.layout != torch.strided or source_param.layout != torch.strided):
            raise ValueError(f"Polyak parameter {name!r} requires dense floating-point or complex tensors.")
        if (target_param.shape != source_param.shape or
                target_param.device != source_param.device or
                target_param.dtype != source_param.dtype):
            raise ValueError(f"Polyak parameter {name!r} must have matching shape, device, and dtype.")
    for name, target_param in targets.items():
        source_param = sources[name]
        target_param.lerp_(source_param, tau)
