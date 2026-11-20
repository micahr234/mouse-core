"""Stochastic model writeback from the selected update arithmetic precision."""

from functools import lru_cache

import torch


@torch.no_grad()
def stochastic_round_bfloat16(value: torch.Tensor, *, generator=None) -> torch.Tensor:
    """Round to adjacent BF16 values with distance-proportional probabilities.

    Exact BF16 values stay exact. NaNs/infinities are preserved; finite values
    outside BF16's finite range saturate (unbiasedness applies inside the range).
    Random low mantissa bits implement the same rule for both signs, including
    subnormals. No converted weights or rounding residuals are retained.
    """
    if value.dtype != torch.float32:
        raise TypeError("Stochastic BF16 rounding requires an FP32 input")
    limit = torch.finfo(torch.bfloat16).max
    finite = value.clamp(-limit, limit).contiguous()
    bits = finite.view(torch.int32)
    random = (torch.randint_like(bits, 0, 65536) if generator is None else
              torch.randint(0, 65536, bits.shape, dtype=torch.int32,
                            device=bits.device, generator=generator))
    rounded = ((bits + random) & -65536).view(torch.float32).to(torch.bfloat16)
    return torch.where(torch.isfinite(value), rounded, value.to(torch.bfloat16))


@torch.no_grad()
def stochastic_round_to(value: torch.Tensor, dtype: torch.dtype, *, generator=None) -> torch.Tensor:
    """Round directly to model precision without an intermediate narrowing cast.

    FP64 probabilities and adjacent target values are evaluated in FP64.
    Finite overflow saturates; exact values and NaNs/infinities are preserved.
    """
    supported = (torch.bfloat16, torch.float32, torch.float64)
    if value.dtype not in supported or dtype not in supported:
        raise TypeError("Stochastic rounding supports float64, float32 and bfloat16")
    if torch.finfo(value.dtype).bits <= torch.finfo(dtype).bits:
        return value.to(dtype)
    if value.dtype == torch.float32:
        return stochastic_round_bfloat16(value, generator=generator)
    limit = torch.finfo(dtype).max
    finite = value.clamp(-limit, limit)
    nearest = finite.to(dtype)
    nearest_wide = nearest.to(torch.float64)
    lower = torch.where(nearest_wide > finite,
                        torch.nextafter(nearest, torch.full_like(nearest, -float('inf'))), nearest)
    upper = torch.where(nearest_wide < finite,
                        torch.nextafter(nearest, torch.full_like(nearest, float('inf'))), nearest)
    lower_wide, upper_wide = lower.double(), upper.double()
    span = upper_wide - lower_wide
    probability = (finite - lower_wide) / torch.where(span == 0, torch.ones_like(span), span)
    random = (torch.rand_like(value) if generator is None else
              torch.rand(value.shape, dtype=value.dtype, device=value.device, generator=generator))
    rounded = torch.where(random < probability, upper, lower)
    return torch.where(torch.isfinite(value), rounded, value.to(dtype))


@torch.no_grad()
def copy_update_(parameter: torch.Tensor, value: torch.Tensor, *, stochastic_rounding=True) -> None:
    """Round only narrowing model writes; otherwise copy exactly."""
    parameter.copy_(stochastic_round_to(value, parameter.dtype) if stochastic_rounding else value)


@torch.no_grad()
def polyak_parameter_(target, source, tau, *, stochastic_rounding=False, arithmetic_dtype=None):
    if arithmetic_dtype is not None:
        if not isinstance(tau, torch.Tensor):
            if tau == 0:
                return
            if tau == 1:
                target.copy_(source)
                return
        if arithmetic_dtype == torch.float32 and stochastic_rounding and target.dtype == torch.bfloat16:
            rate = tau.float() if isinstance(tau, torch.Tensor) else tau
            if target.is_cuda and not isinstance(rate, torch.Tensor):
                rate = scalar_tensor(rate)
            _polyak_kernel(target.device)(target, source, rate)
        else:
            rate = tau.to(arithmetic_dtype) if isinstance(tau, torch.Tensor) else tau
            if target.dtype == source.dtype == arithmetic_dtype:
                target.lerp_(source, rate)
            else:
                value = target.to(arithmetic_dtype).lerp(source.to(arithmetic_dtype), rate)
                copy_update_(target, value, stochastic_rounding=stochastic_rounding)
        return
    if not stochastic_rounding:
        target.lerp_(source, tau)
        return
    if not isinstance(tau, torch.Tensor):
        if tau == 0:
            return
        if tau == 1:
            target.copy_(source)
            return
    if stochastic_rounding and target.dtype == torch.bfloat16:
        rate = tau.float() if isinstance(tau, torch.Tensor) else tau
        if target.is_cuda and not isinstance(rate, torch.Tensor):
            rate = scalar_tensor(rate)
        _polyak_kernel(target.device)(target, source, rate)
    else:
        target.lerp_(source, tau)


@lru_cache(maxsize=128)
def scalar_tensor(value, dtype=torch.float32):
    """Read-only CPU scalar metadata; values can change without recompiling CUDA."""
    return torch.tensor(value, dtype=dtype, device="cpu")


def _polyak_bfloat16_(target, source, tau):
    copy_update_(target, target.float().lerp(source.float(), tau))


_compiled_polyak_bfloat16 = None


def _polyak_kernel(device):
    global _compiled_polyak_bfloat16
    if device.type != "cuda":
        return _polyak_bfloat16_
    if _compiled_polyak_bfloat16 is None:
        _compiled_polyak_bfloat16 = torch.compile(
            _polyak_bfloat16_, fullgraph=True, dynamic=True,
            options={"triton.cudagraphs": False},
        )
    return _compiled_polyak_bfloat16
