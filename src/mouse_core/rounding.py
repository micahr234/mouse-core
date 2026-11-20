"""Unbiased BF16 writeback from FP32 update arithmetic."""

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
    random = torch.randint(0, 65536, bits.shape, dtype=torch.int32,
                           device=bits.device, generator=generator)
    rounded = ((bits + random) & -65536).view(torch.float32).to(torch.bfloat16)
    return torch.where(torch.isfinite(value), rounded, value.to(torch.bfloat16))


@torch.no_grad()
def copy_update_(parameter: torch.Tensor, value: torch.Tensor) -> None:
    """Write an update once, stochastically for BF16 and exactly for FP32."""
    parameter.copy_(stochastic_round_bfloat16(value) if parameter.dtype == torch.bfloat16 else value)


@torch.no_grad()
def polyak_parameter_(target, source, tau, *, stochastic_rounding=False):
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
        copy_update_(target, target.float().lerp(source.float(), rate))
    else:
        target.lerp_(source, tau)
