"""Shared FlexAttention mask, precision and position utilities."""
from __future__ import annotations
from typing import Any, Callable, cast
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_FLASH_DTYPES = frozenset({torch.bfloat16, torch.float16})
_BLOCK_SIZE = 128
# Shared kernels specialize for prefill/decode block geometry, precision, head
# layout and mask captures across models. Keep the budget local to these
# compiled functions; never change the application's global Dynamo settings.
_FLEX_RECOMPILE_LIMIT = 64
# The pinned torch runtime accepts recompile_limit, but its overload stubs omit it.
_compiled_create_block_mask: Any | None = None

def check_autocast_dtype(autocast_dtype: object, dtype: torch.dtype) -> torch.dtype | None:
    """Validate a backbone autocast dtype (train or decode) against the base weight ``dtype``.

    ``None`` disables autocast. Mixed precision accepts fp32 base weights or
    base weights already matching the requested autocast dtype. Matching
    autocast still matters for fp32 LoRA adapters alongside low-precision weights.
    """
    if autocast_dtype is None:
        return None
    if autocast_dtype not in _FLASH_DTYPES:
        raise ValueError(
            f"autocast_dtype must be torch.bfloat16, torch.float16, or None, got {autocast_dtype!r}."
        )
    if dtype is not torch.float32 and dtype is not autocast_dtype:
        raise ValueError(
            f"autocast_dtype={autocast_dtype} requires fp32 or matching base weights, got dtype={dtype}."
        )
    return cast(torch.dtype, autocast_dtype)


def _use_flex_compile(device: torch.device) -> bool:
    """Compiled FlexAttention kernels are CUDA-only (fp32, bf16 and fp16 all compile).

    fp32 is roughly 12x faster compiled than eager and no longer materializes
    the score matrix; ``torch.set_float32_matmul_precision("high")`` (TF32)
    additionally lets the fp32 kernel use tensor cores.
    """
    return device.type == "cuda"


def _is_lora_adapter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name


def module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    """Device and dtype of a module's base weights.

    Skips fp32 LoRA adapters so a LoRA stack still reports the frozen
    base dtype. Raises if the module has no parameters. (The KV cache is
    allocated in the compute dtype: the base dtype, or the backbone's
    ``decode_autocast_dtype`` when that is set.)
    """
    for name, param in module.named_parameters():
        if _is_lora_adapter(name):
            continue
        return param.device, param.dtype
    try:
        param = next(module.parameters())
    except StopIteration as exc:
        raise ValueError("module has no parameters") from exc
    return param.device, param.dtype


def flex_block_mask(
    mask_mod: Callable[..., torch.Tensor],
    *,
    B: int,
    Q_LEN: int,
    KV_LEN: int,
    device: torch.device | str,
    block_size: int,
    compile_masks: bool,
) -> Any:
    """Build a Flex ``BlockMask``. Compiles ``create_block_mask`` once.

    A stable ``mask_mod`` (module-level, holder pattern) lets the compiled
    wrapper reuse its graph. ``dynamic=True`` keeps one graph across
    ``(Q_LEN, KV_LEN)`` pairs.
    """
    global _compiled_create_block_mask
    builder: Any = create_block_mask
    if compile_masks:
        if _compiled_create_block_mask is None:
            _compiled_create_block_mask = cast(Any, torch.compile)(create_block_mask, dynamic=True, recompile_limit=_FLEX_RECOMPILE_LIMIT)
        builder = _compiled_create_block_mask
    return builder(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=str(device),
        BLOCK_SIZE=block_size,
    )


def packed_rope_positions(*, sequence_ids: torch.Tensor) -> torch.Tensor:
    """RoPE position of every token in a flat packed stream ``[L]``.

    Position = number of earlier tokens with the same ``group_id``.
    This is the counting rule cached decode uses and the same neighbourhood
    the attention masks allow (causal within one group, regardless of
    contiguity), so a sequence id that recurs after another id continues
    its own position counter instead of restarting at 0. Sync-free: one
    stable sort plus a cummax, no host ``.item()``.
    """
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    if L == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    seq = sequence_ids.to(device=device, dtype=torch.long)
    order = torch.argsort(seq, stable=True)
    sorted_seq = seq[order]
    arange = torch.arange(L, device=device)
    new_run = torch.ones(L, dtype=torch.bool, device=device)
    new_run[1:] = sorted_seq[1:] != sorted_seq[:-1]
    markers = torch.where(new_run, arange, torch.full_like(arange, -1))
    sorted_pos = arange - torch.cummax(markers, dim=0).values
    positions = torch.empty(L, dtype=torch.long, device=device)
    positions[order] = sorted_pos
    return positions


def _inductor_rejected(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {"InductorError", "LoweringException"} or "InductorError" in name


class _FlexKernel:
    """Call flex_attention; compile on CUDA, fall back to eager if Inductor rejects a call."""

    def __init__(self, device: torch.device) -> None:
        self._eager = flex_attention
        self._compiled = cast(Any, torch.compile)(flex_attention, recompile_limit=_FLEX_RECOMPILE_LIMIT) if _use_flex_compile(device) else None
        self._active = self._compiled or self._eager

    def __call__(self, *args, **kwargs):
        try:
            return self._active(*args, **kwargs)
        except Exception as exc:
            if self._compiled is not None and self._active is self._compiled and _inductor_rejected(exc):
                self._active = self._eager
                return self._eager(*args, **kwargs)
            raise
