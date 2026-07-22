"""FlexAttention packed training over a flat concatenated token stream."""

from __future__ import annotations

import warnings

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

_BLOCK_SIZE = 128
_COMPILED_DTYPES = frozenset({torch.bfloat16, torch.float16})
_kernel_cache: dict[tuple[str, int | None, torch.dtype], "_FlexKernel"] = {}
_warned_unfused = False


def _use_flex_compile(device: torch.device, dtype: torch.dtype) -> bool:
    return device.type == "cuda" and dtype in _COMPILED_DTYPES


class _FlexKernel:
    """Call flex_attention; compile on CUDA bf16/fp16, fall back to eager on failure."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self._eager = flex_attention
        self._compiled = torch.compile(flex_attention) if _use_flex_compile(device, dtype) else None
        self._active = self._compiled or self._eager

    def __call__(self, *args, **kwargs):
        try:
            return self._active(*args, **kwargs)
        except Exception as exc:
            if self._compiled is not None and self._active is self._compiled:
                name = type(exc).__name__
                if name in {"InductorError", "LoweringException"} or "InductorError" in name:
                    self._active = self._eager
                    return self._eager(*args, **kwargs)
            raise


def _get_flex_kernel(device: torch.device, dtype: torch.dtype) -> _FlexKernel:
    key = (device.type, device.index, dtype)
    kern = _kernel_cache.get(key)
    if kern is None:
        kern = _FlexKernel(device, dtype)
        _kernel_cache[key] = kern
    return kern


def flex_packed_forward(
    model: torch.nn.Module,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    output_hidden_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Full prefill over a flat packed sequence ``embeds [L, D]``.

    Attention is causal within positions that share both ``sequence_ids`` and
    ``segment_ids``. No cross-sequence padding.

    RoPE positions reset at each new (sequence, segment) run (contiguous
    positions within a segment get 0,1,2,…).
    """
    if embeds.ndim != 2:
        raise ValueError(f"embeds must be [L, D], got shape {tuple(embeds.shape)}")
    L, _D = embeds.shape
    if sequence_ids.shape != (L,) or segment_ids.shape != (L,):
        raise ValueError("sequence_ids and segment_ids must have shape [L]")

    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    x = embeds.to(device=device, dtype=dtype).unsqueeze(0)  # [1, L, D]
    seq = sequence_ids.to(device=device)
    seg = segment_ids.to(device=device)

    # Per-token RoPE position within its (sequence, segment) run.
    # Sync-free: cummax of run-start markers, no host .item() for n_runs.
    position_ids = torch.zeros(L, dtype=torch.long, device=device)
    if L > 0:
        arange = torch.arange(L, device=device)
        new_run = torch.ones(L, dtype=torch.bool, device=device)
        new_run[1:] = (seq[1:] != seq[:-1]) | (seg[1:] != seg[:-1])
        markers = torch.where(new_run, arange, torch.full_like(arange, -1))
        position_ids = arange - torch.cummax(markers, dim=0).values

    # mask_mod closes over tensors via holder to avoid issues.
    holder = {"seq": seq, "seg": seg}
    compile_masks = _use_flex_compile(device, dtype)

    def mask_mod(b, h, q_idx, kv_idx):
        return (
            (kv_idx <= q_idx)
            & (holder["seq"][q_idx] == holder["seq"][kv_idx])
            & (holder["seg"][q_idx] == holder["seg"][kv_idx])
        )

    global _warned_unfused
    if not compile_masks and device.type == "cuda" and not _warned_unfused:
        warnings.warn(
            "FlexAttention training is running the unfused path because the "
            f"backbone dtype is {dtype}. Use "
            "`model.to(device=device, dtype=preferred_dtype(device))` "
            "(bfloat16/float16 on CUDA) so `flex_attention` compiles. "
            "`Model.to` keeps output heads in float32.",
            stacklevel=2,
        )
        _warned_unfused = True

    block_mask = create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        device=str(device),
        BLOCK_SIZE=_BLOCK_SIZE if L >= _BLOCK_SIZE else max(L, 1),
        _compile=compile_masks and L >= _BLOCK_SIZE,
    )

    flex = _get_flex_kernel(device, dtype)
    pos = position_ids.unsqueeze(0)  # [1, L]
    cos, sin = model.rotary_emb(x, pos)

    h = x
    layer_hiddens: list[torch.Tensor] = []
    for layer in model.layers:
        residual = h
        hn = layer.input_layernorm(h)
        attn = layer.self_attn
        q = attn.q_proj(hn).view(1, L, n_heads, head_dim)
        k = attn.k_proj(hn).view(1, L, n_kv_heads, head_dim)
        q_norm = getattr(attn, "q_norm", None)
        if q_norm is not None:
            q = q_norm(q)
            k = attn.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = attn.v_proj(hn).view(1, L, n_kv_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        o = flex(q, k, v, block_mask=block_mask, scale=attn.scaling, enable_gqa=True)
        o = o.transpose(1, 2).reshape(1, L, -1)
        h = residual + attn.o_proj(o)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        if output_hidden_states:
            layer_hiddens.append(h.squeeze(0))

    h = model.norm(h).squeeze(0)  # [L, D]
    if output_hidden_states:
        return h, tuple(layer_hiddens)
    return h
