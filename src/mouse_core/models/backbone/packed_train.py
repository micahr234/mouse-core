"""Packed variable-length training forward for transformer backbones.

:func:`packed_forward` runs a full (uncached) forward over the flat
concatenated token stream ``embeds [L, D]``. Attention is causal within
positions that share both ``sequence_ids`` and ``grouping_ids``::

    allowed(q, k) = k <= q and sequence_ids[k] == sequence_ids[q]
                           and grouping_ids[k] == grouping_ids[q]

Groups are equality classes of ``(sequence_id, grouping_id)``, not contiguous
runs: an id that recurs later in the stream attends back to its earlier
occurrence and continues its RoPE counter. The stream is stably regrouped
so every class is one contiguous packed segment (original order preserved
inside each class), all decoder layers run in that order, and the outputs
are restored to the original order before returning. Causal attention over
the packed segments is then exactly the predicate above.

Three attention kernels run the same packed segments; ``train_kernel``
(``Backbone.train_kernel``, a required constructor argument, / the
``packed_forward`` argument) selects one so they can be compared on equal
terms. Everything else — the regrouping, RoPE positions, decoder body,
compiled body, gradient checkpointing, the hidden-state contract — is shared.

- ``"varlen"``. CUDA bf16 / fp16 (the frozen-base LoRA configuration):
  :func:`torch.nn.attention.varlen.varlen_attn`, PyTorch's FlashAttention
  varlen kernel — no mask tensor, grouped KV heads read directly, O(L)
  memory. CPU, or CUDA fp32: the same decoder body with
  :func:`torch.nn.functional.scaled_dot_product_attention` over a
  block-causal boolean mask built once per forward — the fp32 reference
  mode (O(L^2) memory), not a fallback.
- ``"padded"``. Each packed segment is right-padded to ``max_seqlen`` and
  run as a dense causal SDPA call ``[n_seg, H, S, Dh]`` (Flash when the
  backend picks it). Pad keys are masked; attention is not truncated.
  Cost tracks ``n_seg * S^2``, so a few long groups beat ``"varlen"``'s
  packed-stream ``L^2`` fallback, and many short groups waste pad.
- ``"flex"``. :func:`torch.nn.attention.flex_attention` with a block-sparse
  mask over the packed segments (128-token blocks; masked blocks are
  skipped). Compiled on CUDA in every dtype — inside the compiled decoder
  body when :func:`install_compiled_decoder` is active, otherwise as a
  compiled kernel of its own — and unfused (scores materialized) on CPU,
  where it is forward-only. The kernel of choice for full fp32 fine-tuning.

:func:`install_compiled_decoder` compiles the per-layer train decoder body
once (``torch.compile(dynamic=True)``) and, on CUDA, the cached-decode
per-layer body as well. The same compiled train function serves every
layer, stream length, group count and kernel. ``Backbone.gradient_checkpointing``
recomputes each layer in backward instead of storing its activations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, cast, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.nn.attention.varlen import varlen_attn
from torch.utils.checkpoint import checkpoint as _checkpoint
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from mouse_core.models.backbone.flex_decode import (
    _BLOCK_SIZE,
    _FlexKernel,
    _use_flex_compile,
    flex_block_mask,
    install_compiled_decode_layer,
    module_device_dtype,
)

TrainKernel = Literal["varlen", "flex", "padded"]


def check_train_kernel(kernel: object) -> TrainKernel:
    """Return ``kernel`` if it names a training attention kernel, else raise ``ValueError``."""
    if kernel not in get_args(TrainKernel):
        raise ValueError(f"train_kernel must be one of {get_args(TrainKernel)}, got {kernel!r}.")
    return cast(TrainKernel, kernel)


_FLASH_DTYPES = frozenset({torch.bfloat16, torch.float16})

_compiled_layer: Any | None = None
_flex_kernels: dict[tuple[str, int | None], _FlexKernel] = {}
_warned_reference_cuda: set[str] = set()


@dataclass(frozen=True)
class _PackingPlan:
    """Stable regrouping of a flat stream by ``(sequence_id, grouping_id)``.

    Packed token ``i`` is original token ``order[i]``;
    ``original = packed[inverse]``. ``cu_seqlens`` are the int32 segment
    boundaries the varlen kernel consumes, ``max_seqlen`` the largest class
    size, and ``position_ids`` the RoPE position of every packed token (its
    index inside its class).
    """

    order: torch.Tensor
    inverse: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    position_ids: torch.Tensor


def _packing_plan(sequence_ids: torch.Tensor, grouping_ids: torch.Tensor) -> _PackingPlan:
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    seq = sequence_ids.to(torch.long)
    grp = grouping_ids.to(torch.long)
    # Lexicographic stable sort by (sequence, grouping) as two stable
    # argsorts: collision-free for any id values, no composite key to overflow.
    by_group = torch.argsort(grp, stable=True)
    order = by_group[torch.argsort(seq[by_group], stable=True)]
    packed_seq = seq[order]
    packed_grp = grp[order]
    arange = torch.arange(L, device=device)
    is_start = torch.ones(L, dtype=torch.bool, device=device)
    is_start[1:] = (packed_seq[1:] != packed_seq[:-1]) | (packed_grp[1:] != packed_grp[:-1])
    # Two host syncs per forward (segment count and longest segment); both
    # are inputs the kernel needs as Python ints / a sized tensor.
    starts = arange[is_start]
    cu_seqlens = torch.cat([starts, arange.new_tensor([L])]).to(torch.int32)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    segment_start = torch.cummax(torch.where(is_start, arange, arange.new_tensor(-1)), dim=0).values
    inverse = torch.empty_like(order)
    inverse[order] = arange
    return _PackingPlan(
        order=order,
        inverse=inverse,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        position_ids=arange - segment_start,
    )


def _segment_ids(plan: _PackingPlan) -> torch.Tensor:
    """``[L]`` packed-segment index of every packed token."""
    lengths = (plan.cu_seqlens[1:] - plan.cu_seqlens[:-1]).to(torch.long)
    return torch.repeat_interleave(torch.arange(lengths.shape[0], device=lengths.device), lengths)


def _block_causal_mask(plan: _PackingPlan) -> torch.Tensor:
    """``[L, L]`` boolean mask over the packed order: causal within each segment."""
    L = plan.order.shape[0]
    arange = torch.arange(L, device=plan.order.device)
    segment = _segment_ids(plan)
    return (segment[:, None] == segment[None, :]) & (arange[None, :] <= arange[:, None])


# Stable mask_mod identity (reads the current stream's segment ids from a
# holder) so the compiled ``create_block_mask`` is traced once, not per call.
_segment_holder: dict[str, torch.Tensor] = {}


def _segment_mask_mod(b, h, q_idx, kv_idx):
    seg = _segment_holder["segment"]
    return (kv_idx <= q_idx) & (seg[q_idx] == seg[kv_idx])


def _flex_block_mask(plan: _PackingPlan, device: torch.device) -> BlockMask:
    """Block-sparse Flex mask over the packed order: causal within each segment."""
    _segment_holder["segment"] = _segment_ids(plan)
    L = plan.order.shape[0]
    return flex_block_mask(
        _segment_mask_mod,
        B=1,
        Q_LEN=L,
        KV_LEN=L,
        device=device,
        block_size=_BLOCK_SIZE,
        compile_masks=_use_flex_compile(device),
    )


def _pad_packed(x: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """``x [L, H, D]`` → right-padded ``[n_seg, H, S, D]``."""
    L, h, d = x.shape
    n_seg = cu_seqlens.numel() - 1
    idx = torch.arange(L, device=x.device)
    ends = cu_seqlens[1:].to(dtype=idx.dtype)
    seg = torch.searchsorted(ends, idx, right=True)
    pos = idx - cu_seqlens.to(dtype=idx.dtype)[seg]
    out = x.new_zeros(n_seg, h, max_seqlen, d)
    out[seg, :, pos] = x
    return out


def _unpad_packed(x: torch.Tensor, cu_seqlens: torch.Tensor, length: int) -> torch.Tensor:
    """``x [n_seg, H, S, D]`` → packed ``[L, H, D]``."""
    idx = torch.arange(length, device=x.device)
    ends = cu_seqlens[1:].to(dtype=idx.dtype)
    seg = torch.searchsorted(ends, idx, right=True)
    pos = idx - cu_seqlens.to(dtype=idx.dtype)[seg]
    return x[seg, :, pos]


def _padded_causal_mask(cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """``[n_seg, 1, S, S]`` bool: causal within each segment, pad keys dropped."""
    device = cu_seqlens.device
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
    t = torch.arange(max_seqlen, device=device)
    valid = t.unsqueeze(0) < lengths.unsqueeze(1)
    causal = t[:, None] >= t[None, :]
    return (valid.unsqueeze(2) & valid.unsqueeze(1) & causal).unsqueeze(1)


def _flex_fn(device: torch.device) -> Callable[..., torch.Tensor]:
    """Flex kernel for the eager body: compiled on CUDA, eager on CPU.

    Inside the compiled decoder body the plain ``flex_attention`` is lowered
    with the rest of the layer, so no separate kernel is needed there.
    """
    if _compiled_layer is not None:
        return flex_attention
    key = (device.type, device.index)
    kern = _flex_kernels.get(key)
    if kern is None:
        kern = _flex_kernels[key] = _FlexKernel(device)
    return kern


def _decoder_layer(
    layer: Any,
    h: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    attn_mask: torch.Tensor | None,
    block_mask: BlockMask | None,
    flex_fn: Callable[..., torch.Tensor] | None,
    padded: bool,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """One Llama / Qwen3 decoder layer over a packed stream ``h [L, D]``.

    ``block_mask`` selects FlexAttention; ``padded`` selects rectangular
    causal SDPA; otherwise ``attn_mask is None`` selects flash varlen and a
    mask selects packed-stream SDPA. Compiled by :func:`install_compiled_decoder`.
    """
    L = h.shape[0]
    attn = layer.self_attn
    residual = h
    hn = layer.input_layernorm(h)
    q = attn.q_proj(hn).view(L, n_heads, head_dim)
    k = attn.k_proj(hn).view(L, n_kv_heads, head_dim)
    v = attn.v_proj(hn).view(L, n_kv_heads, head_dim)
    q_norm = getattr(attn, "q_norm", None)  # Qwen3 normalizes q/k before RoPE; Llama does not
    if q_norm is not None:
        q = q_norm(q)
        k = attn.k_norm(k)
    # HF RoPE takes [B, H, L, Dh].
    q, k = apply_rotary_pos_emb(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), cos, sin
    )
    gqa = n_heads != n_kv_heads
    if block_mask is not None:
        assert flex_fn is not None
        # Q_LEN == KV_LEN here, so Inductor's short-query "decoding" kernel is
        # never the right choice; with dynamic shapes it also has no valid
        # autotune configs for no-grad forwards under 128 tokens. Force the
        # main kernel.
        o = flex_fn(
            q,
            k,
            v.transpose(0, 1).unsqueeze(0),
            block_mask=block_mask,
            scale=attn.scaling,
            enable_gqa=gqa,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )[0].transpose(0, 1)
    elif padded:
        q_p = _pad_packed(q[0].transpose(0, 1), cu_seqlens, max_seqlen)
        k_p = _pad_packed(k[0].transpose(0, 1), cu_seqlens, max_seqlen)
        v_p = _pad_packed(v, cu_seqlens, max_seqlen)
        o = _unpad_packed(
            F.scaled_dot_product_attention(
                q_p,
                k_p,
                v_p,
                attn_mask=attn_mask,
                is_causal=attn_mask is None,
                scale=attn.scaling,
                enable_gqa=gqa,
            ),
            cu_seqlens,
            L,
        )
    elif attn_mask is None:
        o = cast(
            torch.Tensor,
            varlen_attn(
                q[0].transpose(0, 1),
                k[0].transpose(0, 1),
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                scale=attn.scaling,
                window_size=(-1, 0),
                enable_gqa=gqa,
            ),
        )
    else:
        o = F.scaled_dot_product_attention(
            q,
            k,
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=attn_mask,
            scale=attn.scaling,
            enable_gqa=gqa,
        )[0].transpose(0, 1)
    h = residual + attn.o_proj(o.reshape(L, n_heads * head_dim))
    return h + layer.mlp(layer.post_attention_layernorm(h))


def install_compiled_decoder() -> bool:
    """Compile the per-layer train and (on CUDA) decode decoder bodies.

    One ``torch.compile(dynamic=True)`` function serves every train layer of
    every backbone in the process, any stream length and group count, and
    ``output_hidden_states=True``. A second compiled function serves cached
    decode. Idempotent; returns True if anything was installed.
    """
    global _compiled_layer
    installed = False
    if _compiled_layer is None:
        _compiled_layer = torch.compile(_decoder_layer, dynamic=True)
        installed = True
    if install_compiled_decode_layer():
        installed = True
    return installed


@overload
def packed_forward(
    *,
    model: torch.nn.Module,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
    train_kernel: TrainKernel,
    output_hidden_states: Literal[False] = False,
    checkpoint: bool = False,
) -> torch.Tensor: ...


@overload
def packed_forward(
    *,
    model: torch.nn.Module,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
    train_kernel: TrainKernel,
    output_hidden_states: Literal[True],
    checkpoint: bool = False,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]: ...


@overload
def packed_forward(
    *,
    model: torch.nn.Module,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
    train_kernel: TrainKernel,
    output_hidden_states: bool,
    checkpoint: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]: ...


def packed_forward(
    *,
    model: torch.nn.Module,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
    train_kernel: TrainKernel,
    output_hidden_states: bool = False,
    checkpoint: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Full uncached forward over a flat packed stream ``embeds [L, D]``.

    Args:
        model: A ``transformers`` decoder stack (``Qwen3Model`` /
            ``LlamaModel``) with ``layers``, ``rotary_emb``, ``norm``.
        embeds: Token embeddings ``[L, D]`` in stream order; cast to the
            model's base dtype.
        sequence_ids: ``[L]`` sequence id per token.
        grouping_ids: ``[L]`` grouping id per token (any integer values).
        train_kernel: ``"varlen"`` (flash varlen on CUDA bf16/fp16,
            masked SDPA otherwise), ``"padded"`` (dense causal SDPA on
            segments padded to ``max_seqlen``), or ``"flex"`` (FlexAttention
            block mask, compiled on CUDA); see the module docstring.
            Required so the kernel is always an explicit choice; ``Model``
            passes ``Backbone.train_kernel``.
        output_hidden_states: Also return every layer's output (before the
            final norm) for layerwise heads.
        checkpoint: Recompute each layer in backward instead of storing its
            activations (``Backbone.gradient_checkpointing``).

    Returns:
        Final normalized hidden states ``[L, D]`` in the original stream
        order, plus the per-layer tuple when ``output_hidden_states``.
    """
    if embeds.ndim != 2:
        raise ValueError(f"embeds must be [L, D], got shape {tuple(embeds.shape)}")
    L, _D = embeds.shape
    if sequence_ids.shape != (L,):
        raise ValueError("sequence_ids must have shape [L]")
    if grouping_ids.shape != (L,):
        raise ValueError("grouping_ids must have shape [L]")
    check_train_kernel(train_kernel)

    hf = cast(Any, model)  # HF stacks are nn.Module; pyright sees children as Tensor|Module
    if getattr(hf.config, "use_sliding_window", False):
        raise ValueError("packed_forward does not support sliding-window attention.")
    device, dtype = module_device_dtype(hf)
    x = embeds.to(device=device, dtype=dtype)
    n_layers = len(hf.layers)

    if L == 0:
        out = hf.norm(x)
        return (out, (x,) * n_layers) if output_hidden_states else out

    plan = _packing_plan(sequence_ids.to(device), grouping_ids.to(device))
    fused = device.type == "cuda" and dtype in _FLASH_DTYPES
    attn_mask: torch.Tensor | None = None
    block_mask: BlockMask | None = None
    flex_fn: Callable[..., torch.Tensor] | None = None
    padded = train_kernel == "padded"
    if train_kernel == "flex":
        block_mask = _flex_block_mask(plan, device)
        flex_fn = _flex_fn(device)
    elif train_kernel == "padded":
        lengths = plan.cu_seqlens[1:] - plan.cu_seqlens[:-1]
        if not bool((lengths == plan.max_seqlen).all()):
            attn_mask = _padded_causal_mask(plan.cu_seqlens, plan.max_seqlen)
    elif train_kernel == "varlen" and not fused:
        attn_mask = _block_causal_mask(plan)
        if device.type == "cuda" and not _warned_reference_cuda:
            warnings.warn(
                f"packed_forward is running the fp32 reference attention (masked SDPA, O(L^2) "
                f"memory) because the backbone dtype is {dtype} and train_kernel is "
                '"varlen". For fp32 training prefer train_kernel="flex" (block-sparse, '
                "compiled on CUDA); the flash varlen kernel needs a bf16/fp16 backbone.",
                stacklevel=2,
            )
            _warned_reference_cuda.add(train_kernel)

    cfg = hf.config
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(cfg.head_dim)

    h = x[plan.order]
    cos, sin = hf.rotary_emb(h.unsqueeze(0), plan.position_ids.unsqueeze(0))
    body = _compiled_layer if _compiled_layer is not None else _decoder_layer
    recompute = checkpoint and torch.is_grad_enabled()
    layer_hiddens: list[torch.Tensor] = []
    for layer in hf.layers:
        args = (
            layer, h, cos, sin, plan.cu_seqlens, plan.max_seqlen,
            attn_mask, block_mask, flex_fn, padded, n_heads, n_kv_heads, head_dim,
        )
        h = _checkpoint(body, *args, use_reentrant=False) if recompute else body(*args)
        if output_hidden_states:
            layer_hiddens.append(h[plan.inverse])
    out = hf.norm(h)[plan.inverse]
    if output_hidden_states:
        return out, tuple(layer_hiddens)
    return out
