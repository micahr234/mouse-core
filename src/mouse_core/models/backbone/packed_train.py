"""Packed variable-length training forward for transformer backbones.

:func:`packed_forward` runs a full (uncached) forward over the flat
concatenated token stream ``embeds [L, D]``. Attention is causal within
one ``group_id`` (one dataloader sample)::

    allowed(q, k) = k <= q and sequence_ids[k] == sequence_ids[q]

Groups are equality classes of ``group_id``, not necessarily
contiguous runs: an id that recurs later in the stream attends back to
its earlier occurrence and continues its RoPE counter. The stream is
stably regrouped so every class is one contiguous packed segment
(original order preserved inside each class), all decoder layers run in
that order, and the outputs are restored to the original order before
returning. Causal attention over the packed segments is then exactly
the predicate above.

FlexAttention is the only attention implementation. It uses a block-sparse
mask over packed segments, compiled on CUDA and eager on CPU. The installed
PyTorch CPU implementation supports inference only; training requires CUDA.
There is no SDPA or FlashAttention varlen fallback.

Mixed precision is declared at construction, never inferred:
``train_autocast_dtype=torch.bfloat16`` (or ``torch.float16``) on a fp32 backbone
wraps the decoder stack in :func:`torch.autocast` — matmuls run in the
autocast dtype while weights, norms, RoPE, and the residual stream stay
fp32. ``decode_autocast_dtype`` is the separate declaration for cached decode
(``ContinuousDecodeSession`` runs its layers under it and allocates the KV pool
in that dtype), so bf16 train can pair with exact fp32 eval decode.
``packed_forward`` raises if it is called inside an ambient
``torch.autocast`` region, so the declared precision is always the actual
precision.

:func:`install_compiled_decoder` compiles the per-layer train decoder body
once (``torch.compile(dynamic=True)``) and, on CUDA, the full cached-decode
stack as well. The same compiled train function serves every layer, stream
length, and group count. ``Backbone.gradient_checkpointing``
recomputes each layer in backward instead of storing its activations.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, cast, overload

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from torch.utils.checkpoint import checkpoint as _checkpoint
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from mouse_core.models.backbone.flex_utils import (
    _BLOCK_SIZE,
    _FlexKernel,
    _use_flex_compile,
    check_autocast_dtype,
    flex_block_mask,
    module_device_dtype,
)

_compiled_layer: Any | None = None
_flex_kernels: dict[tuple[str, int | None], _FlexKernel] = {}


@dataclass(frozen=True)
class _PackingPlan:
    """Stable regrouping of a flat stream by ``group_id``.

    Packed token ``i`` is original token ``order[i]``;
    ``original = packed[inverse]``. ``sequence_ids`` identify the packed
    groups directly, and ``position_ids`` give the RoPE position
    of every packed token (its index inside its class).
    """

    order: torch.Tensor
    inverse: torch.Tensor
    sequence_ids: torch.Tensor
    position_ids: torch.Tensor


def _packing_plan(sequence_ids: torch.Tensor) -> _PackingPlan:
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    seq = sequence_ids.to(torch.long)
    order = torch.argsort(seq, stable=True)
    packed_seq = seq[order]
    arange = torch.arange(L, device=device)
    is_start = torch.ones(L, dtype=torch.bool, device=device)
    is_start[1:] = packed_seq[1:] != packed_seq[:-1]
    segment_start = torch.cummax(torch.where(is_start, arange, arange.new_tensor(-1)), dim=0).values
    inverse = torch.empty_like(order)
    inverse[order] = arange
    return _PackingPlan(
        order=order,
        inverse=inverse,
        sequence_ids=packed_seq,
        position_ids=arange - segment_start,
    )


# Stable mask_mod identity (reads the current stream's segment ids from a
# holder) so the compiled ``create_block_mask`` is traced once, not per call.
_segment_holder: dict[str, torch.Tensor] = {}
_segment_lock = threading.Lock()


def _segment_mask_mod(b, h, q_idx, kv_idx):
    seg = _segment_holder["segment"]
    return (kv_idx <= q_idx) & (seg[q_idx] == seg[kv_idx])


def _run_layer_rebound(
    body: Callable[..., torch.Tensor],
    segment: torch.Tensor,
    *args: object,
) -> torch.Tensor:
    """Run one decoder layer with its segment IDs.

    Gradient checkpointing re-executes this during ``backward()``. By then a
    later ``packed_forward`` (reasoner generation passes, gradient
    accumulation, a second model on another batch) may have replaced
    ``_segment_holder["segment"]``, so the recompute must rebind the ids this
    forward's BlockMask was built from before the kernel reads them.
    """
    with _segment_lock:
        _segment_holder["segment"] = segment
    return body(*args)


def _flex_block_mask(segment: torch.Tensor, L: int, device: torch.device) -> BlockMask:
    """Block-sparse causal mask within complete episodes."""
    with _segment_lock:
        _segment_holder["segment"] = segment
        return flex_block_mask(
            _segment_mask_mod,
            B=1,
            Q_LEN=L,
            KV_LEN=L,
            device=device,
            block_size=_BLOCK_SIZE,
            compile_masks=_use_flex_compile(device),
        )


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


def _attention_inputs(
    layer: Any, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    n_heads: int, n_kv_heads: int, head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared Q/K/V projections and RoPE for packed training and HF decoding."""
    L = h.shape[0]
    attn = layer.self_attn
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
    if q.dtype != v.dtype:
        # autocast_dtype over fp32 weights: the projections emit the autocast
        # dtype but the fp32 RoPE cos/sin promote q/k back to fp32.
        # Attention inputs must have the same dtype, so
        # follow v (the projection output dtype). RoPE itself stays fp32.
        q = q.to(v.dtype)
        k = k.to(v.dtype)
    return q, k, v.transpose(0, 1).unsqueeze(0)


def _attention_output(
    layer: Any, h: torch.Tensor, attention_output: torch.Tensor,
) -> torch.Tensor:
    """Shared attention residual and feed-forward block."""
    o = attention_output[0].transpose(0, 1).flatten(1)
    h = h + layer.self_attn.o_proj(o)
    return h + layer.mlp(layer.post_attention_layernorm(h))


def _decoder_layer(
    layer: Any,
    h: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    block_mask: BlockMask,
    flex_fn: Callable[..., torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """One packed-compatible decoder layer, shared with cached inference."""
    q, k, v = _attention_inputs(layer, h, cos, sin, n_heads, n_kv_heads, head_dim)
    output = flex_fn(
        q, k, v, block_mask=block_mask, scale=layer.self_attn.scaling,
        enable_gqa=n_heads != n_kv_heads,
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
    )
    return _attention_output(layer, h, output)


def install_compiled_decoder() -> bool:
    """Compile the per-layer training body; inference compiles when a session opens.

    Idempotent; returns True when a compiled training body is installed.
    """
    global _compiled_layer
    installed = False
    if _compiled_layer is None:
        _compiled_layer = torch.compile(_decoder_layer, dynamic=True)
        installed = True
    return installed


class _DecoderStack(Protocol):
    """Llama / Qwen3 decoder ``Backbone.model`` points at.

    Read-only properties keep ``norm`` and ``layers`` covariant, so the
    concrete output module and ``ModuleList`` match. A generic HuggingFace module
    is stored with a cast.
    """

    @property
    def config(self) -> Any: ...

    @property
    def layers(self) -> nn.ModuleList: ...

    @property
    def norm(self) -> nn.Module: ...

    def get_input_embeddings(self) -> nn.Module: ...

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]: ...

    def modules(self) -> Iterator[nn.Module]: ...

    def named_modules(
        self,
        memo: set[nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Module]]: ...

    def named_children(self) -> Iterator[tuple[str, nn.Module]]: ...

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> Any: ...


_PackedModel = nn.Module | _DecoderStack


@overload
def packed_forward(
    *,
    model: _PackedModel,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    autocast_dtype: torch.dtype | None = None,
    output_hidden_states: Literal[False] = False,
    checkpoint: bool = False,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor: ...


@overload
def packed_forward(
    *,
    model: _PackedModel,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    autocast_dtype: torch.dtype | None = None,
    output_hidden_states: Literal[True],
    checkpoint: bool = False,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]: ...


@overload
def packed_forward(
    *,
    model: _PackedModel,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    autocast_dtype: torch.dtype | None = None,
    output_hidden_states: bool,
    checkpoint: bool = False,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]: ...


def packed_forward(
    *,
    model: _PackedModel,
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    autocast_dtype: torch.dtype | None = None,
    output_hidden_states: bool = False,
    checkpoint: bool = False,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Full uncached forward over a flat packed stream ``embeds [L, D]``.

    Always uses FlexAttention. CUDA supports training; CPU supports inference
    under ``torch.no_grad()`` only.

    Args:
        model: A ``transformers`` decoder stack (``Qwen3Model`` /
            ``LlamaModel``) with ``layers``, ``rotary_emb``, ``norm``.
        embeds: Token embeddings ``[L, D]`` in stream order; cast to the
            model's base dtype.
        sequence_ids: ``[L]`` group id per token. One group is one sample.
        position_ids: Optional explicit ``[L]`` rotary positions in stream order.
        autocast_dtype: ``None`` runs everything in the model's base dtype.
            ``torch.bfloat16`` / ``torch.float16`` (fp32 or matching base weights)
            wraps the decoder stack in ``torch.autocast``: matmuls run in this
            dtype, weights / norms / RoPE / residuals stay fp32. ``Model``
            passes ``Backbone.train_autocast_dtype``. Calling ``packed_forward``
            inside an ambient ``torch.autocast`` region raises.
        output_hidden_states: Also return every layer's output (before the
            stack's output module, which is identity for MOUSE backbones).
        checkpoint: Recompute each layer in backward instead of storing its
            activations (``Backbone.gradient_checkpointing``).

    Returns:
        Final hidden states ``[L, D]`` in the original stream
        order, plus the per-layer tuple when ``output_hidden_states``.
    """
    if embeds.ndim != 2:
        raise ValueError(f"embeds must be [L, D], got shape {tuple(embeds.shape)}")
    L, _D = embeds.shape
    if sequence_ids.shape != (L,):
        raise ValueError("sequence_ids must have shape [L]")

    hf = cast(Any, model)  # HF stacks are nn.Module; pyright sees children as Tensor|Module
    device, dtype = module_device_dtype(hf)
    check_autocast_dtype(autocast_dtype, dtype)
    if torch.is_autocast_enabled(device.type):
        raise RuntimeError(
            "packed_forward was called inside an active torch.autocast region. "
            "Mixed precision is declared at backbone construction "
            "(train_autocast_dtype= / decode_autocast_dtype=), never inferred from "
            "ambient state; remove the surrounding torch.autocast."
        )
    if position_ids is not None and position_ids.shape != (L,):
        raise ValueError("position_ids must have shape [L]")
    x = embeds.to(device=device, dtype=dtype)
    n_layers = len(hf.layers)

    if L == 0:
        out = hf.norm(x)
        return (out, (x,) * n_layers) if output_hidden_states else out

    plan = _packing_plan(sequence_ids.to(device))
    flex_segment = plan.sequence_ids
    block_mask = _flex_block_mask(flex_segment, L, device)
    flex_fn = _flex_fn(device)

    cfg = hf.config
    n_heads = int(cfg.num_attention_heads)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", n_heads))
    head_dim = getattr(cfg, "head_dim", None)
    head_dim = int(head_dim) if head_dim is not None else int(cfg.hidden_size) // n_heads

    h = x[plan.order]
    # RoPE cos/sin are computed outside autocast and stay fp32 alongside the
    # residual stream; ``_decoder_layer`` casts q/k to the compute dtype.
    positions = plan.position_ids if position_ids is None else position_ids.to(device)[plan.order]
    cos, sin = hf.rotary_emb(h.unsqueeze(0), positions.unsqueeze(0))
    body = _compiled_layer if _compiled_layer is not None else _decoder_layer
    recompute = checkpoint and torch.is_grad_enabled()
    layer_hiddens: list[torch.Tensor] = []
    run_ctx = (
        nullcontext() if autocast_dtype is None else torch.autocast(device.type, dtype=autocast_dtype)
    )
    with run_ctx:
        for layer in hf.layers:
            args = (
                layer, h, cos, sin, block_mask, flex_fn, n_heads, n_kv_heads, head_dim,
            )
            h = (
                _checkpoint(_run_layer_rebound, body, flex_segment, *args, use_reentrant=False)
                if recompute
                else body(*args)
            )
            if output_hidden_states:
                layer_hiddens.append(h[plan.inverse])
        out = hf.norm(h)[plan.inverse]
    if output_hidden_states:
        return out, tuple(layer_hiddens)
    return out
