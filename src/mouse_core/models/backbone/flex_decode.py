"""FlexAttention cached decoding for transformer backbones.

:class:`FlexDecodeSession` is the cached-decode engine behind
``Model.forward(batch, cache=..., use_cache=True)``. It decodes a batch of
independent sequences incrementally, where on **every** call each sequence may
contribute any number of new steps — including zero. There is no lockstep or
uniform-length assumption anywhere.

How it works:

* Inputs arrive left-padded to the call's longest row (``Model.forward`` pads
  ragged batches for the encoder); the session receives the padded token
  embeddings plus each row's *real* token count.
* K/V for real tokens are written into a per-sequence region of a batched
  cache buffer ``[layers, B, kv_heads, capacity, head_dim]`` at that
  sequence's own length offset. Pad tokens are never written.
* Attention runs through :func:`torch.nn.attention.flex_attention` with a
  BlockMask that keeps each query inside its own sequence's causal prefix.
  Masked blocks are *skipped*, not computed-and-discarded, so each sequence's
  decode cost scales with its own history rather than the batch maximum.
* RoPE positions are per-sequence token counters, so every row decodes exactly
  as it would alone with a private cache (pinned by ``tests/test_kv_cache.py``).

The session wraps the backbone's ``transformers`` model in place (shared
weights, decoder loop reimplemented) and supports both ``Qwen3Model`` and
``LlamaModel`` layer layouts (Qwen3 adds q/k RMSNorm). Sessions are
inference-only (``torch.no_grad``): one session per rollout segment, dropped
when the segment ends.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Identical implementations; either import works for both architectures.
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

_BLOCK_SIZE = 128  # compiled CUDA FlexAttention requires >= 128
_COMPILED_DTYPES = frozenset({torch.bfloat16, torch.float16})


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _use_flex_compile(device: torch.device, dtype: torch.dtype) -> bool:
    """Compiled FlexAttention kernels are CUDA-only and bf16/fp16-only in practice."""
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
            # fp32 CUDA compile can fail on ragged prefills (InductorError); eager works.
            if self._compiled is not None and self._active is self._compiled:
                name = type(exc).__name__
                if name in {"InductorError", "LoweringException"} or "InductorError" in name:
                    self._active = self._eager
                    return self._eager(*args, **kwargs)
            raise


class FlexDecodeSession:
    """Incremental decoder over ``batch_size`` independently-growing sequences.

    Create via ``backbone.decode_session(batch_size)``; ``Model.forward``
    does this automatically and carries the session inside its ``cache`` dict.

    Args:
        model: A ``transformers`` decoder stack (``Qwen3Model`` / ``LlamaModel``)
            with ``layers``, ``rotary_emb``, and ``norm`` attributes. Used in
            place; not modified.
        batch_size: Number of sequences decoded by this session.
        capacity: Initial per-sequence KV slots (rounded up to the attention
            block size). Grows automatically when a sequence outruns it.
    """

    def __init__(self, model: torch.nn.Module, batch_size: int, capacity: int = _BLOCK_SIZE) -> None:
        if getattr(model.config, "use_sliding_window", False):
            raise ValueError("FlexDecodeSession does not support sliding-window attention.")
        self.model = model
        cfg = model.config
        self.B = batch_size
        self.cap = _round_up(max(capacity, 1), _BLOCK_SIZE)
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        param = next(model.parameters())
        self.device, self.dtype = param.device, param.dtype
        self._flex = _FlexKernel(self.device, self.dtype)
        self._compile_masks = _use_flex_compile(self.device, self.dtype)

        n_layers = len(model.layers)
        self.k_cache = torch.zeros(
            n_layers, self.B, self.n_kv_heads, self.cap, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.lengths = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # Stable mask_mod identity (reads a per-call position table) lets
        # torch.compile reuse the traced mask across calls instead of
        # re-guarding on fresh closures.
        #
        # Important: close over a dict holder, not ``self``. A ``mask_mod`` that
        # captures ``self`` creates a reference cycle (session → mask_mod →
        # session). Cyclic GC may not run between rollout and train, so the KV
        # buffers stay allocated and online training OOMs after a few cycles.
        q_pos_holder: dict[str, torch.Tensor] = {
            "t": torch.zeros(0, 0, dtype=torch.long, device=self.device)
        }
        self._q_pos_holder = q_pos_holder

        def mask_mod(b, h, q_idx, kv_idx):
            # Causal within each sequence, offset by its cached history.
            # Pad queries carry a clamped position (a prefix of real slots),
            # so they stay finite; their K/V are never written and their
            # outputs are discarded by the caller.
            return kv_idx <= q_pos_holder["t"][b, q_idx]

        self._mask_mod = mask_mod

    # ------------------------------------------------------------------

    def _grow(self, needed: int) -> None:
        new_cap = _round_up(max(needed, 2 * self.cap), _BLOCK_SIZE)
        for name in ("k_cache", "v_cache"):
            old = getattr(self, name)
            new = torch.zeros(
                *old.shape[:3], new_cap, self.head_dim, device=self.device, dtype=self.dtype
            )
            new[..., : self.cap, :] = old
            setattr(self, name, new)
        self.cap = new_cap

    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        embeds: torch.Tensor,
        lengths: list[int],
        *,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Decode one chunk per sequence.

        Args:
            embeds: Left-padded token embeddings ``[B, S, D]``; row ``b``'s
                real tokens are the trailing ``lengths[b]`` positions. ``S``
                is this call's longest row — unrelated to any other call.
            lengths: Real token count per row, ``0 <= lengths[b] <= S``.
            output_hidden_states: Also return every layer's hidden states
                (for layerwise heads).

        Returns:
            Hidden states ``[B, S, D]`` (values at pad positions are
            meaningless), plus a tuple of per-layer hidden states when
            ``output_hidden_states=True``.
        """
        B, S, _ = embeds.shape
        if B != self.B:
            raise ValueError(f"Session was created for batch_size={self.B}, got {B}.")
        n = torch.tensor(lengths, dtype=torch.long, device=self.device)
        if int((self.lengths + n).max()) > self.cap:
            self._grow(int((self.lengths + n).max()))

        x = embeds.to(self.device, self.dtype)
        pad = (S - n)[:, None]  # leading pad tokens per row
        col = torch.arange(S, device=self.device)[None]

        # Per-token position within its own sequence: len_before + (col - pad).
        # Pad columns get earlier/negative values; clamp keeps RoPE and the
        # mask finite (pad outputs are discarded by the caller either way).
        pos = self.lengths[:, None] + col - pad
        self._q_pos_holder["t"] = pos.clamp_min(0)

        # Real tokens are the trailing lengths[b] columns; only they are
        # written to the cache, at their own sequence's slots.
        real_rows, real_cols = (col >= pad).nonzero(as_tuple=True)
        cache_slots = pos[real_rows, real_cols]

        block_mask = create_block_mask(
            self._mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=self.cap,
            device=str(self.device), BLOCK_SIZE=_BLOCK_SIZE,
            _compile=self._compile_masks,
        )

        cos, sin = self.model.rotary_emb(x, pos.clamp_min(0))

        h = x
        layer_hiddens: list[torch.Tensor] = []
        for li, layer in enumerate(self.model.layers):
            residual = h
            hn = layer.input_layernorm(h)
            attn = layer.self_attn
            q = attn.q_proj(hn).view(B, S, self.n_heads, self.head_dim)
            k = attn.k_proj(hn).view(B, S, self.n_kv_heads, self.head_dim)
            q_norm = getattr(attn, "q_norm", None)  # Qwen3 yes, Llama no
            if q_norm is not None:
                q = q_norm(q)
                k = attn.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = attn.v_proj(hn).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            self.k_cache[li][real_rows, :, cache_slots] = k[real_rows, :, real_cols]
            self.v_cache[li][real_rows, :, cache_slots] = v[real_rows, :, real_cols]

            o = self._flex(
                q, self.k_cache[li], self.v_cache[li],
                block_mask=block_mask, scale=attn.scaling, enable_gqa=True,
            )
            o = o.transpose(1, 2).reshape(B, S, -1)
            h = residual + attn.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            if output_hidden_states:
                layer_hiddens.append(h)

        h = self.model.norm(h)
        self.lengths = self.lengths + n
        if output_hidden_states:
            return h, tuple(layer_hiddens)
        return h
