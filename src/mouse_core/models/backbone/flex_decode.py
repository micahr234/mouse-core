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
* K/V live in a **paged pool** shared by the whole batch:
  ``[layers, kv_heads, n_pages * 128, head_dim]``. Each sequence owns
  ``ceil(len / 128)`` pages (at least one), mapped through a page table from
  its logical slot ``0..len`` to a physical slot in the pool, so a batch with
  one long row and many short rows costs the sum of the rows' lengths, not
  ``B * max_len``. Pad tokens are never written. The pool doubles when it
  runs out of free pages; :meth:`reset_rows` returns a row's pages to the
  pool.
* Attention runs through :func:`torch.nn.attention.flex_attention` with a
  BlockMask that keeps each query inside its own sequence's causal prefix
  **and** the same grouping-id run (``grouping_ids``). The mask is built in
  logical coordinates (cheap: ``[B, S, max_len]``) and its block indices are
  remapped through the page table, so the kernel only visits the pages a row
  owns; masked blocks are *skipped*, not computed-and-discarded, and each
  sequence's decode cost scales with its own history rather than the batch
  maximum.
* RoPE positions are per-``(sequence, grouping_id)`` token counters, so a new
  grouping id starts at position 0 even if other-id KV slots remain (those
  slots are masked out), and an id that recurs later continues its own
  counter. This is the same rule the uncached train paths use
  (:func:`packed_rope_positions`), so decode matches a full forward
  (pinned by ``tests/test_kv_cache.py``).
* :meth:`reset_rows` zeros selected ``lengths`` and frees their pages so a
  cleared stream (e.g. task boundary) can restart at position 0 without
  rebuilding the rest of the batch.
* Page bookkeeping is host-side (the caller already passes ``lengths`` as a
  Python list), so a decode step issues no device→host sync; new pages are
  assigned with one small device write, only when a row crosses a page
  boundary.
* The per-layer body is compiled on CUDA in two pieces (like the train-side
  :func:`~mouse_core.models.backbone.packed_train.install_compiled_decoder`):
  pre (norms, QKV, RoPE, KV scatter) and post (o-proj, MLP). FlexAttention
  stays the existing compiled kernel so the scatter is visible. Page-table
  growth, mask build, and address setup stay in eager Python.
* After compile warmup, the common incremental step (every row adds one
  token, ``S=1``) is captured in a CUDA graph. Prefills and rebuilds stay
  on the compiled-but-not-graphed path.

The session wraps the backbone's ``transformers`` model in place (shared
weights, decoder loop reimplemented) and supports both ``Qwen3Model`` and
``LlamaModel`` layer layouts (Qwen3 adds q/k RMSNorm). Sessions are
inference-only (``torch.no_grad``): one session per rollout segment, dropped
when the segment ends.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Callable, Literal, cast, get_args

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

# Identical implementations; either import works for both architectures.
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

DecodeKernel = Literal["flex"]
"""Cached-decode attention kernel. ``"flex"`` (paged FlexAttention) is the only
kernel that reads K/V through a page table; the choice is explicit so that a
second one can be added without changing call sites."""


def check_decode_kernel(kernel: object) -> DecodeKernel:
    """Return ``kernel`` if it names a decode kernel, else raise ``ValueError``."""
    if kernel not in get_args(DecodeKernel):
        raise ValueError(f"decode_kernel must be one of {get_args(DecodeKernel)}, got {kernel!r}.")
    return cast(DecodeKernel, kernel)


_BLOCK_SIZE = 128  # compiled CUDA FlexAttention requires >= 128
_compiled_create_block_mask: Any | None = None


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


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
    base dtype (the dtype the KV cache is allocated in). Raises if the
    module has no parameters.
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

    A stable ``mask_mod`` (holder pattern, not a fresh closure) lets the
    compiled wrapper reuse its graph across steps.
    """
    global _compiled_create_block_mask
    builder: Any = create_block_mask
    if compile_masks:
        if _compiled_create_block_mask is None:
            _compiled_create_block_mask = torch.compile(create_block_mask)
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


def packed_rope_positions(
    *,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
) -> torch.Tensor:
    """RoPE position of every token in a flat packed stream ``[L]``.

    Position = number of earlier tokens with the same ``(sequence_id,
    grouping_id)``. This is the counting rule cached decode uses
    (:func:`_decode_rope_positions`) and the same neighbourhood the attention
    masks allow (causal within the same sequence and grouping id, regardless
    of contiguity), so a grouping id that recurs after another id continues
    its own position counter instead of restarting at 0. Sync-free: one
    stable sort plus a cummax, no host ``.item()``.
    """
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    if L == 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    seq = sequence_ids.to(device=device, dtype=torch.long)
    grp = grouping_ids.to(device=device, dtype=torch.long)
    grp = grp - grp.min()
    key = seq * (grp.max() + 1) + grp  # unique per (sequence, grouping) pair
    order = torch.argsort(key, stable=True)
    sorted_key = key[order]
    arange = torch.arange(L, device=device)
    new_run = torch.ones(L, dtype=torch.bool, device=device)
    new_run[1:] = sorted_key[1:] != sorted_key[:-1]
    markers = torch.where(new_run, arange, torch.full_like(arange, -1))
    sorted_pos = arange - torch.cummax(markers, dim=0).values
    positions = torch.empty(L, dtype=torch.long, device=device)
    positions[order] = sorted_pos
    return positions


def _decode_rope_positions(
    *,
    chunk_grouping_ids: torch.Tensor,
    real: torch.Tensor,
    cached_grouping_ids: torch.Tensor,
    prior_lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-token RoPE positions for a left-padded decode chunk.

    Position = (same-``grouping_id`` count in the cache prefix) + (earlier
    real tokens in this chunk with the same id). Pad columns stay 0. Matches
    :func:`packed_rope_positions` on the equivalent flat stream.

    Built by flattening each row's cache prefix plus this chunk's real tokens
    and calling :func:`packed_rope_positions` (stable sort + cummax). The old
    pairwise form allocated ``[B, S, S]`` and ``[B, S, cache]`` bool tables —
    16 GiB at ``B=8, S=16384`` — and OOMed on long prefills.
    """
    B, S = chunk_grouping_ids.shape
    cap = cached_grouping_ids.shape[-1]
    device = chunk_grouping_ids.device
    out = torch.zeros(B, S, dtype=torch.long, device=device)
    if B == 0 or S == 0:
        return out

    row = torch.arange(B, device=device)
    cache_ok = torch.arange(cap, device=device).unsqueeze(0) < prior_lengths.unsqueeze(1)
    cache_seq = row.unsqueeze(1).expand(B, cap)[cache_ok]
    cache_grp = cached_grouping_ids[cache_ok]
    chunk_seq = row.unsqueeze(1).expand(B, S)[real]
    chunk_grp = chunk_grouping_ids[real]
    if cache_seq.numel() == 0 and chunk_seq.numel() == 0:
        return out

    pos = packed_rope_positions(
        sequence_ids=torch.cat([cache_seq, chunk_seq]),
        grouping_ids=torch.cat([cache_grp, chunk_grp]),
    )
    n_cache = cache_seq.numel()
    out[real] = pos[n_cache:]
    return out


def _inductor_rejected(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {"InductorError", "LoweringException"} or "InductorError" in name


def _decode_pre(
    layer: Any,
    h: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    addr: torch.Tensor,
    real_rows: torch.Tensor,
    real_cols: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Norms, QKV, RoPE, KV scatter. Compiled separately from attention.

    Returning after the in-place scatter makes the write visible to FlexAttention
    (a single compiled graph can read the pre-write cache and diverge from train).
    """
    B, S, _ = h.shape
    hn = layer.input_layernorm(h)
    attn = layer.self_attn
    q = attn.q_proj(hn).view(B, S, n_heads, head_dim)
    k = attn.k_proj(hn).view(B, S, n_kv_heads, head_dim)
    q_norm = getattr(attn, "q_norm", None)  # Qwen3 yes, Llama no
    if q_norm is not None:
        q = q_norm(q)
        k = attn.k_norm(k)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = attn.v_proj(hn).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k_cache[:, addr] = k[real_rows, :, real_cols].transpose(0, 1)
    v_cache[:, addr] = v[real_rows, :, real_cols].transpose(0, 1)
    return h, q, attn.scaling


def _decode_post(layer: Any, residual: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Output projection and MLP. Compiled separately from attention."""
    B, S, _ = residual.shape
    o = o.transpose(1, 2).reshape(B, S, -1)
    h = residual + layer.self_attn.o_proj(o)
    return h + layer.mlp(layer.post_attention_layernorm(h))


_compiled_decode_pre: Any | None = None
_compiled_decode_post: Any | None = None


def install_compiled_decode_layer() -> bool:
    """Compile the per-layer cached-decode pre/post bodies. CUDA only; idempotent.

    Attention stays outside those graphs so the KV scatter is visible.
    Returns True on the first successful install.
    """
    global _compiled_decode_pre, _compiled_decode_post
    if _compiled_decode_pre is not None:
        return False
    if not torch.cuda.is_available():
        return False
    _compiled_decode_pre = torch.compile(_decode_pre, dynamic=True)
    _compiled_decode_post = torch.compile(_decode_post, dynamic=True)
    return True


_BLOCK_MASK_TENSORS: tuple[str, ...] = (
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
)


def _copy_block_mask(dst: BlockMask, src: BlockMask) -> bool:
    """Copy ``src`` kv/q tables into ``dst``. False if a shape differs."""
    for name in _BLOCK_MASK_TENSORS:
        d = getattr(dst, name, None)
        s = getattr(src, name, None)
        if d is None and s is None:
            continue
        if d is None or s is None or d.shape != s.shape or d.dtype != s.dtype:
            return False
        d.copy_(s)
    return True


class _FlexKernel:
    """Call flex_attention; compile on CUDA, fall back to eager if Inductor rejects a call."""

    def __init__(self, device: torch.device) -> None:
        self._eager = flex_attention
        self._compiled = torch.compile(flex_attention) if _use_flex_compile(device) else None
        self._active = self._compiled or self._eager

    def __call__(self, *args, **kwargs):
        try:
            return self._active(*args, **kwargs)
        except Exception as exc:
            if self._compiled is not None and self._active is self._compiled and _inductor_rejected(exc):
                self._active = self._eager
                return self._eager(*args, **kwargs)
            raise


class FlexDecodeSession:
    """Incremental decoder over ``batch_size`` independently-growing sequences.

    Create via ``backbone.decode_session(batch_size)``; ``Model.forward``
    does this automatically and carries the session inside its ``cache``.

    The KV cache is a paged pool shared by all rows (see module docstring).
    It starts with one 128-token page per row and doubles whenever a row
    needs a page and none is free, so no capacity has to be chosen up front.

    Args:
        model: A ``transformers`` decoder stack (``Qwen3Model`` / ``LlamaModel``)
            with ``layers``, ``rotary_emb``, and ``norm`` attributes. Used in
            place; not modified.
        batch_size: Number of sequences decoded by this session.

    Attributes:
        lengths: Tokens cached per row, ``[B]`` (device tensor).
        n_pages: Physical pages in the pool.
        pages_in_use: Pages currently owned by rows (each row holds >= 1).
        k_cache / v_cache: ``[layers, kv_heads, n_pages * 128, head_dim]``.
    """

    def __init__(self, model: torch.nn.Module, batch_size: int) -> None:
        # HF decoder stacks are ``nn.Module``; pyright treats children as Tensor|Module.
        hf = cast(Any, model)
        if getattr(hf.config, "use_sliding_window", False):
            raise ValueError("FlexDecodeSession does not support sliding-window attention.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.model = hf
        cfg = hf.config
        self.B = batch_size
        self.page = _BLOCK_SIZE
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)

        self.device, self.dtype = module_device_dtype(hf)
        self._flex = _FlexKernel(self.device)
        self._compile_masks = _use_flex_compile(self.device)
        self._use_compiled_layer = self.device.type == "cuda"

        # Physical pool: one page per row to start (a row always owns >= 1 page
        # so logical block 0 maps to real memory even for pad-only queries).
        n_layers = len(hf.layers)
        self.n_pages = self.B
        self.k_cache = torch.zeros(
            n_layers, self.n_kv_heads, self.n_pages * self.page, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # Logical (per-row) tables, ``logical_cap`` slots wide.
        self.logical_cap = self.page
        self.grouping_ids = torch.zeros(self.B, self.logical_cap, dtype=torch.long, device=self.device)
        self.lengths = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self._lengths_host: list[int] = [0] * self.B

        # Page table. Host lists drive allocation without device syncs; the
        # device tensors mirror them for addressing and the mask.
        self._row_pages: list[list[int]] = [[b] for b in range(self.B)]
        self._free_pages: list[int] = []
        self.page_table = torch.arange(self.B, device=self.device)[:, None]  # [B, logical_cap // page]
        page_logical = torch.zeros(self.n_pages, dtype=torch.long, device=self.device)
        page_row = torch.arange(self.B, device=self.device)

        # Stable mask_mod identity (reads per-call tables) lets torch.compile
        # reuse the traced mask across calls instead of re-guarding on fresh
        # closures.
        #
        # Important: close over a dict holder, not ``self``. A ``mask_mod`` that
        # captures ``self`` creates a reference cycle (session → mask_mod →
        # session). Cyclic GC may not run between rollout and train, so the KV
        # buffers stay allocated and online training OOMs after a few cycles.
        holder: dict[str, torch.Tensor] = {
            "t": torch.zeros(0, 0, dtype=torch.long, device=self.device),
            "q_mask": torch.zeros(0, 0, dtype=torch.long, device=self.device),
            "kv_mask": self.grouping_ids,
            "page_logical": page_logical,
            "page_row": page_row,
        }
        self._mask_holder = holder
        page = self.page

        def mask_mod(b, h, q_idx, kv_idx):
            # Logical coordinates: causal within each sequence, offset by its
            # cached history, and only within the same grouping-id run. Pad
            # queries carry a clamped position (a prefix of real slots), so
            # they stay finite; their K/V are never written and their outputs
            # are discarded by the caller.
            q_pos = holder["t"][b, q_idx]
            return (kv_idx <= q_pos) & (
                holder["kv_mask"][b, kv_idx] == holder["q_mask"][b, q_idx]
            )

        def physical_mask_mod(b, h, q_idx, kv_idx):
            # Same predicate over pool addresses: map the page back to the
            # row's logical slot. The owner check makes the eager (dense)
            # path exact; the compiled kernel only ever visits owned pages.
            blk = kv_idx // page
            logical_kv = holder["page_logical"][blk] * page + kv_idx % page
            q_pos = holder["t"][b, q_idx]
            return (
                (holder["page_row"][blk] == b)
                & (logical_kv <= q_pos)
                & (holder["kv_mask"][b, logical_kv] == holder["q_mask"][b, q_idx])
            )

        self._mask_mod = mask_mod
        self._physical_mask_mod = physical_mask_mod

        # S=1 CUDA graph: captured after compile warmup; rebuilt when the
        # pool is replaced or BlockMask table shapes change.
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_disabled = False
        self._g_h: torch.Tensor | None = None
        self._g_cos: torch.Tensor | None = None
        self._g_sin: torch.Tensor | None = None
        self._g_addr: torch.Tensor | None = None
        self._g_rows: torch.Tensor | None = None
        self._g_cols: torch.Tensor | None = None
        self._g_mask: BlockMask | None = None
        self._g_out: torch.Tensor | None = None
        self._g_hiddens: list[torch.Tensor] | None = None
        self._g_cache_id: int | None = None

    # ------------------------------------------------------------------

    @property
    def pages_in_use(self) -> int:
        return self.n_pages - len(self._free_pages)

    def _grow_logical(self, needed: int) -> None:
        new_cap = _round_up(max(needed, 2 * self.logical_cap), self.page)
        new_ids = torch.zeros(self.B, new_cap, dtype=torch.long, device=self.device)
        new_ids[:, : self.logical_cap] = self.grouping_ids
        self.grouping_ids = new_ids
        self._mask_holder["kv_mask"] = new_ids
        new_table = torch.full((self.B, new_cap // self.page), -1, dtype=torch.long, device=self.device)
        new_table[:, : self.page_table.shape[1]] = self.page_table
        self.page_table = new_table
        self.logical_cap = new_cap

    def _grow_pool(self, min_pages: int) -> None:
        new_n = max(min_pages, 2 * self.n_pages)
        old_len = self.n_pages * self.page
        for name in ("k_cache", "v_cache"):
            old = getattr(self, name)
            new = torch.zeros(
                old.shape[0], self.n_kv_heads, new_n * self.page, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            new[:, :, :old_len] = old
            setattr(self, name, new)
        self._invalidate_graph()
        fill = torch.zeros(new_n - self.n_pages, dtype=torch.long, device=self.device)
        self._mask_holder["page_logical"] = torch.cat([self._mask_holder["page_logical"], fill])
        self._mask_holder["page_row"] = torch.cat([self._mask_holder["page_row"], fill - 1])
        self._free_pages.extend(range(self.n_pages, new_n))
        self.n_pages = new_n

    def _reserve(self, needed: list[int]) -> None:
        """Give every row enough pages for ``needed[b]`` tokens."""
        pending: list[tuple[int, int]] = []  # (row, logical block)
        for b, need in enumerate(needed):
            have = len(self._row_pages[b])
            want = -(-need // self.page)
            pending.extend((b, blk) for blk in range(have, want))
        if not pending:
            return
        if len(pending) > len(self._free_pages):
            self._grow_pool(self.n_pages + len(pending) - len(self._free_pages))
        rows, blks, phys = [], [], []
        for b, blk in pending:
            p = self._free_pages.pop()
            self._row_pages[b].append(p)
            rows.append(b)
            blks.append(blk)
            phys.append(p)
        rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        blks_t = torch.tensor(blks, dtype=torch.long, device=self.device)
        phys_t = torch.tensor(phys, dtype=torch.long, device=self.device)
        self.page_table[rows_t, blks_t] = phys_t
        self._mask_holder["page_logical"][phys_t] = blks_t
        self._mask_holder["page_row"][phys_t] = rows_t

    def _physical_block_mask(self, logical: BlockMask, q_len: int) -> BlockMask:
        """Remap a logical BlockMask's kv block indices through the page table."""
        table = self.page_table

        def remap(idx: torch.Tensor) -> torch.Tensor:
            flat = idx.reshape(idx.shape[0], -1).to(torch.long)
            # Entries past ``kv_num_blocks`` are padding the kernel never reads;
            # clamp so they stay valid addresses if they hit an unassigned block.
            return torch.gather(table, 1, flat).view(idx.shape).clamp_min(0).to(torch.int32)

        full_num = logical.full_kv_num_blocks
        full_idx = logical.full_kv_indices
        return BlockMask.from_kv_blocks(
            logical.kv_num_blocks,
            remap(logical.kv_indices),
            full_num,
            None if full_idx is None else remap(full_idx),
            BLOCK_SIZE=logical.BLOCK_SIZE,
            mask_mod=self._physical_mask_mod,
            seq_lengths=(q_len, self.n_pages * self.page),
            compute_q_blocks=False,
        )

    def _invalidate_graph(self) -> None:
        self._graph = None
        self._g_mask = None
        self._g_cache_id = None

    def _update_mask_tables(self, t: torch.Tensor, q_mask: torch.Tensor) -> None:
        """Write query tables; copy in-place when the CUDA graph closed over them."""
        ht = self._mask_holder["t"]
        hq = self._mask_holder["q_mask"]
        if (
            ht.shape == t.shape
            and hq.shape == q_mask.shape
            and ht.dtype == t.dtype
            and hq.dtype == q_mask.dtype
            and ht.device == t.device
        ):
            ht.copy_(t)
            hq.copy_(q_mask)
        else:
            self._mask_holder["t"] = t.contiguous()
            self._mask_holder["q_mask"] = q_mask.contiguous()
        self._mask_holder["kv_mask"] = self.grouping_ids

    def _run_layers(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        addr: torch.Tensor,
        real_rows: torch.Tensor,
        real_cols: torch.Tensor,
        block_mask: BlockMask,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        use_compiled = self._use_compiled_layer and _compiled_decode_pre is not None
        pre = _compiled_decode_pre if use_compiled else _decode_pre
        post = _compiled_decode_post if use_compiled else _decode_post
        flex_fn: Callable[..., torch.Tensor] = self._flex
        layer_hiddens: list[torch.Tensor] = []
        h_in = h
        try:
            for li, layer in enumerate(self.model.layers):
                residual, q, scale = pre(
                    layer, h, cos, sin,
                    self.k_cache[li], self.v_cache[li],
                    addr, real_rows, real_cols,
                    self.n_heads, self.n_kv_heads, self.head_dim,
                )
                o = flex_fn(
                    q, self.k_cache[li][None], self.v_cache[li][None],
                    block_mask=block_mask, scale=scale, enable_gqa=True,
                )
                h = post(layer, residual, o)
                layer_hiddens.append(h)
        except Exception as exc:
            if use_compiled and _inductor_rejected(exc):
                self._use_compiled_layer = False
                return self._run_layers(h_in, cos, sin, addr, real_rows, real_cols, block_mask)
            raise
        return self.model.norm(h), layer_hiddens

    def _graph_input_ready(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        addr: torch.Tensor,
        real_rows: torch.Tensor,
        real_cols: torch.Tensor,
    ) -> bool:
        bufs = (self._g_h, self._g_cos, self._g_sin, self._g_addr, self._g_rows, self._g_cols)
        srcs = (h, cos, sin, addr, real_rows, real_cols)
        if any(b is None for b in bufs):
            return False
        return all(b.shape == s.shape and b.dtype == s.dtype for b, s in zip(bufs, srcs))

    def _copy_graph_inputs(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        addr: torch.Tensor,
        real_rows: torch.Tensor,
        real_cols: torch.Tensor,
        block_mask: BlockMask,
    ) -> bool:
        if not self._graph_input_ready(h, cos, sin, addr, real_rows, real_cols):
            return False
        assert self._g_h is not None and self._g_cos is not None and self._g_sin is not None
        assert self._g_addr is not None and self._g_rows is not None and self._g_cols is not None
        self._g_h.copy_(h)
        self._g_cos.copy_(cos)
        self._g_sin.copy_(sin)
        self._g_addr.copy_(addr)
        self._g_rows.copy_(real_rows)
        self._g_cols.copy_(real_cols)
        if self._g_mask is None or not _copy_block_mask(self._g_mask, block_mask):
            return False
        return True

    def _try_s1_graph(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        addr: torch.Tensor,
        real_rows: torch.Tensor,
        real_cols: torch.Tensor,
        block_mask: BlockMask,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        """Replay or capture the S=1 layer stack. None → use the compiled path."""
        if (
            self._graph_disabled
            or self.device.type != "cuda"
            or not self._use_compiled_layer
            or _compiled_decode_pre is None
            or h.shape[1] != 1
            or addr.numel() != self.B
        ):
            return None
        if (
            self._graph is not None
            and self._g_cache_id == id(self.k_cache)
            and self._copy_graph_inputs(h, cos, sin, addr, real_rows, real_cols, block_mask)
        ):
            assert self._graph is not None and self._g_out is not None and self._g_hiddens is not None
            self._graph.replay()
            return self._g_out, self._g_hiddens
        return self._capture_s1_graph(h, cos, sin, addr, real_rows, real_cols, block_mask)

    def _capture_s1_graph(
        self,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        addr: torch.Tensor,
        real_rows: torch.Tensor,
        real_cols: torch.Tensor,
        block_mask: BlockMask,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        self._g_h = h.clone()
        self._g_cos = cos.clone()
        self._g_sin = sin.clone()
        self._g_addr = addr.clone()
        self._g_rows = real_rows.clone()
        self._g_cols = real_cols.clone()
        self._g_mask = block_mask
        if not self._copy_graph_inputs(h, cos, sin, addr, real_rows, real_cols, block_mask):
            self._invalidate_graph()
            return self._run_layers(h, cos, sin, addr, real_rows, real_cols, block_mask)

        def _restore_static() -> None:
            assert self._g_h is not None and self._g_cos is not None and self._g_sin is not None
            assert self._g_addr is not None and self._g_rows is not None and self._g_cols is not None
            self._g_h.copy_(h)
            self._g_cos.copy_(cos)
            self._g_sin.copy_(sin)
            self._g_addr.copy_(addr)
            self._g_rows.copy_(real_rows)
            self._g_cols.copy_(real_cols)

        def _run_static() -> tuple[torch.Tensor, list[torch.Tensor]]:
            assert self._g_h is not None and self._g_cos is not None and self._g_sin is not None
            assert self._g_addr is not None and self._g_rows is not None and self._g_cols is not None
            assert self._g_mask is not None
            return self._run_layers(
                self._g_h, self._g_cos, self._g_sin,
                self._g_addr, self._g_rows, self._g_cols, self._g_mask,
            )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _restore_static()
                out, hiddens = _run_static()
        torch.cuda.current_stream().wait_stream(side)
        if not self._use_compiled_layer or _compiled_decode_pre is None:
            return out, hiddens
        _restore_static()
        try:
            graph = torch.cuda.CUDAGraph()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.cuda.graph(graph):
                    self._g_out, self._g_hiddens = _run_static()
            empty = any("CUDA Graph is empty" in str(w.message) for w in caught)
        except Exception:
            empty = True
        if empty or self._g_out is None or not torch.allclose(self._g_out.float(), out.float(), atol=5e-2, rtol=5e-2):
            self._graph_disabled = True
            self._invalidate_graph()
            return out, hiddens
        self._graph = graph
        self._g_cache_id = id(self.k_cache)
        assert self._g_hiddens is not None
        return self._g_out, self._g_hiddens

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart decode positions for selected rows (or the whole batch).

        Sets ``lengths[b] = 0`` so the next tokens for those rows write at
        position 0 and attend only to the new prefix, and returns all but the
        row's first page to the pool. Other rows are unchanged. Stale K/V /
        grouping-id slots in the kept page are ignored by the attention mask
        and overwritten as the row grows again. Use this when a stream's
        context is cleared (e.g. task boundary) without rebuilding the batch.
        """
        idx = list(range(self.B)) if rows is None else list(rows)
        if not idx:
            return
        freed: list[int] = []
        for b in idx:
            pages = self._row_pages[b]
            freed.extend(pages[1:])
            self._row_pages[b] = pages[:1]
            self._lengths_host[b] = 0
        self._free_pages.extend(freed)
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        self.lengths[idx_t] = 0
        self.page_table[idx_t, 1:] = -1
        if freed:
            freed_t = torch.as_tensor(freed, dtype=torch.long, device=self.device)
            self._mask_holder["page_logical"][freed_t] = 0
            self._mask_holder["page_row"][freed_t] = -1

    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        *,
        embeds: torch.Tensor,
        lengths: list[int],
        grouping_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Decode one chunk per sequence.

        Args:
            embeds: Left-padded token embeddings ``[B, S, D]``; row ``b``'s
                real tokens are the trailing ``lengths[b]`` positions. ``S``
                is this call's longest row — unrelated to any other call.
            lengths: Real token count per row, ``0 <= lengths[b] <= S``.
            grouping_ids: Left-padded **absolute** per-token grouping ids
                ``[B, S]`` matching ``embeds`` (from the data pipeline
                ``grouping_id`` field; pad columns ignored).
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
        if self.device.type == "cuda":
            install_compiled_decode_layer()
        if grouping_ids.shape != (B, S):
            raise ValueError(
                f"grouping_ids must have shape [{B}, {S}], got {tuple(grouping_ids.shape)}."
            )
        if len(lengths) != B:
            raise ValueError(f"lengths has {len(lengths)} entries for batch_size={B}.")
        needed = [prior + int(c) for prior, c in zip(self._lengths_host, lengths)]
        if max(needed) > self.logical_cap:
            self._grow_logical(max(needed))
        self._reserve(needed)
        n = torch.tensor(lengths, dtype=torch.long, device=self.device)

        x = embeds.to(self.device, self.dtype)
        mid = grouping_ids.to(device=self.device, dtype=torch.long)
        pad = (S - n)[:, None]  # leading pad tokens per row
        col = torch.arange(S, device=self.device)[None]

        # Cache slot within its own sequence: len_before + (col - pad).
        # Pad columns get earlier/negative values; clamp keeps the causal mask
        # finite (pad outputs are discarded by the caller either way).
        cache_pos = self.lengths[:, None] + col - pad
        self._update_mask_tables(cache_pos.clamp_min(0), mid)

        # Real tokens are the trailing lengths[b] columns; only they are
        # written to the cache, at their own sequence's slots.
        real_rows, real_cols = (col >= pad).nonzero(as_tuple=True)
        cache_slots = cache_pos[real_rows, real_cols]
        # Pool address of each real token's slot: its row's page for that block.
        phys_page = self.page_table[real_rows, cache_slots // self.page]
        addr = phys_page * self.page + cache_slots % self.page

        # Write mask ids before building the mask so same-mask KV checks see
        # this chunk's slots (queries may attend within the new prefix).
        self.grouping_ids[real_rows, cache_slots] = mid[real_rows, real_cols]

        # RoPE is per (sequence, mask) run: same-id count in the cache prefix
        # plus earlier same-id tokens in this chunk. Vectorized — no per-row
        # host sync. ``grouping_ids`` already includes this chunk's writes;
        # ``self.lengths`` is still the prior length, so new slots are not
        # double-counted (they appear in the in-chunk term instead).
        rope_pos = _decode_rope_positions(
            chunk_grouping_ids=mid,
            real=col >= pad,
            cached_grouping_ids=self.grouping_ids,
            prior_lengths=self.lengths,
        )

        logical_mask = flex_block_mask(
            self._mask_mod,
            B=B,
            Q_LEN=S,
            KV_LEN=self.logical_cap,
            device=self.device,
            block_size=self.page,
            compile_masks=self._compile_masks,
        )
        block_mask = self._physical_block_mask(logical_mask, S)

        cos, sin = self.model.rotary_emb(x, rope_pos)

        graphed = self._try_s1_graph(x, cos, sin, addr, real_rows, real_cols, block_mask)
        if graphed is None:
            h, layer_hiddens = self._run_layers(
                x, cos, sin, addr, real_rows, real_cols, block_mask,
            )
        else:
            h, layer_hiddens = graphed

        self.lengths = self.lengths + n
        self._lengths_host = needed
        if output_hidden_states:
            return h, tuple(layer_hiddens)
        return h
