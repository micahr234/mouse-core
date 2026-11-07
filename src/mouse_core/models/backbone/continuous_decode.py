"""Hugging Face continuous batching for embedding-to-hidden-state RL inference.

The Transformers scheduler chooses token chunks and its paged allocator owns KV
storage. Mouse supplies the existing decoder layers, a sequence-causal Flex mask,
and hidden states instead of the language-model logits expected by generate_batch.
No vocabulary head, token sampling, or second model is involved.
"""
from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, FlexKernelOptions, flex_attention
from transformers.generation import ContinuousBatchingConfig
from transformers.generation.continuous_batching.cache import PagedAttentionCache, FullAttentionCacheAllocator
from transformers.generation.continuous_batching.distributed import DistributedHelper
from transformers.generation.continuous_batching.requests import RequestState
from transformers.generation.continuous_batching.scheduler import FIFOScheduler

from mouse_core.models.backbone.flex_utils import (
    _BLOCK_SIZE, _FLEX_RECOMPILE_LIMIT, _use_flex_compile, check_autocast_dtype,
    module_device_dtype,
)
from mouse_core.models.backbone.packed_train import _attention_inputs, _attention_output

Readout = torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]
_PAGE_SIZE = ContinuousBatchingConfig().page_size
_compiled_layer: Any = None
_compiled_inputs: Any = None


def _decode_layer(layer, h, q, n_heads, n_kv_heads, head_dim,
                  key_cache, value_cache, query_index, query_width, block_mask):
    """Attend directly to HF-owned KV pages; pad only the new query tokens."""
    batch = block_mask.kv_num_blocks.shape[0]
    queries = q.new_zeros((batch * query_width, n_heads, head_dim))
    queries.index_copy_(0, query_index, q[0].transpose(0, 1))
    kernel_options: FlexKernelOptions | None = (
        {"BLOCK_M": _BLOCK_SIZE}
        if query_width < 128 and query_width * (n_heads // n_kv_heads) > _BLOCK_SIZE
        else None
    )
    attention_output = flex_attention(
        queries.view(batch, query_width, n_heads, head_dim).transpose(1, 2),
        key_cache.transpose(0, 1).unsqueeze(0),
        value_cache.transpose(0, 1).unsqueeze(0),
        block_mask=block_mask, scale=layer.self_attn.scaling,
        enable_gqa=n_heads != n_kv_heads, kernel_options=kernel_options,
    )
    output = attention_output.transpose(1, 2).reshape(batch * query_width, n_heads, head_dim)
    output = output.index_select(0, query_index).transpose(0, 1).unsqueeze(0)
    return _attention_output(layer, h, output)


def _physical_indices(allocator, request_id, positions):
    """Vectorized lookup into HF's per-request pages (no allocation policy)."""
    slots = positions
    pages = np.asarray(allocator.block_table[request_id], dtype=np.int64)
    return (pages[slots // allocator.tokens_per_page] * allocator.block_physical_stride
            + slots % allocator.tokens_per_page)


@dataclass
class _Submission:
    embeds: torch.Tensor
    output_hidden_states: bool
    outputs: list[torch.Tensor] = field(default_factory=list)
    layers: list[tuple[torch.Tensor, ...]] = field(default_factory=list)


def _paged_mask(past_lengths, query_lengths, owners, logical_pages, stride, page_size):
    def mask(b, h, q, k):
        block, offset = k // stride, k % stride
        logical_k = logical_pages[block] * page_size + offset
        position = past_lengths[b] + q
        return ((owners[block] == b) & (offset < page_size)
                & (logical_k < past_lengths[b] + query_lengths[b])
                & (logical_k <= position))
    return mask


class ContinuousDecodeSession:
    """Bounded, token-budgeted inference with persistent independent episode rows.

    Create with ``backbone.decode_session``. ``forward`` accepts the
    left-padded embedding layout; projections and MLPs process only real
    tokens, while attention pads the scheduled environment rows. Alternatively,
    call ``submit(row, embeds)`` as environments become ready and ``step()`` to run
    one scheduled batch and retrieve completed inputs.
    Calls must be serialized by the caller; there is no background worker.

    ``cache_capacity_tokens`` is the shared pool capacity, in multiples of HF's
    default page size; each live episode rounds its allocation up to a page. ``max_batch_tokens``
    limits queries in one forward. Neither setting changes the attention mask.
    HF's full-attention allocator retains each outer episode until reset_rows
    returns its pages to the shared pool and restarts rotary positions at zero.
    Insufficient shared pool capacity also raises before submitting. The cache
    never discards keys the model's training mask would allow, or recomputes a
    truncated history.

    Close after rollout, before updating model/embedding weights. A backbone
    weight change while cache entries exist raises; resetting ALL rows explicitly
    starts a new weight version. The caller must also reset after embedding or
    preprocessing changes, which are outside this session's ownership.
    """

    def __init__(
        self, model: torch.nn.Module, *, batch_size: int,
        max_batch_tokens: int, cache_capacity_tokens: int,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        for name, value in (("batch_size", batch_size), ("max_batch_tokens", max_batch_tokens),
                            ("cache_capacity_tokens", cache_capacity_tokens)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if cache_capacity_tokens % _PAGE_SIZE:
            raise ValueError(f"cache_capacity_tokens must be a multiple of {_PAGE_SIZE} (HF's default KV page size).")
        self.model: Any = model
        cfg = self.model.config
        if getattr(cfg, "use_sliding_window", False) or any(
            kind != "full_attention" for kind in (getattr(cfg, "layer_types", None) or [])
        ):
            raise ValueError("Continuous decoding requires full attention in every layer.")
        self.B = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.cache_capacity_tokens = cache_capacity_tokens
        self.device, self.dtype = module_device_dtype(model)
        self.autocast_dtype = check_autocast_dtype(autocast_dtype, self.dtype)
        self._cache: PagedAttentionCache | None = PagedAttentionCache(
            cfg,
            ContinuousBatchingConfig(
                max_batch_tokens=max_batch_tokens,
                num_blocks=cache_capacity_tokens // _PAGE_SIZE,
                max_memory_percent=1.0,
                allow_block_sharing=False,
                max_blocks_per_request=0,
                auto_switch_to_flash=False,
                use_cuda_graph=False,
                use_async_batching=False,
            ),
            device=self.device, dtype=autocast_dtype or self.dtype,
            distributed_helper=DistributedHelper(None, None),
            model_supports_logits_to_keep=True,
        )
        self._scheduler: FIFOScheduler | None = FIFOScheduler(
            self._cache, safety_margin=0.0, max_requests_per_batch=batch_size,
        )
        self._lengths = [0] * batch_size
        self._pending: dict[int, _Submission] = {}
        self._versions = self._weight_versions()
        global _compiled_layer, _compiled_inputs
        if _use_flex_compile(self.device):
            if _compiled_layer is None:
                # Separate from training to keep inference specialization out of
                # the checkpoint-recomputation compilation budget.
                _compiled_layer = cast(Any, torch.compile)(
                    _decode_layer, dynamic=True, fullgraph=True,
                    recompile_limit=_FLEX_RECOMPILE_LIMIT,
                )
            if _compiled_inputs is None:
                _compiled_inputs = cast(Any, torch.compile)(
                    _attention_inputs, dynamic=True, fullgraph=True,
                    recompile_limit=_FLEX_RECOMPILE_LIMIT,
                )
            self._inputs = _compiled_inputs
            self._layer = _compiled_layer
        else:
            self._inputs = _attention_inputs
            self._layer = _decode_layer

    @property
    def lengths(self) -> tuple[int, ...]:
        """Number of processed tokens in each outer episode (no device sync)."""
        return tuple(self._lengths)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def _weight_versions(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.model.parameters())

    def _check(self) -> tuple[PagedAttentionCache, FIFOScheduler]:
        if self._cache is None or self._scheduler is None:
            raise RuntimeError("Continuous decode session is closed.")
        if self.model.training:
            raise RuntimeError("Continuous decoding requires model.eval().")
        if torch.is_autocast_enabled(self.device.type):
            raise RuntimeError("Declare decode_autocast_dtype on the backbone; remove ambient torch.autocast.")
        if self._weight_versions() != self._versions:
            raise RuntimeError("Backbone weights changed; close the cache or reset all rows before decoding.")
        return self._cache, self._scheduler

    def _row(self, row: int) -> None:
        if type(row) is not int or not 0 <= row < self.B:
            raise ValueError(f"row must be an integer in [0, {self.B}).")

    def _validate_submission(self, row: int, embeds: torch.Tensor) -> None:
        self._row(row)
        if row in self._pending:
            raise ValueError(f"Row {row} already has an unfinished input.")
        if embeds.ndim != 2 or embeds.shape[1] != self.model.config.hidden_size or not embeds.shape[0]:
            raise ValueError("submit requires nonempty embeddings [tokens, hidden_dim].")

    def _check_capacity(self, additions: dict[int, int]) -> None:
        # Reserve all pending tokens, including chunks the scheduler has not run.
        lengths = list(self._lengths)
        for row, pending in self._pending.items():
            processed = sum(x.shape[0] for x in pending.outputs)
            lengths[row] += pending.embeds.shape[0] - processed
        for row, count in additions.items():
            lengths[row] += count
        needed = sum((n + _PAGE_SIZE - 1) // _PAGE_SIZE for n in lengths) * _PAGE_SIZE
        if needed > self.cache_capacity_tokens:
            raise ValueError(
                f"KV capacity exceeded: {needed} tokens including page rounding, "
                f"capacity {self.cache_capacity_tokens}. Increase cache_capacity_tokens "
                "or reset completed outer episodes; keys visible to the mask cannot be evicted."
            )

    def submit(self, row: int, embeds: torch.Tensor, *, output_hidden_states: bool = False) -> None:
        """Queue the next embedding chunk for one row, retaining its earlier KV."""
        self._check()
        self._validate_submission(row, embeds)
        self._check_capacity({row: embeds.shape[0]})
        self._enqueue(row, embeds, output_hidden_states=output_hidden_states)

    def _enqueue(self, row: int, embeds: torch.Tensor, *, output_hidden_states: bool) -> None:
        """Queue an input after the public entry point has validated the batch."""
        assert self._scheduler is not None
        self._pending[row] = _Submission(embeds.detach().to(device=self.device, dtype=self.dtype), output_hidden_states)
        # HF schedules lists of IDs. These are local embedding indices, never
        # vocabulary IDs; prefix sharing MUST remain disabled for numeric inputs.
        state = RequestState(request_id=str(row), initial_tokens=list(range(embeds.shape[0])))
        state.position_offset = self._lengths[row]
        self._scheduler.add_waiting_request(state)

    @torch.no_grad()
    def step(self) -> dict[int, Readout]:
        """Run at most max_batch_tokens real queries; return completed submissions.

        A completed submission remains cached while its environment takes an
        action. Submit the next input on that row, or reset it at the outer reset.
        Other rows may arrive or reset between calls, including during a prefill.
        """
        cache, scheduler = self._check()
        if not self._pending:
            return {}
        try:
            scheduled, _, _, _ = scheduler.schedule_batch(self.max_batch_tokens, cache.max_tokens_read)
            if not scheduled:
                raise RuntimeError("Hugging Face could not schedule a batch within the reserved KV capacity.")
            states = [future.state for future in scheduled]
            rows = [int(state.request_id) for state in states]
            chunks = [len(state.tokens_to_process) for state in states]
            # Each scheduled input is a contiguous slice of its submitted embeddings.
            h = torch.cat([self._pending[row].embeds[state.tokens_to_process[0]:state.tokens_to_process[-1] + 1]
                           for row, state in zip(rows, states, strict=True)])
            allocator = cast(FullAttentionCacheAllocator, cache.layer_to_allocator[0])
            # HF blocks interleave layers and K/V pages. Translate page indices,
            # not their contents; the same block map applies to each shifted layer view.
            past_lengths = np.asarray([state.position_offset for state in states], dtype=np.int64)
            query_lengths = np.asarray(chunks, dtype=np.int64)
            query_width = max(chunks)
            page_size, stride = allocator.tokens_per_page, allocator.block_physical_stride
            cache_rows = allocator._kv_token_views[allocator.layer_indices[-1]][1].shape[0]
            num_blocks = (cache_rows + stride - 1) // stride
            owners = np.full(num_blocks, -1, dtype=np.int64)
            logical_pages = np.zeros(num_blocks, dtype=np.int64)
            q_blocks = (query_width + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            max_pages = max(len(allocator.block_table[state.request_id]) for state in states)
            indices = np.zeros((len(rows), 1, q_blocks, max_pages), dtype=np.int32)
            block_counts = np.zeros((len(rows), 1, q_blocks), dtype=np.int32)
            full_indices = np.zeros_like(indices)
            full_counts = np.zeros_like(block_counts)
            columns = [[] for _ in range(3)]
            for owner, (state, count) in enumerate(zip(states, chunks, strict=True)):
                past = state.position_offset
                pages = np.asarray(allocator.block_table[state.request_id], dtype=np.int64)
                owners[pages] = owner
                logical_pages[pages] = np.arange(len(pages))
                for tile in range((count + _BLOCK_SIZE - 1) // _BLOCK_SIZE):
                    used = (past + min(count, (tile + 1) * _BLOCK_SIZE) + page_size - 1) // page_size
                    full = min(used, (past + tile * _BLOCK_SIZE + 1) // page_size)
                    block_counts[owner, 0, tile] = used - full
                    indices[owner, 0, tile, :used - full] = pages[full:used] * (stride // page_size)
                    full_counts[owner, 0, tile] = full
                    full_indices[owner, 0, tile, :full] = pages[:full] * (stride // page_size)
                positions = np.arange(past, past + count, dtype=np.int64)
                values = (positions, _physical_indices(allocator, state.request_id, positions),
                          owner * query_width + np.arange(count, dtype=np.int64))
                for column, value in zip(columns, values, strict=True):
                    column.append(value)
            metadata = [past_lengths, query_lengths, owners, logical_pages,
                        *(np.concatenate(column) for column in columns)]
            past, counts, owners_gpu, logical_gpu, qp, wi, qi = torch.from_numpy(np.concatenate(metadata)).to(self.device).split(
                [len(column) for column in metadata])
            block_arrays = [block_counts, indices, full_counts, full_indices]
            blocks_gpu = torch.from_numpy(np.concatenate([array.ravel() for array in block_arrays])).to(self.device)
            count_gpu, index_gpu, full_count_gpu, full_index_gpu = blocks_gpu.split([array.size for array in block_arrays])
            mask = BlockMask.from_kv_blocks(
                count_gpu.view(block_counts.shape), index_gpu.view(indices.shape),
                full_count_gpu.view(full_counts.shape), full_index_gpu.view(full_indices.shape),
                BLOCK_SIZE=(_BLOCK_SIZE, page_size),
                mask_mod=_paged_mask(past, counts, owners_gpu, logical_gpu, stride, page_size),
                seq_lengths=(query_width, cache_rows), compute_q_blocks=False,
            )
            write_only = wi[:0]
            cos, sin = self.model.rotary_emb(h.unsqueeze(0), qp.unsqueeze(0))
            cfg = self.model.config
            head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            layer_outputs = []
            need_layers = any(self._pending[row].output_hidden_states for row in rows)
            context = nullcontext() if self.autocast_dtype is None else torch.autocast(self.device.type, dtype=self.autocast_dtype)
            with context:
                for layer_index, layer in enumerate(self.model.layers):
                    # HF owns these strided views, including page/trash storage.
                    key_cache, value_cache = allocator._kv_token_views[layer_index]
                    q, k, v = self._inputs(
                        layer, h, cos, sin, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim,
                    )
                    # New full-attention pages never overwrite live episode history.
                    allocator.update(k[0].transpose(0, 1), v[0].transpose(0, 1), layer_index, write_only, wi)
                    h = self._layer(
                        layer, h, q, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim,
                        key_cache[:cache_rows], value_cache[:cache_rows], qi, query_width, mask,
                    )
                    if need_layers:
                        layer_outputs.append(h.split(chunks))
                outputs = self.model.norm(h).split(chunks)
            completed = {}
            for index, (row, state, count, output) in enumerate(zip(rows, states, chunks, outputs, strict=True)):
                pending = self._pending[row]
                pending.outputs.append(output)
                if pending.output_hidden_states:
                    pending.layers.append(tuple(layer[index] for layer in layer_outputs))
                state.position_offset += count
                self._lengths[row] += count
                if not state.remaining_prefill_tokens:
                    out = torch.cat(pending.outputs)
                    completed[row] = (out, tuple(torch.cat(parts) for parts in zip(*pending.layers, strict=True))) if pending.output_hidden_states else out
                    # finish_request would free KV: an RL environment pauses here,
                    # it has not finished its outer episode.
                    scheduler.active_requests.pop(state.request_id)
                    del self._pending[row]
            return completed
        except Exception:
            # A failed layer may have written only part of the cache. Never reuse it.
            self.close()
            raise

    @torch.no_grad()
    def forward(self, *, embeds: torch.Tensor, lengths: list[int], output_hidden_states: bool = False) -> Readout:
        """Process a left-padded batch; zero-length rows do not enter the decoder."""
        self._check()
        if self._pending:
            raise RuntimeError("Drain pending submit/step inputs before calling forward.")
        if (embeds.ndim != 3 or embeds.shape[0] != self.B
                or embeds.shape[2] != self.model.config.hidden_size or len(lengths) != self.B
                or any(type(n) is not int or not 0 <= n <= embeds.shape[1] for n in lengths)):
            raise ValueError("forward requires [batch_size, tokens, hidden_dim] and valid per-row lengths.")
        self._check_capacity(dict(enumerate(lengths)))
        output = torch.zeros(embeds.shape, device=self.device, dtype=self.dtype)
        layers = tuple(torch.zeros_like(output) for _ in self.model.layers) if output_hidden_states else ()
        for row, count in enumerate(lengths):
            if count:
                self._enqueue(row, embeds[row, -count:], output_hidden_states=output_hidden_states)
        while self._pending:
            for row, result in self.step().items():
                if isinstance(result, tuple):
                    out, layer_hiddens = result
                    for target, hidden in zip(layers, layer_hiddens, strict=True):
                        target[row, -lengths[row]:] = hidden
                else:
                    out = result
                output[row, -lengths[row]:] = out
        return (output, layers) if output_hidden_states else output

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Release completed episodes; reset rotary positions independently per row."""
        if self._cache is None or self._scheduler is None:
            raise RuntimeError("Continuous decode session is closed.")
        selected = list(range(self.B)) if rows is None else list(rows)
        for row in selected:
            self._row(row)
        if set(selected) == set(range(self.B)):
            self._cache.reset()
            self._scheduler.reset()
            self._pending.clear()
            self._lengths = [0] * self.B
            # A changed layout requires allocating a new pool, even after reset.
            if module_device_dtype(self.model) != (self.device, self.dtype):
                self.close()
                raise RuntimeError("Model device/dtype changed; create a new decode session.")
            self._versions = self._weight_versions()
        else:
            self._check()
            for row in selected:
                self._scheduler.set_request_cancellation(str(row))
                self._pending.pop(row, None)
                self._lengths[row] = 0
            self._scheduler.clear_cancelled_requests()

    def close(self) -> None:
        """Release the HF scheduler, KV pool, and pending embeddings. Idempotent."""
        self._pending.clear()
        self._scheduler = None
        self._cache = None
