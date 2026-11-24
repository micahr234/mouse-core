"""Hugging Face continuous batching for embedding-to-hidden-state RL inference.

The Transformers scheduler chooses token chunks and its paged allocator owns KV
storage. Mouse supplies the existing decoder layers, a sequence-causal Flex mask,
and hidden states instead of the language-model logits expected by generate_batch.
No vocabulary head, token sampling, or second model is involved.
"""
from __future__ import annotations

from collections import OrderedDict
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
_GRAPH_REUSE_THRESHOLD = 8
_MAX_DECODE_GRAPHS = 16
_MAX_TRACKED_SHAPES = 256


def _decode_layer(layer, h, q, n_heads, n_kv_heads, head_dim,
                  key_cache, value_cache, query_index, query_width, block_mask, attention_tiles=(16, 128)):
    """Attend directly to HF-owned KV pages; pad only the new query tokens."""
    batch = block_mask.kv_num_blocks.shape[0]
    queries = q.new_zeros((batch * query_width, n_heads, head_dim))
    queries.index_copy_(0, query_index, q[0].transpose(0, 1))
    queries = queries.view(batch, query_width, n_heads, head_dim).transpose(1, 2)
    keys = key_cache.transpose(0, 1).unsqueeze(0)
    values = value_cache.transpose(0, 1).unsqueeze(0)
    # Explicit tiles avoid an autotuning search.
    # General Flex keeps the batch dimension dynamic, unlike FlexDecoding.
    kernel_options: FlexKernelOptions = {"BACKEND": "TRITON"}
    if attention_tiles[0] is not None:
        kernel_options["BLOCK_M"] = attention_tiles[0]
    if attention_tiles[1] is not None:
        kernel_options["BLOCK_N"] = attention_tiles[1]
    attention_output = flex_attention(
        queries, keys, values,
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
    return (pages[slots // allocator.tokens_per_page] * allocator.tokens_per_page
            + slots % allocator.tokens_per_page)


@dataclass
class _Submission:
    embeds: torch.Tensor
    output_hidden_states: bool
    prefix_length: int = 0
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
    tokens, while attention pads the scheduled environment rows. ``forward_chunks``
    accepts a mapping of rows to unpadded chunks and validates the batch once.
    Alternatively, call ``submit(row, embeds)`` as environments become ready and ``step()`` to run
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

    ``generation_state`` holds caller-owned tensors per row and is cleared on
    row reset and close, together with the corresponding KV history.

    Close after rollout, before updating model/embedding weights. A backbone
    weight change while cache entries exist raises; resetting ALL rows explicitly
    starts a new weight version. The caller must also reset after embedding or
    preprocessing changes, which are outside this session's ownership.
    """

    def __init__(
        self, model: torch.nn.Module, *, batch_size: int,
        max_batch_tokens: int, cache_capacity_tokens: int,
        autocast_dtype: torch.dtype | None = None,
        residual_dtype: torch.dtype | None = None,
        attention_tiles: tuple[int | None, int | None] = (16, 128),
    ) -> None:
        for name, value in (("batch_size", batch_size), ("max_batch_tokens", max_batch_tokens),
                            ("cache_capacity_tokens", cache_capacity_tokens)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if cache_capacity_tokens % _PAGE_SIZE:
            raise ValueError(f"cache_capacity_tokens must be a multiple of {_PAGE_SIZE} (HF's default KV page size).")
        from mouse_core.models.backbone.flex_utils import validate_attention_tiles
        self.attention_tiles = validate_attention_tiles(*attention_tiles)
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
        from mouse_core.models.backbone.flex_utils import check_residual_dtype
        self.residual_dtype = check_residual_dtype(residual_dtype, self.dtype)
        self.autocast_dtype = check_autocast_dtype(autocast_dtype, self.dtype)
        self._precision = (self.dtype, self.autocast_dtype, self.residual_dtype)
        self._backbone_ref: Any = None
        try:
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
        except MemoryError as exc:
            raise MemoryError(
                "Cannot allocate the continuous decode cache with "
                f"cache_capacity_tokens={cache_capacity_tokens}, "
                f"max_batch_tokens={max_batch_tokens}, batch_size={batch_size}, "
                f"cache_dtype={autocast_dtype or self.dtype}. "
                "The entire KV pool is allocated at session creation, independently "
                "of the training batch size. Reduce cache_capacity_tokens while "
                "retaining enough space for all live outer episodes, or reduce "
                f"concurrent episodes and the pool together. Original error: {exc}"
            ) from exc
        # HF owns page allocation and request scheduling. Mouse owns the contents:
        # reinterpret the same allocation as layer-major pages (no copy/pool).
        # Full attention and disabled prefix sharing are required above; HF's
        # interleaved update/copy APIs must not be used with this content layout.
        allocator = cast(FullAttentionCacheAllocator, self._cache.layer_to_allocator[0])
        self._kv_cache = allocator.cache_tensor.view(
            len(self.model.layers), 2, -1, allocator.num_key_value_heads, allocator.head_dim,
        )
        self._scheduler: FIFOScheduler | None = FIFOScheduler(
            self._cache, safety_margin=0.0, max_requests_per_batch=batch_size,
        )
        self._lengths = [0] * batch_size
        self.generation_state: list[dict[str, torch.Tensor]] = [{} for _ in range(batch_size)]
        self._pending: dict[int, _Submission] = {}
        self._deferred: dict[int, torch.Tensor] = {}
        self._module_layout = tuple((module, module._modules.copy()) for module in self.model.modules())
        self._versions = self._weight_versions()
        self._graphs = OrderedDict()
        self._graph_pool = torch.cuda.graph_pool_handle() if self.device.type == "cuda" else None
        self._graph_counts = OrderedDict()
        self._capture_stream = None
        global _compiled_layer, _compiled_inputs
        if _use_flex_compile(self.device):
            if _compiled_layer is None:
                # Separate from training to keep inference specialization out of
                # the checkpoint-recomputation compilation budget.
                _compiled_layer = cast(Any, torch.compile)(
                    _decode_layer, dynamic=True, fullgraph=True,
                    recompile_limit=_FLEX_RECOMPILE_LIMIT,
                    options={
                        "max_autotune": False,
                        "max_autotune_gemm": False,
                        # The session owns whole-transformer CUDA graphs.
                        "triton.cudagraphs": False,
                    },
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
        """History length, including deferred feedback (no device sync)."""
        lengths = [n + len(self._deferred.get(row, ())) for row, n in enumerate(self._lengths)]
        for row, pending in self._pending.items():
            processed = sum(len(output) for output in pending.outputs)
            lengths[row] += max(0, pending.prefix_length - processed)
        return tuple(lengths)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def _weight_versions(self):
        # Avoid recursively rebuilding parameter names for every decode chunk.
        # Still inspect the live parameter dictionaries, and rebuild the flat
        # module list if a child is added, removed, or replaced.
        if any(module._modules != children for module, children in self._module_layout):
            self._module_layout = tuple((module, module._modules.copy()) for module in self.model.modules())
        return tuple((id(p), p._version, p.device, p.dtype)
                     for module, _ in self._module_layout
                     for p in module._parameters.values() if p is not None)

    def _check(self) -> tuple[PagedAttentionCache, FIFOScheduler]:
        if self._cache is None or self._scheduler is None:
            raise RuntimeError("Continuous decode session is closed.")
        if (self.dtype, self.autocast_dtype, self.residual_dtype) != self._precision:
            raise RuntimeError("Decode precision changed; create a new decode session")
        backbone = self._backbone_ref() if self._backbone_ref is not None else None
        if backbone is not None and (backbone.autocast_dtype, backbone.residual_dtype) != self._precision[1:]:
            raise RuntimeError("Backbone precision changed; create a new decode session")
        if self.model.training:
            raise RuntimeError("Continuous decoding requires model.eval().")
        if torch.is_autocast_enabled(self.device.type):
            raise RuntimeError("Declare autocast_dtype on the backbone; remove ambient torch.autocast.")
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
        lengths = [n + len(self._deferred.get(row, ())) for row, n in enumerate(self._lengths)]
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

    def defer(self, row: int, embeds: torch.Tensor) -> None:
        """Retain output-free history and process it with the row's next input.

        No token is dropped: deferred embeddings precede the next submission at
        their original rotary positions. Resetting the episode discards them.
        """
        self.defer_chunks({row: embeds})

    def defer_chunks(self, chunks: dict[int, torch.Tensor]) -> None:
        """Retain a batch of output-free suffixes after one weight/capacity check.

        Validate every row before changing deferred history. Each suffix is copied
        so caller mutation cannot change a later observation's cached context.
        """
        self._check()
        for row, embeds in chunks.items():
            self._validate_submission(row, embeds)
        self._check_capacity({row: len(embeds) for row, embeds in chunks.items()})
        prepared = {}
        for row, embeds in chunks.items():
            embeds = embeds.detach().to(device=self.device, dtype=self.residual_dtype)
            previous = self._deferred.get(row)
            prepared[row] = embeds.clone() if previous is None else torch.cat((previous, embeds))
        self._deferred.update(prepared)

    def _enqueue(self, row: int, embeds: torch.Tensor, *, output_hidden_states: bool) -> None:
        """Queue an input after the public entry point has validated the batch."""
        assert self._scheduler is not None
        embeds = embeds.detach().to(device=self.device, dtype=self.residual_dtype)
        prefix = self._deferred.pop(row, None)
        prefix_length = 0 if prefix is None else len(prefix)
        if prefix is not None:
            embeds = torch.cat((prefix, embeds))
        self._pending[row] = _Submission(embeds, output_hidden_states, prefix_length)
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
        return self._step(*self._check())

    def _step(self, cache, scheduler) -> dict[int, Readout]:
        """Run one scheduler chunk after the public operation validated weights."""
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
            # The HF page table addresses layer-major KV pages directly.
            past_lengths = np.asarray([state.position_offset for state in states], dtype=np.int64)
            query_lengths = np.asarray(chunks, dtype=np.int64)
            query_width = max(chunks)
            page_size = stride = allocator.tokens_per_page
            assert self._kv_cache is not None
            cache_rows = self._kv_cache.shape[2]
            num_blocks = (cache_rows + stride - 1) // stride
            owners = np.full(num_blocks, -1, dtype=np.int64)
            logical_pages = np.zeros(num_blocks, dtype=np.int64)
            q_blocks = (query_width + _BLOCK_SIZE - 1) // _BLOCK_SIZE
            # Page-index tables cover the fixed pool. Actual block counts still
            # bound every attention read; history growth only updates indices,
            # rather than invalidating a captured decoder graph's tensor shapes.
            max_pages = self.cache_capacity_tokens // page_size
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
            metadata_gpu = torch.from_numpy(np.concatenate(metadata)).to(self.device)
            block_arrays = [block_counts, indices, full_counts, full_indices]
            blocks_gpu = torch.from_numpy(np.concatenate([array.ravel() for array in block_arrays])).to(self.device)
            need_layers = any(self._pending[row].output_hidden_states for row in rows)
            output, hidden = self._decode(
                h, metadata_gpu, blocks_gpu,
                tuple(len(column) for column in metadata), tuple(array.shape for array in block_arrays),
                query_width, cache_rows, stride, page_size, allocator, need_layers,
            )
            outputs = output.split(chunks)
            layer_outputs = [layer.split(chunks) for layer in hidden]
            completed = {}
            for index, (row, state, count, output) in enumerate(zip(rows, states, chunks, outputs, strict=True)):
                pending = self._pending[row]
                pending.outputs.append(output)
                if pending.output_hidden_states:
                    pending.layers.append(tuple(layer[index] for layer in layer_outputs))
                state.position_offset += count
                self._lengths[row] += count
                if not state.remaining_prefill_tokens:
                    # A single chunk already owns stable decoder output storage.
                    # Only split prefills need a concatenation; copying every row
                    # adds a GPU launch per request at every generation step.
                    def join(parts):
                        return (parts[0] if len(parts) == 1 else torch.cat(parts))[pending.prefix_length:]
                    out = join(pending.outputs)
                    completed[row] = (out, tuple(join(parts) for parts in zip(*pending.layers, strict=True))) if pending.output_hidden_states else out
                    # finish_request would free KV: an RL environment pauses here,
                    # it has not finished its outer episode.
                    scheduler.active_requests.pop(state.request_id)
                    del self._pending[row]
            return completed
        except Exception:
            # A failed layer may have written only part of the cache. Never reuse it.
            self.close()
            raise

    def _decode(self, h, metadata, blocks, sizes, shapes, query_width, cache_rows,
                stride, page_size, allocator, need_layers):
        kv_cache = self._kv_cache
        assert kv_cache is not None
        # Capture only shapes that recur. Ragged scheduler batches and all KV
        # mappings remain unchanged; no projection/MLP padding is introduced.
        signature = (h.shape, h.dtype, sizes, shapes, query_width, need_layers)
        use_graph = False
        uses = 0
        if _use_flex_compile(self.device):
            uses = self._graph_counts.pop(signature, 0) + 1
            if len(self._graph_counts) >= _MAX_TRACKED_SHAPES:
                self._graph_counts.popitem(last=False)
            self._graph_counts[signature] = uses
            use_graph = signature in self._graphs or uses >= _GRAPH_REUSE_THRESHOLD
        entry = self._graphs.pop(signature, None) if use_graph else None
        if entry is not None:
            graph, static, result = entry
            for dest, source in zip(static, (h, metadata, blocks), strict=True):
                dest.copy_(source)
            graph.replay()
            self._graphs[signature] = entry
        else:
            static = tuple(t.clone() for t in (h, metadata, blocks)) if use_graph else (h, metadata, blocks)
            initial, meta, block_data = static
            past, counts, owners, logical, qp, wi, qi = meta.split(sizes)
            count, index, full_count, full_index = [
                t.view(shape) for t, shape in zip(
                    block_data.split([int(np.prod(shape)) for shape in shapes]), shapes, strict=True)]
            mask = BlockMask.from_kv_blocks(
                count, index, full_count, full_index,
                BLOCK_SIZE=(_BLOCK_SIZE, page_size),
                mask_mod=_paged_mask(past, counts, owners, logical, stride, page_size),
                seq_lengths=(query_width, cache_rows), compute_q_blocks=False,
            )
            cfg = self.model.config
            head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            def run():
                current = initial
                cos, sin = self.model.rotary_emb(current.new_empty((1, 0, current.shape[-1]), dtype=self.dtype), qp.unsqueeze(0))
                hidden = []
                context = nullcontext() if self.autocast_dtype is None else torch.autocast(self.device.type, dtype=self.autocast_dtype)
                with context:
                    for layer_index, layer in enumerate(self.model.layers):
                        key_cache, value_cache = kv_cache[layer_index].unbind(0)
                        q, k, v = self._inputs(layer, current, cos, sin, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim)
                        key_cache.index_copy_(0, wi, k[0].transpose(0, 1))
                        value_cache.index_copy_(0, wi, v[0].transpose(0, 1))
                        current = self._layer(layer, current, q, cfg.num_attention_heads, cfg.num_key_value_heads,
                                              head_dim, key_cache, value_cache, qi, query_width, mask, self.attention_tiles)
                        if need_layers:
                            hidden.append(current)
                    return self.model.norm(current), tuple(hidden)
            if use_graph:
                if len(self._graphs) >= _MAX_DECODE_GRAPHS:
                    self._graphs.popitem(last=False)
                # Warm up compilation and library work outside capture. Writes
                # target the same new KV positions, so repeating them is safe.
                stream = self._capture_stream
                first_capture = stream is None
                if stream is None:
                    stream = self._capture_stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                # This exact shape has already run before capture. Initialize
                # stream-local library state only on the first use of the shared
                # capture stream, rather than rerunning every captured shape.
                if first_capture or uses < 2:
                    with torch.cuda.stream(stream):
                        run()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                # Graphs run serially and outputs are copied before any replay:
                # their intermediate storage can share one bounded pool.
                with torch.cuda.graph(graph, stream=stream, pool=self._graph_pool):
                    result = run()
                graph.replay()
                self._graphs[signature] = (graph, static, result)
            else:
                result = run()
        if use_graph:
            # Pending chunks and callers must not alias the next replay output.
            return result[0].clone(), tuple(t.clone() for t in result[1])
        return result

    @torch.no_grad()
    def forward_chunks(self, chunks: dict[int, torch.Tensor], *,
                       output_hidden_states: bool = False) -> dict[int, Readout]:
        """Process unpadded row inputs, retaining each row's complete KV history.

        Validate the entire batch before submitting any row. Ascending row order
        matches ``forward`` scheduling even if the mapping was built out of order.
        Results contain only submitted rows and exclude deferred prefix outputs.
        """
        cache, scheduler = self._check()
        if self._pending:
            raise RuntimeError("Drain pending submit/step inputs before calling forward_chunks.")
        for row, embeds in chunks.items():
            self._validate_submission(row, embeds)
        self._check_capacity({row: len(embeds) for row, embeds in chunks.items()})
        for row in sorted(chunks):
            self._enqueue(row, chunks[row], output_hidden_states=output_hidden_states)
        result: dict[int, Readout] = {}
        while self._pending:
            result.update(self._step(cache, scheduler))
        return result

    @torch.no_grad()
    def forward(self, *, embeds: torch.Tensor, lengths: list[int], output_hidden_states: bool = False) -> Readout:
        """Process a left-padded batch; zero-length rows do not enter the decoder."""
        cache, scheduler = self._check()
        if self._pending:
            raise RuntimeError("Drain pending submit/step inputs before calling forward.")
        if (embeds.ndim != 3 or embeds.shape[0] != self.B
                or embeds.shape[2] != self.model.config.hidden_size or len(lengths) != self.B
                or any(type(n) is not int or not 0 <= n <= embeds.shape[1] for n in lengths)):
            raise ValueError("forward requires [batch_size, tokens, hidden_dim] and valid per-row lengths.")
        self._check_capacity(dict(enumerate(lengths)))
        output = torch.zeros(embeds.shape, device=self.device, dtype=self.residual_dtype)
        layers = tuple(torch.zeros_like(output) for _ in self.model.layers) if output_hidden_states else ()
        for row, count in enumerate(lengths):
            if count:
                self._enqueue(row, embeds[row, -count:], output_hidden_states=output_hidden_states)
        while self._pending:
            # forward is one serialized operation with fixed weights. Public
            # step still validates every independently submitted operation.
            for row, result in self._step(cache, scheduler).items():
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
            for state in self.generation_state:
                state.clear()
            self._graphs.clear()
            self._graph_counts.clear()
            self._cache.reset()
            self._scheduler.reset()
            self._pending.clear()
            self._deferred.clear()
            self._lengths = [0] * self.B
            # A changed layout requires allocating a new pool, even after reset.
            if module_device_dtype(self.model) != (self.device, self.dtype):
                self.close()
                raise RuntimeError("Model device/dtype changed; create a new decode session.")
            self._versions = self._weight_versions()
        else:
            self._check()
            for row in selected:
                self.generation_state[row].clear()
                self._scheduler.set_request_cancellation(str(row))
                self._pending.pop(row, None)
                self._deferred.pop(row, None)
                self._lengths[row] = 0
            self._scheduler.clear_cancelled_requests()

    def close(self) -> None:
        """Release the HF scheduler, KV pool, and pending embeddings. Idempotent."""
        self._graphs.clear()
        self._graph_counts.clear()
        self._pending.clear()
        self._deferred.clear()
        for state in self.generation_state:
            state.clear()
        self._scheduler = None
        self._kv_cache = None
        self._cache = None
        self._capture_stream = None
