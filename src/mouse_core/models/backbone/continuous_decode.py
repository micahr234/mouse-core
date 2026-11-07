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

import torch
from torch.nn.attention.flex_attention import flex_attention
from transformers.generation import ContinuousBatchingConfig
from transformers.generation.continuous_batching.cache import PagedAttentionCache
from transformers.generation.continuous_batching.distributed import DistributedHelper
from transformers.generation.continuous_batching.requests import RequestState
from transformers.generation.continuous_batching.scheduler import FIFOScheduler

from mouse_core.models.backbone.flex_utils import (
    _BLOCK_SIZE, _FLEX_RECOMPILE_LIMIT, _use_flex_compile, check_autocast_dtype,
    flex_block_mask, module_device_dtype,
)
from mouse_core.models.backbone.packed_train import _attention_inputs, _attention_output
from mouse_core.models.backbone.attention_window import attention_window_size

Readout = torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]
_PAGE_SIZE = 16
_compiled_inputs: Any = None
_compiled_output: Any = None


def _continuous_flex_attention(q, k, v, **kwargs):
    # Keep this entry point separate from packed training's compiled function.
    # Otherwise inference dtype/layout variants consume its Dynamo recompile
    # budget and can change the kernel selected during checkpoint recomputation.
    return flex_attention(q, k, v, **kwargs)


@dataclass
class _Submission:
    embeds: torch.Tensor
    output_hidden_states: bool
    outputs: list[torch.Tensor] = field(default_factory=list)
    layers: list[tuple[torch.Tensor, ...]] = field(default_factory=list)


def _packed_mask(q_owner, k_owner, q_position, k_position, window_size):
    # The closure's tensor captures become inputs to the compiled Flex kernel.
    # Clamping protects Flex's padded block indices, including short batches.
    def mask(b, h, q, k):
        qi = q.clamp(max=q_owner.shape[0] - 1)
        ki = k.clamp(max=k_owner.shape[0] - 1)
        distance = q_position[qi] - k_position[ki]
        allowed = ((q < q_owner.shape[0]) & (k < k_owner.shape[0])
                   & (q_owner[qi] == k_owner[ki]) & (distance >= 0))
        return allowed & (distance < window_size)
    return mask


class ContinuousDecodeSession:
    """Bounded, token-budgeted inference with persistent independent episode rows.

    Create with ``backbone.decode_session``. ``forward`` accepts the
    left-padded embedding layout; decoder layers only process
    real tokens. Alternatively, call ``submit(row, embeds)`` as environments become
    ready and ``step()`` to run one scheduled batch and retrieve completed inputs.
    Calls must be serialized by the caller; there is no background worker.

    ``cache_capacity_tokens`` is the shared pool capacity, in multiples of 16;
    each live episode rounds its allocation up to a page. ``max_batch_tokens``
    limits queries in one forward. Neither setting changes the attention mask.
    HF's circular allocator retains at most distance + 1 tokens
    per row (rounded to pages); absolute positions keep advancing. Chunked input
    reads the old KV before overwriting it, including chunks longer than the ring.
    Insufficient capacity raises before submitting. The cache never discards keys
    the model's training mask would allow, or recomputes a truncated history.

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
            raise ValueError("cache_capacity_tokens must be a multiple of 16 (the KV page size).")
        self.model: Any = model
        cfg = self.model.config
        self._window_size = attention_window_size(cfg)
        if self._window_size is None:
            raise ValueError("Core requires a finite attention window.")
        self.B = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.cache_capacity_tokens = cache_capacity_tokens
        self.device, self.dtype = module_device_dtype(model)
        self.autocast_dtype = check_autocast_dtype(autocast_dtype, self.dtype)
        # A tensor bound keeps different window lengths on the same compiled
        # mask graph; full attention uses an effectively unlimited bound.
        self._mask_window = torch.tensor(
            self._window_size,
            device=self.device, dtype=torch.long,
        )
        self._cache: PagedAttentionCache | None = PagedAttentionCache(
            cfg,
            ContinuousBatchingConfig(
                max_batch_tokens=max_batch_tokens,
                num_blocks=cache_capacity_tokens // _PAGE_SIZE,
                max_memory_percent=1.0,
                page_size=_PAGE_SIZE,
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
        global _compiled_inputs, _compiled_output
        if _use_flex_compile(self.device):
            if _compiled_inputs is None:
                _compiled_inputs = torch.compile(_attention_inputs, dynamic=True)
                _compiled_output = torch.compile(_attention_output, dynamic=True)
            self._project, self._finish_layer = _compiled_inputs, _compiled_output
            self._flex = cast(Any, torch.compile)(
                _continuous_flex_attention, dynamic=True, fullgraph=True,
                recompile_limit=_FLEX_RECOMPILE_LIMIT,
            )
        else:
            self._project, self._finish_layer = _attention_inputs, _attention_output
            self._flex = flex_attention

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
        if attention_window_size(self.model.config) != self._window_size:
            raise RuntimeError("Attention window changed; close the cache and create a new session.")
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
        retained = [min(n, self._window_size) for n in lengths]
        needed = sum((n + _PAGE_SIZE - 1) // _PAGE_SIZE for n in retained) * _PAGE_SIZE
        if needed > self.cache_capacity_tokens:
            raise ValueError(
                f"KV capacity exceeded: {needed} tokens including page rounding, "
                f"capacity {self.cache_capacity_tokens}. Increase cache_capacity_tokens "
                "or reset completed outer episodes; keys visible to the mask cannot be evicted."
            )

    def submit(self, row: int, embeds: torch.Tensor, *, output_hidden_states: bool = False) -> None:
        """Queue the next embedding chunk for one row, retaining its earlier KV."""
        _, scheduler = self._check()
        self._validate_submission(row, embeds)
        self._check_capacity({row: embeds.shape[0]})
        self._pending[row] = _Submission(embeds.detach().to(device=self.device, dtype=self.dtype), output_hidden_states)
        # HF schedules lists of IDs. These are local embedding indices, never
        # vocabulary IDs; prefix sharing MUST remain disabled for numeric inputs.
        state = RequestState(request_id=str(row), initial_tokens=list(range(embeds.shape[0])))
        state.position_offset = self._lengths[row]
        scheduler.add_waiting_request(state)

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
            q_owner, k_owner, q_pos, k_pos, read, write = [], [], [], [], [], []
            allocator = cache.layer_to_allocator[0]
            for owner, (state, count) in enumerate(zip(states, chunks, strict=True)):
                past = state.position_offset
                q_owner.extend([owner] * count)
                key_count = allocator.get_seqlen_k(past, count)
                k_owner.extend([owner] * key_count)
                q_pos.extend(range(past, past + count))
                k_pos.extend(range(past + count - key_count, past + count))
                read.extend(allocator.get_read_indices(state.request_id, past, count))
                write.extend(allocator.get_write_indices(state.request_id, past, count))
            qo, ko, qp, kp, ri, wi = [torch.tensor(v, device=self.device, dtype=torch.long)
                                     for v in (q_owner, k_owner, q_pos, k_pos, read, write)]
            mask = flex_block_mask(
                _packed_mask(qo, ko, qp, kp, self._mask_window), B=1, Q_LEN=len(q_pos), KV_LEN=len(k_pos),
                device=self.device, block_size=_BLOCK_SIZE, compile_masks=_use_flex_compile(self.device),
            )
            cos, sin = self.model.rotary_emb(h.unsqueeze(0), qp.unsqueeze(0))
            cfg = self.model.config
            head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            layer_outputs = []
            need_layers = any(self._pending[row].output_hidden_states for row in rows)
            context = nullcontext() if self.autocast_dtype is None else torch.autocast(self.device.type, dtype=self.autocast_dtype)
            with context:
                for layer_index, layer in enumerate(self.model.layers):
                    q, k, v = self._project(
                        layer, h, cos, sin, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim,
                    )
                    k, v = allocator.update(
                        k[0].transpose(0, 1), v[0].transpose(0, 1), layer_index, ri, wi,
                    )
                    attention_output = self._flex(
                        q.contiguous(), k.transpose(0, 1).unsqueeze(0).contiguous(),
                        v.transpose(0, 1).unsqueeze(0).contiguous(),
                        block_mask=mask, scale=layer.self_attn.scaling,
                        enable_gqa=cfg.num_attention_heads != cfg.num_key_value_heads,
                        kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
                    )
                    h = self._finish_layer(layer, h, attention_output)
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
                self.submit(row, embeds[row, -count:], output_hidden_states=output_hidden_states)
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
            if (module_device_dtype(self.model) != (self.device, self.dtype)
                    or attention_window_size(self.model.config) != self._window_size):
                self.close()
                raise RuntimeError("Model device/dtype or attention window changed; create a new decode session.")
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
