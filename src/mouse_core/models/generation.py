"""Shared inference scheduler for token and latent generation."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

import torch

from mouse_core.models.backbone.base import BaseBackbone
from mouse_core.models.backbone.continuous_decode import ContinuousDecodeSession


@torch.no_grad()
def generate(
    *, backbone: BaseBackbone, prompts: Sequence[torch.Tensor],
    advance: Callable[[list[int], torch.Tensor], Sequence[torch.Tensor | None]],
    backend: Literal["uncached", "continuous"], max_active_requests: int,
    max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
    cache: ContinuousDecodeSession | None = None, cache_rows: Sequence[int] | None = None,
) -> None:
    """Generate with bounded concurrency and immediate reuse of finished slots.

    Each prompt is a nonempty [tokens, hidden_dim] embedding tensor. After each
    forward, ``advance(request_ids, hidden)`` receives stable original request IDs
    and their last hidden states [active_requests, hidden_dim]. It records outputs
    and returns each request's next nonempty embedding chunk, or None to finish.
    The callback owns sampling, stopping, and output storage, including a finite
    generation cap. A replacement prompt enters on the next iteration alongside
    continuing requests; cache rows and rotary positions reset before reuse.

    A caller-owned cache retains episode history between calls. ``cache_rows``
    maps prompts to distinct episode rows. All generated tokens remain cached;
    the caller resets episodes and closes its cache before changing weights.
    Continuous mode caches prefixes; uncached mode recomputes full histories.
    The caller must disable dropout and keep weights fixed for the entire call.
    This synchronous inference function does not overlap optimizer updates.
    """
    if backend not in ("uncached", "continuous"):
        raise ValueError("backend must be uncached or continuous")
    if type(max_active_requests) is not int or max_active_requests < 1:
        raise ValueError("max_active_requests must be a positive integer")
    def check(chunk: torch.Tensor) -> None:
        if chunk.ndim != 2 or not len(chunk) or chunk.shape[-1] != backbone.hidden_dim:
            raise ValueError("Generation chunks must have nonempty shape [tokens, hidden_dim]")
    for prompt in prompts:
        check(prompt)
    if not prompts:
        return
    owned_cache = cache is None
    if cache is not None:
        if backend != "continuous":
            raise ValueError("A persistent cache requires the continuous backend")
        if cache_rows is None or len(cache_rows) != len(prompts) or len(set(cache_rows)) != len(prompts):
            raise ValueError("cache_rows must map each prompt to a distinct cache row")
        for row in cache_rows:
            cache._row(row)
        cache._check()
    elif cache_rows is not None:
        raise ValueError("cache_rows requires a caller-owned cache")
    slots = min(len(prompts), max_active_requests)
    active: list[int | None] = list(range(slots))
    pending = slots
    chunks = list(prompts[:slots])
    histories = list(chunks)
    if owned_cache and backend == "continuous":
        cache = backbone.decode_session(batch_size=slots, max_batch_tokens=max_batch_tokens,
                                        cache_capacity_tokens=cache_capacity_tokens)
    try:
        while any(request is not None for request in active):
            live = [slot for slot, request in enumerate(active) if request is not None]
            requests = [cast(int, active[slot]) for slot in live]
            if cache is not None:
                physical = [slot if owned_cache else cast(Sequence[int], cache_rows)[request]
                            for slot, request in zip(live, requests, strict=True)]
                lengths = [0] * cache.B
                for slot, row in zip(live, physical, strict=True):
                    lengths[row] = len(chunks[slot])
                padded = prompts[0].new_zeros((cache.B, max(lengths), backbone.hidden_dim))
                for slot, row in zip(live, physical, strict=True):
                    padded[row, -lengths[row]:] = chunks[slot]
                hidden = cast(torch.Tensor, backbone(padded, cache=cache, lengths=lengths))[physical, -1]
            else:
                rows = [histories[slot] for slot in live]
                lengths = torch.tensor([len(row) for row in rows], device=rows[0].device)
                ids = torch.repeat_interleave(torch.arange(len(rows), device=rows[0].device), lengths)
                features = cast(torch.Tensor, backbone(torch.cat(rows), sequence_ids=ids))
                hidden = features[lengths.cumsum(0) - 1]
            next_chunks = advance(requests, hidden)
            if len(next_chunks) != len(live):
                raise ValueError("advance must return one chunk or None for each active request")
            reset: list[int] = []
            for slot, chunk in zip(live, next_chunks, strict=True):
                if chunk is None:
                    reset.append(slot if owned_cache else cast(Sequence[int], cache_rows)[cast(int, active[slot])])
                    if pending < len(prompts):
                        active[slot] = pending
                        chunks[slot] = histories[slot] = prompts[pending]
                        pending += 1
                    else:
                        active[slot] = None
                        chunks[slot] = histories[slot] = prompts[0][:0]
                else:
                    check(chunk)
                    chunks[slot] = chunk
                    if cache is None:
                        histories[slot] = torch.cat((histories[slot], chunk))
            if cache is not None and reset:
                if owned_cache:
                    cache.reset_rows(reset)
    except BaseException:
        # A failed callback may leave a partially generated suffix; never reuse it.
        if cache is not None:
            cache.close()
        raise
    finally:
        if cache is not None and owned_cache:
            cache.close()
