"""Unit tests for grow-then-rebuild KV cache policy helpers."""

from __future__ import annotations

import numpy as np

from mouse_core.models.kv_policy import (
    cache_needs_rebuild,
    rebuild_starts,
    resolve_cache_bounds,
)


def test_resolve_cache_bounds_defaults() -> None:
    max_c, start_c = resolve_cache_bounds(512, None)
    assert max_c == 512
    assert start_c == 256


def test_resolve_cache_bounds_clamps_start() -> None:
    max_c, start_c = resolve_cache_bounds(512, 512)
    assert max_c == 512
    assert start_c == 511


def test_rebuild_starts_prefers_start_cache() -> None:
    ends = np.array([600, 600])
    starts = rebuild_starts(ends, context_start=0, start_cache=256, max_cache=512)
    assert starts.tolist() == [344, 344]


def test_rebuild_starts_respects_context_start() -> None:
    ends = np.array([100])
    starts = rebuild_starts(ends, context_start=80, start_cache=256, max_cache=512)
    assert starts.tolist() == [80]


def test_cache_needs_rebuild_on_overflow() -> None:
    assert cache_needs_rebuild(
        has_cache=True,
        cached_starts=np.array([0]),
        cached_ends=np.array([512]),
        ends=np.array([513]),
        context_start=np.array([0]),
        max_cache=512,
        batch_complete=True,
    )


def test_cache_allows_grow_within_max() -> None:
    assert not cache_needs_rebuild(
        has_cache=True,
        cached_starts=np.array([100]),
        cached_ends=np.array([356]),
        ends=np.array([357]),
        context_start=np.array([0]),
        max_cache=512,
        batch_complete=True,
    )


def test_cache_needs_rebuild_on_context_clear() -> None:
    assert cache_needs_rebuild(
        has_cache=True,
        cached_starts=np.array([0]),
        cached_ends=np.array([10]),
        ends=np.array([20]),
        context_start=np.array([15]),
        max_cache=512,
        batch_complete=True,
    )
