"""Grow-then-rebuild KV cache policy for cached inference.

Instead of sliding the window (and rebuilding) every step once history exceeds
``max_cache``, keep a contiguous cached segment that grows from a ``start_cache``
prefill until it hits ``max_cache``, then rebuild by prefilling the latest
``start_cache`` steps again.
"""

from __future__ import annotations

import numpy as np


def resolve_cache_bounds(max_cache: int, start_cache: int | None = None) -> tuple[int, int]:
    """Return ``(max_cache, start_cache)`` with grow room when possible.

    ``start_cache`` defaults to ``max_cache // 2`` (at least 1). When
    ``start_cache >= max_cache``, it is clamped to ``max(1, max_cache - 1)`` so
    a full cache can still accept at least one incremental step before the next
    rebuild (except ``max_cache == 1``, which rebuilds every step).
    """
    max_cache = max(1, int(max_cache))
    if start_cache is None:
        start_cache = max(1, max_cache // 2)
    else:
        start_cache = max(1, int(start_cache))
    if start_cache >= max_cache:
        start_cache = max(1, max_cache - 1)
    return max_cache, start_cache


def rebuild_starts(
    *,
    ends: np.ndarray,
    context_start: np.ndarray | int,
    start_cache: int,
    max_cache: int,
) -> np.ndarray:
    """Inclusive history indices for a rebuild prefill of up to ``start_cache`` steps."""
    ends = np.asarray(ends, dtype=np.int64)
    ctx = np.asarray(context_start, dtype=np.int64)
    span = min(int(start_cache), int(max_cache))
    return np.maximum(ctx, ends - span)


def cache_needs_rebuild(
    *,
    has_cache: bool,
    cached_starts: np.ndarray,
    cached_ends: np.ndarray,
    ends: np.ndarray,
    context_start: np.ndarray,
    max_cache: int,
    batch_complete: bool,
) -> bool:
    """Whether the grow-rebuild policy must prefill instead of appending."""
    if not has_cache or not batch_complete:
        return True
    ends = np.asarray(ends, dtype=np.int64)
    cached_starts = np.asarray(cached_starts, dtype=np.int64)
    cached_ends = np.asarray(cached_ends, dtype=np.int64)
    context_start = np.asarray(context_start, dtype=np.int64)
    if ends.shape != cached_starts.shape or ends.shape != cached_ends.shape:
        return True
    if np.any(ends < cached_ends):
        return True
    if np.any(context_start > cached_starts):
        return True
    # Appending would make the cached span exceed max_cache.
    if np.any(ends - cached_starts > int(max_cache)):
        return True
    return False
