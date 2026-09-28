"""Example-local ``when`` / ``sample_*`` callables for the notebooks.

These are FrozenLake-shaped convenience predicates. The library only
exposes the generic callable hooks; define your own (or import from
here in notebooks / benches run from the repo root).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mouse_core.data.conditions import GROUP_START_KEY, SampleFn
from mouse_core.data.modality import values_equal


def when_group_start(ctx: Mapping[str, Any]) -> bool:
    """Emit only as pack-time ``group_start_*`` tokens."""
    return bool(ctx[GROUP_START_KEY])


def when_reward_nonzero(ctx: Mapping[str, Any]) -> bool:
    """Emit when ``reward`` is present and not ``0.0``."""
    return "reward" in ctx and not values_equal(ctx["reward"], 0.0)


def when_episode_done_nonzero(ctx: Mapping[str, Any]) -> bool:
    """Emit when ``episode_done`` is present and not ``0``."""
    return "episode_done" in ctx and not values_equal(ctx["episode_done"], 0)


def when_step_index_zero(ctx: Mapping[str, Any]) -> bool:
    """Emit when ``step_index`` is present and equals ``0``."""
    return "step_index" in ctx and values_equal(ctx["step_index"], 0)


def when_step_index_zero_or_group_start(ctx: Mapping[str, Any]) -> bool:
    """Emit on ``step_index == 0`` (ordinary) or at group start."""
    step0 = "step_index" in ctx and values_equal(ctx["step_index"], 0)
    return bool(step0) | bool(ctx[GROUP_START_KEY])


def full_task_start(cols: Mapping[str, Any]) -> np.ndarray:
    """Legal starts: ``episode_index == 0`` and ``step_index == 0``."""
    return (np.asarray(cols["episode_index"]) == 0) & (
        np.asarray(cols["step_index"]) == 0
    )


def full_task_end(cols: Mapping[str, Any]) -> np.ndarray:
    """Inclusive end rows: ``task_done != 0``."""
    return np.asarray(cols["task_done"]) != 0


def after_field_ne(*, field: str, value: object) -> SampleFn:
    """Start after each ``field != value`` row (plus store index 0)."""

    def _after(cols: Mapping[str, Any]) -> np.ndarray:
        codes = np.asarray(cols[field])
        n = len(codes)
        mask = np.zeros(n, dtype=bool)
        if n == 0:
            return mask
        mask[0] = True
        if n > 1:
            mask[1:] = codes[:-1] != value
        return mask

    _after.__name__ = f"after_{field}_ne"
    _after.__qualname__ = f"after_field_ne.<locals>._after"
    return _after
