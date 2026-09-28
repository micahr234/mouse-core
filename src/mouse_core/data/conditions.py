"""Callable emission / sample-window conditions.

Tokenizer ``when=`` and ``DataLoader`` ``sample_start`` / ``sample_end``
take callables — not equals / not_equals / group_start dicts.

Tokenizer ``when``
-----------------
A ``when`` callable receives a per-step context mapping and returns a
Python ``bool`` (or 0-d numpy / torch bool). The context is the step
dict plus a boolean ``group_start`` key injected by the tokenizer:

* Evaluate with ``group_start=False`` for ordinary step tokens.
* Evaluate with ``group_start=True`` for pack-time ``group_start_*``
  tokens. When the ordinary call is false and the group-start call is
  true, tokens ride ``group_start_*`` for
  :func:`~mouse_core.data.token_batch.pack_token_batch`.

OR of several reasons to emit is written in the callable with ``|`` /
``or``::

    when=lambda ctx: (ctx.get("step_index") == 0) | ctx["group_start"]

Omit ``when`` (``None``) and the field always emits on the ordinary run.

Named module-level functions (or the helpers below) round-trip through
:func:`~mouse_core.data.tokenizer.save_tokenizer` /
:func:`~mouse_core.data.tokenizer.load_tokenizer` via an import path.
Lambdas work at runtime but cannot be saved.

DataLoader ``sample_start`` / ``sample_end``
-------------------------------------------
Each callable receives a column mapping ``name → 1-d array`` over the
rows being considered and must return a boolean numpy array of the
same length. ``sample_start`` marks legal window starts (the matching
row itself). ``sample_end`` marks the inclusive end row; if set but
never met before ``sequence_length`` / store end, sampling raises.
``None`` leaves starts unrestricted / never truncates early.

Full-task FrozenLake windows::

    sample_start=full_task_start
    sample_end=full_task_end
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from mouse_core.data.modality import values_equal

# Per-step tokenizer gate: ctx → bool.
WhenFn = Callable[[Mapping[str, Any]], bool]
# DataLoader window predicate: column arrays → bool mask.
SampleFn = Callable[[Mapping[str, Any]], np.ndarray]

GROUP_START_KEY = "group_start"


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
    """Start after each ``field != value`` row (plus store index 0).

    Matches the old ``SampleBoundary`` start semantics. Returns a named
    module-level-style callable only when used as a factory result —
    prefer writing the mask inline when the logic is one-off.
    """

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


def when_ref(fn: WhenFn) -> str:
    """Import path for a named ``when`` callable (``module:qualname``)."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise TypeError(
            "tokenizer when= must be a named module-level function to "
            "save/load; got "
            f"{fn!r}. Prefer helpers in mouse_core.data.conditions "
            "(e.g. when_group_start) or a def at module scope."
        )
    return f"{module}:{qualname}"


def resolve_when_ref(ref: str) -> WhenFn:
    """Import a ``when`` callable from ``module:qualname``."""
    if ":" not in ref:
        raise ValueError(
            f"tokenizer when ref must be 'module:qualname', got {ref!r}"
        )
    module_name, _, qualname = ref.partition(":")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"tokenizer when ref {ref!r} is not callable")
    return obj
