"""Callable emission / sample-window condition types and save/load refs.

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

Named module-level functions round-trip through
:func:`~mouse_core.data.tokenizer.save_tokenizer` /
:func:`~mouse_core.data.tokenizer.load_tokenizer` via an import path.
Lambdas work at runtime but cannot be saved. Define any named
predicates inline in the notebook or caller module — this package
does not ship convenience helpers.

DataLoader ``sample_start`` / ``sample_end``
-------------------------------------------
Each callable receives a column mapping ``name → 1-d array`` over the
rows being considered and must return a boolean numpy array of the
same length. ``sample_start`` marks legal window starts (the matching
row itself). ``sample_end`` marks the inclusive end row; if set but
never met before ``sequence_length`` / store end for a chosen start,
that draw is skipped and another start is sampled. Exhaustion or a
yielded window that lacks an end match raises. ``None`` leaves starts
unrestricted / never truncates early.

Example full-task FrozenLake windows (predicates defined by the
caller)::

    def full_task_start(cols):
        return (np.asarray(cols["episode_index"]) == 0) & (
            np.asarray(cols["step_index"]) == 0
        )

    def full_task_end(cols):
        return np.asarray(cols["task_done"]) != 0

    sample_start=full_task_start
    sample_end=full_task_end
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

# Per-step tokenizer gate: ctx → bool.
WhenFn = Callable[[Mapping[str, Any]], bool]
# DataLoader window predicate: column arrays → bool mask.
SampleFn = Callable[[Mapping[str, Any]], np.ndarray]

GROUP_START_KEY = "group_start"


def when_ref(fn: WhenFn) -> str:
    """Import path for a named ``when`` callable (``module:qualname``)."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise TypeError(
            "tokenizer when= must be a named module-level function to "
            "save/load; got "
            f"{fn!r}. Prefer a def at module scope in the caller."
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
    return cast(WhenFn, obj)
