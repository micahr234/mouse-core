"""Callable emission condition types and save/load refs.

Tokenizer ``when=`` takes a callable — not an equals / not_equals dict.
DataLoader samples are not callables: ``sample_field`` names the column
whose contiguous equal values are one sample.

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

DataLoader ``sample_field``
--------------------------
A sample is every contiguous row that shares one value in
``sample_field``. Each sample is one group. Attention and loss stop
where ``group_id`` changes. Pass exactly one of ``samples_budget`` (steps) or
``token_budget`` (packed tokens). A sample that does not fit is left
out. Examples pass ``batch_size=1`` and ``sample_field="task_index"``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, cast

# Per-step tokenizer gate: ctx → bool.
WhenFn = Callable[[Mapping[str, Any]], bool]

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
