"""Callable emission condition types and save/load refs.

Tokenizer ``when=`` takes a callable — not an equals / not_equals dict.
DataLoader accepts ``sample_field`` for contiguous equal-value groups or
``sample_ranges`` for a domain-specific source-event grouping function.

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
Notebook and script functions use ``__main__:name`` references. Define the same
named conditions in the loading notebook or script before calling ``load_tokenizer``.
Lambdas and nested functions work at runtime but cannot be saved.

DataLoader ``sample_field``
--------------------------
A sample is every contiguous row that shares one value in
``sample_field``. Each sample is one group. Attention and loss stop
where ``group_id`` changes. Pass exactly one of ``events_budget`` (source events) or
``token_budget`` (packed tokens). A sample that does not fit is left
out. Examples pass ``batch_size=1`` and ``sample_ranges=environment_episode_ranges``.
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
            "tokenizer when= must be a named top-level function to "
            "save/load; got "
            f"{fn!r}. Define the condition at notebook or module scope."
        )
    ref = f"{module}:{qualname}"
    try:
        resolved = resolve_when_ref(ref)
    except (ImportError, AttributeError, ValueError) as exc:
        raise TypeError(f"tokenizer when= function is not importable as {ref!r}") from exc
    if resolved != fn:
        raise TypeError(f"tokenizer when= function does not match its import reference {ref!r}")
    return ref


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
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            if module_name == "__main__":
                raise ValueError(
                    f"Define tokenizer condition {qualname!r} in this notebook or script "
                    "before loading the tokenizer"
                ) from exc
            raise
    if not callable(obj):
        raise TypeError(f"tokenizer when ref {ref!r} is not callable")
    return cast(WhenFn, obj)
