"""Save/load references for named format callbacks.

Callbacks receive the event payload and return text; an empty string emits no
tokens. They may declare ``*, context`` to receive separate runtime settings. Define notebook callbacks before loading their saved tokenizer. Lambdas
and nested functions work at runtime but cannot be saved.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any, cast


def callable_ref(fn: Callable[..., Any]) -> str:
    """Import path for a named tokenizer callback (``module:qualname``)."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        raise TypeError(
            "tokenizer callback must be a named top-level function to "
            "save/load; got "
            f"{fn!r}. Define the callback at notebook or module scope."
        )
    ref = f"{module}:{qualname}"
    try:
        resolved = resolve_callable_ref(ref)
    except (ImportError, AttributeError, ValueError) as exc:
        raise TypeError(
            f"tokenizer callback function is not importable as {ref!r}"
        ) from exc
    if resolved != fn:
        raise TypeError(
            f"tokenizer callback function does not match its import reference {ref!r}"
        )
    return ref


def resolve_callable_ref(ref: str) -> Callable[..., Any]:
    """Import a tokenizer callback from ``module:qualname``."""
    if ":" not in ref:
        raise ValueError(
            f"tokenizer callback ref must be 'module:qualname', got {ref!r}"
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
                    f"Define tokenizer callback {qualname!r} in this notebook or script "
                    "before loading the tokenizer"
                ) from exc
            raise
    if not callable(obj):
        raise TypeError(f"tokenizer callback ref {ref!r} is not callable")
    return cast(Callable[..., Any], obj)


def accepts_context(fn: Callable[..., Any]) -> bool:
    """Inspect once at setup; context is an explicit keyword-only opt-in."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "context"
            and parameter.kind == inspect.Parameter.KEYWORD_ONLY
        )
        for parameter in parameters
    )
