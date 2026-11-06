"""Compose event pipeline callables with optional runtime context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mouse_core.data.callbacks import accepts_context


class _Compose:
    """Callable pipeline with optional ``reseed`` forwarding to stages."""

    __slots__ = ("_stages", "_context_stages")

    def __init__(self, stages: tuple[Callable[..., Any], ...]) -> None:
        self._stages = stages
        self._context_stages = tuple(accepts_context(stage) for stage in stages)

    def __call__(self, value: Any, *, context=None) -> Any:
        for stage, uses_context in zip(self._stages, self._context_stages):
            value = stage(value, context=context) if uses_context else stage(value)
        return value

    def reseed(self, *, generation: int | None = None) -> None:
        """Call ``reseed(generation=...)`` on every stage that defines it."""
        for stage in self._stages:
            fn = getattr(stage, "reseed", None) or getattr(
                getattr(stage, "__self__", None), "reseed", None
            )
            if callable(fn):
                fn(generation=generation)


def compose(*, stages: Sequence[Callable[..., Any]]) -> _Compose:
    """Return ``fn`` such that ``fn(x) == stages[-1](...stages[0](x)...)``.

    ``pipeline(events, context=...)`` forwards runtime context to stages declaring
    ``*, context`` (or ``**kwargs``). Other stages receive only their normal input.
    Signature inspection happens once when constructing the pipeline.

    The result is callable and exposes ``reseed(generation=None)``, which
    forwards to any stage that defines ``reseed`` (e.g.
    :class:`~mouse_core.data.augmenter.Augmenter`). ``DataLoader`` calls it
    once per sampled segment with a generation unique to that segment:
    ``batch_index * (budget + 1) + attempt``,
    where ``budget`` is ``events_budget`` or ``token_budget``.

    Train includes the augmenter; eval leaves it out so the model sees raw
    values::

        train_transform = compose(stages=(augmenter.events, tokenizer.encode))
        eval_transform = tokenizer.encode
        step_tokens = train_transform(step)
        train_transform.reseed()
        step_tokens = eval_transform(step)
    """
    if not stages:
        raise ValueError("compose requires at least one stage")
    return _Compose(tuple(stages))
