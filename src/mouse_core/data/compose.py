"""Compose per-step pipeline callables."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _Compose:
    """Callable pipeline with optional ``reseed`` forwarding to stages."""

    __slots__ = ("_stages",)

    def __init__(self, stages: tuple[Callable[[Any], Any], ...]) -> None:
        self._stages = stages

    def __call__(self, value: Any) -> Any:
        for stage in self._stages:
            value = stage(value)
        return value

    def reseed(self, seed: int | None = None) -> None:
        """Call ``reseed`` on every stage that defines it."""
        for stage in self._stages:
            fn = getattr(stage, "reseed", None)
            if callable(fn):
                fn(seed=seed)


def compose(*stages: Callable[[Any], Any]) -> _Compose:
    """Return ``fn`` such that ``fn(x) == stages[-1](...stages[0](x)...)``.

    The result is callable and exposes ``reseed(...)``, which forwards to any
    stage that defines ``reseed`` (e.g. :class:`~mouse_core.data.augmenter.Augmenter`).

    Train includes the augmenter; eval leaves it out so the model sees raw
    values::

        train_transform = compose(augmenter, selector, tokenizer)
        eval_transform = compose(selector, tokenizer)
        step_tokens = train_transform(step)
        train_transform.reseed()
        step_tokens = eval_transform(step)
    """
    if not stages:
        raise ValueError("compose requires at least one stage")
    return _Compose(stages)
