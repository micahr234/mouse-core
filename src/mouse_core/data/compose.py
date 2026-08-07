"""Compose per-step pipeline callables."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def _accepts_augment(stage: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(stage).parameters
    except (TypeError, ValueError):
        return False
    return "augment" in params


class _Compose:
    """Callable pipeline with optional ``reseed`` forwarding to stages."""

    __slots__ = ("_stages", "_accepts_augment")

    def __init__(self, stages: tuple[Callable[[Any], Any], ...]) -> None:
        self._stages = stages
        self._accepts_augment = tuple(_accepts_augment(stage) for stage in stages)

    def __call__(self, value: Any, *, augment: bool = True) -> Any:
        for stage, accepts in zip(self._stages, self._accepts_augment):
            value = stage(value, augment=augment) if accepts else stage(value)
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
    Calling with ``augment=False`` forwards that flag to stages whose signature
    declares an ``augment`` parameter (the :class:`Augmenter` skips itself), so
    eval / decode can reuse the train pipeline on raw values.

    Typical train/eval pipeline::

        transform = compose(augmenter, grouper, selector, tokenizer)
        step_tokens = transform(step)                 # train: augmented
        transform.reseed()                            # new draws for the next batch
        step_tokens = transform(step, augment=False)  # eval / decode: raw values
    """
    if not stages:
        raise ValueError("compose requires at least one stage")
    return _Compose(stages)
