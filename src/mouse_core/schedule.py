"""Schedules over optimizer steps.

``ExponentialDecay`` multiplies by ``decay`` every step: ``value * decay**x``
(``decay == 1`` holds ``value``). ``Piecewise`` maps a scalar to a value through a sorted list of
``(x, y)`` knots. Between knots it interpolates ``"linear"``,
``"geometric"`` (linear in ``log(y)``; use for rates that span orders of
magnitude), or ``"constant"`` (hold the left knot). Outside the knots it
clamps to the first / last ``y``. One knot is a constant.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Literal, Sequence

Interpolation = Literal["linear", "geometric", "constant"]


@dataclass(frozen=True)
class ExponentialDecay:
    """``x -> value * decay**x``: the value is multiplied by ``decay`` each step.

    Args:
        value: Value at ``x = 0``.
        decay: Per-step multiplicative factor in ``(0, 1]``. ``1`` holds
            ``value``.
    """

    value: float
    decay: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value}.")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError(f"decay must be in (0, 1], got {self.decay}.")

    def __call__(self, x: float) -> float:
        if self.decay == 1.0:
            return float(self.value)
        return float(self.value) * self.decay**x


@dataclass(frozen=True)
class Piecewise:
    """Piecewise schedule ``x -> y`` through ``knots``.

    Args:
        knots: ``(x, y)`` pairs with strictly increasing ``x``. At least one.
        interpolation: ``"linear"`` lerps ``y``; ``"geometric"`` lerps
            ``log(y)`` (all ``y`` must be ``> 0``); ``"constant"`` holds each
            knot's ``y`` until the next knot.
    """

    knots: tuple[tuple[float, float], ...]
    interpolation: Interpolation = "linear"

    def __init__(
        self,
        knots: Sequence[tuple[float, float]],
        *,
        interpolation: Interpolation = "linear",
    ) -> None:
        if interpolation not in ("linear", "geometric", "constant"):
            raise ValueError(
                "interpolation must be 'linear', 'geometric', or 'constant', "
                f"got {interpolation!r}."
            )
        pairs = tuple((float(x), float(y)) for x, y in knots)
        if not pairs:
            raise ValueError("Piecewise needs at least one knot.")
        for (x0, _), (x1, _) in zip(pairs, pairs[1:]):
            if not x1 > x0:
                raise ValueError(f"knot x must be strictly increasing, got {x0} then {x1}.")
        for x, y in pairs:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"knots must be finite, got ({x}, {y}).")
            if interpolation == "geometric" and y <= 0.0:
                raise ValueError(f"geometric interpolation needs y > 0, got y={y} at x={x}.")
        object.__setattr__(self, "knots", pairs)
        object.__setattr__(self, "interpolation", interpolation)

    @property
    def xs(self) -> tuple[float, ...]:
        return tuple(x for x, _ in self.knots)

    @property
    def ys(self) -> tuple[float, ...]:
        return tuple(y for _, y in self.knots)

    def __call__(self, x: float) -> float:
        """Value at ``x``; clamps to the first / last knot outside the range."""
        xs = self.xs
        ys = self.ys
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_right(xs, x)  # xs[i-1] <= x < xs[i]
        x0, y0 = xs[i - 1], ys[i - 1]
        x1, y1 = xs[i], ys[i]
        if self.interpolation == "constant":
            return y0
        t = (x - x0) / (x1 - x0)
        if self.interpolation == "geometric":
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
        return y0 + t * (y1 - y0)
