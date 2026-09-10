"""Tests for ``mouse_core.schedule``."""

from __future__ import annotations

import math

import pytest

from mouse_core.schedule import ExponentialDecay, Piecewise


def test_exponential_decay_unit_decay_is_constant() -> None:
    f = ExponentialDecay(0.0005)
    assert f.decay == 1.0
    assert f(0) == 0.0005
    assert f(1e9) == 0.0005


def test_exponential_decay_multiplies_by_decay_each_step() -> None:
    f = ExponentialDecay(0.01, 0.5)
    assert f(0) == pytest.approx(0.01)
    assert f(1) == pytest.approx(0.005)
    assert f(2) == pytest.approx(0.0025)
    assert f(3) == pytest.approx(f(2) * 0.5)
    slow = ExponentialDecay(0.01, 0.99995)
    assert slow(20000) == pytest.approx(0.01 * 0.99995**20000)
    assert slow(100000) > 0.0


def test_exponential_decay_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        ExponentialDecay(float("nan"))
    with pytest.raises(ValueError, match=r"decay must be in \(0, 1\]"):
        ExponentialDecay(1.0, 0.0)
    with pytest.raises(ValueError, match=r"decay must be in \(0, 1\]"):
        ExponentialDecay(1.0, 1.5)
    with pytest.raises(ValueError, match=r"decay must be in \(0, 1\]"):
        ExponentialDecay(1.0, float("nan"))


def test_single_knot_is_constant() -> None:
    f = Piecewise([(0, 0.5)])
    assert f(-10) == 0.5
    assert f(0) == 0.5
    assert f(1e9) == 0.5


def test_clamps_outside_knots() -> None:
    f = Piecewise([(10, 1.0), (20, 3.0)])
    assert f(0) == 1.0
    assert f(10) == 1.0
    assert f(20) == 3.0
    assert f(25) == 3.0


def test_linear_interpolates_between_knots() -> None:
    f = Piecewise([(0, 0.0), (10, 1.0), (20, 3.0)])
    assert f(5) == pytest.approx(0.5)
    assert f(15) == pytest.approx(2.0)
    assert f(10) == pytest.approx(1.0)


def test_geometric_is_linear_in_log_space() -> None:
    f = Piecewise([(0, 0.01), (20000, 0.0005)], interpolation="geometric")
    assert f(0) == pytest.approx(0.01)
    assert f(20000) == pytest.approx(0.0005)
    assert f(10000) == pytest.approx(math.sqrt(0.01 * 0.0005))
    # Monotone over the segment.
    ys = [f(x) for x in range(0, 20001, 500)]
    assert all(a > b for a, b in zip(ys, ys[1:]))


def test_constant_holds_left_knot() -> None:
    f = Piecewise([(0, 1.0), (10, 2.0), (20, 4.0)], interpolation="constant")
    assert f(0) == 1.0
    assert f(9.99) == 1.0
    assert f(10) == 2.0
    assert f(19) == 2.0
    assert f(20) == 4.0
    assert f(99) == 4.0


def test_accepts_int_knots_and_exposes_axes() -> None:
    f = Piecewise(((0, 1), (5, 2)))
    assert f.xs == (0.0, 5.0)
    assert f.ys == (1.0, 2.0)
    assert f.knots == ((0.0, 1.0), (5.0, 2.0))
    assert f.interpolation == "linear"


def test_rejects_bad_knots() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Piecewise([])
    with pytest.raises(ValueError, match="strictly increasing"):
        Piecewise([(0, 1.0), (0, 2.0)])
    with pytest.raises(ValueError, match="strictly increasing"):
        Piecewise([(5, 1.0), (1, 2.0)])
    with pytest.raises(ValueError, match="finite"):
        Piecewise([(0, float("nan"))])
    with pytest.raises(ValueError, match="y > 0"):
        Piecewise([(0, 1.0), (1, 0.0)], interpolation="geometric")
    with pytest.raises(ValueError, match="interpolation must be"):
        Piecewise([(0, 1.0)], interpolation="cubic")  # type: ignore[arg-type]


def test_is_hashable_and_frozen() -> None:
    f = Piecewise([(0, 1.0), (1, 2.0)])
    hash(f)
    with pytest.raises(AttributeError):
        f.interpolation = "constant"  # type: ignore[misc]


def test_top_level_exports() -> None:
    from mouse_core import ExponentialDecay as Exp
    from mouse_core import Piecewise as Pw

    assert Exp is ExponentialDecay
    assert Pw is Piecewise
