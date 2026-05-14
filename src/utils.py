"""Shared utilities (serialization, small pure helpers)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch


class ProbSchedule:
    """Piecewise-linear probability schedule over integer steps.

    Constructed from a sequence of (step, prob) knots stored in ``EnvConfig``.
    A knot step may be:

    - A non-negative ``int``: absolute step index.
    - A ``float`` in ``[0.0, 1.0]``: fraction of the total steps, resolved to
      ``round(frac * (max_steps - 1))`` at call time (e.g. ``1.0`` = last step).

    When no knots are provided every call returns 1.0.
    The prefix before the first explicit knot defaults to 1.0.

    Usage::

        sched = ProbSchedule(env_cfg.action_source_loop_prob_schedule)
        prob = sched(loop_step_idx, max_steps=cfg.max_steps)
    """

    def __init__(self, knots: tuple[tuple[int | float, float], ...] | None) -> None:
        self._knots: tuple[tuple[int | float, float], ...] = knots or ()

    def __bool__(self) -> bool:
        return bool(self._knots)

    def __call__(self, step: int, max_steps: int) -> float:
        """Return interpolated probability at ``step`` out of ``max_steps``."""
        if not self._knots:
            return 1.0
        last = max(0, max_steps - 1)
        by_step: dict[int, float] = {}
        for s, p in self._knots:
            s_resolved = min(round(s * last), last) if isinstance(s, float) else min(int(s), last)
            by_step[s_resolved] = float(p)
        resolved = [(k, by_step[k]) for k in sorted(by_step)]
        if not resolved:
            return 1.0
        if resolved[0][0] > 0:
            resolved.insert(0, (0, 1.0))
        return self._lerp(knots=resolved, t=step)

    def horizon(self, max_episode_steps: int | None) -> int:
        """Best-effort horizon for episode-relative schedules.

        Returns ``max_episode_steps`` when set, otherwise infers from the
        highest absolute (int) knot step, falling back to 10 000.
        """
        if max_episode_steps is not None and int(max_episode_steps) >= 1:
            return int(max_episode_steps)
        if self._knots:
            int_steps = [s for s, _ in self._knots if isinstance(s, int)]
            if int_steps:
                return int(max(2, max(int_steps) + 1))
        return 10_000

    @staticmethod
    def _lerp(knots: list[tuple[int, float]], t: int) -> float:
        t = int(t)
        if t <= knots[0][0]:
            return float(knots[0][1])
        if t >= knots[-1][0]:
            return float(knots[-1][1])
        for i in range(len(knots) - 1):
            s0, p0 = knots[i]
            s1, p1 = knots[i + 1]
            if s0 <= t <= s1:
                if s1 == s0:
                    return float(p1)
                w = (float(t) - float(s0)) / (float(s1) - float(s0))
                return (1.0 - w) * float(p0) + w * float(p1)
        return float(knots[-1][1])


def map_payload_to_json_str(value: Any) -> str | None:
    """Serialize a map payload (dict, numpy arrays, nested structures) to a JSON string.

    Used by environments for ``info["map"]`` so rollout datasets get a stable string column.
    ``None`` stays ``None``; strings are returned unchanged (already JSON text).
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value

    def _to_builtin(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): _to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_builtin(v) for v in obj]
        return obj

    return json.dumps(_to_builtin(value), sort_keys=True)


def greedy_accuracy(*, logits: torch.Tensor, q_star_tok: torch.Tensor) -> torch.Tensor:
    """Fraction of steps where ``argmax(logits)`` matches greedy ``q_star`` action."""
    A = logits.shape[-1]
    return (
        logits.reshape(-1, A).argmax(dim=-1) == q_star_tok.reshape(-1, A).argmax(dim=-1)
    ).float().mean()
