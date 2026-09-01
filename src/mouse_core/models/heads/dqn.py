"""DiscreteActionValueHead: per-discrete-action values (e.g. Q(s,a))."""

from __future__ import annotations

from mouse_core.models.heads.swiglu import SwiGLUHead


class DiscreteActionValueHead(SwiGLUHead):
    """Head that outputs a value for each discrete action.

    Same architecture as :class:`SwiGLUHead`. Pair with
    :class:`~mouse_core.polyak.PolyakAverager` for a delayed copy used as the
    DQN bootstrap target.
    """
