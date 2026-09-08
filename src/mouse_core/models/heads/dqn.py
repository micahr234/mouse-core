"""DiscreteActionValueHead: per-discrete-action values (e.g. Q(s,a))."""

from __future__ import annotations

from mouse_core.models.heads.swiglu import SwiGLUHead


class DiscreteActionValueHead(SwiGLUHead):
    """Head that outputs a value for each discrete action.

    Same architecture as :class:`SwiGLUHead`. Pair with
    :meth:`~mouse_core.models.base.Model.delayed_copy` for a delayed copy used
    as the DQN bootstrap target.
    """
