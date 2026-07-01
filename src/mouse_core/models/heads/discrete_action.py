"""DiscreteActionHead: per-discrete-action logits for policy / SpObjective distillation."""

from __future__ import annotations

from mouse_core.models.heads.swiglu import SwiGLUHead


class DiscreteActionHead(SwiGLUHead):
    """Head that outputs logits for each discrete action.

    Same architecture as :class:`SwiGLUHead`; no target network. Pair with
    :class:`~mouse_core.models.heads.dqn.DiscreteActionValueHead` when a
    teacher learns Q-values and a student learns action logits (e.g. online
    distillation via :class:`~mouse_core.objectives.SpObjective`).
    """
