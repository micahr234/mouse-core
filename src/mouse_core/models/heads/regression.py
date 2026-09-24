"""RegressionHead: raw scalar outputs (Q-values, V(s), any regression target)."""

from __future__ import annotations

from mouse_core.models.heads.swiglu import _MlpHead


class RegressionHead(_MlpHead):
    """Head that outputs raw scalars — one per ``out_features``.

    Use ``out_features=A`` for per-action values (Q-learning), or
    ``out_features=1`` for a state-value baseline or the DQN offset
    ``w(h)`` (``DqnObjective`` ``w=``). Pair multi-action
    outputs with :meth:`~mouse_core.models.base.Model.copy` when
    an objective bootstraps from a delayed copy.
    """
