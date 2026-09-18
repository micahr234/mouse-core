"""ClassificationHead: unnormalized logits (policy, behavior, distillation)."""

from __future__ import annotations

from mouse_core.models.heads.swiglu import _MlpHead


class ClassificationHead(_MlpHead):
    """Head that outputs unnormalized logits — one per class / discrete action.

    Pair with a scalar :class:`~mouse_core.models.heads.regression.RegressionHead`
    (``out_features=1``) for :class:`~mouse_core.objectives.PpoObjective`,
    use alone for :class:`~mouse_core.objectives.GrpoObjective`, or with a
    teacher :class:`~mouse_core.models.heads.regression.RegressionHead` when
    a student learns action logits (e.g. :class:`~mouse_core.objectives.SpObjective`).
    """
