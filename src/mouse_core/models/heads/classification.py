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

    ``propagate_gradient`` in ``[0, 1]`` scales backbone gradient through
    the pooled hidden state: ``1`` full flow, ``0`` none. The head's own
    parameters still train.
    """
