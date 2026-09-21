"""Base type for MOUSE objective objects.

All objectives are plain Python objects: instantiate with hyperparameters,
then call with ``objective_data=`` and ``predictions=`` to get a loss and metrics.

Example — custom objective::

    from mouse_core.objectives.base import Objective
    import torch

    class MyObjective(Objective):
        def __init__(self, *, temperature: float):
            self.temperature = temperature

        def __call__(
            self,
            *,
            objective_data: dict[str, torch.Tensor],
            predictions: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            ...
            return loss, {"my_objective": loss.item()}
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Objective(ABC):
    """Abstract base for all MOUSE objective objects.

    Subclass this and implement :meth:`__call__` to create a custom objective.
    Instantiate with hyperparameters; call with ``objective_data=`` and
    the prediction tensor for the head being trained.
    """

    @abstractmethod
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute a scalar loss and return diagnostic metrics.

        Args:
            objective_data: ``dict[str, Tensor]`` of tokenizer ``objective_fields``
                (``action``, ``reward``, ``episode_done``, ``task_done``, …),
                keyed by flat step index with ``sequence_id``.
            predictions: Tensor for the head this objective trains, taken
                from :meth:`~mouse_core.models.base.Model.forward` (index
                ``ModelOutput.predictions`` with
                :func:`~mouse_core.models.heads.base.prediction_key`).
                DQN-family subclasses also take ``delayed_predictions=``;
                PPO takes ``value_predictions=``; Retrace takes both
                ``delayed_predictions=`` and ``behavior_predictions=``.

        Returns:
            ``(scalar_loss, metrics)`` where ``metrics`` is a ``dict[str, float]``
            ready for logging.
        """
        ...
