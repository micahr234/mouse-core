"""Base type for MOUSE objective objects.

All objectives are plain Python objects: instantiate with hyperparameters,
then call with ``(objective_data, predictions)`` to get a loss and metrics.

Example — custom objective::

    from mouse_core.objectives.base import Objective
    from tensordict import TensorDict
    import torch

    class MyObjective(Objective):
        def __init__(self, temperature: float = 1.0):
            self.temperature = temperature

        def __call__(
            self,
            objective_data: TensorDict,
            predictions: TensorDict,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            ...
            return loss, {"my_objective": loss.item()}
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class Objective(ABC):
    """Abstract base for all MOUSE objective objects.

    Subclass this and implement :meth:`__call__` to create a custom objective.
    Instantiate with hyperparameters; call with ``(objective_data, predictions)``.
    """

    @abstractmethod
    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute a scalar loss and return diagnostic metrics.

        Args:
            objective_data: ``TensorDict[N]`` of the modality tensors extracted
                by the encoder (action, reward, done, observation, etc.),
                keyed by flat step index with ``sequence_id``.
            predictions: ``TensorDict[N]`` of model head outputs from
                :meth:`~mouse_core.models.base.Model.forward`.

        Returns:
            ``(scalar_loss, metrics)`` where ``metrics`` is a ``dict[str, float]``
            ready for logging.
        """
        ...
