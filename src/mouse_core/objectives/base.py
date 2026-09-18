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
            predictions: dict[str, torch.Tensor],
            delayed_predictions: dict[str, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            ...
            return loss, {"my_objective": loss.item()}
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from mouse_core.models.heads.base import BaseHead, prediction_key


class Objective(ABC):
    """Abstract base for all MOUSE objective objects.

    Subclass this and implement :meth:`__call__` to create a custom objective.
    Instantiate with hyperparameters; call with ``objective_data=`` and
    ``predictions=``.
    """

    @abstractmethod
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        delayed_predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute a scalar loss and return diagnostic metrics.

        Args:
            objective_data: ``dict[str, Tensor]`` of tokenizer ``objective_fields``
                (``action``, ``reward``, ``episode_done``, ``task_done``, …),
                keyed by flat step index with ``sequence_id``.
            predictions: ``dict[str, Tensor]`` of model head outputs from
                :meth:`~mouse_core.models.base.Model.forward`.
            delayed_predictions: Delayed-model head outputs. Required by DQN
                family objectives; ignored by PPO, GRPO, SP, and SV.

        Returns:
            ``(scalar_loss, metrics)`` where ``metrics`` is a ``dict[str, float]``
            ready for logging.
        """
        ...


def require_head(*, head: object, what: str) -> BaseHead:
    """Require ``head`` to be a :class:`BaseHead` already attached to a Model."""
    if not isinstance(head, BaseHead):
        raise TypeError(f"{what} must be a BaseHead instance, got {type(head).__name__}.")
    prediction_key(head=head)
    return head


def predictions_for(*, head: BaseHead, predictions: dict[str, torch.Tensor], who: str) -> torch.Tensor:
    """Return ``predictions`` for ``head``'s bound storage key."""
    key = prediction_key(head=head)
    if key not in predictions.keys():
        raise KeyError(f"{who} expects predictions[{key!r}] for the given head.")
    return predictions[key]
