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
            delayed_predictions: torch.Tensor | None = None,
            value_predictions: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            ...
            return loss, {"my_objective": loss.item()}
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


def _require_prediction(
    value: torch.Tensor | None, *, owner: str, name: str
) -> torch.Tensor:
    """Return ``value`` or raise the same ``TypeError`` a missing argument would."""
    if value is None:
        raise TypeError(
            f"{owner}.__call__() missing 1 required keyword-only argument: {name!r}"
        )
    return value


def _reject_predictions(owner: str, **unused: torch.Tensor | None) -> None:
    """Raise when a call passes a prediction tensor this objective does not read."""
    for name, value in unused.items():
        if value is not None:
            raise TypeError(
                f"{owner}.__call__() got an unexpected keyword argument {name!r}"
            )


class Objective(ABC):
    """Abstract base for all MOUSE objective objects.

    Subclass this and implement :meth:`__call__` to create a custom objective.
    Instantiate with hyperparameters; call with ``objective_data=`` and
    the prediction tensor for the head being trained. Objectives that read
    another head also take that tensor: DQN takes ``delayed_predictions=``
    and PPO takes ``value_predictions=``. A custom subclass must accept the
    same optional parameters (pass ``None`` for a tensor it does not read).
    """

    @abstractmethod
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
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
            delayed_predictions: Delayed Q for DQN. ``None`` on
                objectives that do not read it. Omitting it on an objective
                that does read it raises ``TypeError``.
            value_predictions: Value-head tensor for PPO. ``None`` on
                objectives that do not read it.

        Returns:
            ``(scalar_loss, metrics)`` where ``metrics`` is a ``dict[str, float]``
            ready for logging.
        """
        ...
