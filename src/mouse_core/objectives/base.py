"""Base type for MOUSE objective objects.

All objectives are plain Python objects: instantiate with hyperparameters,
then call with ``objective_data=``, ``predictions=``, and any inputs required
by that objective to get a loss and metrics.

Example — custom objective::

    from mouse_core.objectives.base import Objective
    import torch

    class MyObjective(Objective[...]):
        def __init__(self, *, temperature: float):
            self.temperature = temperature

        def __call__(
            self,
            *,
            objective_data: dict[str, torch.Tensor],
            predictions: torch.Tensor,
            targets: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
            loss = ((predictions - targets) ** 2).mean()
            return loss, {"my_objective": loss.detach()}
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


class Objective[**P](ABC):
    """Abstract base for all MOUSE objective objects.

    Subclass this and implement :meth:`__call__` to create a custom objective.
    Instantiate with hyperparameters; call with ``objective_data=`` and
    the prediction tensor for the head being trained. Objectives that read
    another head also take that tensor: DQN takes ``delayed_predictions=``
    and PPO takes ``value_predictions=``. Supervised objectives take
    ``targets=`` (action ids for SP, Q vectors for SV). DQN, PPO, and GRPO
    require ``group_id=`` (int64 ``[N]``) beside ``objective_data``.
    ``DqnObjective`` also requires ``reward_center=`` on that call:
    a 0-dim float32 tensor, or ``None``.

    ``P`` describes the subclass's call signature; there is no single
    set of optional prediction arguments shared by all objectives.
    Subclasses with their own keyword-only signatures use ``Objective[...]``
    and declare their required inputs on ``__call__``. Use the concrete
    objective type when calling it so those requirements remain checked.
    """

    @abstractmethod
    def __call__(
        self,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        """Compute a scalar loss and return diagnostic metrics.

        Concrete objectives declare the required keyword-only arguments.
        ``objective_data`` holds tokenizer ``objective_fields`` keyed by
        flat step index; ``predictions`` comes from the matching head in
        ``ModelOutput.predictions``. Additional group, prediction, and
        target inputs depend on the objective.

        Returns:
            ``(scalar_loss, metrics)`` where ``metrics`` holds detached
            0-dim tensors (or floats) for logging — prefer tensors so the
            train step can defer host materialization until log time —
            and, when an objective exposes them, detached tensors built
            during the loss (for example DQN ``backup``).
        """
        ...
