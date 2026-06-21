"""Base types for MOUSE objective functions.

All objective functions share the same call signature — take a batch and model output,
return a scalar loss and a metrics dict.  Use :class:`ObjectiveConfig` as the base for
custom config dataclasses, and :class:`ObjectiveFunction` as the typing interface.

Example — custom objective::

    from dataclasses import dataclass
    from mouse_core.objectives.base import ObjectiveConfig, ObjectiveFunction
    from tensordict import TensorDict
    import torch

    @dataclass(frozen=True)
    class MyObjectiveConfig(ObjectiveConfig):
        weight: float = 1.0
        temperature: float = 1.0

    def my_objective(
        step_stream: TensorDict,
        out: TensorDict,
        cfg: MyObjectiveConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...
        return loss, {"my_objective": loss.item()}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class ObjectiveConfig:
    """Base dataclass for objective configurations.

    Subclass this and add your own hyperparameters.  Use ``frozen=True`` to keep
    configs immutable and safe to share across training steps.
    """


class ObjectiveFunction(Protocol):
    """Protocol describing the expected signature of all MOUSE objective functions.

    Any callable matching this signature can be used interchangeably with the
    built-in objectives (``dqn_objective``, ``sp_objective``, etc.).
    """

    def __call__(
        self,
        step_stream: TensorDict,
        out: TensorDict,
        cfg: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute a scalar loss and return diagnostic metrics.

        Args:
            step_stream: Batch of step records ``[B, S]`` from
                :class:`~mouse_core.data.dataloader.DataLoader`.
            out: Model output TensorDict ``[B, S]`` from
                :meth:`~mouse_core.models.base.Model.forward`.
            cfg: Frozen config dataclass (subclass of :class:`ObjectiveConfig`).

        Returns:
            Tuple of ``(scalar_loss, metrics)`` where ``metrics`` is a
            ``dict[str, float]`` ready for logging to W&B / TensorBoard.
        """
        ...
