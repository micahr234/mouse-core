"""Base types for MOUSE loss functions.

All loss functions share the same call signature — take a batch and model output,
return a scalar loss and a metrics dict.  Use :class:`LossConfig` as the base for
custom config dataclasses, and :class:`LossFunction` as the typing interface.

Example — custom loss::

    from dataclasses import dataclass
    from mouse_core.losses.base import LossConfig, LossFunction
    from tensordict import TensorDict
    import torch

    @dataclass(frozen=True)
    class MyLossConfig(LossConfig):
        weight: float = 1.0
        temperature: float = 1.0

    def my_loss(
        step_stream: TensorDict,
        out: TensorDict,
        cfg: MyLossConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...
        return loss, {"my_loss": loss.item()}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class LossConfig:
    """Base dataclass for loss configurations.

    Subclass this and add your own hyperparameters.  Use ``frozen=True`` to keep
    configs immutable and safe to share across training steps.
    """


class LossFunction(Protocol):
    """Protocol describing the expected signature of all MOUSE loss functions.

    Any callable matching this signature can be used interchangeably with the
    built-in losses (``dqn_loss``, ``sp_loss``, etc.).
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
                :class:`~mouse_core.data.batch.PrefetchBatchifier`.
            out: Model output TensorDict ``[B, S]`` from
                :meth:`~mouse_core.models.base.Model.forward`.
            cfg: Frozen config dataclass (subclass of :class:`LossConfig`).

        Returns:
            Tuple of ``(scalar_loss, metrics)`` where ``metrics`` is a
            ``dict[str, float]`` ready for logging to W&B / TensorBoard.
        """
        ...
