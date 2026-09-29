"""Constant offset removed from every DQN action value (reward centering).

The same split as the delayed model and
:func:`~mouse_core.polyak.model_polyak`. :class:`RewardCentering` is
the buffer (the constant). :func:`reward_centering_polyak` is the copy
step that writes it, and it keeps no state. The scalar starts at 0.
Pass ``center`` as ``reward_center=`` on the ``DqnObjective`` call:
the objective shapes every reward slot by the constant potential
``center`` (a slot whose discount is ``γ`` subtracts
``(1 − γ)·center``), which telescopes so the head learns
``Q − center`` at every state–action — the policy ordering never
changes, episodic or continuing, for any ``center``. The loss detaches
it, so it gets no gradient. After the optimizer step,
``reward_centering_polyak(center=, tau=, values=)`` sets
``center ← τ·mean(values) + (1−τ)·center`` — feed the in-run backup
rows (``metrics["in_run_backup"]``)
so it tracks the mean action value.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RewardCentering(nn.Module):
    """Constant offset ``center`` removed from every action value.

    The buffer only. ``center`` starts at 0 and is not a parameter.
    Pass it as ``reward_center=`` on the ``DqnObjective`` call
    (``None`` leaves the residual plain). The objective subtracts
    ``(1 − γ)·center`` per reward slot (constant-potential shaping),
    which telescopes to the same offset ``center`` on every action
    value: the head learns ``Q − center`` and greedy actions are
    unchanged for any ``center``. In a continuing task with constant
    ``γ < 1`` this is classic reward centering with average-reward
    estimate ``(1 − γ)·center``. The loss detaches it.
    :func:`reward_centering_polyak` writes the buffer.
    """

    center: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "center", torch.zeros((), dtype=torch.float32), persistent=True
        )


@torch.no_grad()
def reward_centering_polyak(
    *,
    center: RewardCentering,
    tau: float,
    values: torch.Tensor,
) -> None:
    """Polyak-average a :class:`RewardCentering` toward ``mean(values)``.

    Keeps no state. The constant stays on ``center``. Does not run a
    forward. After each ``optimizer.step()`` call with this step's
    ``tau`` and the values to track::

        reward_centering = RewardCentering()
        ...
        reward_centering_polyak(
            center=reward_centering,
            tau=0.01,
            values=metrics["in_run_backup"],
        )

    ``center ← τ·mean(values) + (1−τ)·center``. ``τ = 0`` keeps the
    center; ``τ = 1`` replaces it with the mean. ``tau`` is a float in
    ``[0, 1]``. ``values`` is a non-empty float32 tensor on the same
    device as ``center`` — the in-run backup rows
    (``metrics["in_run_backup"]``), so
    ``center`` tracks the mean action value the head would otherwise
    carry as a constant offset. The constant is not on the delayed
    model and is not an AdamW parameter.
    """
    if not isinstance(center, RewardCentering):
        raise TypeError(
            "reward_centering_polyak writes a RewardCentering buffer, got "
            f"{type(center).__name__}."
        )
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}.")
    if not isinstance(values, torch.Tensor):
        raise TypeError(
            f"values must be a float32 tensor, got {type(values).__name__}."
        )
    if values.dtype != torch.float32:
        raise TypeError(f"values must be float32, got {values.dtype}.")
    if values.numel() == 0:
        raise ValueError("values must contain at least one value.")
    buf = center.center
    if values.device != buf.device:
        raise ValueError(
            f"values device {values.device} must match center device "
            f"{buf.device}."
        )
    if tau == 0.0:
        return
    buf.lerp_(values.detach().mean(), tau)
