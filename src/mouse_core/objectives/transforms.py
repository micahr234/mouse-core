"""Shared reward, value, and discount callables for the DQN-family and PPO objectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class Discount(Protocol):
    """Per-step Bellman discount from unpacked ``objective_data`` columns.

    Called as ``discount(**objective_data)``. Named arguments are the
    column tensors. Extra columns must be accepted (``**_``). Must
    return a floating tensor of shape ``[N]`` — the discount applied
    to the bootstrap (and the continued return) out of each step.
    """

    def __call__(self, **columns: torch.Tensor) -> torch.Tensor: ...


class Reward(Protocol):
    """Per-step reward from the unpacked ``objective_data`` columns.

    Called as ``reward(**objective_data)``. Named arguments are the
    column tensors. Extra columns must be accepted (``**_``). Must
    return a floating tensor of shape ``[N]`` — the per-step reward
    used in the TD / GAE target. Does not mutate ``objective_data``.
    """

    def __call__(self, **columns: torch.Tensor) -> torch.Tensor: ...


class Value(Protocol):
    """Per-step affine on a value / Q tensor from unpacked ``objective_data``.

    Called as ``value(value=..., **objective_data)``. ``value`` is the
    online or delayed prediction (``[P, A]``, ``[P, L, A]``, or
    ``[N]``). Extra columns must be accepted (``**_``). Must return a
    floating tensor of the same shape — the TD / GAE value. Does not
    mutate ``objective_data`` or the prediction tensors.
    """

    def __call__(self, **columns: torch.Tensor) -> torch.Tensor: ...


def _as_scale(value: float | None) -> float:
    return 1.0 if value is None else float(value)


def _as_shift(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def affine_reward(
    *,
    scale: float | None,
    shift: float | None,
) -> Reward:
    """Column affine: ``scale * reward + shift``.

    Every argument must be passed. ``None`` is identity for that
    argument (``scale`` → ``1.0``, ``shift`` → ``0.0``). Both ``None``
    returns the ``reward`` column unchanged.

    Args:
        scale: Multiplier applied to the ``reward`` column. ``None``
            is ``1.0``.
        shift: Offset added after ``scale``. ``None`` is ``0.0``.
    """
    scale_ = _as_scale(scale)
    shift_ = _as_shift(shift)

    def reward(*, reward: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        if scale_ == 1.0 and shift_ == 0.0:
            return reward
        return scale_ * reward + shift_

    return reward


def boundary_reward(
    *,
    scale: float | None,
    shift: float | None,
    reward_episode_terminal_scale: float | None,
    reward_episode_terminal_shift: float | None,
    reward_episode_truncated_scale: float | None,
    reward_episode_truncated_shift: float | None,
    reward_task_terminal_scale: float | None,
    reward_task_terminal_shift: float | None,
    reward_task_truncated_scale: float | None,
    reward_task_truncated_shift: float | None,
) -> Reward:
    """Done-code affine: ``(scale × episode scale × task scale) * reward + shift + episode shift + task shift``.

    Every argument must be passed. ``None`` is identity for that
    argument (scales / extras that multiply → ``1.0``, shifts →
    ``0.0``). All ``None`` returns the ``reward`` column unchanged.

    Both done fields use codes ``0`` / ``1`` / ``2``. Every transition is
    ``scale * reward + shift``. Scale extras are ``1.0`` and shift extras
    are ``0.0`` when the matching code is ``0``. When a task ends both
    extras fire (e.g. ``episode_done=1`` and ``task_done=2``): scales
    multiply and shifts add.

    Args:
        scale: Multiplier applied to the ``reward`` column. ``None``
            is ``1.0``.
        shift: Offset added after ``scale``. ``None`` is ``0.0``.
            A running transition (both codes ``0``) uses only
            ``scale * reward + shift``.
        reward_episode_terminal_scale: Extra scale when the episode
            terminates (``episode_done == 1``). ``None`` is ``1.0``.
        reward_episode_terminal_shift: Extra shift when the episode
            terminates (``episode_done == 1``). ``None`` is ``0.0``.
        reward_episode_truncated_scale: Extra scale when the episode is
            truncated (``episode_done == 2``). ``None`` is ``1.0``.
        reward_episode_truncated_shift: Extra shift when the episode is
            truncated (``episode_done == 2``). ``None`` is ``0.0``.
        reward_task_terminal_scale: Extra scale when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
            ``None`` is ``1.0``.
        reward_task_terminal_shift: Extra shift when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
            ``None`` is ``0.0``.
        reward_task_truncated_scale: Extra scale when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``). ``None`` is ``1.0``.
        reward_task_truncated_shift: Extra shift when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``). ``None`` is ``0.0``.
    """
    scale_ = _as_scale(scale)
    shift_ = _as_shift(shift)
    reward_episode_terminal_scale_ = _as_scale(reward_episode_terminal_scale)
    reward_episode_terminal_shift_ = _as_shift(reward_episode_terminal_shift)
    reward_episode_truncated_scale_ = _as_scale(reward_episode_truncated_scale)
    reward_episode_truncated_shift_ = _as_shift(reward_episode_truncated_shift)
    reward_task_terminal_scale_ = _as_scale(reward_task_terminal_scale)
    reward_task_terminal_shift_ = _as_shift(reward_task_terminal_shift)
    reward_task_truncated_scale_ = _as_scale(reward_task_truncated_scale)
    reward_task_truncated_shift_ = _as_shift(reward_task_truncated_shift)
    identity = (
        scale_ == 1.0
        and shift_ == 0.0
        and reward_episode_terminal_scale_ == 1.0
        and reward_episode_terminal_shift_ == 0.0
        and reward_episode_truncated_scale_ == 1.0
        and reward_episode_truncated_shift_ == 0.0
        and reward_task_terminal_scale_ == 1.0
        and reward_task_terminal_shift_ == 0.0
        and reward_task_truncated_scale_ == 1.0
        and reward_task_truncated_shift_ == 0.0
    )

    def reward(
        *,
        reward: torch.Tensor,
        episode_done: torch.Tensor,
        task_done: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        if identity:
            return reward
        episode_scales = torch.tensor(
            [1.0, reward_episode_terminal_scale_, reward_episode_truncated_scale_],
            dtype=torch.float32,
            device=reward.device,
        )
        episode_shifts = torch.tensor(
            [0.0, reward_episode_terminal_shift_, reward_episode_truncated_shift_],
            dtype=torch.float32,
            device=reward.device,
        )
        task_scales = torch.tensor(
            [1.0, reward_task_terminal_scale_, reward_task_truncated_scale_],
            dtype=torch.float32,
            device=reward.device,
        )
        task_shifts = torch.tensor(
            [0.0, reward_task_terminal_shift_, reward_task_truncated_shift_],
            dtype=torch.float32,
            device=reward.device,
        )
        return (
            scale_ * episode_scales[episode_done] * task_scales[task_done] * reward
            + shift_
            + episode_shifts[episode_done]
            + task_shifts[task_done]
        )

    return reward


def affine_value(
    *,
    scale: float | None,
    shift: float | None,
) -> Value:
    """Prediction affine: ``scale * value + shift``.

    Every argument must be passed. ``None`` is identity for that
    argument (``scale`` → ``1.0``, ``shift`` → ``0.0``). Both ``None``
    returns the value / Q tensor unchanged.

    Args:
        scale: Multiplier applied to the value / Q tensor. ``None``
            is ``1.0``.
        shift: Offset added after ``scale``. ``None`` is ``0.0``.
    """
    scale_ = _as_scale(scale)
    shift_ = _as_shift(shift)

    def value(*, value: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        if scale_ == 1.0 and shift_ == 0.0:
            return value
        return scale_ * value + shift_

    return value


def boundary_value(
    *,
    scale: float | None,
    shift: float | None,
    value_episode_terminal_scale: float | None,
    value_episode_terminal_shift: float | None,
    value_episode_truncated_scale: float | None,
    value_episode_truncated_shift: float | None,
    value_task_terminal_scale: float | None,
    value_task_terminal_shift: float | None,
    value_task_truncated_scale: float | None,
    value_task_truncated_shift: float | None,
) -> Value:
    """Done-code affine: ``(scale × episode scale × task scale) * value + shift + episode shift + task shift``.

    Every argument must be passed. ``None`` is identity for that
    argument (scales / extras that multiply → ``1.0``, shifts →
    ``0.0``). All ``None`` returns the value / Q tensor unchanged.

    Both done fields use codes ``0`` / ``1`` / ``2``. Every row is
    ``scale * value + shift``. Scale extras are ``1.0`` and shift extras
    are ``0.0`` when the matching code is ``0``. When a task ends both
    extras fire (e.g. ``episode_done=1`` and ``task_done=2``): scales
    multiply and shifts add. Per-row extras broadcast over remaining
    value dimensions (``[P, A]``, ``[P, L, A]``).

    Args:
        scale: Multiplier applied to the value / Q tensor. ``None``
            is ``1.0``.
        shift: Offset added after ``scale``. ``None`` is ``0.0``.
            A running row (both codes ``0``) uses only
            ``scale * value + shift``.
        value_episode_terminal_scale: Extra scale when the episode
            terminates (``episode_done == 1``). ``None`` is ``1.0``.
        value_episode_terminal_shift: Extra shift when the episode
            terminates (``episode_done == 1``). ``None`` is ``0.0``.
        value_episode_truncated_scale: Extra scale when the episode is
            truncated (``episode_done == 2``). ``None`` is ``1.0``.
        value_episode_truncated_shift: Extra shift when the episode is
            truncated (``episode_done == 2``). ``None`` is ``0.0``.
        value_task_terminal_scale: Extra scale when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
            ``None`` is ``1.0``.
        value_task_terminal_shift: Extra shift when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
            ``None`` is ``0.0``.
        value_task_truncated_scale: Extra scale when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``). ``None`` is ``1.0``.
        value_task_truncated_shift: Extra shift when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``). ``None`` is ``0.0``.
    """
    scale_ = _as_scale(scale)
    shift_ = _as_shift(shift)
    value_episode_terminal_scale_ = _as_scale(value_episode_terminal_scale)
    value_episode_terminal_shift_ = _as_shift(value_episode_terminal_shift)
    value_episode_truncated_scale_ = _as_scale(value_episode_truncated_scale)
    value_episode_truncated_shift_ = _as_shift(value_episode_truncated_shift)
    value_task_terminal_scale_ = _as_scale(value_task_terminal_scale)
    value_task_terminal_shift_ = _as_shift(value_task_terminal_shift)
    value_task_truncated_scale_ = _as_scale(value_task_truncated_scale)
    value_task_truncated_shift_ = _as_shift(value_task_truncated_shift)
    identity = (
        scale_ == 1.0
        and shift_ == 0.0
        and value_episode_terminal_scale_ == 1.0
        and value_episode_terminal_shift_ == 0.0
        and value_episode_truncated_scale_ == 1.0
        and value_episode_truncated_shift_ == 0.0
        and value_task_terminal_scale_ == 1.0
        and value_task_terminal_shift_ == 0.0
        and value_task_truncated_scale_ == 1.0
        and value_task_truncated_shift_ == 0.0
    )

    def value(
        *,
        value: torch.Tensor,
        episode_done: torch.Tensor,
        task_done: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        if identity:
            return value
        episode_scales = torch.tensor(
            [1.0, value_episode_terminal_scale_, value_episode_truncated_scale_],
            dtype=torch.float32,
            device=value.device,
        )
        episode_shifts = torch.tensor(
            [0.0, value_episode_terminal_shift_, value_episode_truncated_shift_],
            dtype=torch.float32,
            device=value.device,
        )
        task_scales = torch.tensor(
            [1.0, value_task_terminal_scale_, value_task_truncated_scale_],
            dtype=torch.float32,
            device=value.device,
        )
        task_shifts = torch.tensor(
            [0.0, value_task_terminal_shift_, value_task_truncated_shift_],
            dtype=torch.float32,
            device=value.device,
        )
        row_scale = scale_ * episode_scales[episode_done] * task_scales[task_done]
        row_shift = shift_ + episode_shifts[episode_done] + task_shifts[task_done]
        while row_scale.ndim < value.ndim:
            row_scale = row_scale.unsqueeze(-1)
            row_shift = row_shift.unsqueeze(-1)
        return row_scale * value + row_shift

    return value


def boundary_discount(
    *,
    gamma_step: float | None,
    gamma_episode_terminal: float | None,
    gamma_episode_truncated: float | None,
    gamma_task_terminal: float | None,
    gamma_task_truncated: float | None,
) -> Discount:
    """Done-code lookup: ``gamma_step`` × episode extra × task extra.

    Every argument must be passed. ``None`` is identity for that
    argument (``1.0``). All ``None`` returns ``1`` at every step.

    Both fields use codes ``0`` / ``1`` / ``2``. Every transition is
    multiplied by ``gamma_step``. Episode and task extras are ``1.0`` when
    the matching code is ``0``. When a task ends both extras fire (e.g.
    ``episode_done=1`` and ``task_done=2``) and the product is used; a
    factor of ``0.0`` zeros the whole bootstrap.

    Args:
        gamma_step: Always multiplied. ``None`` is ``1.0``. A running
            transition (both codes ``0``) uses only this factor.
        gamma_episode_terminal: Extra factor when the episode terminates
            (``episode_done == 1``). ``None`` is ``1.0``.
        gamma_episode_truncated: Extra factor when the episode is truncated
            (``episode_done == 2``). ``None`` is ``1.0``.
        gamma_task_terminal: Extra factor when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
            ``None`` is ``1.0``.
        gamma_task_truncated: Extra factor when the task is truncated
            (``task_done == 2``; last episode of ``max_task_episodes``).
            ``None`` is ``1.0``. ``0.0`` zeros the bootstrap.
    """
    gamma_step_ = _as_scale(gamma_step)
    gamma_episode_terminal_ = _as_scale(gamma_episode_terminal)
    gamma_episode_truncated_ = _as_scale(gamma_episode_truncated)
    gamma_task_terminal_ = _as_scale(gamma_task_terminal)
    gamma_task_truncated_ = _as_scale(gamma_task_truncated)
    identity = (
        gamma_step_ == 1.0
        and gamma_episode_terminal_ == 1.0
        and gamma_episode_truncated_ == 1.0
        and gamma_task_terminal_ == 1.0
        and gamma_task_truncated_ == 1.0
    )

    def discount(
        *,
        episode_done: torch.Tensor,
        task_done: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        if identity:
            return torch.ones(
                episode_done.shape, dtype=torch.float32, device=episode_done.device
            )
        episode_gammas = torch.tensor(
            [1.0, gamma_episode_terminal_, gamma_episode_truncated_],
            dtype=torch.float32,
            device=episode_done.device,
        )
        task_gammas = torch.tensor(
            [1.0, gamma_task_terminal_, gamma_task_truncated_],
            dtype=torch.float32,
            device=episode_done.device,
        )
        return gamma_step_ * episode_gammas[episode_done] * task_gammas[task_done]

    return discount


def _require_transform(value: object, *, name: str) -> Callable[..., torch.Tensor]:
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value)}.")
    return value


def _apply_transform(
    *,
    transform: Callable[..., torch.Tensor],
    name: str,
    objective_data: dict[str, torch.Tensor],
    N: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Evaluate ``transform`` on unpacked ``objective_data`` and cast."""
    values = transform(**objective_data)
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must return a Tensor, got {type(values)}.")
    if values.shape != torch.Size([N]):
        raise ValueError(
            f"{name} must return shape [{N}], got {tuple(values.shape)}."
        )
    return values.to(dtype=dtype, device=device)


def _align_objective_rows(
    *,
    objective_data: dict[str, torch.Tensor],
    step_of: torch.Tensor,
    N: int,
) -> dict[str, torch.Tensor]:
    """Index length-``N`` columns by ``step_of`` so they match ``[P]`` rows."""
    aligned: dict[str, torch.Tensor] = {}
    for key, values in objective_data.items():
        if values.shape[:1] == torch.Size([N]):
            aligned[key] = values[step_of]
        else:
            aligned[key] = values
    return aligned


def _apply_value(
    *,
    transform: Callable[..., torch.Tensor],
    name: str,
    value: torch.Tensor,
    objective_data: dict[str, torch.Tensor],
    step_of: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Evaluate ``transform`` on ``value=`` plus row-aligned ``objective_data``."""
    columns = _align_objective_rows(
        objective_data=objective_data, step_of=step_of, N=N
    )
    values = transform(**columns, value=value)
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must return a Tensor, got {type(values)}.")
    if values.shape != value.shape:
        raise ValueError(
            f"{name} must return shape {tuple(value.shape)}, "
            f"got {tuple(values.shape)}."
        )
    return values.to(dtype=value.dtype, device=value.device)
