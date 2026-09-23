"""Shared reward, value, discount, and gate callables for the DQN-family and PPO objectives."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Protocol, cast

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


def affine_reward(
    *,
    scale: float,
    shift: float,
) -> Reward:
    """Column affine: ``scale * reward + shift``.

    Args:
        scale: Multiplier applied to the ``reward`` column.
        shift: Offset added after ``scale``.
    """

    def reward(*, reward: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return scale * reward + shift

    return reward


def boundary_reward(
    *,
    scale: float,
    shift: float,
    reward_episode_terminal_scale: float,
    reward_episode_terminal_shift: float,
    reward_episode_truncated_scale: float,
    reward_episode_truncated_shift: float,
    reward_task_terminal_scale: float,
    reward_task_terminal_shift: float,
    reward_task_truncated_scale: float,
    reward_task_truncated_shift: float,
) -> Reward:
    """Done-code affine: ``(scale × episode scale × task scale) * reward + shift + episode shift + task shift``.

    Both done fields use codes ``0`` / ``1`` / ``2``. Every transition is
    ``scale * reward + shift``. Scale extras are ``1.0`` and shift extras
    are ``0.0`` when the matching code is ``0``. When a task ends both
    extras fire (e.g. ``episode_done=1`` and ``task_done=2``): scales
    multiply and shifts add.

    Args:
        scale: Multiplier applied to the ``reward`` column.
        shift: Offset added after ``scale``. A running transition
            (both codes ``0``) uses only ``scale * reward + shift``.
        reward_episode_terminal_scale: Extra scale when the episode
            terminates (``episode_done == 1``).
        reward_episode_terminal_shift: Extra shift when the episode
            terminates (``episode_done == 1``).
        reward_episode_truncated_scale: Extra scale when the episode is
            truncated (``episode_done == 2``).
        reward_episode_truncated_shift: Extra shift when the episode is
            truncated (``episode_done == 2``).
        reward_task_terminal_scale: Extra scale when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
        reward_task_terminal_shift: Extra shift when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
        reward_task_truncated_scale: Extra scale when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``).
        reward_task_truncated_shift: Extra shift when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``).
    """

    def reward(
        *,
        reward: torch.Tensor,
        episode_done: torch.Tensor,
        task_done: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        episode_scales = torch.tensor(
            [1.0, reward_episode_terminal_scale, reward_episode_truncated_scale],
            dtype=torch.float32,
            device=reward.device,
        )
        episode_shifts = torch.tensor(
            [0.0, reward_episode_terminal_shift, reward_episode_truncated_shift],
            dtype=torch.float32,
            device=reward.device,
        )
        task_scales = torch.tensor(
            [1.0, reward_task_terminal_scale, reward_task_truncated_scale],
            dtype=torch.float32,
            device=reward.device,
        )
        task_shifts = torch.tensor(
            [0.0, reward_task_terminal_shift, reward_task_truncated_shift],
            dtype=torch.float32,
            device=reward.device,
        )
        return (
            scale * episode_scales[episode_done] * task_scales[task_done] * reward
            + shift
            + episode_shifts[episode_done]
            + task_shifts[task_done]
        )

    return reward


def affine_value(
    *,
    scale: float,
    shift: float,
) -> Value:
    """Prediction affine: ``scale * value + shift``.

    Args:
        scale: Multiplier applied to the value / Q tensor.
        shift: Offset added after ``scale``.
    """

    def value(*, value: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return scale * value + shift

    return value


def boundary_value(
    *,
    scale: float,
    shift: float,
    value_episode_terminal_scale: float,
    value_episode_terminal_shift: float,
    value_episode_truncated_scale: float,
    value_episode_truncated_shift: float,
    value_task_terminal_scale: float,
    value_task_terminal_shift: float,
    value_task_truncated_scale: float,
    value_task_truncated_shift: float,
) -> Value:
    """Done-code affine: ``(scale × episode scale × task scale) * value + shift + episode shift + task shift``.

    Both done fields use codes ``0`` / ``1`` / ``2``. Every row is
    ``scale * value + shift``. Scale extras are ``1.0`` and shift extras
    are ``0.0`` when the matching code is ``0``. When a task ends both
    extras fire (e.g. ``episode_done=1`` and ``task_done=2``): scales
    multiply and shifts add. Per-row extras broadcast over remaining
    value dimensions (``[P, A]``, ``[P, L, A]``).

    Args:
        scale: Multiplier applied to the value / Q tensor.
        shift: Offset added after ``scale``. A running row
            (both codes ``0``) uses only ``scale * value + shift``.
        value_episode_terminal_scale: Extra scale when the episode
            terminates (``episode_done == 1``).
        value_episode_terminal_shift: Extra shift when the episode
            terminates (``episode_done == 1``).
        value_episode_truncated_scale: Extra scale when the episode is
            truncated (``episode_done == 2``).
        value_episode_truncated_shift: Extra shift when the episode is
            truncated (``episode_done == 2``).
        value_task_terminal_scale: Extra scale when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
        value_task_terminal_shift: Extra shift when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
        value_task_truncated_scale: Extra scale when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``).
        value_task_truncated_shift: Extra shift when the task is
            truncated (``task_done == 2``; last episode of
            ``max_task_episodes``).
    """

    def value(
        *,
        value: torch.Tensor,
        episode_done: torch.Tensor,
        task_done: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        episode_scales = torch.tensor(
            [1.0, value_episode_terminal_scale, value_episode_truncated_scale],
            dtype=torch.float32,
            device=value.device,
        )
        episode_shifts = torch.tensor(
            [0.0, value_episode_terminal_shift, value_episode_truncated_shift],
            dtype=torch.float32,
            device=value.device,
        )
        task_scales = torch.tensor(
            [1.0, value_task_terminal_scale, value_task_truncated_scale],
            dtype=torch.float32,
            device=value.device,
        )
        task_shifts = torch.tensor(
            [0.0, value_task_terminal_shift, value_task_truncated_shift],
            dtype=torch.float32,
            device=value.device,
        )
        row_scale = scale * episode_scales[episode_done] * task_scales[task_done]
        row_shift = shift + episode_shifts[episode_done] + task_shifts[task_done]
        while row_scale.ndim < value.ndim:
            row_scale = row_scale.unsqueeze(-1)
            row_shift = row_shift.unsqueeze(-1)
        return row_scale * value + row_shift

    return value


def boundary_discount(
    *,
    gamma_step: float,
    gamma_episode_terminal: float,
    gamma_episode_truncated: float,
    gamma_task_terminal: float,
    gamma_task_truncated: float,
) -> Discount:
    """Done-code lookup: ``gamma_step`` × episode extra × task extra.

    Both fields use codes ``0`` / ``1`` / ``2``. Every transition is
    multiplied by ``gamma_step``. Episode and task extras are ``1.0`` when
    the matching code is ``0``. When a task ends both extras fire (e.g.
    ``episode_done=1`` and ``task_done=2``) and the product is used; a
    factor of ``0.0`` zeros the whole bootstrap.

    Args:
        gamma_step: Always multiplied. A running transition
            (both codes ``0``) uses only this factor.
        gamma_episode_terminal: Extra factor when the episode terminates
            (``episode_done == 1``).
        gamma_episode_truncated: Extra factor when the episode is truncated
            (``episode_done == 2``).
        gamma_task_terminal: Extra factor when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
        gamma_task_truncated: Extra factor when the task is truncated
            (``task_done == 2``; last episode of ``max_task_episodes``).
            ``0.0`` zeros the bootstrap.
    """

    def discount(
        *,
        episode_done: torch.Tensor,
        task_done: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        episode_gammas = torch.tensor(
            [1.0, gamma_episode_terminal, gamma_episode_truncated],
            dtype=torch.float32,
            device=episode_done.device,
        )
        task_gammas = torch.tensor(
            [1.0, gamma_task_terminal, gamma_task_truncated],
            dtype=torch.float32,
            device=episode_done.device,
        )
        return gamma_step * episode_gammas[episode_done] * task_gammas[task_done]

    return discount


class Gate(Protocol):
    """Continuation of a DQN return, from unpacked ``objective_data``.

    Called as ``gate(**objective_data, q=q, q_delayed=q_delayed)``.
    ``q`` is the detached per-step online Q at each step's last
    head-output row, shape ``[N, A]``, after ``value``. ``q_delayed``
    is the detached per-step delayed Q, same layout, after ``value``.
    Extra columns must be accepted (``**_``). Does not mutate
    ``objective_data``.

    Return ``[N, N]``. Row ``t``, column ``s`` is the continuation at
    absolute step ``s`` for the return that started at ``t``. The value
    at column ``t + 1`` is the continuation through ``s_{t+1}``, the
    same place the reward and γ of that transition are stored. Entries
    with ``s <= t`` are ignored. A ``0`` bootstraps ``V`` at that step.
    The objective then zeros any continuation that would leave the run
    (``sequence_id`` / ``grouping_field``). Values must lie in
    ``[0, 1]``. One row cumprod.
    """

    def __call__(self, **columns: torch.Tensor) -> torch.Tensor: ...


def _require_gate_batch(*, q: torch.Tensor, action: torch.Tensor) -> int:
    """``N`` for a gate. A shorter batch raises; it is not a zero continuation.

    The action taken from step ``t`` is stored at index ``t + 1``, so a
    gate needs at least two steps. ``q`` is ``[N, A]`` and ``action`` is
    ``[N]``.
    """
    if q.ndim != 2:
        raise ValueError(f"q must have shape [N, A], got {tuple(q.shape)}.")
    N = int(q.shape[0])
    if action.ndim != 1 or int(action.shape[0]) != N:
        raise ValueError(
            f"action must have shape [{N}], got {tuple(action.shape)}."
        )
    if N < 2:
        raise ValueError(f"gate needs at least 2 steps, got {N}.")
    return N


def lambda_gate(*, td_lambda: float) -> Gate:
    """λ-return to the end of the run.

    Returns ``[N, N]``. ``out[t, s]`` is λ at absolute step ``s`` for the
    return that started at ``t``, and ``0`` when ``s <= t``.
    ``td_lambda=0`` is the zero matrix (one-step). The objective still
    applies the in-run mask. Fewer than 2 steps, or an ``action`` whose
    length is not ``N``, raises.

    Args:
        td_lambda: λ in ``[0, 1]``.
    """
    if not 0.0 <= float(td_lambda) <= 1.0:
        raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}.")
    lam = float(td_lambda)

    def gate(
        *,
        q: torch.Tensor,
        action: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        N = _require_gate_batch(q=q, action=action)
        t = torch.arange(N, device=q.device).unsqueeze(1)
        s = torch.arange(N, device=q.device).unsqueeze(0)
        fill = q.new_full((), lam)
        return torch.where(s > t, fill, fill.new_zeros(()))

    return gate


def watkins_gate() -> Gate:
    """Hard trace cut where the taken action is not an online argmax.

    Returns ``[N, N]``. Column ``s`` is ``1`` when the action taken from
    step ``s`` is an online max of ``q`` (ties count) and ``0`` otherwise.
    Column ``0`` and the last column are ``0``. The objective still
    applies the in-run mask. Fewer than 2 steps, or an ``action`` whose
    length is not ``N``, raises.
    """

    def gate(
        *,
        q: torch.Tensor,
        action: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        N = _require_gate_batch(q=q, action=action)
        scores = q[:-1]
        taken = action[1:].unsqueeze(-1)
        is_max = scores == scores.amax(dim=-1, keepdim=True)
        greedy = is_max.gather(dim=-1, index=taken).squeeze(-1).to(dtype=q.dtype)
        column = torch.cat([greedy, greedy.new_zeros(1)])
        t = torch.arange(N, device=q.device).unsqueeze(1)
        s = torch.arange(N, device=q.device).unsqueeze(0)
        return torch.where(s > t, column, column.new_zeros(()))

    return gate


def nstep_gate(*, n: int) -> Gate:
    """n-step return. The continuation is ``1`` out to lag ``n``.

    Returns ``[N, N]``. ``out[t, s]`` is ``1`` when ``t < s < t + n``
    and ``0`` otherwise, so each return bootstraps at lag ``n``.
    ``n=1`` is the zero matrix (one-step). An ``n`` at least as long as
    the batch keeps every later step. Fewer than 2 steps, or an
    ``action`` whose length is not ``N``, raises.

    Args:
        n: Horizon, an ``int >= 1``.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an int >= 1, got {n!r}.")
    horizon = n

    def gate(
        *,
        q: torch.Tensor,
        action: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        N = _require_gate_batch(q=q, action=action)
        t = torch.arange(N, device=q.device).unsqueeze(1)
        s = torch.arange(N, device=q.device).unsqueeze(0)
        return ((s > t) & (s < t + horizon)).to(dtype=q.dtype)

    return gate


def value_gap_gate(
    *,
    beta: float,
    normalize: bool,
    delayed: bool,
    eps: float | None = None,
) -> Gate:
    """Soft trace cut from the optimality gap.

    ``V = max_a Q(s, a)`` on the detached per-step Q selected by
    ``delayed``, after ``value``. ``delayed=False`` reads the online Q
    the objective passes as ``q``. ``delayed=True`` reads the delayed Q
    passed as ``q_delayed`` (same ``[N, A]`` layout). ``V`` here is
    that max. The bootstrap state value stays the delayed one the
    objective already computes. ``normalize=True`` divides by the range
    of the selected Q at that state:

    ``c = exp(-beta * (V - Q(s, a_taken)) / (max_a Q - min_a Q + eps))``.

    ``normalize=False`` leaves that range term out and uses the raw gap,
    ``c = exp(-beta * (V - Q(s, a_taken)))``. ``eps`` is required when
    ``normalize`` is ``True`` and is rejected when ``normalize`` is
    ``False``. Returns ``[N, N]``. Column ``s`` is that continuation through step
    ``s`` for every earlier start, the same step ``watkins_gate`` marks
    with its greedy flag. Column ``0`` and the last column are ``0``
    (no action leaves the batch). A zero gap — the taken action is a
    max of the selected Q, ties included — continues with ``c = 1``. A
    larger gap shrinks the continuation toward a bootstrap of the
    delayed state value. The objective still applies the in-run mask.
    Fewer than 2 steps, or an ``action`` whose length is not ``N``,
    raises.

    Args:
        beta: Positive scale on the gap. With ``normalize=True``, a
            taken action at the minimum of the selected Q continues
            near ``exp(-beta)`` when the range is large next to
            ``eps``. With ``normalize=False``, a gap of ``1`` continues
            at ``exp(-beta)``.
        normalize: ``True`` divides the gap by ``max_a Q - min_a Q + eps``.
            ``False`` uses the raw gap.
        delayed: ``False`` reads online Q (``q``). ``True`` reads
            delayed Q (``q_delayed``).
        eps: Positive floor added to ``max_a Q - min_a Q``. Required
            when ``normalize`` is ``True``. Omit it when ``normalize``
            is ``False``.
    """
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError(f"beta must be a real number, got {type(beta)}.")
    scale = float(beta)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"beta must be finite and > 0, got {beta}.")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool, got {type(normalize)}.")
    if not isinstance(delayed, bool):
        raise TypeError(f"delayed must be a bool, got {type(delayed)}.")
    floor: float | None
    if normalize:
        if eps is None:
            raise ValueError("eps is required when normalize is True.")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError(f"eps must be a real number, got {type(eps)}.")
        floor = float(eps)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError(f"eps must be finite and > 0, got {eps}.")
    else:
        if eps is not None:
            raise ValueError("eps is used only when normalize is True.")
        floor = None

    def gate(
        *,
        q: torch.Tensor,
        action: torch.Tensor,
        q_delayed: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        N = _require_gate_batch(q=q, action=action)
        scores = (q_delayed if delayed else q)[:-1]
        taken = action[1:].unsqueeze(-1)
        q_taken = scores.gather(dim=-1, index=taken).squeeze(-1)
        q_max = scores.amax(dim=-1)
        gap = q_max - q_taken
        if floor is not None:
            q_min = scores.amin(dim=-1)
            gap = gap / (q_max - q_min + floor)
        cont = torch.exp(-scale * gap)
        cont[0] = 0
        column = torch.cat([cont, cont.new_zeros(1)])
        t = torch.arange(N, device=scores.device).unsqueeze(1)
        s = torch.arange(N, device=scores.device).unsqueeze(0)
        return torch.where(s > t, column, column.new_zeros(()))

    return gate


def general_gate(*, gates: Sequence[Gate]) -> Gate:
    """Element-wise product of continuation matrices.

    Each gate is called as ``gate(**objective_data, q=q, q_delayed=q_delayed)``.
    The result is the product of those ``[N, N]`` matrices, so a step continues
    only where every gate continues. One gate returns that gate's
    matrix. An empty sequence, a non-sequence, or a non-callable
    raises. A matrix that is not ``[N, N]`` raises. Fewer than 2 steps,
    or an ``action`` whose length is not ``N``, raises.

    Args:
        gates: Gates to multiply, in order.
    """
    if isinstance(gates, (str, bytes)) or not isinstance(gates, Sequence):
        raise TypeError(f"gates must be a sequence of gates, got {type(gates)}.")
    if len(gates) < 1:
        raise ValueError("general_gate needs at least one gate.")
    for item in gates:
        if not callable(item):
            raise TypeError(f"gates must be callables, got {type(item)}.")
    chosen = tuple(gates)

    def gate(
        *,
        q: torch.Tensor,
        action: torch.Tensor,
        **columns: torch.Tensor,
    ) -> torch.Tensor:
        N = _require_gate_batch(q=q, action=action)

        def matrix_of(item: Gate) -> torch.Tensor:
            matrix = item(q=q, action=action, **columns)
            if not isinstance(matrix, torch.Tensor):
                raise TypeError(
                    f"gate must return a Tensor, got {type(matrix)}."
                )
            if matrix.shape != (N, N):
                raise ValueError(
                    f"gate must return shape [{N}, {N}], got {tuple(matrix.shape)}."
                )
            return matrix

        acc = matrix_of(chosen[0])
        for item in chosen[1:]:
            acc.mul_(matrix_of(item))
        return acc

    return gate


def _require_transform(
    value: Callable[..., torch.Tensor] | None, *, name: str
) -> Callable[..., torch.Tensor] | None:
    if value is None:
        return None
    if not callable(value):
        raise TypeError(f"{name} must be callable, got {type(value)}.")
    return cast(Callable[..., torch.Tensor], value)


def _apply_transform(
    *,
    transform: Callable[..., torch.Tensor] | None,
    name: str,
    objective_data: dict[str, torch.Tensor],
    N: int,
    dtype: torch.dtype,
    device: torch.device | str,
    identity: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``transform`` on unpacked ``objective_data`` and cast.

    ``transform is None`` skips the call and uses ``identity``.
    """
    values = identity if transform is None else transform(**objective_data)
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
    transform: Callable[..., torch.Tensor] | None,
    name: str,
    value: torch.Tensor,
    objective_data: dict[str, torch.Tensor],
    step_of: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Evaluate ``transform`` on ``value=`` plus row-aligned ``objective_data``.

    ``transform is None`` skips the call and returns ``value``.
    """
    if transform is None:
        return value
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
