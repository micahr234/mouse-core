"""One-step two-head DQN TD objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


def _require_done_codes(
    objective_data: TensorDict,
    *,
    episode_done_key: str,
    task_done_key: str,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and validate mouse-gym ``episode_done`` / ``task_done`` columns."""
    codes: list[torch.Tensor] = []
    for key in (episode_done_key, task_done_key):
        values = objective_data[key]
        if values.dtype != torch.int64:
            raise TypeError(f"{key} must be int64, got {values.dtype}.")
        if values.shape != torch.Size([N]):
            raise ValueError(
                f"objective expects {key} shape [{N}], got {tuple(values.shape)}."
            )
        if bool((values < 0).any() or (values > 2).any()):
            raise ValueError(f"{key} codes must be 0, 1, or 2.")
        codes.append(values)
    return codes[0], codes[1]


def _boundary_discounts(
    *,
    episode_done: torch.Tensor,
    task_done: torch.Tensor,
    gamma_step: float,
    gamma_episode_terminal: float,
    gamma_episode_truncated: float,
    gamma_task_terminal: float,
    gamma_task_truncated: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Per-transition discount from mouse-gym ``episode_done`` / ``task_done``.

    Both fields use codes ``0`` / ``1`` / ``2``. The bootstrap is multiplied by
    the episode gamma, then by the task gamma. ``task_done == 0`` uses ``1.0``
    so a mid-task step is unchanged by the task factor. When a task ends both
    fire (e.g. ``episode_done=1`` and ``task_done=2``) and the product is used;
    a task gamma of ``0.0`` zeros the whole bootstrap.
    """
    episode_gammas = torch.tensor(
        [gamma_step, gamma_episode_terminal, gamma_episode_truncated],
        dtype=dtype,
        device=device,
    )
    task_gammas = torch.tensor(
        [1.0, gamma_task_terminal, gamma_task_truncated],
        dtype=dtype,
        device=device,
    )
    return episode_gammas[episode_done] * task_gammas[task_done]


def _pair_weight(
    objective_data: TensorDict,
    N: int,
    device: torch.device | str | None,
    *,
    grouping_field: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[N-1]`` weights: ``1.0`` when ``(i, i+1)`` share a run, else ``0.0``.

    A run is the same ``sequence_id`` and, when ``grouping_field`` is set and
    present, the same grouping column. Batches without those columns skip the
    corresponding check (every pair stays weight ``1``).
    """
    if device is None:
        device = torch.device("cpu")
    if N < 2:
        return torch.zeros(0, dtype=dtype, device=device)
    same_run = torch.ones(N - 1, dtype=torch.bool, device=device)
    if "sequence_id" in objective_data.keys():
        sequence_id = objective_data["sequence_id"]
        if sequence_id.shape != torch.Size([N]):
            raise ValueError(
                f"sequence_id must have shape [{N}], got {tuple(sequence_id.shape)}."
            )
        same_run &= sequence_id[1:] == sequence_id[:-1]
    if grouping_field is not None and grouping_field in objective_data.keys():
        grouping = objective_data[grouping_field]
        if grouping.shape != torch.Size([N]):
            raise ValueError(
                f"{grouping_field} must have shape [{N}], got {tuple(grouping.shape)}."
            )
        same_run &= grouping[1:] == grouping[:-1]
    return same_run.to(dtype=dtype)


def _weighted_mean(values: torch.Tensor, pair_weight: torch.Tensor) -> torch.Tensor:
    """``(w * x).sum() / w.sum().clamp(min=1)``. All-zero weights yield ``0``."""
    weight = pair_weight.to(dtype=values.dtype)
    return (weight * values).sum() / weight.sum().clamp(min=1)


def _in_run_stats(
    values: torch.Tensor,
    pair_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean / std / min / max over in-run pairs. All-zero weights yield zeros."""
    in_run = pair_weight > 0
    zero = values.new_zeros(())
    if not bool(in_run.any()):
        return zero, zero, zero, zero
    selected = values[in_run]
    std = selected.std() if selected.numel() > 1 else zero
    return selected.mean(), std, selected.min(), selected.max()


class DqnObjective(Objective):
    """One-step Bellman TD objective with a delayed target network.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss. Fill
    ``predictions["action_value_target"]`` via
    :meth:`~mouse_core.polyak.PolyakAverager.write_targets` after the online
    forward.

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is set
    and present, the same grouping column (typically ``task_index``). Neighbor
    reads (action / reward / done / next Q at ``i+1``) must stay in-run: an
    out-of-run pair still has a loss term, but it is multiplied by ``0`` so
    output ``i`` does not affect the scalar loss or the gradient. If every
    weight is ``0`` the loss is ``0``. Episode resets inside a run
    (``episode_done`` 1/2, then a reset frame) are still in-run and may train.
    Gamma is the Bellman discount from the done codes at ``i+1`` inside a
    same-run pair — it is not a run mask.

    The ``episode_done`` / ``task_done`` codes stored at ``i+1`` determine the
    discount applied to the bootstrap value. Both factors always multiply:
    ``V ← episode_gamma * task_gamma * V``. ``task_done == 0`` uses task
    factor ``1.0``.

    Those columns arrive in ``objective_data`` only if they are listed in the
    tokenizer ``objective_fields`` keep-list (input fields are not auto-copied).
    ``task_done`` is an objective column only — do not add it as a tokenizer
    input field or embedder modality, or it will be fed to the transformer::

        tokenizer = NumericTokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "episode_done"},
                {"input_field": "task_done"},
            ],
        )

    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | episode_done | task_done | Meaning                          | Discount                                      |
    +==============+===========+==================================+===============================================+
    | 0            | 0         | Running                          | ``gamma_step``                                |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 1            | 0         | Episode terminated (mid-task)    | ``gamma_episode_terminal``                    |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 2            | 0         | Episode truncated (mid-task)     | ``gamma_episode_truncated``                   |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 1            | 2         | Last episode terminated          | ``gamma_episode_terminal * gamma_task_truncated`` |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 2            | 2         | Last episode truncated           | ``gamma_episode_truncated * gamma_task_truncated`` |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 1 or 2       | 1         | Task terminated (reserved)       | episode gamma ``* gamma_task_terminal``       |
    +--------------+-----------+----------------------------------+-----------------------------------------------+

    Args:
        gamma_step: Discount factor for running (non-terminal) transitions
            (``episode_done == 0``).
        gamma_episode_terminal: Discount applied when the episode terminates
            naturally (``episode_done == 1``). Set to ``1.0`` to bootstrap
            across episode boundaries (recommended for multi-episode MOUSE
            tasks).
        gamma_episode_truncated: Discount applied when the episode is truncated
            (``episode_done == 2``). Set to ``1.0`` to bootstrap across
            episode boundaries.
        gamma_task_terminal: Extra discount when the task terminates
            (``task_done == 1``; reserved, unused by mouse-gym today).
            Multiplies the episode discount. ``task_done == 0`` uses ``1.0``.
        gamma_task_truncated: Extra discount when the task is truncated
            (``task_done == 2``; last episode of ``episodes_per_task``).
            Multiplies the episode discount. ``0.0`` zeros the bootstrap.
        action_key: Key in ``objective_data`` that holds the integer action.
        reward_key: Key in ``objective_data`` that holds the per-step reward.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        cql_weight: Alpha coefficient for the Conservative Q-Learning penalty.
            ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.
    """

    def __init__(
        self,
        *,
        gamma_step: float = 0.99,
        gamma_episode_terminal: float = 0.0,
        gamma_episode_truncated: float = 0.0,
        gamma_task_terminal: float = 0.0,
        gamma_task_truncated: float = 0.0,
        action_key: str = "action",
        reward_key: str = "reward",
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        grouping_field: str | None = None,
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        self.gamma_step = gamma_step
        self.gamma_episode_terminal = gamma_episode_terminal
        self.gamma_episode_truncated = gamma_episode_truncated
        self.gamma_task_terminal = gamma_task_terminal
        self.gamma_task_truncated = gamma_task_truncated
        self.action_key = action_key
        self.reward_key = reward_key
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.grouping_field = grouping_field
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q: torch.Tensor = predictions["action_value"]
        q_target: torch.Tensor = predictions["action_value_target"]

        if q.ndim != 2:
            raise ValueError(
                f"DQN expects action_value shape [N, A], got {tuple(q.shape)}."
            )
        N, A = q.shape
        device = q.device
        value_dtype = q.dtype

        if N < 2:
            raise ValueError("Not enough valid q values in data.")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.shape != torch.Size([N]):
            raise ValueError(f"DQN objective expects action shape [{N}], got {tuple(action.shape)}.")

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([N]):
            raise ValueError(f"DQN objective expects reward shape [{N}], got {tuple(reward.shape)}.")

        episode_done, task_done = _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=N,
        )

        pair_weight = _pair_weight(
            objective_data,
            N,
            device,
            grouping_field=self.grouping_field,
            dtype=value_dtype,
        )

        # Each token at position i encodes (obs_i, action_{i-1}, reward_{i-1},
        # episode_done_{i-1}, task_done_{i-1}), i.e. the action, reward, and
        # done codes stored at i are the ones that *produced* obs_i, not the
        # ones taken *from* obs_i.  The transition out of state i is therefore
        # described by the fields stored at i+1.
        curr_q = q[:-1, :]              # [N-1, A]  Q(s_i)
        next_actions = action[1:]       # [N-1]     a_i (stored at i+1)
        next_rewards = reward[1:]       # [N-1]     r_i (stored at i+1)
        next_q_target = q_target[1:, :]  # [N-1, A]  Q_target(s_{i+1})

        discount = _boundary_discounts(
            episode_done=episode_done[1:],
            task_done=task_done[1:],
            gamma_step=self.gamma_step,
            gamma_episode_terminal=self.gamma_episode_terminal,
            gamma_episode_truncated=self.gamma_episode_truncated,
            gamma_task_terminal=self.gamma_task_terminal,
            gamma_task_truncated=self.gamma_task_truncated,
            dtype=value_dtype,
            device=device,
        )

        q_values = curr_q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [N-1]
        next_max_q_target = next_q_target.amax(dim=-1)                                  # [N-1]

        td_target = next_rewards + discount * next_max_q_target

        loss = (q_values - td_target.detach()) ** 2

        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty = torch.logsumexp(curr_q, dim=-1) - q_values
            loss = loss + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = _weighted_mean(cql_penalty.detach(), pair_weight)

        loss = _weighted_mean(loss, pair_weight)

        curr_max_q = curr_q.amax(dim=-1)  # [N-1]  max online Q at s_i
        q_mean, q_std, q_min, q_max = _in_run_stats(curr_max_q.detach(), pair_weight)
        named: dict[str, torch.Tensor] = {
            "q_values_mean":   q_mean,
            "q_values_std":    q_std,
            "q_values_min":    q_min,
            "q_values_max":    q_max,
            "action_value":    loss.detach(),
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
