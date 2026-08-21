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


def _valid_transitions(
    objective_data: TensorDict,
    N: int,
    device: torch.device | str | None,
    *,
    grouping_field: str | None = None,
) -> torch.Tensor:
    """Boolean ``[N-1]`` mask: True where the pair ``(i, i+1)`` is a real transition.

    Adjacent steps belonging to different sequences (different ``sequence_id``)
    or different grouping runs (``grouping_field`` column) are excluded. Batches
    without those columns skip the corresponding check.
    """
    if device is None:
        device = torch.device("cpu")
    if N < 2:
        return torch.zeros(0, dtype=torch.bool, device=device)
    valid = torch.ones(N - 1, dtype=torch.bool, device=device)
    used_grouping = False
    if "sequence_id" in objective_data.keys():
        sequence_id = objective_data["sequence_id"]
        if sequence_id.shape != torch.Size([N]):
            raise ValueError(
                f"sequence_id must have shape [{N}], got {tuple(sequence_id.shape)}."
            )
        valid &= sequence_id[1:] == sequence_id[:-1]
    if grouping_field is not None and grouping_field in objective_data.keys():
        grouping = objective_data[grouping_field]
        if grouping.shape != torch.Size([N]):
            raise ValueError(
                f"{grouping_field} must have shape [{N}], got {tuple(grouping.shape)}."
            )
        valid &= grouping[1:] == grouping[:-1]
        used_grouping = True
    if ("sequence_id" in objective_data.keys() or used_grouping) and (not valid.any()):
        raise ValueError(
            "No valid transitions: every consecutive pair in the batch "
            "crosses a sequence or grouping boundary."
        )
    return valid


class DqnObjective(Objective):
    """One-step Bellman TD objective with a frozen target network.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss.

    Every consecutive pair ``(i, i+1)`` that shares ``sequence_id`` and
    ``grouping_field`` (when set and present) is a valid TD transition.
    Cross-sequence and cross-grouping pairs are excluded. The
    ``episode_done`` / ``task_done`` codes stored at ``i+1`` determine the
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
                "action",
                "reward",
                "episode_done",
                "task_done",
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
        tau: Polyak coefficient for target-network updates.
            Pass to ``model.polyak_update(action_value_tau=objective.tau)`` after
            each optimizer step.
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
        tau: float = 0.01,
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
        self.tau = tau
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

        valid = _valid_transitions(
            objective_data, N, device, grouping_field=self.grouping_field
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
            cql_penalty_mean = cql_penalty.detach()[valid].mean()

        loss = loss[valid].mean()

        curr_max_q = curr_q.amax(dim=-1)  # [N-1]  max online Q at s_i
        curr_max_det = curr_max_q.detach()[valid]
        curr_max_std = (
            curr_max_det.std()
            if curr_max_det.numel() > 1
            else torch.zeros((), device=device, dtype=value_dtype)
        )
        named: dict[str, torch.Tensor] = {
            "q_values_mean":   curr_max_det.mean(),
            "q_values_std":    curr_max_std,
            "q_values_min":    curr_max_det.min(),
            "q_values_max":    curr_max_det.max(),
            "action_value":    loss.detach(),
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
