"""One-step two-head DQN TD objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


def _valid_transitions(
    objective_data: TensorDict,
    N: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Boolean ``[N-1]`` mask: True where the pair ``(i, i+1)`` is a real transition.

    Adjacent steps belonging to different sequences (different ``sequence_id``)
    are excluded. Batches without a ``sequence_id`` column are fully valid.
    """
    if device is None:
        device = torch.device("cpu")
    if N < 2:
        return torch.zeros(0, dtype=torch.bool, device=device)
    valid = torch.ones(N - 1, dtype=torch.bool, device=device)
    if "sequence_id" in objective_data.keys():
        sequence_id = objective_data["sequence_id"]
        if sequence_id.shape != torch.Size([N]):
            raise ValueError(
                f"sequence_id must have shape [{N}], got {tuple(sequence_id.shape)}."
            )
        valid &= sequence_id[1:] == sequence_id[:-1]
        if not valid.any():
            raise ValueError(
                "No valid transitions: every consecutive pair in the batch "
                "crosses a sequence boundary."
            )
    return valid


class DqnObjective(Objective):
    """One-step Bellman TD objective with a frozen target network.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss.

    Every consecutive pair ``(i, i+1)`` that shares a ``sequence_id`` is a valid
    TD transition. Cross-sequence pairs are excluded. The done code stored at
    ``i+1`` determines the discount applied to the bootstrap value:

    +------+--------------------------------------+------------------------------+
    | done | Meaning                              | Discount parameter           |
    +======+======================================+==============================+
    | 0    | Running (non-terminal)               | ``gamma_step``               |
    +------+--------------------------------------+------------------------------+
    | 1    | Episode terminated (not last in task)| ``gamma_episode_terminal``   |
    +------+--------------------------------------+------------------------------+
    | 2    | Episode truncated (not last in task) | ``gamma_episode_truncated``  |
    +------+--------------------------------------+------------------------------+
    | 3    | Task terminated (last episode done)  | ``gamma_task_terminal``      |
    +------+--------------------------------------+------------------------------+
    | 4    | Task truncated (last episode trunc.) | ``gamma_task_truncated``     |
    +------+--------------------------------------+------------------------------+

    Args:
        gamma_step: Discount factor for running (non-terminal) transitions (``done == 0``).
        gamma_episode_terminal: Discount applied when the episode terminates naturally
            within a task (``done == 1``). Set to ``1.0`` to bootstrap across
            episode boundaries (recommended for multi-episode MOUSE tasks).
        gamma_episode_truncated: Discount applied when the episode is truncated within a
            task (``done == 2``). Set to ``1.0`` to bootstrap across episode
            boundaries.
        gamma_task_terminal: Discount applied when the task terminates naturally
            (``done == 3``).
        gamma_task_truncated: Discount applied when the task is truncated
            (``done == 4``).
        tau: Polyak coefficient for target-network updates.
            Pass to ``model.polyak_update(action_value_tau=objective.tau)`` after
            each optimizer step.
        action_key: Key in ``objective_data`` that holds the integer action.
        reward_key: Key in ``objective_data`` that holds the per-step reward.
        done_key: Key in ``objective_data`` that holds the integer done code.
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
        done_key: str = "done",
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
        self.done_key = done_key
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

        done = objective_data[self.done_key]
        if done.dtype != torch.int64:
            raise TypeError(f"done must be int64, got {done.dtype}.")
        if done.shape != torch.Size([N]):
            raise ValueError(f"DQN objective expects done shape [{N}], got {tuple(done.shape)}.")

        valid = _valid_transitions(objective_data, N, device)

        # Each token at position i encodes (obs_i, action_{i-1}, reward_{i-1}, done_{i-1}),
        # i.e. the action, reward, and done stored at i are the ones that *produced* obs_i,
        # not the ones taken *from* obs_i.  The transition out of state i is therefore
        # described by the fields stored at i+1.
        curr_q = q[:-1, :]              # [N-1, A]  Q(s_i)
        next_actions = action[1:]       # [N-1]     a_i (stored at i+1)
        next_rewards = reward[1:]       # [N-1]     r_i (stored at i+1)
        next_done = done[1:]            # [N-1]     done code at i+1
        next_q_target = q_target[1:, :]  # [N-1, A]  Q_target(s_{i+1})

        # Vectorized discount: one gamma per done code (0–4).
        gammas = torch.tensor(
            [
                self.gamma_step,
                self.gamma_episode_terminal,
                self.gamma_episode_truncated,
                self.gamma_task_terminal,
                self.gamma_task_truncated,
            ],
            dtype=value_dtype,
            device=device,
        )
        discount = gammas[next_done]  # [N-1]

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
