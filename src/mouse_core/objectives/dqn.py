"""One-step two-head DQN TD objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


def _valid_transitions(
    objective_data: TensorDict,
    B: int,
    S: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean ``[B, S-1]`` mask: True where the pair ``(t, t+1)`` is a real transition.

    ``DataLoader(pack=True)`` stitches sequences from independently sampled
    segments and flags the first row of each appended segment with
    ``is_seam=1``. A pair whose row ``t+1`` starts a new segment straddles two
    unrelated slices, so it must not be trained on. Batches without an
    ``is_seam`` column (unpacked loaders, manual batches) are fully valid.
    """
    valid = torch.ones(B, S - 1, dtype=torch.bool, device=device)
    if "is_seam" in objective_data.keys():
        is_seam = objective_data["is_seam"]
        if is_seam.shape != torch.Size([B, S]):
            raise ValueError(
                f"is_seam must have shape [{B}, {S}], got {tuple(is_seam.shape)}."
            )
        valid &= is_seam[:, 1:] == 0
        if not valid.any():
            raise ValueError(
                "No valid transitions: every consecutive pair in the batch "
                "crosses a packed-segment seam."
            )
    return valid


class DqnObjective(Objective):
    """One-step Bellman TD objective with a frozen target network.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss.

    Every consecutive pair ``(t, t+1)`` within a sampled sequence is a valid
    TD transition, unless ``objective_data`` carries the DataLoader's pack-mode
    ``is_seam`` flag and row ``t+1`` starts a new packed segment — those pairs
    straddle independently sampled slices and are excluded from the loss. The
    done code stored at ``t+1`` determines the discount applied to the
    bootstrap value:

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

        B, S, A = q.shape
        device = q.device
        value_dtype = q.dtype

        if S < 2:
            raise ValueError("Not enough valid q values in data.")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.shape != torch.Size([B, S]):
            raise ValueError(f"DQN objective expects action shape [{B}, {S}], got {tuple(action.shape)}.")

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([B, S]):
            raise ValueError(f"DQN objective expects reward shape [{B}, {S}], got {tuple(reward.shape)}.")

        done = objective_data[self.done_key]
        if done.dtype != torch.int64:
            raise TypeError(f"done must be int64, got {done.dtype}.")
        if done.shape != torch.Size([B, S]):
            raise ValueError(f"DQN objective expects done shape [{B}, {S}], got {tuple(done.shape)}.")

        valid = _valid_transitions(objective_data, B, S, device)

        # Each token at position t encodes (obs_t, action_{t-1}, reward_{t-1}, done_{t-1}),
        # i.e. the action, reward, and done stored at t are the ones that *produced* obs_t,
        # not the ones taken *from* obs_t.  The transition out of state t is therefore
        # described by the fields stored at t+1.
        curr_q        = q[:, :-1, :]              # [B, S-1, A]  Q(s_t)
        next_actions  = action[:, 1:]             # [B, S-1]     a_t (stored at t+1)
        next_rewards  = reward[:, 1:]             # [B, S-1]     r_t (stored at t+1)
        next_done     = done[:, 1:]  # [B, S-1]  done code at t+1
        next_q_target = q_target[:, 1:, :]        # [B, S-1, A]  Q_target(s_{t+1})

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
        discount = gammas[next_done]  # [B, S-1]

        q_values          = curr_q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
        next_max_q_target = next_q_target.amax(dim=-1)                                           # [B, S-1]

        td_target = next_rewards + discount * next_max_q_target

        loss = (q_values - td_target.detach()) ** 2

        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale      = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty  = torch.logsumexp(curr_q, dim=-1) - q_values
            loss         = loss + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = cql_penalty.detach()[valid].mean()

        loss = loss[valid].mean()

        q_det = q_values.detach()[valid]
        td_det = td_target.detach()[valid]
        q_std = q_det.std() if q_det.numel() > 1 else torch.zeros((), device=device, dtype=value_dtype)
        named: dict[str, torch.Tensor] = {
            "q_values_mean":   q_det.mean(),
            "q_values_std":    q_std,
            "q_values_min":    q_det.min(),
            "q_values_max":    q_det.max(),
            "q_values_target": td_det.mean(),
            "action_value":    loss.detach(),
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
