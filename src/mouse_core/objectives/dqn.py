"""One-step two-head DQN TD objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


class DqnObjective(Objective):
    """One-step Bellman TD objective with a frozen target network.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss.

    Args:
        gamma: Discount factor for non-terminal transitions.
        gamma_terminal: Discount applied when the episode terminates naturally
            (``done == 1``). Set to ``0.0`` to use no bootstrap on terminal steps.
        gamma_truncated: Discount applied when the episode is truncated by a
            time limit (``done == 2``). Set to ``0.0`` to use no bootstrap on
            truncated steps.
        tau: Polyak coefficient for target-network updates.
            Pass to ``model.polyak_update(action_value_tau=objective.tau)`` after
            each optimizer step.
        normalize_reward_mean: Subtract per-sequence mean from rewards before
            computing TD targets.
        normalize_reward_std: Divide rewards by per-sequence std before computing
            TD targets.
        normalize_reward_eps: Numerical floor used in std normalization.
        normalize_reward_std_target: Scale factor applied after std normalization.
        use_episodic_reward: Use ``objective_data["reward_episodic"]`` instead of
            ``objective_data["reward"]`` as the TD reward signal.
        action_key: Key in ``objective_data`` that holds the integer action.
        cql_weight: Alpha coefficient for the Conservative Q-Learning penalty.
            ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.
        reward_scale: Multiply reward before computing the TD target.
        reward_shift: Additive offset applied after scaling in the TD target.
    """

    def __init__(
        self,
        *,
        gamma: float = 0.99,
        gamma_terminal: float = 0.0,
        gamma_truncated: float = 0.0,
        tau: float = 0.01,
        normalize_reward_mean: bool = False,
        normalize_reward_std: bool = False,
        normalize_reward_eps: float = 1e-8,
        normalize_reward_std_target: float = 1.0,
        use_episodic_reward: bool = False,
        action_key: str = "action",
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
    ) -> None:
        self.gamma = gamma
        self.gamma_terminal = gamma_terminal
        self.gamma_truncated = gamma_truncated
        self.tau = tau
        self.normalize_reward_mean = normalize_reward_mean
        self.normalize_reward_std = normalize_reward_std
        self.normalize_reward_eps = normalize_reward_eps
        self.normalize_reward_std_target = normalize_reward_std_target
        self.use_episodic_reward = use_episodic_reward
        self.action_key = action_key
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift

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

        action = objective_data[self.action_key].to(dtype=torch.long)
        if action.ndim == 3 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        if action.ndim != 2:
            raise ValueError(f"DQN objective expects action shape [B, S], got {tuple(action.shape)}.")
        if self.use_episodic_reward:
            if "reward_episodic" not in objective_data.keys():
                raise KeyError(
                    "use_episodic_reward=True but 'reward_episodic' is not in the batch. "
                    "Ensure your dataset includes the 'reward_episodic' column."
                )
            reward = objective_data["reward_episodic"].to(dtype=value_dtype)
        else:
            reward = objective_data["reward"].to(dtype=value_dtype)
        terminals = (objective_data["done"] == 1).to(dtype=value_dtype)
        truncateds = (objective_data["done"] == 2).to(dtype=value_dtype)

        # Each token at position t encodes (obs_t, action_{t-1}, reward_{t-1}, done_{t-1}),
        # i.e. the action and reward stored at t are the ones that *produced* obs_t, not
        # the ones taken *from* obs_t.  Therefore the action, reward, and done that
        # correspond to the transition out of state t are stored one step ahead at t+1.
        # Consecutive (s, s+1) pairs within each batch row.
        curr_q = q[:, :-1, :]              # [B, S-1, A]  Q(s_t)
        next_q_target = q_target[:, 1:, :] # [B, S-1, A]  Q_target(s_{t+1})
        next_actions = action[:, 1:]        # [B, S-1]     a_t (stored at t+1)
        next_rewards = reward[:, 1:]        # [B, S-1]     r_t (stored at t+1)
        next_terminals = terminals[:, 1:]   # [B, S-1]     terminal_t (stored at t+1)
        next_truncateds = truncateds[:, 1:] # [B, S-1]     truncated_t (stored at t+1)

        q_values = curr_q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
        next_max_q_target = next_q_target.amax(dim=-1)                                  # [B, S-1]

        if self.normalize_reward_mean:
            next_rewards = next_rewards - next_rewards.mean(dim=1, keepdim=True)
        if self.normalize_reward_std:
            next_rewards = (next_rewards / (next_rewards.std(dim=1, keepdim=True) + self.normalize_reward_eps)) * self.normalize_reward_std_target

        discount = (
            self.gamma * (1.0 - next_terminals - next_truncateds)
            + self.gamma_terminal * next_terminals
            + self.gamma_truncated * next_truncateds
        )
        next_rewards_adjusted = next_rewards * self.reward_scale + self.reward_shift
        td_target = next_rewards_adjusted + discount * next_max_q_target
        td_target = td_target.to(dtype=q_values.dtype)

        loss = (q_values - td_target.detach()) ** 2

        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty = torch.logsumexp(curr_q, dim=-1) - q_values
            loss = loss + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = cql_penalty.detach().mean()

        loss = loss.mean()

        q_det = q_values.detach()
        named: dict[str, torch.Tensor] = {
            "q_values_mean": q_det.mean(),
            "q_values_std":  q_det.std(),
            "q_values_min":  q_det.min(),
            "q_values_max":  q_det.max(),
            "q_values_target": td_target.detach().mean(),
            "action_value":  loss.detach(),
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
