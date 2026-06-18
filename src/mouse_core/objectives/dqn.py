"""One-step two-head DQN TD objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mouse_core.objectives.base import ObjectiveConfig
import torch.nn.functional as F
from tensordict import TensorDict


@dataclass(frozen=True)
class DqnObjectiveConfig(ObjectiveConfig):
    """Symmetric two-head one-step TD at PREDICTION (see ``dqn_objective``)."""

    weight: float = 0.0  # omit ``loop.dqn.weight`` or set 0 = do not compute DQN objective (YAML default)
    gamma: float = 0.99
    gamma_terminal: float = 0.0  # discount on max Q(s') when episode terminates naturally
    gamma_truncated: float = 0.0  # discount on max Q(s') when episode is truncated (time limit)
    tau: float = 0.01  # Polyak toward target head; θ_target ← τ·θ_online + (1−τ)·θ_target
    normalize_reward_mean: bool = False  # per-sequence-row subtract mean of TD rewards
    normalize_reward_std: bool = False  # per-sequence-row divide by std of TD rewards
    normalize_reward_eps: float = 1e-8  # numerical floor for variance and division in TD reward normalization
    normalize_reward_std_target: float = 1.0  # multiply normalized TD rewards by this (after mean/std norm; no effect if both norm flags false)
    use_xformed_reward: bool = False  # use xformed_reward instead of reward as the TD target signal
    cql_weight: float = 0.0  # α in CQL penalty; 0 = disabled
    cql_scale_q_eps: float = 1.0  # additive floor in CQL Q-scaling: scale = |td_target| + cql_scale_q_eps
    reward_scale: float = 1.0  # multiply reward before TD target: td_target = r * reward_scale + reward_shift + γ * max Q(s')
    reward_shift: float = 0.0  # additive offset applied after scaling in TD target


def dqn_objective(
    step_stream: TensorDict,
    out: TensorDict,
    cfg: DqnObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, float]]:

    q: torch.Tensor = out["dqn"]
    q_target: torch.Tensor = out["dqn_target"]

    B, S, A = q.shape
    device = q.device
    value_dtype = q.dtype

    if S < 2:
        raise ValueError("Not enough valid q values in data.")

    action = step_stream["action"].to(dtype=torch.long)
    if cfg.use_xformed_reward:
        if "xformed_reward" not in step_stream.keys():
            raise KeyError(
                "use_xformed_reward=True but 'xformed_reward' is not in the batch. "
                "Ensure your dataset includes the 'xformed_reward' column."
            )
        reward = step_stream["xformed_reward"].to(dtype=value_dtype)
    else:
        reward = step_stream["reward"].to(dtype=value_dtype)
    terminals = (step_stream["done"] == 1).to(dtype=value_dtype)
    truncateds = (step_stream["done"] == 2).to(dtype=value_dtype)

    # Each token at position t encodes (obs_t, action_{t-1}, reward_{t-1}, done_{t-1}),
    # i.e. the action and reward stored at t are the ones that *produced* obs_t, not
    # the ones taken *from* obs_t.  Therefore the action, reward, and done that
    # correspond to the transition out of state t are stored one step ahead at t+1.
    # Consecutive (s, s+1) pairs within each batch row.
    curr_q = q[:, :-1, :]          # [B, S-1, A]  Q(s_t)
    next_q_target = q_target[:, 1:, :]  # [B, S-1, A]  Q_target(s_{t+1})
    next_actions = action[:, 1:]   # [B, S-1]     a_t (stored at t+1)
    next_rewards = reward[:, 1:]   # [B, S-1]     r_t (stored at t+1)
    next_terminals = terminals[:, 1:]  # [B, S-1]     terminal_t (stored at t+1)
    next_truncateds = truncateds[:, 1:]  # [B, S-1]     truncated_t (stored at t+1)

    q_values = curr_q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [B, S-1]
    next_max_q_target = next_q_target.amax(dim=-1)                           # [B, S-1]

    if cfg.normalize_reward_mean:
        next_rewards = next_rewards - next_rewards.mean(dim=1, keepdim=True)
    if cfg.normalize_reward_std:
        next_rewards = (next_rewards / (next_rewards.std(dim=1, keepdim=True) + cfg.normalize_reward_eps)) * cfg.normalize_reward_std_target

    discount = (
        cfg.gamma * (1.0 - next_terminals - next_truncateds)
        + cfg.gamma_terminal * next_terminals
        + cfg.gamma_truncated * next_truncateds
    )
    next_rewards_adjusted = next_rewards * cfg.reward_scale + cfg.reward_shift
    td_target = next_rewards_adjusted + discount * next_max_q_target
    td_target = td_target.to(dtype=q_values.dtype)

    loss = (q_values - td_target.detach()) ** 2

    cql_penalty_mean: torch.Tensor | None = None
    if cfg.cql_weight > 0.0:
        # CQL penalty ∝ Q while TD loss ∝ Q², so a fixed weight becomes ineffective
        # as Q grows.  Multiplying by q_scale brings CQL up to Q² so the ratio
        # cfg.cql_weight stays constant throughout training.
        q_scale = (td_target.abs() + cfg.cql_scale_q_eps).detach()
        cql_penalty = torch.logsumexp(curr_q, dim=-1) - q_values
        loss = loss + cfg.cql_weight * q_scale * cql_penalty
        cql_penalty_mean = cql_penalty.detach().mean()

    loss = loss.mean()

    q_det = q_values.detach()
    named: dict[str, torch.Tensor] = {
        "q_values_mean": q_det.mean(),
        "q_values_std":  q_det.std(),
        "q_values_min":  q_det.min(),
        "q_values_max":  q_det.max(),
        "q_values_target": td_target.detach().mean(),
        "dqn":           loss.detach(),
    }
    if cql_penalty_mean is not None:
        named["cql_penalty"] = cql_penalty_mean

    metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))

    return loss, metrics
