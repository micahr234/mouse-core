"""One-step two-head vector DQN cosine-similarity objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import ObjectiveConfig
from mouse_core.models.heads.vec_dqn import rope_rotate, vec_dqn_scores


@dataclass(frozen=True)
class VecDqnObjectiveConfig(ObjectiveConfig):
    """Vector-DQN cosine-similarity objective at PREDICTION (see ``vec_dqn_objective``)."""

    weight: float = 0.0  # omit ``loop.vec_dqn.weight`` or set 0 = do not compute (YAML default)
    tau: float = 0.01  # Polyak toward target head; θ_target ← τ·θ_online + (1−τ)·θ_target
    reward_scale: float = 1.0  # maps reward to rotation angle: θ = reward * reward_scale
    reward_shift: float = 0.0  # additive offset applied after scaling: θ = reward * reward_scale + reward_shift
    normalize_reward_mean: bool = False  # per-sequence-row subtract mean of rewards
    normalize_reward_std: bool = False  # per-sequence-row divide by std of rewards
    normalize_reward_eps: float = 1e-8  # numerical floor for variance and division in reward normalization
    normalize_reward_std_target: float = 1.0  # multiply normalized rewards by this (after mean/std norm; no effect if both norm flags false)
    use_xformed_reward: bool = False  # use xformed_reward instead of reward as the rotation signal


def vec_dqn_objective(
    step_stream: TensorDict,
    online_vecs: torch.Tensor,
    target_vecs: torch.Tensor,
    cfg: VecDqnObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, float]]:

    B, S, A, D = online_vecs.shape
    device = online_vecs.device
    dtype = torch.float32

    if S < 2:
        raise ValueError("Not enough valid vec_dqn vectors in data.")

    action = step_stream["action"].to(dtype=torch.long)
    if cfg.use_xformed_reward:
        if "xformed_reward" not in step_stream.keys():
            raise KeyError(
                "use_xformed_reward=True but 'xformed_reward' is not in the batch. "
                "Ensure your dataset includes the 'xformed_reward' column."
            )
        reward = step_stream["xformed_reward"].to(dtype=dtype)
    else:
        reward = step_stream["reward"].to(dtype=dtype)
    online_vecs = online_vecs.to(dtype=dtype)
    target_vecs = target_vecs.to(dtype=dtype)

    # Each token at position t encodes (obs_t, action_{t-1}, reward_{t-1}, done_{t-1}),
    # i.e. the action and reward stored at t are the ones that *produced* obs_t, not
    # the ones taken *from* obs_t.  Therefore the action, reward, and done that
    # correspond to the transition out of state t are stored one step ahead at t+1.
    # Consecutive (s, s+1) pairs within each batch row.
    curr_vecs = online_vecs[:, :-1, :, :]   # [B, S-1, A, D]  vecs(s_t)
    next_vecs = target_vecs[:, 1:, :, :]    # [B, S-1, A, D]  vecs_target(s_{t+1})
    next_actions = action[:, 1:]            # [B, S-1]         a_t (stored at t+1)
    next_rewards = reward[:, 1:]            # [B, S-1]         r_t (stored at t+1)

    # curr: vector for the executed action at s_t (what we train).
    # next: vector for the GREEDY best action at s_{t+1} (bootstrap target),
    #       selected with the same rotate-90 scoring used at inference.
    action_idx_exp = next_actions.unsqueeze(-1).unsqueeze(-1).expand(B, S - 1, 1, D)
    curr_action_vecs = curr_vecs.gather(dim=2, index=action_idx_exp).squeeze(2)  # [B, S-1, D]

    greedy_idx = vec_dqn_scores(next_vecs).argmax(dim=-1)                         # [B, S-1]
    greedy_idx_exp = greedy_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S - 1, 1, D)
    next_action_vecs = next_vecs.gather(dim=2, index=greedy_idx_exp).squeeze(2)   # [B, S-1, D]

    if cfg.normalize_reward_mean:
        next_rewards = next_rewards - next_rewards.mean(dim=1, keepdim=True)
    if cfg.normalize_reward_std:
        next_rewards = (next_rewards / (next_rewards.std(dim=1, keepdim=True) + cfg.normalize_reward_eps)) * cfg.normalize_reward_std_target

    theta = next_rewards * cfg.reward_scale + cfg.reward_shift                     # [B, S-1]
    rotated = rope_rotate(x=next_action_vecs, theta=theta)                         # [B, S-1, D]

    # Cosine similarity objective — detach target mirrors td_target.detach() in dqn_objective
    cosine_sim = F.cosine_similarity(curr_action_vecs, rotated.detach(), dim=-1)  # [B, S-1]
    loss = (1.0 - cosine_sim).mean()

    abs_scores = vec_dqn_scores(online_vecs[:, -1].float()).abs() / (math.pi)  # [B, A]
    named: dict[str, torch.Tensor] = {
        "vec_dqn": loss.detach(),
        "vec_dqn_score_abs_min": abs_scores.min().detach(),
        "vec_dqn_score_abs_max": abs_scores.max().detach(),
        "vec_dqn_score_abs_mean": abs_scores.mean().detach(),
    }
    metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))

    return loss, metrics
