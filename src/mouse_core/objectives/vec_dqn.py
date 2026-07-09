"""One-step two-head vector DQN cosine-similarity objective."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import _valid_transitions
from mouse_core.models.heads.vec_dqn import rope_rotate, vector_action_scores


class VecDqnObjective(Objective):
    """Vector-DQN cosine-similarity objective with a frozen target network.

    Reads ``predictions["action_vector"]`` and ``predictions["action_vector_target"]``
    (shape ``[B, S, A, D]``) from the model's vector action-value head.
    Boundary rows are not used as current states because the following row may
    be a reset frame whose input action was ignored. When a transition ends at a
    boundary and a reset row is available inside the sampled sequence, that reset
    row supplies the target vector. Row pairs that straddle a packed-segment seam
    (``is_seam`` from ``DataLoader(pack=True)``) are excluded, as are boundary
    transitions whose substitute target row starts a new segment.

    Args:
        tau: Polyak coefficient for target-network updates.
            Pass to ``model.polyak_update(action_vector_tau=objective.tau)`` after
            each optimizer step.
        reward_scale: Maps reward to rotation angle: ``θ = reward * reward_scale + reward_shift``.
        reward_shift: Additive offset applied after scaling.
        normalize_reward_mean: Subtract per-sequence mean from rewards.
        normalize_reward_std: Divide rewards by per-sequence std.
        normalize_reward_eps: Numerical floor used in std normalization.
        normalize_reward_std_target: Scale factor applied after std normalization.
        use_episodic_reward: Use ``objective_data["reward_episodic"]`` instead of
            ``objective_data["reward"]`` as the rotation signal.
        action_key: Key in ``objective_data`` that holds the integer action.
    """

    def __init__(
        self,
        *,
        tau: float = 0.01,
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        normalize_reward_mean: bool = False,
        normalize_reward_std: bool = False,
        normalize_reward_eps: float = 1e-8,
        normalize_reward_std_target: float = 1.0,
        use_episodic_reward: bool = False,
        action_key: str = "action",
    ) -> None:
        self.tau = tau
        self.reward_scale = reward_scale
        self.reward_shift = reward_shift
        self.normalize_reward_mean = normalize_reward_mean
        self.normalize_reward_std = normalize_reward_std
        self.normalize_reward_eps = normalize_reward_eps
        self.normalize_reward_std_target = normalize_reward_std_target
        self.use_episodic_reward = use_episodic_reward
        self.action_key = action_key

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        online_vecs: torch.Tensor = predictions["action_vector"]
        target_vecs: torch.Tensor = predictions["action_vector_target"]

        B, S, A, D = online_vecs.shape
        dtype = torch.float32

        if S < 2:
            raise ValueError("Not enough valid vec_dqn vectors in data.")

        action = objective_data[self.action_key].to(dtype=torch.long)
        if action.ndim == 3 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        if action.ndim != 2:
            raise ValueError(f"VecDqnObjective expects action shape [B, S], got {tuple(action.shape)}.")
        if self.use_episodic_reward:
            if "reward_episodic" not in objective_data.keys():
                raise KeyError(
                    "use_episodic_reward=True but 'reward_episodic' is not in the batch. "
                    "Ensure your dataset includes the 'reward_episodic' column."
                )
            reward = objective_data["reward_episodic"].to(dtype=dtype)
        else:
            reward = objective_data["reward"].to(dtype=dtype)
        done = objective_data["done"]
        online_vecs = online_vecs.to(dtype=dtype)
        target_vecs = target_vecs.to(dtype=dtype)

        # Each token at position t encodes (obs_t, action_{t-1}, reward_{t-1}, done_{t-1}),
        # i.e. the action and reward stored at t are the ones that *produced* obs_t, not
        # the ones taken *from* obs_t.  Therefore the action, reward, and done that
        # correspond to the transition out of state t are stored one step ahead at t+1.
        curr_vecs = online_vecs[:, :-1, :, :]   # [B, S-1, A, D]  vecs(s_t)
        # Cloned because the boundary substitution below writes into it; a plain
        # slice is a view that would mutate predictions["action_vector_target"].
        next_vecs = target_vecs[:, 1:, :, :].clone()  # [B, S-1, A, D]  vecs_target(s_{t+1})
        next_actions = action[:, 1:]             # [B, S-1]         a_t (stored at t+1)
        next_rewards = reward[:, 1:]             # [B, S-1]         r_t (stored at t+1)

        boundary_at_next = done[:, 1:] != 0
        if S > 2:
            next_vecs[:, :-1, :, :] = torch.where(
                boundary_at_next[:, :-1].view(B, S - 2, 1, 1),
                target_vecs[:, 2:, :, :],
                next_vecs[:, :-1, :, :],
            )
        valid_transition = done[:, :-1] == 0
        valid_transition[:, -1] = valid_transition[:, -1] & ~boundary_at_next[:, -1]

        # Exclude pairs straddling a packed-segment seam (is_seam at t+1), and
        # boundary transitions whose substituted target row (t+2) starts a new
        # segment — that row belongs to an unrelated slice.
        valid_transition &= _valid_transitions(objective_data, B, S, device=done.device)
        if S > 2 and "is_seam" in objective_data.keys():
            seam_at_substitute = objective_data["is_seam"][:, 2:] != 0
            valid_transition[:, :-1] &= ~(boundary_at_next[:, :-1] & seam_at_substitute)

        if not valid_transition.any():
            raise ValueError("VecDqnObjective: batch contains no valid transitions.")

        action_idx_exp = next_actions.unsqueeze(-1).unsqueeze(-1).expand(B, S - 1, 1, D)
        curr_action_vecs = curr_vecs.gather(dim=2, index=action_idx_exp).squeeze(2)  # [B, S-1, D]

        greedy_idx = vector_action_scores(next_vecs).argmax(dim=-1)                        # [B, S-1]
        greedy_idx_exp = greedy_idx.unsqueeze(-1).unsqueeze(-1).expand(B, S - 1, 1, D)
        next_action_vecs = next_vecs.gather(dim=2, index=greedy_idx_exp).squeeze(2)  # [B, S-1, D]

        if self.normalize_reward_mean:
            next_rewards = next_rewards - next_rewards.mean(dim=1, keepdim=True)
        if self.normalize_reward_std:
            next_rewards = (next_rewards / (next_rewards.std(dim=1, keepdim=True) + self.normalize_reward_eps)) * self.normalize_reward_std_target

        theta = next_rewards * self.reward_scale + self.reward_shift          # [B, S-1]
        rotated = rope_rotate(x=next_action_vecs, theta=theta)                # [B, S-1, D]

        cosine_sim = F.cosine_similarity(curr_action_vecs, rotated.detach(), dim=-1)  # [B, S-1]
        loss = (1.0 - cosine_sim)[valid_transition].mean()

        abs_scores = vector_action_scores(online_vecs[:, -1].float()).abs() / (math.pi)  # [B, A]
        named: dict[str, torch.Tensor] = {
            "action_vector": loss.detach(),
            "action_vector_score_abs_min": abs_scores.min().detach(),
            "action_vector_score_abs_max": abs_scores.max().detach(),
            "action_vector_score_abs_mean": abs_scores.mean().detach(),
        }
        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
