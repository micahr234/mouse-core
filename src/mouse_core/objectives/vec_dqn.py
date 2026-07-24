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
    (shape ``[N, A, D]``) from the model's vector action-value head.
    Boundary rows are not used as current states because the following row may
    be a reset frame whose input action was ignored. When a transition ends at a
    boundary and a reset row is available inside the same sequence, that reset
    row supplies the target vector. Row pairs that straddle a sequence
    boundary (different ``sequence_id``) are excluded, as are boundary
    transitions whose substitute target row belongs to a different sequence.

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

        if online_vecs.ndim != 3:
            raise ValueError(
                f"VecDqnObjective expects action_vector shape [N, A, D], "
                f"got {tuple(online_vecs.shape)}."
            )
        N, A, D = online_vecs.shape
        dtype = torch.float32

        if N < 2:
            raise ValueError("Not enough valid vec_dqn vectors in data.")

        action = objective_data[self.action_key].to(dtype=torch.long)
        if action.ndim == 2 and action.shape[-1] == 1:
            action = action.squeeze(-1)
        if action.ndim != 1:
            raise ValueError(
                f"VecDqnObjective expects action shape [N], got {tuple(action.shape)}."
            )
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

        curr_vecs = online_vecs[:-1, :, :]   # [N-1, A, D]
        next_vecs = target_vecs[1:, :, :].clone()  # [N-1, A, D]
        next_actions = action[1:]             # [N-1]
        next_rewards = reward[1:]             # [N-1]

        boundary_at_next = done[1:] != 0
        if N > 2:
            next_vecs[:-1, :, :] = torch.where(
                boundary_at_next[:-1].view(N - 2, 1, 1),
                target_vecs[2:, :, :],
                next_vecs[:-1, :, :],
            )
        valid_transition = done[:-1] == 0
        valid_transition[-1] = valid_transition[-1] & ~boundary_at_next[-1]

        valid_transition &= _valid_transitions(objective_data, N, device=done.device)
        if N > 2 and "sequence_id" in objective_data.keys():
            sequence_id = objective_data["sequence_id"]
            substitute_differs = sequence_id[2:] != sequence_id[1:-1]
            valid_transition[:-1] &= ~(boundary_at_next[:-1] & substitute_differs)

        if not valid_transition.any():
            raise ValueError("VecDqnObjective: batch contains no valid transitions.")

        action_idx_exp = next_actions.unsqueeze(-1).unsqueeze(-1).expand(N - 1, 1, D)
        curr_action_vecs = curr_vecs.gather(dim=1, index=action_idx_exp).squeeze(1)  # [N-1, D]

        greedy_idx = vector_action_scores(next_vecs).argmax(dim=-1)  # [N-1]
        greedy_idx_exp = greedy_idx.unsqueeze(-1).unsqueeze(-1).expand(N - 1, 1, D)
        next_action_vecs = next_vecs.gather(dim=1, index=greedy_idx_exp).squeeze(1)  # [N-1, D]

        if self.normalize_reward_mean or self.normalize_reward_std:
            sid = (
                objective_data["sequence_id"][:-1]
                if "sequence_id" in objective_data.keys()
                else torch.zeros(N - 1, dtype=torch.long, device=next_rewards.device)
            )
            next_rewards = _normalize_per_sequence(do_mean=self.normalize_reward_mean, do_std=self.normalize_reward_std, eps=self.normalize_reward_eps, std_target=self.normalize_reward_std_target, x=next_rewards, sequence_id=sid)

        theta = next_rewards * self.reward_scale + self.reward_shift          # [N-1]
        rotated = rope_rotate(x=next_action_vecs, theta=theta)                # [N-1, D]

        cosine_sim = F.cosine_similarity(curr_action_vecs, rotated.detach(), dim=-1)  # [N-1]
        loss = (1.0 - cosine_sim)[valid_transition].mean()

        curr_scores = vector_action_scores(curr_vecs).abs() / math.pi  # [N-1, A]
        curr_max_score = curr_scores.amax(dim=-1)  # [N-1]
        curr_max_det = curr_max_score.detach()[valid_transition]
        named: dict[str, torch.Tensor] = {
            "action_vector": loss.detach(),
            "action_vector_score_abs_min": curr_max_det.min(),
            "action_vector_score_abs_max": curr_max_det.max(),
            "action_vector_score_abs_mean": curr_max_det.mean(),
        }
        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics


def _normalize_per_sequence(
    *,
    x: torch.Tensor,
    sequence_id: torch.Tensor,
    do_mean: bool,
    do_std: bool,
    eps: float,
    std_target: float,
) -> torch.Tensor:
    """Normalize ``x [N]`` within each ``sequence_id`` group."""
    out = x.clone()
    for sid in sequence_id.unique():
        mask = sequence_id == sid
        chunk = out[mask]
        if do_mean:
            chunk = chunk - chunk.mean()
        if do_std:
            chunk = (chunk / (chunk.std() + eps)) * std_target
        out[mask] = chunk
    return out
