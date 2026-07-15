"""Clipped GRPO policy objective with group-relative advantages (no critic)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import _valid_transitions


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Z-score rewards within one GRPO group.

    Args:
        rewards: ``[G]`` scalar scores for the ``G`` trajectories in a group
            (e.g. undiscounted return of each branch completion).
        eps: Numerical floor on the group standard deviation.

    Returns:
        ``[G]`` advantages ``(r - mean) / (std + eps)``. When ``G == 1`` or all
        rewards are identical, returns zeros (no learning signal).
    """
    if rewards.ndim != 1:
        raise ValueError(
            f"group_relative_advantages expects a 1-D group tensor, got shape "
            f"{tuple(rewards.shape)}."
        )
    G = int(rewards.shape[0])
    if G == 0:
        raise ValueError("group_relative_advantages: empty group.")
    if G == 1:
        return torch.zeros_like(rewards)
    mean = rewards.mean()
    # Population std so a two-sample group is well-defined.
    std = rewards.std(correction=0)
    if float(std.item()) < eps:
        return torch.zeros_like(rewards)
    return (rewards - mean) / (std + eps)


class GrpoObjective(Objective):
    """Clipped GRPO policy objective (no value / critic head).

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)``.

    Requires a policy head only:

    * ``predictions["action"]`` — ``[B, S, A]`` discrete policy logits

    Advantages are **not** estimated from a learned baseline. The caller
    computes them with :func:`group_relative_advantages` over a group of
    ``G`` branch completions that shared the same env snapshot and context
    prefix, then stamps the scalar onto every completion step (and usually
    ``0`` on the shared prefix)::

        adv = group_relative_advantages(torch.tensor(returns))  # [G]
        for g, rows in enumerate(branches):
            for row in rows:                     # or suffix-only
                row["advantage"] = float(adv[g])

        predictions, objective_data, _ = model(batch, segment_ids=segment_ids)
        objective_data["old_log_prob"] = batch_field(batch, "old_log_prob", ...)
        objective_data["advantage"] = batch_field(batch, "advantage", ...)
        loss, metrics = objective(objective_data, predictions)

    Timing matches :class:`~mouse_core.objectives.dqn.DqnObjective`: token
    ``t`` is state ``s_t``; action / behavior log-prob / advantage at ``t+1``
    describe the transition out of ``s_t``. Pack-mode ``segment_id`` seams
    are excluded.

    Args:
        clip_eps: PPO-style ratio clip ε.
        ent_coef: Weight on the policy entropy bonus (subtracted from the loss).
        action_key: Key in ``objective_data`` for integer actions.
        old_log_prob_key: Key in ``objective_data`` for behavior log-probs.
        advantage_key: Key in ``objective_data`` for group-relative advantages.
        predictions_key: Key in ``predictions`` for policy logits.
        num_actions: If set, only the first ``num_actions`` logits participate.
    """

    def __init__(
        self,
        *,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        action_key: str = "action",
        old_log_prob_key: str = "old_log_prob",
        advantage_key: str = "advantage",
        predictions_key: str = "action",
        num_actions: int | None = None,
    ) -> None:
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.action_key = action_key
        self.old_log_prob_key = old_log_prob_key
        self.advantage_key = advantage_key
        self.predictions_key = predictions_key
        self.num_actions = num_actions

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions[self.predictions_key]

        if logits.ndim != 3:
            raise ValueError(
                f"GRPO expects {self.predictions_key!r} logits shape [B, S, A], "
                f"got {tuple(logits.shape)}."
            )
        B, S, A = logits.shape
        device = logits.device
        dtype = logits.dtype

        if self.num_actions is not None:
            if self.num_actions <= 0 or self.num_actions > A:
                raise ValueError(
                    f"num_actions must be in 1..{A}, got {self.num_actions}."
                )
            logits = logits[..., : self.num_actions]
            A = self.num_actions

        if S < 2:
            raise ValueError("Not enough valid steps in data for GRPO (need S >= 2).")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.shape != torch.Size([B, S]):
            raise ValueError(
                f"GRPO objective expects action shape [{B}, {S}], "
                f"got {tuple(action.shape)}."
            )

        if self.advantage_key not in objective_data.keys():
            raise KeyError(
                f"GRPO requires objective_data[{self.advantage_key!r}] "
                "(stamp group-relative advantages onto rows, then inject with "
                "batch_field)."
            )
        advantage_full = objective_data[self.advantage_key]
        if advantage_full.shape != torch.Size([B, S]):
            raise ValueError(
                f"GRPO expects {self.advantage_key!r} shape [{B}, {S}], "
                f"got {tuple(advantage_full.shape)}."
            )
        advantage = advantage_full[:, 1:].to(dtype=dtype)

        valid = _valid_transitions(objective_data, B, S, device)

        next_actions = action[:, 1:]
        curr_logits = logits[:, :-1, :]

        log_probs_all = F.log_softmax(curr_logits, dim=-1)
        new_log_prob = log_probs_all.gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)

        if self.old_log_prob_key in objective_data.keys():
            old_log_prob_full = objective_data[self.old_log_prob_key]
            if old_log_prob_full.shape != torch.Size([B, S]):
                raise ValueError(
                    f"GRPO expects {self.old_log_prob_key!r} shape [{B}, {S}], "
                    f"got {tuple(old_log_prob_full.shape)}."
                )
            old_log_prob = old_log_prob_full[:, 1:].to(dtype=dtype)
        else:
            old_log_prob = new_log_prob.detach()

        ratio = (new_log_prob - old_log_prob).exp()
        surr1 = ratio * advantage
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2)[valid].mean()

        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)[valid].mean()
        loss = policy_loss - self.ent_coef * entropy

        with torch.no_grad():
            clipfrac = (
                ((ratio[valid] - 1.0).abs() > self.clip_eps).to(dtype=dtype).mean()
            )
            approx_kl = (old_log_prob[valid] - new_log_prob[valid]).mean()

        named: dict[str, torch.Tensor] = {
            "grpo": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "entropy": entropy.detach(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "advantage_mean": advantage[valid].mean(),
        }
        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
