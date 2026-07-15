"""Clipped PPO objective with GAE advantages."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import _valid_transitions


def batch_field(
    batch: list[list[dict[str, Any]]],
    key: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract a scalar field from nested ``[B][S]`` row dicts into a ``[B, S]`` tensor.

    Used to inject rollout-only columns (e.g. ``old_log_prob``) into
    ``objective_data`` without declaring them as encoder modalities.
    """
    if not batch:
        raise ValueError("batch_field: batch is empty.")
    B = len(batch)
    S = len(batch[0])
    if S == 0:
        raise ValueError("batch_field: sequences are empty.")
    out = torch.empty(B, S, dtype=dtype, device=device)
    for b, rows in enumerate(batch):
        if len(rows) != S:
            raise ValueError(
                f"batch_field: ragged batch — row {b} has length {len(rows)}, expected {S}."
            )
        for s, row in enumerate(rows):
            if key not in row:
                raise KeyError(
                    f"batch_field: row [{b}][{s}] is missing key {key!r}."
                )
            out[b, s] = row[key]
    return out


def sample_discrete_action(
    logits: torch.Tensor,
    *,
    num_actions: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample actions from a categorical policy and return ``(actions, log_probs)``.

    Args:
        logits: ``[..., A]`` unnormalized action scores (typically the last-step
            slice of ``predictions["action"]``).
        num_actions: If set, only the first ``num_actions`` logits are used.
    """
    if num_actions is not None:
        logits = logits[..., :num_actions]
    log_probs_all = F.log_softmax(logits, dim=-1)
    probs = log_probs_all.exp()
    actions = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1).reshape(
        probs.shape[:-1]
    )
    log_probs = log_probs_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    return actions, log_probs


def _gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    valid: torch.Tensor,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over valid consecutive pairs.

    Args:
        rewards: ``[B, S-1]`` rewards for transitions out of states ``0..S-2``.
        values: ``[B, S]`` value predictions ``V(s_t)``.
        discounts: ``[B, S-1]`` per-transition discount (from done-code gammas).
        valid: ``[B, S-1]`` mask — False at pack-segment seams.
        gae_lambda: GAE λ.

    Returns:
        ``(advantages, returns)`` each ``[B, S-1]``. Invalid positions are zero.
    """
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype
    advantages = torch.zeros(B, T, device=device, dtype=dtype)
    gae = torch.zeros(B, device=device, dtype=dtype)
    for t in range(T - 1, -1, -1):
        delta = rewards[:, t] + discounts[:, t] * values[:, t + 1] - values[:, t]
        gae = delta + discounts[:, t] * gae_lambda * gae
        gae = torch.where(valid[:, t], gae, torch.zeros_like(gae))
        advantages[:, t] = gae
    returns = advantages + values[:, :-1]
    return advantages, returns


class PpoObjective(Objective):
    """Clipped PPO policy+value objective with GAE.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss.

    Requires dual heads on the model:

    * ``predictions["action"]`` — ``[B, S, A]`` discrete policy logits
    * ``predictions["value"]`` — ``[B, S, 1]`` or ``[B, S]`` scalar state values

    Every consecutive pair ``(t, t+1)`` is a decision point, using the same
    timing convention as :class:`~mouse_core.objectives.dqn.DqnObjective`:
    token ``t`` encodes state ``s_t``, and the action / reward / done / behavior
    log-prob stored at ``t+1`` describe the transition out of ``s_t``. Pack-mode
    ``segment_id`` seams are excluded.

    Done-code discounts match the DQN table (``gamma_step`` for ``done==0``, …,
    ``gamma_task_truncated`` for ``done==4``).

    For multi-epoch PPO, store behavior log-probs during rollout (same step as
    ``action``) and inject them before the objective call::

        predictions, objective_data, _ = model(batch, segment_ids=segment_ids)
        objective_data["old_log_prob"] = batch_field(
            batch, "old_log_prob", device=objective_data.device
        )
        loss, metrics = objective(objective_data, predictions)

    When ``old_log_prob`` is absent, the detached current log-probs are used
    (ratio = 1) — suitable for a single pass over a freshly collected batch.

    Args:
        gamma_step: Discount for running transitions (``done == 0``).
        gamma_episode_terminal: Discount when an episode ends inside a task
            (``done == 1``).
        gamma_episode_truncated: Discount when an episode is truncated inside a
            task (``done == 2``).
        gamma_task_terminal: Discount when a task ends (``done == 3``).
        gamma_task_truncated: Discount when a task is truncated (``done == 4``).
        gae_lambda: GAE λ (``1.0`` = Monte Carlo returns within the discount).
        clip_eps: PPO ratio clip ε.
        vf_coef: Weight on the value-function MSE term.
        ent_coef: Weight on the policy entropy bonus (subtracted from the loss).
        normalize_advantage: If True, standardize advantages over valid pairs.
        action_key: Key in ``objective_data`` for integer actions.
        reward_key: Key in ``objective_data`` for rewards.
        done_key: Key in ``objective_data`` for done codes.
        old_log_prob_key: Key in ``objective_data`` for behavior log-probs.
        predictions_key: Key in ``predictions`` for policy logits.
        value_key: Key in ``predictions`` for scalar values.
        num_actions: If set, only the first ``num_actions`` logits participate.
    """

    def __init__(
        self,
        *,
        gamma_step: float = 0.99,
        gamma_episode_terminal: float = 0.0,
        gamma_episode_truncated: float = 0.0,
        gamma_task_terminal: float = 0.0,
        gamma_task_truncated: float = 0.0,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantage: bool = True,
        action_key: str = "action",
        reward_key: str = "reward",
        done_key: str = "done",
        old_log_prob_key: str = "old_log_prob",
        predictions_key: str = "action",
        value_key: str = "value",
        num_actions: int | None = None,
    ) -> None:
        self.gamma_step = gamma_step
        self.gamma_episode_terminal = gamma_episode_terminal
        self.gamma_episode_truncated = gamma_episode_truncated
        self.gamma_task_terminal = gamma_task_terminal
        self.gamma_task_truncated = gamma_task_truncated
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantage = normalize_advantage
        self.action_key = action_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.old_log_prob_key = old_log_prob_key
        self.predictions_key = predictions_key
        self.value_key = value_key
        self.num_actions = num_actions

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions[self.predictions_key]
        values_raw: torch.Tensor = predictions[self.value_key]

        if logits.ndim != 3:
            raise ValueError(
                f"PPO expects {self.predictions_key!r} logits shape [B, S, A], "
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

        if values_raw.shape[:2] != torch.Size([B, S]):
            raise ValueError(
                f"PPO expects {self.value_key!r} leading shape [{B}, {S}], "
                f"got {tuple(values_raw.shape)}."
            )
        if values_raw.ndim == 3 and values_raw.shape[-1] == 1:
            values = values_raw.squeeze(-1)
        elif values_raw.ndim == 2:
            values = values_raw
        else:
            raise ValueError(
                f"PPO expects {self.value_key!r} shape [{B}, {S}] or [{B}, {S}, 1], "
                f"got {tuple(values_raw.shape)}."
            )
        values = values.to(dtype=dtype)

        if S < 2:
            raise ValueError("Not enough valid steps in data for PPO (need S >= 2).")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.shape != torch.Size([B, S]):
            raise ValueError(
                f"PPO objective expects action shape [{B}, {S}], got {tuple(action.shape)}."
            )

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([B, S]):
            raise ValueError(
                f"PPO objective expects reward shape [{B}, {S}], got {tuple(reward.shape)}."
            )

        done = objective_data[self.done_key]
        if done.dtype != torch.int64:
            raise TypeError(f"done must be int64, got {done.dtype}.")
        if done.shape != torch.Size([B, S]):
            raise ValueError(
                f"PPO objective expects done shape [{B}, {S}], got {tuple(done.shape)}."
            )

        valid = _valid_transitions(objective_data, B, S, device)

        # Transition out of state t is described by fields stored at t+1.
        next_actions = action[:, 1:]
        next_rewards = reward[:, 1:].to(dtype=dtype)
        next_done = done[:, 1:]
        curr_logits = logits[:, :-1, :]
        curr_values = values[:, :-1]

        gammas = torch.tensor(
            [
                self.gamma_step,
                self.gamma_episode_terminal,
                self.gamma_episode_truncated,
                self.gamma_task_terminal,
                self.gamma_task_truncated,
            ],
            dtype=dtype,
            device=device,
        )
        discounts = gammas[next_done]

        advantages, returns = _gae_advantages(
            rewards=next_rewards,
            values=values,
            discounts=discounts,
            valid=valid,
            gae_lambda=self.gae_lambda,
        )

        log_probs_all = F.log_softmax(curr_logits, dim=-1)
        new_log_prob = log_probs_all.gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)

        if self.old_log_prob_key in objective_data.keys():
            old_log_prob_full = objective_data[self.old_log_prob_key]
            if old_log_prob_full.shape != torch.Size([B, S]):
                raise ValueError(
                    f"PPO expects {self.old_log_prob_key!r} shape [{B}, {S}], "
                    f"got {tuple(old_log_prob_full.shape)}."
                )
            old_log_prob = old_log_prob_full[:, 1:].to(dtype=dtype)
        else:
            old_log_prob = new_log_prob.detach()

        adv = advantages
        if self.normalize_advantage:
            adv_valid = adv[valid]
            if adv_valid.numel() > 1:
                adv = (adv - adv_valid.mean()) / (adv_valid.std(correction=0) + 1e-8)
            elif adv_valid.numel() == 1:
                adv = adv - adv_valid.mean()

        ratio = (new_log_prob - old_log_prob).exp()
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2)[valid].mean()

        value_loss = ((curr_values - returns.detach()) ** 2)[valid].mean()

        entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)[valid].mean()

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        with torch.no_grad():
            clipfrac = (
                ((ratio[valid] - 1.0).abs() > self.clip_eps).to(dtype=dtype).mean()
            )
            approx_kl = (old_log_prob[valid] - new_log_prob[valid]).mean()
            # Explained variance of returns by value predictions.
            ret_v = returns[valid]
            val_v = curr_values[valid]
            ret_var = ret_v.var(correction=0)
            if ret_var > 0:
                explained_var = 1.0 - ((ret_v - val_v) ** 2).mean() / ret_var
            else:
                explained_var = torch.zeros((), device=device, dtype=dtype)

        named: dict[str, torch.Tensor] = {
            "ppo": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy": entropy.detach(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "explained_variance": explained_var,
            "advantage_mean": advantages[valid].mean(),
            "value_mean": curr_values[valid].mean(),
        }
        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
