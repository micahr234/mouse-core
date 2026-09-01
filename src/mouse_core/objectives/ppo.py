"""Clipped PPO objective with GAE advantages."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _boundary_discounts,
    _pair_weight,
    _require_done_codes,
    _weighted_mean,
)


def sample_discrete_action(
    *,
    logits: torch.Tensor,
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
        rewards: ``[N-1]`` rewards for transitions out of states ``0..N-2``.
        values: ``[N]`` value predictions ``V(s_i)``.
        discounts: ``[N-1]`` per-transition discount (from episode/task done codes).
        valid: ``[N-1]`` mask — False at run boundaries (different
            ``sequence_id`` or grouping).
        gae_lambda: GAE λ.

    Returns:
        ``(advantages, returns)`` each ``[N-1]``, detached from autograd:
        both are regression / weighting targets, so the policy surrogate must
        not differentiate through the value head via the advantage. Invalid
        positions are zero.
    """
    with torch.no_grad():
        values = values.detach()
        T = rewards.shape[0]
        device = rewards.device
        dtype = rewards.dtype
        advantages = torch.zeros(T, device=device, dtype=dtype)
        gae = torch.zeros((), device=device, dtype=dtype)
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + discounts[t] * values[t + 1] - values[t]
            gae = delta + discounts[t] * gae_lambda * gae
            gae = torch.where(valid[t], gae, torch.zeros_like(gae))
            advantages[t] = gae
        returns = advantages + values[:-1]
    return advantages, returns


class PpoObjective(Objective):
    """Clipped PPO policy+value objective with GAE.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)`` to compute the loss.

    Requires dual heads on the model:

    * ``predictions["action"]`` — ``[N, A]`` discrete policy logits
    * ``predictions["value"]`` — ``[N, 1]`` or ``[N]`` scalar state values

    A run is the same ``sequence_id`` and, when ``grouping_field=`` is set,
    the same grouping column. Neighbor reads must stay in-run: out-of-run
    pairs are multiplied by ``0`` (all-zero weights → loss ``0``). Timing
    matches :class:`~mouse_core.objectives.dqn.DqnObjective`: token ``i``
    encodes state ``s_i``, and the action / reward / episode-done /
    task-done / behavior log-prob stored at ``i+1`` describe the transition
    out of ``s_i``.

    Discounts match the DQN table: the bootstrap is multiplied by the episode
    gamma, then by the task gamma (``1.0`` when ``task_done==0``).

    ``task_done`` and ``old_log_prob`` are objective columns only — not
    tokenizer or embedder input. Stamp behavior log-probs on rollout rows
    (same step as ``action``) and include them in the tokenizer ``objective_fields``
    keep-list so they land in ``objective_data``::

        tokenizer = NumericTokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "episode_done"},
                {"input_field": "task_done"},
                {"input_field": "old_log_prob"},
            ],
        )
        inputs, objective_data = loader.next_batch()
        predictions, _ = model(inputs)
        loss, metrics = objective(objective_data.to(device), predictions)

    When ``old_log_prob`` is absent, the detached current log-probs are used
    (ratio = 1) — suitable for a single pass over a freshly collected batch.

    Args:
        gamma_step: Discount for running transitions (``episode_done == 0``).
        gamma_episode_terminal: Discount when an episode ends (``episode_done == 1``).
        gamma_episode_truncated: Discount when an episode is truncated
            (``episode_done == 2``).
        gamma_task_terminal: Extra discount when a task ends (``task_done == 1``);
            multiplies the episode discount. ``task_done == 0`` uses ``1.0``.
        gamma_task_truncated: Extra discount when a task is truncated
            (``task_done == 2``); multiplies the episode discount.
        gae_lambda: GAE λ (``1.0`` = Monte Carlo returns within the discount).
        clip_eps: PPO ratio clip ε.
        vf_coef: Weight on the value-function MSE term.
        ent_coef: Weight on the policy entropy bonus (subtracted from the loss).
        normalize_advantage: If True, standardize advantages over valid pairs.
        action_key: Key in ``objective_data`` for integer actions.
        reward_key: Key in ``objective_data`` for rewards.
        episode_done_key: Key in ``objective_data`` for episode-done codes.
        task_done_key: Key in ``objective_data`` for task-done codes.
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
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        old_log_prob_key: str = "old_log_prob",
        predictions_key: str = "action",
        value_key: str = "value",
        num_actions: int | None = None,
        grouping_field: str | None = None,
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
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.old_log_prob_key = old_log_prob_key
        self.predictions_key = predictions_key
        self.value_key = value_key
        self.num_actions = num_actions
        self.grouping_field = grouping_field

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions[self.predictions_key]
        values_raw: torch.Tensor = predictions[self.value_key]

        if logits.ndim != 2:
            raise ValueError(
                f"PPO expects {self.predictions_key!r} logits shape [N, A], "
                f"got {tuple(logits.shape)}."
            )
        N, A = logits.shape
        device = logits.device
        dtype = logits.dtype

        if self.num_actions is not None:
            if self.num_actions <= 0 or self.num_actions > A:
                raise ValueError(
                    f"num_actions must be in 1..{A}, got {self.num_actions}."
                )
            logits = logits[..., : self.num_actions]
            A = self.num_actions

        if values_raw.shape[0] != N:
            raise ValueError(
                f"PPO expects {self.value_key!r} leading size [{N}], "
                f"got {tuple(values_raw.shape)}."
            )
        if values_raw.ndim == 2 and values_raw.shape[-1] == 1:
            values = values_raw.squeeze(-1)
        elif values_raw.ndim == 1:
            values = values_raw
        else:
            raise ValueError(
                f"PPO expects {self.value_key!r} shape [{N}] or [{N}, 1], "
                f"got {tuple(values_raw.shape)}."
            )
        values = values.to(dtype=dtype)

        if N < 2:
            raise ValueError("Not enough valid steps in data for PPO (need N >= 2).")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.shape != torch.Size([N]):
            raise ValueError(
                f"PPO objective expects action shape [{N}], got {tuple(action.shape)}."
            )

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([N]):
            raise ValueError(
                f"PPO objective expects reward shape [{N}], got {tuple(reward.shape)}."
            )

        episode_done, task_done = _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=N,
        )

        pair_weight = _pair_weight(
            objective_data,
            N,
            device,
            grouping_field=self.grouping_field,
            dtype=dtype,
        )
        valid = pair_weight > 0

        next_actions = action[1:]
        next_rewards = reward[1:].to(dtype=dtype)
        curr_logits = logits[:-1, :]
        curr_values = values[:-1]

        discounts = _boundary_discounts(
            episode_done=episode_done[1:],
            task_done=task_done[1:],
            gamma_step=self.gamma_step,
            gamma_episode_terminal=self.gamma_episode_terminal,
            gamma_episode_truncated=self.gamma_episode_truncated,
            gamma_task_terminal=self.gamma_task_terminal,
            gamma_task_truncated=self.gamma_task_truncated,
            dtype=dtype,
            device=device,
        )

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
            if old_log_prob_full.shape != torch.Size([N]):
                raise ValueError(
                    f"PPO expects {self.old_log_prob_key!r} shape [{N}], "
                    f"got {tuple(old_log_prob_full.shape)}."
                )
            old_log_prob = old_log_prob_full[1:].to(dtype=dtype)
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
        policy_loss = _weighted_mean(-torch.min(surr1, surr2), pair_weight)

        value_loss = _weighted_mean((curr_values - returns.detach()) ** 2, pair_weight)

        entropy = _weighted_mean(
            -(log_probs_all.exp() * log_probs_all).sum(dim=-1), pair_weight
        )

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        with torch.no_grad():
            clipfrac = _weighted_mean(
                ((ratio - 1.0).abs() > self.clip_eps).to(dtype=dtype), pair_weight
            )
            approx_kl = _weighted_mean(old_log_prob - new_log_prob, pair_weight)
            ret_v = returns[valid]
            val_v = curr_values[valid]
            if ret_v.numel() > 0:
                ret_var = ret_v.var(correction=0)
                if ret_var > 0:
                    explained_var = 1.0 - ((ret_v - val_v) ** 2).mean() / ret_var
                else:
                    explained_var = torch.zeros((), device=device, dtype=dtype)
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
            "advantage_mean": _weighted_mean(advantages, pair_weight),
            "value_mean": _weighted_mean(curr_values, pair_weight),
        }
        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
