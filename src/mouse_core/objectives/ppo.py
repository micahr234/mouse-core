"""Clipped PPO objective with GAE advantages."""

from __future__ import annotations

from typing import cast, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction
from mouse_core.objectives.dqn import (
    _pair_weight,
    _require_action_ids,
    _require_done_codes,
    _require_step_aligned_predictions,
    _weighted_mean,
    _zero_off_data_factor_scan,
)
from mouse_core.objectives.transforms import (
    Discount,
    Reward,
    Value,
    _apply_transform,
    _apply_value,
    _require_transform,
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
    bootstrap_cutoff: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over valid consecutive pairs.

    Args:
        rewards: ``[N-1]`` rewards for transitions out of states ``0..N-2``.
        values: ``[N]`` value predictions ``V(s_i)``.
        discounts: ``[N-1]`` per-transition discount (from episode/task done codes).
        valid: ``[N-1]`` mask — False at run boundaries (different
            ``sequence_id`` or grouping).
        gae_lambda: GAE λ.
        bootstrap_cutoff: ``True`` adds ``V`` at a state whose next step is not an
            in-run pair (end of the batch, or a run break) and keeps the
            step. ``False`` drops the step when the factor on that off-data
            ``V`` is non-zero. A zero factor leaves the value out of the
            advantage, so the step stays. Reaching past the sample is not
            enough. ``V`` at a state that still has a later in-run step is
            unchanged. A ``0`` discount (true terminal) removes the value
            either way.

    Returns:
        ``(advantages, returns, participate)`` each ``[N-1]``. Advantages
        and returns are detached from autograd: both are regression /
        weighting targets, so the policy surrogate must not differentiate
        through the value head via the advantage. Invalid positions are
        zero. ``participate`` is ``1`` for a step that stays in the loss
        and in logged metrics.
    """
    with torch.no_grad():
        values = values.detach()
        T = rewards.shape[0]
        device = rewards.device
        dtype = rewards.dtype
        next_values = values[1:]
        # Continuation after s_{t+1} is sampled only when pair t+1 stays in-run.
        sampled = torch.zeros(T, dtype=torch.bool, device=device)
        if T > 1:
            sampled[:-1] = valid[1:]
        if bootstrap_cutoff:
            participate = torch.ones(T, dtype=dtype, device=device)
        else:
            next_values = torch.where(
                sampled, next_values, torch.zeros_like(next_values)
            )
            # λ = 0 does not carry a later cutoff value. A non-zero λ does,
            # and that factor drops the step when the value is off-data.
            carries = sampled if gae_lambda != 0.0 else torch.zeros(
                T, dtype=torch.bool, device=device
            )
            participate = _zero_off_data_factor_scan(
                discount=discounts,
                in_sample=sampled,
                carries=carries,
            )
        advantages = torch.zeros(T, device=device, dtype=dtype)
        gae = torch.zeros((), device=device, dtype=dtype)
        for t in range(T - 1, -1, -1):
            delta = rewards[t] + discounts[t] * next_values[t] - values[t]
            gae = delta + discounts[t] * gae_lambda * gae
            gae = torch.where(valid[t], gae, torch.zeros_like(gae))
            advantages[t] = gae
        returns = advantages + values[:-1]
    return advantages, returns, participate


class PpoObjective(Objective):
    """Clipped PPO policy+value objective with GAE.

    Instantiate with hyperparameters, then call with
    ``objective_data=`` and ``predictions=`` to compute the loss.

    Call with the policy logits as ``predictions=`` and the value
    tensor as ``value_predictions=``:

    * ``predictions`` — ``[N, A]`` discrete policy logits
    * ``value_predictions`` — ``[N, 1]`` or ``[N]`` scalar state values

    A run is the same ``sequence_id`` and, when ``grouping_field=`` is set,
    the same grouping column. Neighbor reads must stay in-run: out-of-run
    pairs are multiplied by ``0`` (all-zero weights → loss ``0``). Timing
    matches :class:`~mouse_core.objectives.dqn.DqnObjective`: token ``i``
    encodes state ``s_i``, and the action / reward / episode-done /
    task-done / behavior log-prob stored at ``i+1`` describe the transition
    out of ``s_i``.

    ``reward(**objective_data)`` supplies the per-step reward; the value
    stored at ``i+1`` is ``r_t``. ``affine_reward`` is the column
    affine; ``boundary_reward`` applies episode / task scale and shift extras.
    ``value(value=..., **objective_data)`` supplies the per-step affine
    on the value-head output. ``affine_value`` is the prediction affine;
    ``boundary_value`` applies episode / task scale and shift extras.
    Discounts match ``DqnObjective``:
    ``discount(**objective_data)``.

    ``task_done`` and ``old_log_prob`` are objective columns only — not
    tokenizer input. Stamp behavior log-probs on rollout rows
    (same step as ``action``) and include them in the tokenizer ``objective_fields``
    keep-list so they land in ``objective_data``::

        tokenizer = Tokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "episode_done"},
                {"input_field": "task_done"},
                {"input_field": "old_log_prob"},
            ],
        )
        from mouse_core.data import to_device
        from mouse_core.models import prediction_key
        inputs, objective_data = loader.next_batch()
        out = model(inputs)
        loss, metrics = objective(
            objective_data=to_device(data=objective_data, device=device),
            predictions=out.predictions[prediction_key(head=policy_head)],
            value_predictions=out.predictions[prediction_key(head=value_head)],
        )

    When ``old_log_prob`` is absent, the detached current log-probs are used
    (ratio = 1) — suitable for a single pass over a freshly collected batch.

    Args:
        discount: Per-step γ from unpacked ``objective_data`` columns.
            ``boundary_discount`` is the standard ``gamma_step`` × extra
            lookup (``None`` skips the call and uses ``1``);
            any ``discount(**objective_data) -> [N]`` is accepted.
        reward: Per-step reward from unpacked ``objective_data`` columns.
            ``affine_reward`` is the column affine; ``boundary_reward``
            applies episode / task scale and shift extras
            (``None`` skips the call);
            any ``reward(**objective_data) -> [N]`` is accepted.
        value: Per-step affine on the value-head output from unpacked
            ``objective_data`` columns plus ``value=``. ``affine_value``
            is the prediction affine; ``boundary_value`` applies episode
            / task scale and shift extras
            (``None`` skips the call);
            any ``value(value=..., **objective_data)`` returning the same
            shape is accepted.
        gae_lambda: GAE λ (``1.0`` = Monte Carlo returns within the discount).
        bootstrap_cutoff: Required. ``True`` adds ``V`` where the continuation
            leaves the sampled run (end of the batch, or a
            ``sequence_id`` / ``grouping_field`` break): a chunk
            boundary, time limit, or truncation whose rest was not
            sampled. That step stays in the loss and in logged metrics.
            ``False`` drops the step from both when the factor on that
            off-data value is non-zero. A zero factor leaves the value
            out of the target, so the step stays. Reaching past the
            sample is not enough. ``V`` at a state that still has a
            later in-run step is unchanged. A true terminal is unchanged
            either way: its γ is ``0``, so the factor is already ``0``.
        clip_eps: PPO ratio clip ε.
        vf_coef: Weight on the value-function MSE term.
        ent_coef: Weight on the policy entropy bonus (subtracted from the loss).
        normalize_advantage: If True, standardize advantages over valid pairs.
        action_key: Key in ``objective_data`` for integer actions.
        episode_done_key: Key in ``objective_data`` for episode-done codes.
        task_done_key: Key in ``objective_data`` for task-done codes.
        old_log_prob_key: Key in ``objective_data`` for behavior log-probs.
        num_actions: If set, only the first ``num_actions`` logits participate.
        grouping_field: Step column that isolates runs (typically
            ``task_index``). Required. Pass ``None`` only when the batch
            has no grouping isolation — omitting it is an error, not a
            silent skip.
    """

    def __init__(
        self,
        *,
        discount: Discount | None,
        reward: Reward | None,
        value: Value | None,
        gae_lambda: float = 0.95,
        bootstrap_cutoff: bool,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantage: bool = True,
        action_key: str = "action",
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        old_log_prob_key: str = "old_log_prob",
        num_actions: int | None = None,
        grouping_field: str | None,
    ) -> None:
        self.discount = _require_transform(discount, name="discount")
        self.reward = _require_transform(reward, name="reward")
        self.value = _require_transform(value, name="value")
        self.gae_lambda = gae_lambda
        self.bootstrap_cutoff = bool(bootstrap_cutoff)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantage = normalize_advantage
        self.action_key = action_key
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.old_log_prob_key = old_log_prob_key
        self.num_actions = num_actions
        self.grouping_field = grouping_field

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        value_predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        value_predictions: torch.Tensor,
        delayed_predictions: None = None,
        behavior_predictions: None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
        behavior_predictions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        _reject_predictions(
            "PpoObjective",
            delayed_predictions=delayed_predictions,
            behavior_predictions=behavior_predictions,
        )
        logits: torch.Tensor = predictions
        values_raw: torch.Tensor = _require_prediction(
            value_predictions, owner="PpoObjective", name="value_predictions"
        )

        if logits.ndim != 2:
            raise ValueError(
                f"PPO expects policy logits shape [N, A], "
                f"got {tuple(logits.shape)}."
            )
        N, A = logits.shape
        device = cast(torch.device, logits.device)
        dtype = cast(torch.dtype, logits.dtype)

        if self.num_actions is not None:
            if self.num_actions <= 0 or self.num_actions > A:
                raise ValueError(
                    f"num_actions must be in 1..{A}, got {self.num_actions}."
                )
            logits = logits[..., : self.num_actions]
            A = self.num_actions

        if values_raw.shape[0] != N:
            raise ValueError(
                f"PPO expects value leading size [{N}], "
                f"got {tuple(values_raw.shape)}."
            )
        if values_raw.ndim == 2 and values_raw.shape[-1] == 1:
            values = values_raw.squeeze(-1)
        elif values_raw.ndim == 1:
            values = values_raw
        else:
            raise ValueError(
                f"PPO expects value shape [{N}] or [{N}, 1], "
                f"got {tuple(values_raw.shape)}."
            )
        values = values.to(dtype=dtype)

        if N < 2:
            raise ValueError("Not enough valid steps in data for PPO (need N >= 2).")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"PPO objective expects action shape [N], got {tuple(action.shape)}."
            )
        _require_step_aligned_predictions(
            objective_data, n_pred=N, n_steps=int(action.shape[0]), who="PPO"
        )
        if action.shape != torch.Size([N]):
            raise ValueError(
                f"PPO objective expects action shape [{N}], got {tuple(action.shape)}."
            )
        _require_action_ids(action, A)

        reward = _apply_transform(
            transform=self.reward,
            name="reward",
            objective_data=objective_data,
            N=N,
            dtype=dtype,
            device=device,
            identity=objective_data["reward"],
        )

        _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=N,
        )
        values = _apply_value(
            transform=self.value,
            name="value",
            value=values,
            objective_data=objective_data,
            step_of=torch.arange(N, device=device, dtype=torch.int64),
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

        discount_all = _apply_transform(
            transform=self.discount,
            name="discount",
            objective_data=objective_data,
            N=N,
            dtype=dtype,
            device=device,
            identity=torch.ones(N, dtype=dtype, device=device),
        )
        discounts = discount_all[1:]

        advantages, returns, participate = _gae_advantages(
            rewards=next_rewards,
            values=values,
            discounts=discounts,
            valid=valid,
            gae_lambda=self.gae_lambda,
            bootstrap_cutoff=self.bootstrap_cutoff,
        )
        pair_weight = pair_weight * participate
        valid = pair_weight > 0

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
