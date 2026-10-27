"""Clipped GRPO policy objective with group-relative advantages (no critic)."""

from __future__ import annotations

import math
from typing import cast, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives._actions import canonical_actions, executed_actions

from mouse_core.objectives.base import Objective, _reject_predictions
from mouse_core.objectives.dqn import (
    _pair_weight,
    _require_action_ids,
    _require_step_aligned_predictions,
    _weighted_mean,
)


def group_relative_advantages(
    *,
    rewards: torch.Tensor,
    eps: float = 1e-4,
    scaling: str = "group",
    batch_rewards: torch.Tensor | None = None,
) -> torch.Tensor:
    """Center rewards within a group, with configurable reward scaling.

    Args:
        rewards: ``[G]`` scalar scores for the ``G`` trajectories in a group
            (undiscounted reward through episode completion for each branch).
        eps: Stabilizer added to the standard deviation.
        scaling: Group std, collection-wide batch std, none, or MaxRL group mean.
        batch_rewards: All branch returns in the collection for batch scaling.

    Returns:
        ``[G]`` centered rewards, optionally divided by ``std + eps``. When ``G == 1`` or all
        rewards are identical, returns zeros (no learning signal).
    """
    if rewards.ndim != 1:
        raise ValueError(
            f"group_relative_advantages expects a 1-D group tensor, got shape "
            f"{tuple(rewards.shape)}."
        )
    if scaling not in ("group", "batch", "none", "maxrl"):
        raise ValueError("scaling must be group, batch, none, or maxrl")
    if rewards.numel() == 0:
        raise ValueError("group_relative_advantages: empty group.")
    if scaling == "maxrl" and (not bool(torch.isfinite(rewards).all()) or bool((rewards < 0).any())):
        raise ValueError("maxrl scaling requires finite nonnegative rewards")
    centered = rewards - rewards.mean()
    if scaling == "maxrl":
        mean = rewards.mean()
        return centered / (mean + eps) if bool(mean > 0) else torch.zeros_like(rewards)
    if scaling == "none":
        return centered
    population = rewards
    if scaling == "batch":
        if batch_rewards is None or batch_rewards.ndim != 1 or batch_rewards.numel() == 0:
            raise ValueError("batch scaling requires nonempty 1-D batch_rewards")
        population = batch_rewards
    std = population.std(correction=0)
    return centered / (std + eps)


@torch.no_grad()
def sampling_mismatch_weights(*, training_log_probs, sampling_log_probs, group_id,
                              executed, mode, ratio_min=None, ratio_max=3.0):
    if mode not in ("none", "action_truncate", "action_mask", "episode_truncate", "episode_mask"):
        raise ValueError("unknown sampling mismatch correction mode")
    if ratio_min is not None and (not math.isfinite(ratio_min) or ratio_min < 0):
        raise ValueError("ratio_min must be finite and nonnegative")
    if ratio_max is not None and (not math.isfinite(ratio_max) or ratio_max <= 0):
        raise ValueError("ratio_max must be finite and positive")
    if ratio_min is not None and ratio_max is not None and ratio_min >= ratio_max:
        raise ValueError("ratio_min must be less than ratio_max")
    shape = training_log_probs.shape
    if len(shape) != 1 or any(x.shape != shape for x in (sampling_log_probs, group_id, executed)):
        raise ValueError("mismatch fields must be aligned one-dimensional tensors")
    result = torch.ones_like(training_log_probs, dtype=torch.float64)
    if mode == "none":
        return result
    if ratio_min is None and ratio_max is None:
        raise ValueError("mismatch correction requires at least one bound")
    active = executed.bool()
    difference = training_log_probs[active].double() - sampling_log_probs[active].double()
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("executed action log probabilities must be finite")
    if mode.startswith("episode_"):
        _, inverse = torch.unique_consecutive(group_id[active], return_inverse=True)
        sums = difference.new_zeros(int(inverse.max()) + 1 if inverse.numel() else 0)
        sums.scatter_add_(0, inverse, difference)
        difference = sums[inverse]
    lower = math.log(ratio_min) if ratio_min else -math.inf
    upper = math.log(ratio_max) if ratio_max is not None else math.inf
    if mode.endswith("truncate"):
        weights = difference.clamp(min=lower, max=upper).exp()
    else:
        keep = (difference >= lower) & (difference <= upper)
        weights = torch.where(keep, difference, 0.0).exp() * keep
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("mismatch weights overflow; set a finite upper bound")
    result[active] = weights
    return result


class GrpoObjective(Objective[...]):
    """Clipped GRPO policy objective (no value / critic head).

    Instantiate with hyperparameters, then call with
    ``objective_data=``, ``group_id=`` (int64 ``[N]``, one id per step,
    returned beside ``objective_data``), and ``predictions=``.

    Call with the policy logits as ``predictions=``:

    * ``predictions`` — ``[N, A]`` discrete policy logits

    Advantages are **not** estimated from a learned baseline. The caller
    computes them with :func:`group_relative_advantages` over a group of
    ``G`` completed-episode branches that shared the same initial env snapshot,
    seed, and random state. Each starts at episode step zero with only the initial
    tokenizer prompt and observation, then builds its own episode history. Collection
    finishes every branch before computing these advantages, then stamps the
    scalar onto every action step (and ``0`` on the initial reset row)::

        adv = group_relative_advantages(rewards=torch.tensor(returns))  # [G]
        for g, rows in enumerate(branches):
            for row in rows:                     # or suffix-only
                row["advantage"] = float(adv[g])

        tokenizer = Tokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "old_log_prob"},
                {"input_field": "advantage"},
            ],
        )
        from mouse_core.data import to_device
        inputs, objective_data, group_id = loader.next_batch()
        out = model(inputs)
        losses, metrics = objective(
            objective_data=to_device(data=objective_data, device=device),
            group_id=group_id.to(device),
            predictions=out["predictions"]["action"],
        )
        loss = torch.stack(tuple(losses.values())).sum()

    ``old_log_prob`` and ``advantage`` are objective columns only — not
    tokenizer input fields.

    Timing matches :class:`~mouse_core.objectives.dqn.DqnObjective`: token
    ``i`` is state ``s_i``; action / behavior log-prob / advantage at ``i+1``
    describe the transition out of ``s_i``. A run is the same ``group_id``
    (one dataloader sample). Neighbor
    reads must stay in-run: out-of-run pairs are multiplied by ``0``
    (all-zero weights → loss ``0``).

    Packing nullable ``action`` fields derives the ``[N]`` batch mask
    ``objective_data["action_executed"]``; raw rows do not need that field.
    A false entry marks an action-independent reset transition. Its numeric
    action is only a placeholder; the transition contributes no policy loss,
    entropy, or KL statistics. Direct tensor callers may omit the mask when
    every action is executed.

    Args:
        clip_ratio_min: Lower probability-ratio clip bound, in (0, 1].
        clip_ratio_max: Upper probability-ratio clip bound, at least 1.
        ent_coef: Weight on the policy entropy bonus (subtracted from the loss).
        action_key: Key in ``objective_data`` for integer actions.
        old_log_prob_key: Key in ``objective_data`` for behavior log-probs.
        importance_sampling_level: Per-action ratios or a geometric-mean ratio per complete episode branch.
        sampling_weight_key: Optional detached sampling-correction weights; only the policy loss is weighted.
        advantage_key: Key in ``objective_data`` for group-relative advantages.
        num_actions: If set, only the first ``num_actions`` logits participate.
    """

    def __init__(
        self,
        *,
        clip_ratio_min: float = 0.8,
        clip_ratio_max: float = 1.2,
        ent_coef: float = 0.01,
        action_key: str = "action",
        old_log_prob_key: str = "old_log_prob",
        advantage_key: str = "advantage",
        sampling_weight_key: str | None = None,
        importance_sampling_level: str = "action",
        num_actions: int | None = None,
    ) -> None:
        if not (math.isfinite(clip_ratio_min) and math.isfinite(clip_ratio_max)
                and 0 < clip_ratio_min <= 1 <= clip_ratio_max):
            raise ValueError("clip ratios must satisfy 0 < clip_ratio_min <= 1 <= clip_ratio_max and be finite")
        if importance_sampling_level not in ("action", "episode"):
            raise ValueError("importance_sampling_level must be action or episode")
        self.importance_sampling_level = importance_sampling_level
        self.clip_ratio_min = clip_ratio_min
        self.clip_ratio_max = clip_ratio_max
        self.ent_coef = ent_coef
        self.action_key = action_key
        self.old_log_prob_key = old_log_prob_key
        self.advantage_key = advantage_key
        self.sampling_weight_key = sampling_weight_key
        self.num_actions = num_actions

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
        backup_value_predictions: None = None,
        backup_action_predictions: None = None,
        value_predictions: None = None,
        targets: None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
        backup_value_predictions: torch.Tensor | None = None,
        backup_action_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "GrpoObjective",
            backup_value_predictions=backup_value_predictions,
            backup_action_predictions=backup_action_predictions,
            value_predictions=value_predictions,
            targets=targets,
        )
        logits: torch.Tensor = predictions

        if logits.ndim != 2:
            raise ValueError(
                f"GRPO expects policy logits shape [N, A], "
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

        if N < 2:
            raise ValueError("Not enough valid steps in data for GRPO (need N >= 2).")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"GRPO objective expects action shape [N], "
                f"got {tuple(action.shape)}."
            )
        _require_step_aligned_predictions(
            objective_data, n_pred=N, n_steps=int(action.shape[0]), who="GRPO"
        )
        if action.shape != torch.Size([N]):
            raise ValueError(
                f"GRPO objective expects action shape [{N}], "
                f"got {tuple(action.shape)}."
            )
        executed = executed_actions(objective_data, action)
        action = canonical_actions(objective_data, action)
        _require_action_ids(action, A)
        objective_data = {**objective_data, self.action_key: action}

        if self.advantage_key not in objective_data.keys():
            raise KeyError(
                f"GRPO requires objective_data[{self.advantage_key!r}] "
                "(stamp group-relative advantages onto rows, then include them "
                "in tokenizer objective_fields)."
            )
        advantage_full = objective_data[self.advantage_key]
        if advantage_full.shape != torch.Size([N]):
            raise ValueError(
                f"GRPO expects {self.advantage_key!r} shape [{N}], "
                f"got {tuple(advantage_full.shape)}."
            )
        advantage = advantage_full[1:].to(dtype=dtype)

        pair_weight = _pair_weight(
            group_id=group_id,
            N=N,
            device=device,
            dtype=dtype,
        )
        pair_weight = pair_weight * executed[1:].to(dtype=dtype)
        active = pair_weight > 0
        # Slice before softmax/exp: even NaN logits or stale log-probabilities
        # on a reset row cannot contaminate losses or gradients through 0 * NaN.
        pair_weight = pair_weight[active]
        advantage = advantage[active]
        next_actions = action[1:][active]
        curr_logits = logits[:-1, :][active]

        log_probs_all = F.log_softmax(curr_logits, dim=-1)
        new_log_prob = log_probs_all.gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)

        if self.sampling_weight_key is not None and self.old_log_prob_key not in objective_data:
            raise KeyError("sampling correction requires frozen training log probabilities")
        if self.old_log_prob_key in objective_data.keys():
            old_log_prob_full = objective_data[self.old_log_prob_key]
            if old_log_prob_full.shape != torch.Size([N]):
                raise ValueError(
                    f"GRPO expects {self.old_log_prob_key!r} shape [{N}], "
                    f"got {tuple(old_log_prob_full.shape)}."
                )
            old_log_prob = old_log_prob_full[1:][active].to(dtype=dtype)
        else:
            old_log_prob = new_log_prob.detach()

        log_ratio = new_log_prob - old_log_prob
        if self.importance_sampling_level == "episode":
            _, inverse, counts = torch.unique_consecutive(
                group_id[1:][active], return_inverse=True, return_counts=True
            )
            sums = log_ratio.new_zeros(counts.shape).scatter_add(0, inverse, log_ratio)
            log_ratio = (sums / counts.to(dtype=dtype))[inverse]
        ratio = log_ratio.exp()
        surr1 = ratio * advantage
        surr2 = ratio.clamp(self.clip_ratio_min, self.clip_ratio_max) * advantage
        surrogate = -torch.min(surr1, surr2)
        if self.sampling_weight_key is not None:
            weights = objective_data[self.sampling_weight_key]
            if weights.shape != torch.Size([N]):
                raise ValueError("sampling weights must have one value per step")
            weights = weights[1:][active].detach().to(dtype=dtype)
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("sampling weights must be finite and nonnegative")
            surrogate = surrogate * weights
        policy_loss = _weighted_mean(surrogate, pair_weight)

        entropy = _weighted_mean(
            -(log_probs_all.exp() * log_probs_all).sum(dim=-1), pair_weight
        )
        loss = policy_loss - self.ent_coef * entropy

        with torch.no_grad():
            clipfrac = _weighted_mean(
                (((ratio < self.clip_ratio_min) & (advantage < 0))
                 | ((ratio > self.clip_ratio_max) & (advantage > 0))).to(dtype=dtype), pair_weight
            )
            approx_kl = _weighted_mean(old_log_prob - new_log_prob, pair_weight)

        named: dict[str, torch.Tensor] = {
            "grpo": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "entropy": entropy.detach(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "advantage_mean": _weighted_mean(advantage, pair_weight),
        }
        metrics: dict[str, float | torch.Tensor] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return {"grpo": loss}, metrics
