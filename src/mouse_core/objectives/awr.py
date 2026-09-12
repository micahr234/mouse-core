"""Advantage-Weighted Regression with a Monte Carlo Q critic."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _affine,
    _boundary_discounts,
    _head_output_layout,
    _pair_values_to_rows,
    _pair_weight,
    _require_action_ids,
    _require_done_codes,
    _td_lambda_targets,
    _weighted_mean,
)


class AwrObjective(Objective):
    """Advantage-Weighted Regression (Peng et al., 2019) with a Q critic.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions)``. ``delayed_predictions`` is ignored
    (AWR has no target network). Requires dual heads:

    * ``predictions["action_value"]`` — ``[P, A]`` Q-values
    * ``predictions["action"]`` — ``[P, A]`` policy logits

    The critic target is the in-run Monte Carlo return
    ``G_i = r + γ G_{i+1}`` with no value bootstrap (``V = 0`` at every
    cut, including the pack-window edge). ``γ`` is the same done-code
    table as :class:`~mouse_core.objectives.dqn.DqnObjective`:
    ``gamma_step`` on a running step, ``gamma_episode_*`` at an episode
    boundary, then multiplied by ``gamma_task_*`` at a task boundary
    (``1.0`` when ``task_done == 0``). A ``0`` gamma stops the sum; a
    non-zero gamma carries later rewards through, discounted. Timing and
    pair masks match DQN: token ``i`` is ``s_i``; action / reward / done
    at ``i+1`` describe the transition out of ``s_i``.

    Q regresses onto ``G``. The policy is weighted log-likelihood
    ``w = min(exp(A / β), ω_max)`` with ``A = G − max_a Q(s, a)``. ``G``
    and ``V(s)`` are detached so the policy does not differentiate through
    the critic via the weights. There is no entropy term, CQL, Watkins
    trace, or advantage normalization.

    Q rows are **per head-output token** (``[P, A]``), not per step: a step
    may own several head-output tokens. The ``head_output_count`` column
    stamped by ``pack_token_batch`` maps rows to steps. Every head-output
    row of step ``i`` trains toward the same ``G``.

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is
    set and present, the same grouping column. Out-of-run pairs still have
    a loss term, but it is multiplied by ``0``. If every weight is ``0``
    the loss is ``0``.

    Those columns arrive in ``objective_data`` only if they are listed in
    the tokenizer ``objective_fields`` keep-list (input fields are not
    auto-copied). ``task_done`` is an objective column only::

        tokenizer = Tokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "episode_done"},
                {"input_field": "task_done"},
            ],
        )

    Act with the policy: build the model with ``action_head="action"``.
    ``get_action(..., temperature=0)`` is greedy-π; ``temperature>0``
    samples.

    Args:
        advantage_temperature: AWR temperature ``β`` in ``exp(A / β)``.
            Must be ``> 0``.
        weight_clip: Maximum advantage weight ``ω_max``. Must be ``> 0``.
        policy_coef: Weight on the policy term. Must be ``>= 0``.
        gamma_step: Discount applied to the rest of ``G`` on a running
            step (``episode_done == 0``).
        gamma_episode_terminal: Discount on later rewards when the episode
            terminates (``episode_done == 1``). ``0.0`` stops ``G``.
        gamma_episode_truncated: Discount on later rewards when the episode
            is truncated (``episode_done == 2``). ``0.0`` stops ``G``.
        gamma_task_terminal: Extra discount when the task terminates
            (``task_done == 1``); multiplies the episode discount.
            ``task_done == 0`` uses ``1.0``.
        gamma_task_truncated: Extra discount when the task is truncated
            (``task_done == 2``); multiplies the episode discount.
            ``0.0`` stops ``G`` at the task cut.
        action_key: Key in ``objective_data`` for the integer action.
        reward_key: Key in ``objective_data`` for the per-step reward.
        reward_scale: Multiplier applied to ``reward`` before the return.
        reward_shift: Offset added after ``reward_scale``.
        q_scale: Multiplier applied to ``action_value``.
        q_shift: Offset added after ``q_scale``.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        grouping_field: Extra run-boundary column (e.g. ``task_index``).
    """

    def __init__(
        self,
        *,
        advantage_temperature: float,
        weight_clip: float,
        policy_coef: float,
        gamma_step: float = 0.99,
        gamma_episode_terminal: float = 0.0,
        gamma_episode_truncated: float = 0.0,
        gamma_task_terminal: float = 0.0,
        gamma_task_truncated: float = 0.0,
        action_key: str = "action",
        reward_key: str = "reward",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        q_scale: float = 1.0,
        q_shift: float = 0.0,
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        grouping_field: str | None = None,
    ) -> None:
        if float(advantage_temperature) <= 0.0:
            raise ValueError(
                f"advantage_temperature must be > 0, got {advantage_temperature}."
            )
        if float(weight_clip) <= 0.0:
            raise ValueError(f"weight_clip must be > 0, got {weight_clip}.")
        if float(policy_coef) < 0.0:
            raise ValueError(f"policy_coef must be >= 0, got {policy_coef}.")
        self.advantage_temperature = float(advantage_temperature)
        self.weight_clip = float(weight_clip)
        self.policy_coef = float(policy_coef)
        self.gamma_step = gamma_step
        self.gamma_episode_terminal = gamma_episode_terminal
        self.gamma_episode_truncated = gamma_episode_truncated
        self.gamma_task_terminal = gamma_task_terminal
        self.gamma_task_truncated = gamma_task_truncated
        self.action_key = action_key
        self.reward_key = reward_key
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self.q_scale = float(q_scale)
        self.q_shift = float(q_shift)
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.grouping_field = grouping_field

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if "action_value" not in predictions.keys():
            raise KeyError("AwrObjective requires predictions['action_value'].")
        if "action" not in predictions.keys():
            raise KeyError(
                "AwrObjective requires predictions['action'] policy logits."
            )

        q: torch.Tensor = predictions["action_value"]
        logits: torch.Tensor = predictions["action"]

        if q.ndim != 2:
            raise ValueError(
                f"AWR expects action_value shape [P, A], got {tuple(q.shape)}."
            )
        if logits.shape != q.shape:
            raise ValueError(
                f"AWR expects action logits shape {tuple(q.shape)} to match "
                f"action_value, got {tuple(logits.shape)}."
            )
        if q.dtype != torch.float32:
            raise TypeError(
                "AWR expects float32 action_value (heads always run in fp32), "
                f"got {q.dtype}."
            )
        if logits.dtype != torch.float32:
            raise TypeError(
                f"AWR expects float32 action logits, got {logits.dtype}."
            )
        q = _affine(q, scale=self.q_scale, shift=self.q_shift)
        P, A = q.shape
        device = q.device
        value_dtype = q.dtype

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"AWR objective expects action shape [N], got {tuple(action.shape)}."
            )
        N = int(action.shape[0])
        _require_action_ids(action, A)

        if N < 2:
            raise ValueError("Not enough valid steps in data for AWR (need N >= 2).")

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([N]):
            raise ValueError(
                f"AWR objective expects reward shape [{N}], got {tuple(reward.shape)}."
            )
        reward = _affine(reward, scale=self.reward_scale, shift=self.reward_shift)

        episode_done, task_done = _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=N,
        )

        step_of, _ = _head_output_layout(
            objective_data, N=N, P=P, device=device
        )

        pair_weight = _pair_weight(
            objective_data,
            N,
            device,
            grouping_field=self.grouping_field,
            dtype=value_dtype,
        )
        row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]

        step_next = (step_of + 1).clamp(max=N - 1)
        next_actions = action[step_next]

        discount_all = _boundary_discounts(
            episode_done=episode_done,
            task_done=task_done,
            gamma_step=self.gamma_step,
            gamma_episode_terminal=self.gamma_episode_terminal,
            gamma_episode_truncated=self.gamma_episode_truncated,
            gamma_task_terminal=self.gamma_task_terminal,
            gamma_task_truncated=self.gamma_task_truncated,
            dtype=value_dtype,
            device=device,
        )

        q_values = q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)
        pair_target = _td_lambda_targets(
            reward=reward,
            discount_all=discount_all,
            v_step=q.new_zeros(N),
            pair_weight=pair_weight,
            td_lambda=1.0,
            greedy_from=None,
        )
        returns = _pair_values_to_rows(pair_target, step_of)

        q_loss = _weighted_mean((q_values - returns) ** 2, row_weight)

        values = q.amax(dim=-1).detach()
        advantages = (returns - values).detach()
        awr_weight = (advantages / self.advantage_temperature).exp().clamp(
            max=self.weight_clip
        )

        log_probs_all = F.log_softmax(logits, dim=-1)
        log_prob = log_probs_all.gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)
        policy_loss = _weighted_mean(-awr_weight * log_prob, row_weight)

        loss = q_loss + self.policy_coef * policy_loss

        named: dict[str, torch.Tensor] = {
            "awr": loss.detach(),
            "q_loss": q_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "advantage_mean": _weighted_mean(advantages, row_weight),
            "weight_mean": _weighted_mean(awr_weight, row_weight),
            "return_mean": _weighted_mean(returns.detach(), row_weight),
        }
        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
