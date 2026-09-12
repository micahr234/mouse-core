"""Episode/task DQN: two Q-heads, one shared next action ``a*``."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _affine,
    _boundary_discounts,
    _greedy_from_online_q,
    _head_output_layout,
    _in_run_stats,
    _pair_values_to_rows,
    _pair_weight,
    _require_action_ids,
    _require_done_codes,
    _td_lambda_targets,
    _weighted_mean,
)


def _require_q(predictions: TensorDict, key: str) -> torch.Tensor:
    """Validate a float32 ``[P, A]`` action-value tensor."""
    q: torch.Tensor = predictions[key]
    if q.ndim != 2:
        raise ValueError(f"DQN expects {key} shape [P, A], got {tuple(q.shape)}.")
    if q.dtype != torch.float32:
        raise TypeError(
            f"DQN expects float32 {key} (heads always run in fp32), got {q.dtype}."
        )
    return q


def _head_td_loss(
    *,
    q: torch.Tensor,
    v_step: torch.Tensor,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    action: torch.Tensor,
    step_of: torch.Tensor,
    pair_weight: torch.Tensor,
    td_lambda: float,
    greedy_from: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE of gathered online Q vs the TD(λ) target.

    Returns ``(scalar_loss, row_weight [P])``.
    """
    N = int(action.shape[0])
    step_next = (step_of + 1).clamp(max=N - 1)
    next_actions = action[step_next]
    q_values = q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)
    pair_target = _td_lambda_targets(
        reward=reward,
        discount_all=discount_all,
        v_step=v_step,
        pair_weight=pair_weight,
        td_lambda=td_lambda,
        greedy_from=greedy_from,
    )
    td_target = _pair_values_to_rows(pair_target, step_of)
    row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]
    loss = _weighted_mean((q_values - td_target) ** 2, row_weight)
    return loss, row_weight


class EpisodeTaskDqnObjective(Objective):
    """Two-head Bellman objective with a shared delayed next action.

    Online Q is ``predictions["action_value_episode"]`` and
    ``predictions["action_value_task"]``. Bootstrap Q comes from the matching
    delayed tensors. One next action
    ``a* = argmax_a (Q_e(s', a) + Q_t(s', a))`` is read from the **delayed**
    sum; both heads gather that ``a*`` instead of taking their own ``max``.

    The episode head is stepwise TD on env reward and does not bootstrap
    across episode boundaries (``gamma_episode_* = 0``). Bootstrap value is
    ``Q_e(s', a*)``.

    The task head predicts return in **subsequent** episodes only (env reward
    is dropped). Intra-episode discount is ``1.0`` and ``task_td_lambda=1``
    so the λ-return skips to the next episode start. At a mid-task episode
    boundary the bootstrap is ``Q_e(s', a*) + Q_t(s', a*)``; mid-episode it
    is ``Q_t(s', a*)``. Task-end discounts zero the bootstrap.

    A run is the same ``sequence_id`` and, when ``grouping_field`` is set and
    present, the same grouping column. Out-of-run pairs have weight ``0``.
    ``action``, ``reward``, ``episode_done``, and ``task_done`` must be in the
    tokenizer ``objective_fields`` keep-list.

    Args:
        gamma_step: Episode-head discount on running transitions.
        gamma_episode_terminal: Episode-head discount when the episode
            terminates. Default ``0`` — remaining return is this episode only.
        gamma_episode_truncated: Episode-head discount when the episode is
            truncated. Default ``0``.
        gamma_task_terminal: Extra discount when the task terminates
            (multiplies the episode discount). Shared by both heads.
        gamma_task_truncated: Extra discount when the task is truncated.
            Shared by both heads. ``0`` zeros the bootstrap.
        task_gamma_step: Task-head discount on running transitions
            (default ``1`` so λ=1 skips to the episode boundary).
        task_gamma_episode_terminal: Task-head discount at a natural episode
            end (default ``1`` — bootstrap the next-episode start).
        task_gamma_episode_truncated: Task-head discount at a truncated
            episode end (default ``1``).
        episode_td_lambda: λ for the episode head (default ``0`` = one-step).
        task_td_lambda: λ for the task head (default ``1`` = skip to the
            next boundary or the end of the run).
        watkins: Cut both λ-traces where the taken action is not the online
            argmax of ``Q_e + Q_t``.
    """

    def __init__(
        self,
        *,
        gamma_step: float = 0.99,
        gamma_episode_terminal: float = 0.0,
        gamma_episode_truncated: float = 0.0,
        gamma_task_terminal: float = 0.0,
        gamma_task_truncated: float = 0.0,
        task_gamma_step: float = 1.0,
        task_gamma_episode_terminal: float = 1.0,
        task_gamma_episode_truncated: float = 1.0,
        action_key: str = "action",
        reward_key: str = "reward",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        q_scale: float = 1.0,
        q_shift: float = 0.0,
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        grouping_field: str | None = None,
        episode_td_lambda: float = 0.0,
        task_td_lambda: float = 1.0,
        watkins: bool = False,
    ) -> None:
        if not 0.0 <= float(episode_td_lambda) <= 1.0:
            raise ValueError(
                f"episode_td_lambda must be in [0, 1], got {episode_td_lambda}."
            )
        if not 0.0 <= float(task_td_lambda) <= 1.0:
            raise ValueError(f"task_td_lambda must be in [0, 1], got {task_td_lambda}.")
        self.gamma_step = gamma_step
        self.gamma_episode_terminal = gamma_episode_terminal
        self.gamma_episode_truncated = gamma_episode_truncated
        self.gamma_task_terminal = gamma_task_terminal
        self.gamma_task_truncated = gamma_task_truncated
        self.task_gamma_step = task_gamma_step
        self.task_gamma_episode_terminal = task_gamma_episode_terminal
        self.task_gamma_episode_truncated = task_gamma_episode_truncated
        self.action_key = action_key
        self.reward_key = reward_key
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self.q_scale = float(q_scale)
        self.q_shift = float(q_shift)
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.grouping_field = grouping_field
        self.episode_td_lambda = float(episode_td_lambda)
        self.task_td_lambda = float(task_td_lambda)
        self.watkins = bool(watkins)

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("EpisodeTaskDqnObjective requires delayed_predictions.")
        q_e = _require_q(predictions, "action_value_episode")
        q_t = _require_q(predictions, "action_value_task")
        q_e_target = _require_q(delayed_predictions, "action_value_episode").detach()
        q_t_target = _require_q(delayed_predictions, "action_value_task").detach()
        if q_e.shape != q_t.shape:
            raise ValueError(
                f"episode action_value shape {tuple(q_e.shape)} must match "
                f"task action_value shape {tuple(q_t.shape)}."
            )
        if q_e_target.shape != q_e.shape or q_t_target.shape != q_t.shape:
            raise ValueError(
                "DQN delayed action_value shapes must match the online heads."
            )

        q_e = _affine(q_e, scale=self.q_scale, shift=self.q_shift)
        q_t = _affine(q_t, scale=self.q_scale, shift=self.q_shift)
        q_e_target = _affine(q_e_target, scale=self.q_scale, shift=self.q_shift)
        q_t_target = _affine(q_t_target, scale=self.q_scale, shift=self.q_shift)
        P, A = q_e.shape
        device = q_e.device
        value_dtype = q_e.dtype

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"DQN objective expects action shape [N], got {tuple(action.shape)}."
            )
        N = int(action.shape[0])
        _require_action_ids(action, A)
        if N < 2:
            raise ValueError("Not enough valid q values in data.")

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([N]):
            raise ValueError(
                f"DQN objective expects reward shape [{N}], got {tuple(reward.shape)}."
            )
        reward = _affine(reward, scale=self.reward_scale, shift=self.reward_shift)

        episode_done, task_done = _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=N,
        )

        step_of, last_rows = _head_output_layout(
            objective_data, N=N, P=P, device=device
        )
        pair_weight = _pair_weight(
            objective_data,
            N,
            device,
            grouping_field=self.grouping_field,
            dtype=value_dtype,
        )

        delayed_e = q_e_target[last_rows]
        delayed_t = q_t_target[last_rows]
        a_star = (delayed_e + delayed_t).argmax(dim=-1)
        idx = a_star.unsqueeze(-1)
        v_e = delayed_e.gather(dim=-1, index=idx).squeeze(-1)
        v_t = delayed_t.gather(dim=-1, index=idx).squeeze(-1)
        # episode_done[i] != 0 ⇒ obs_i is the first frame of the next episode.
        v_task = torch.where(episode_done != 0, v_e + v_t, v_t)

        episode_discount = _boundary_discounts(
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
        task_discount = _boundary_discounts(
            episode_done=episode_done,
            task_done=task_done,
            gamma_step=self.task_gamma_step,
            gamma_episode_terminal=self.task_gamma_episode_terminal,
            gamma_episode_truncated=self.task_gamma_episode_truncated,
            gamma_task_terminal=self.gamma_task_terminal,
            gamma_task_truncated=self.gamma_task_truncated,
            dtype=value_dtype,
            device=device,
        )
        task_reward = torch.zeros_like(reward)

        greedy_from = (
            _greedy_from_online_q(q=q_e + q_t, action=action, last_rows=last_rows)
            if self.watkins
            else None
        )
        loss_e, row_weight = _head_td_loss(
            q=q_e,
            v_step=v_e,
            reward=reward,
            discount_all=episode_discount,
            action=action,
            step_of=step_of,
            pair_weight=pair_weight,
            td_lambda=self.episode_td_lambda,
            greedy_from=greedy_from,
        )
        loss_t, _ = _head_td_loss(
            q=q_t,
            v_step=v_task,
            reward=task_reward,
            discount_all=task_discount,
            action=action,
            step_of=step_of,
            pair_weight=pair_weight,
            td_lambda=self.task_td_lambda,
            greedy_from=greedy_from,
        )
        loss = loss_e + loss_t

        q_e_mean, q_e_std, q_e_min, q_e_max = _in_run_stats(
            q_e.amax(dim=-1).detach(), row_weight
        )
        q_t_mean, q_t_std, q_t_min, q_t_max = _in_run_stats(
            q_t.amax(dim=-1).detach(), row_weight
        )
        named: dict[str, torch.Tensor] = {
            "q_episode_mean": q_e_mean,
            "q_episode_std": q_e_std,
            "q_episode_min": q_e_min,
            "q_episode_max": q_e_max,
            "q_task_mean": q_t_mean,
            "q_task_std": q_t_std,
            "q_task_min": q_t_min,
            "q_task_max": q_t_max,
            "action_value_episode": loss_e.detach(),
            "action_value_task": loss_t.detach(),
        }
        if greedy_from is not None:
            named["watkins_greedy_frac"] = _weighted_mean(greedy_from, pair_weight)

        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
