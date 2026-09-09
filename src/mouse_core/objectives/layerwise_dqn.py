"""Layerwise DQN objective with per-layer discount schedules."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _affine,
    _boundary_discounts,
    _greedy_from_online_q,
    _in_run_stats,
    _pair_values_to_rows,
    _pair_weight,
    _head_output_layout,
    _require_done_codes,
    _td_lambda_targets,
    _weighted_mean,
)


def effective_horizon(gamma: float) -> float:
    """Effective planning horizon ``1 / (1 - gamma)`` for ``gamma < 1``."""
    if gamma >= 1.0:
        return float("inf")
    if gamma <= 0.0:
        return 1.0
    return 1.0 / (1.0 - gamma)


def gamma_from_horizon(horizon: float) -> float:
    """Discount factor with effective horizon ``horizon >= 1``."""
    if not math.isfinite(horizon) or horizon <= 1.0:
        return 0.0
    return 1.0 - 1.0 / horizon


def _build_layer_gamma_schedule(
    *,
    num_layers: int,
    gamma_start: float,
    gamma_deep: float,
) -> list[float]:
    """Build per-layer gammas with linearly increasing effective horizon.

    Layer ``0`` is exactly ``gamma_start``; layer ``L - 1`` is exactly
    ``gamma_deep``. Intermediate layers linearly interpolate horizon:

    ``H_l = H_start + (H_deep - H_start) * (l / (L - 1))``

    ``gamma_l = 1 - 1 / H_l``
    """
    if num_layers < 1:
        raise ValueError(f"num_backbone_layers must be >= 1, got {num_layers}.")
    if num_layers == 1:
        return [gamma_deep]

    if gamma_start == gamma_deep:
        return [gamma_deep] * num_layers

    h_start = effective_horizon(gamma_start)
    h_deep = effective_horizon(gamma_deep)

    if math.isinf(h_start) or math.isinf(h_deep):
        raise ValueError(
            "gamma_start and gamma_deep must be below 1.0 for a finite horizon schedule."
        )

    return [
        gamma_from_horizon(h_start + (h_deep - h_start) * (layer_idx / (num_layers - 1)))
        for layer_idx in range(num_layers)
    ]


class LayerwiseDqnObjective(Objective):
    """Bellman TD(λ) objective on every backbone layer.

    Reads ``predictions["action_value_layerwise"]`` and
    ``delayed_predictions["action_value_layerwise"]`` with shape ``[P, L, A]``
    (one row per head-output token; ``objective_data["head_output_count"]`` maps
    rows to steps). Every head-output row of step ``i`` trains toward the same
    per-layer target; the bootstrap reads step ``i+1``'s last head-output row.
    Delayed Q comes from a delayed :class:`~mouse_core.models.base.Model`
    and is detached
    before the Bellman target, so the TD error does not backprop through it.
    Each layer and each episode/task done-code uses its own discount, built at construction
    from explicit shallow/deep endpoint pairs. A run is the same
    ``sequence_id`` and, when ``grouping_field`` is set and present, the same
    grouping column. Neighbor reads must stay in-run: out-of-run pairs are
    multiplied by ``0`` on every layer (all-zero weights → loss ``0``).
    ``action``,
    ``reward``, ``episode_done``, and ``task_done`` must be in the tokenizer
    ``objective_fields`` keep-list.

    Effective planning horizon is ``H(gamma) = 1 / (1 - gamma)``. Layer ``0`` uses
    each ``gamma_*_start``; the deepest layer uses the deep value
    (``gamma_step``, ``gamma_episode_terminal``, …). Intermediate layers get
    **linearly increasing horizon** (linearly harder targets):

    ``H_l = H_start + (H_deep - H_start) * (l / (L - 1))``

    ``gamma_l = 1 - 1 / H_l``

    Example with ``num_backbone_layers=20``, ``gamma_episode_terminal_start=0.0``,
    ``gamma_episode_terminal=0.99`` (``H_start=1``, ``H_deep=100``):

    +--------+---------------------------+----------+
    | Layer  | gamma_episode_terminal    | Horizon  |
    +========+===========================+==========+
    | 0      | 0.0                       | 1        |
    | 5      | ~0.963                    | ~27      |
    | 10     | ~0.981                    | ~53      |
    | 19     | 0.99                      | 100      |
    +--------+---------------------------+----------+

    ``get_action`` on a model with this head uses the deepest layer's Q-values.

    The per-layer target is the TD(λ) return of
    :class:`~mouse_core.objectives.dqn.DqnObjective` (``td_lambda=0`` is the
    one-step target) built with that layer's discounts; with ``watkins=True``
    the trace is also cut wherever the taken action is not that layer's online
    argmax, and ``metrics["watkins_greedy_frac"]`` reports the deepest layer's
    in-run greedy fraction.

    Args:
        num_backbone_layers: Number of transformer blocks (and Q heads).
        gamma_step_start: Step discount at layer 0 (``episode_done == 0``).
        gamma_step: Step discount at the deepest layer.
        gamma_episode_terminal_start: Episode-terminal discount at layer 0.
        gamma_episode_terminal: Episode-terminal discount at the deepest layer.
        gamma_episode_truncated_start: Episode-truncated discount at layer 0.
        gamma_episode_truncated: Episode-truncated discount at the deepest layer.
        gamma_task_terminal_start: Task-terminal extra discount at layer 0
            (multiplies the episode discount; ``task_done == 0`` uses ``1.0``).
        gamma_task_terminal: Task-terminal extra discount at the deepest layer.
        gamma_task_truncated_start: Task-truncated extra discount at layer 0.
        gamma_task_truncated: Task-truncated extra discount at the deepest layer.
        action_key: Key in ``objective_data`` for the integer action.
        reward_key: Key in ``objective_data`` for per-step reward.
        reward_scale: Multiplier applied to ``reward`` before the TD target
            (default ``1.0``). Does not change ``objective_data``.
        reward_shift: Offset added after ``reward_scale`` (default ``0.0``).
        q_scale: Multiplier applied to online and delayed Q before the TD
            error (default ``1.0``). Same affine on both networks.
        q_shift: Offset added after ``q_scale`` (default ``0.0``).
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        cql_weight: CQL penalty coefficient; ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor when scaling the CQL penalty.
        td_lambda: λ of the TD(λ) target in ``[0, 1]``. ``0.0`` (default) is
            the one-step target; ``1.0`` is the full in-run n-step return.
        watkins: Cut the λ-trace at non-greedy actions (Watkins's Q(λ)).
    """

    def __init__(
        self,
        *,
        num_backbone_layers: int,
        gamma_step_start: float,
        gamma_step: float = 0.99,
        gamma_episode_terminal_start: float = 0.0,
        gamma_episode_terminal: float = 0.0,
        gamma_episode_truncated_start: float = 0.0,
        gamma_episode_truncated: float = 0.0,
        gamma_task_terminal_start: float = 0.0,
        gamma_task_terminal: float = 0.0,
        gamma_task_truncated_start: float = 0.0,
        gamma_task_truncated: float = 0.0,
        action_key: str = "action",
        reward_key: str = "reward",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        q_scale: float = 1.0,
        q_shift: float = 0.0,
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
        grouping_field: str | None = None,
        td_lambda: float = 0.0,
        watkins: bool = False,
    ) -> None:
        if not 0.0 <= float(td_lambda) <= 1.0:
            raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}.")
        self.num_backbone_layers = int(num_backbone_layers)
        self.gamma_step_start = float(gamma_step_start)
        self.gamma_step = float(gamma_step)
        self.gamma_episode_terminal_start = float(gamma_episode_terminal_start)
        self.gamma_episode_terminal = float(gamma_episode_terminal)
        self.gamma_episode_truncated_start = float(gamma_episode_truncated_start)
        self.gamma_episode_truncated = float(gamma_episode_truncated)
        self.gamma_task_terminal_start = float(gamma_task_terminal_start)
        self.gamma_task_terminal = float(gamma_task_terminal)
        self.gamma_task_truncated_start = float(gamma_task_truncated_start)
        self.gamma_task_truncated = float(gamma_task_truncated)
        self.action_key = action_key
        self.reward_key = reward_key
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self.q_scale = float(q_scale)
        self.q_shift = float(q_shift)
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.grouping_field = grouping_field
        self.td_lambda = float(td_lambda)
        self.watkins = bool(watkins)

        build = _build_layer_gamma_schedule
        n = self.num_backbone_layers
        self.layer_gamma_step = build(
            num_layers=n, gamma_start=self.gamma_step_start, gamma_deep=self.gamma_step
        )
        self.layer_gamma_episode_terminal = build(
            num_layers=n,
            gamma_start=self.gamma_episode_terminal_start,
            gamma_deep=self.gamma_episode_terminal,
        )
        self.layer_gamma_episode_truncated = build(
            num_layers=n,
            gamma_start=self.gamma_episode_truncated_start,
            gamma_deep=self.gamma_episode_truncated,
        )
        self.layer_gamma_task_terminal = build(
            num_layers=n,
            gamma_start=self.gamma_task_terminal_start,
            gamma_deep=self.gamma_task_terminal,
        )
        self.layer_gamma_task_truncated = build(
            num_layers=n,
            gamma_start=self.gamma_task_truncated_start,
            gamma_deep=self.gamma_task_truncated,
        )

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q: torch.Tensor = predictions["action_value_layerwise"]
        q_target: torch.Tensor = delayed_predictions["action_value_layerwise"].detach()

        if q.ndim != 3:
            raise ValueError(
                f"Layerwise DQN expects action_value_layerwise shape [P, L, A], "
                f"got {tuple(q.shape)}."
            )
        if q.dtype != torch.float32 or q_target.dtype != torch.float32:
            raise TypeError(
                "Layerwise DQN expects float32 action_value_layerwise (heads always "
                f"run in fp32), got online {q.dtype} and delayed {q_target.dtype}."
            )
        if q_target.shape != q.shape:
            raise ValueError(
                f"Layerwise DQN delayed shape {tuple(q_target.shape)} must "
                f"match online shape {tuple(q.shape)}."
            )
        q = _affine(q, scale=self.q_scale, shift=self.q_shift)
        q_target = _affine(q_target, scale=self.q_scale, shift=self.q_shift)
        P, L, A = q.shape
        device = q.device
        value_dtype = q.dtype

        if L != self.num_backbone_layers:
            raise ValueError(
                f"Layerwise DQN objective expects {self.num_backbone_layers} Q layers "
                f"but predictions have {L}."
            )

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"Layerwise DQN objective expects action shape [N], got {tuple(action.shape)}."
            )
        N = int(action.shape[0])

        if N < 2:
            raise ValueError("Not enough valid q values in data.")

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([N]):
            raise ValueError(
                f"Layerwise DQN objective expects reward shape [{N}], got {tuple(reward.shape)}."
            )
        reward = _affine(
            reward, scale=self.reward_scale, shift=self.reward_shift
        )

        episode_done, task_done = _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=N,
        )

        # A step may own several head-output tokens; every row of step i trains
        # toward the same per-layer target, and the bootstrap reads step i+1's
        # last head-output row.
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
        # Row weight = the (i, i+1) pair weight of the row's step; rows of the
        # final step have no next step and get weight 0.
        row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]  # [P]

        step_next = (step_of + 1).clamp(max=N - 1)      # [P]
        next_actions = action[step_next]                # [P]
        v_step_all = q_target[last_rows].amax(dim=-1)   # [N, L]  V_l(s_i)

        layer_losses: list[torch.Tensor] = []
        layer_curr_max_means: list[torch.Tensor] = []
        cql_penalties: list[torch.Tensor] = []
        deepest_curr_max_q: torch.Tensor | None = None
        deepest_greedy_from: torch.Tensor | None = None

        for layer_idx in range(L):
            discount_all = _boundary_discounts(
                episode_done=episode_done,
                task_done=task_done,
                gamma_step=self.layer_gamma_step[layer_idx],
                gamma_episode_terminal=self.layer_gamma_episode_terminal[layer_idx],
                gamma_episode_truncated=self.layer_gamma_episode_truncated[layer_idx],
                gamma_task_terminal=self.layer_gamma_task_terminal[layer_idx],
                gamma_task_truncated=self.layer_gamma_task_truncated[layer_idx],
                dtype=value_dtype,
                device=device,
            )

            curr_q_layer = q[:, layer_idx, :]  # [P, A]
            q_values = curr_q_layer.gather(
                dim=-1, index=next_actions.unsqueeze(-1)
            ).squeeze(-1)
            curr_max_q = curr_q_layer.amax(dim=-1)
            greedy_from = (
                _greedy_from_online_q(q=curr_q_layer, action=action, last_rows=last_rows)
                if self.watkins
                else None
            )
            pair_target = _td_lambda_targets(
                reward=reward,
                discount_all=discount_all,
                v_step=v_step_all[:, layer_idx],
                pair_weight=pair_weight,
                td_lambda=self.td_lambda,
                greedy_from=greedy_from,
            )
            td_target = _pair_values_to_rows(pair_target, step_of)  # [P]

            loss = (q_values - td_target) ** 2

            if self.cql_weight > 0.0:
                q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
                cql_penalty = torch.logsumexp(curr_q_layer, dim=-1) - q_values
                loss = loss + self.cql_weight * q_scale * cql_penalty
                cql_penalties.append(_weighted_mean(cql_penalty.detach(), row_weight))

            loss = _weighted_mean(loss, row_weight)

            layer_losses.append(loss)
            layer_curr_max_means.append(_weighted_mean(curr_max_q.detach(), row_weight))
            if layer_idx == L - 1:
                deepest_curr_max_q = curr_max_q
                deepest_greedy_from = greedy_from

        total_loss = torch.stack(layer_losses).mean()

        if deepest_curr_max_q is None:
            raise RuntimeError(
                "Layerwise DQN objective did not compute deepest-layer current-state max Q values."
            )
        q_mean, q_std, q_min, q_max = _in_run_stats(
            deepest_curr_max_q.detach(), row_weight
        )

        named: dict[str, torch.Tensor] = {
            "q_values_mean": q_mean,
            "q_values_std": q_std,
            "q_values_min": q_min,
            "q_values_max": q_max,
            "action_value_layerwise": total_loss.detach(),
        }
        for layer_idx, gamma in enumerate(self.layer_gamma_step):
            named[f"layer_{layer_idx}_gamma_step"] = torch.tensor(
                gamma, device=device, dtype=value_dtype
            )
            named[f"layer_{layer_idx}_loss"] = layer_losses[layer_idx].detach()
            named[f"layer_{layer_idx}_q_mean"] = layer_curr_max_means[layer_idx]
        if cql_penalties:
            named["cql_penalty"] = torch.stack(cql_penalties).mean()
        if deepest_greedy_from is not None:
            named["watkins_greedy_frac"] = _weighted_mean(deepest_greedy_from, pair_weight)

        metrics: dict[str, float] = {
            key: (value.item() if value.numel() == 1 else float(value))
            for key, value in named.items()
        }
        return total_loss, metrics
