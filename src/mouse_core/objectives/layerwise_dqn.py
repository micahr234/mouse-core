"""Layerwise DQN objective with per-layer discount schedules."""

from __future__ import annotations

import math

import torch

from mouse_core.models.heads.base import BaseHead
from mouse_core.objectives.base import Objective, predictions_for, require_head
from mouse_core.objectives.dqn import (
    _boltzmann_entropy,
    _greedy_from_online_q,
    _in_run_stats,
    _pair_values_to_rows,
    _pair_weight,
    _head_output_layout,
    _require_action_ids,
    _require_done_codes,
    _require_temperature,
    _soft_state_value,
    _td_lambda_targets,
    _weighted_mean,
)
from mouse_core.objectives.transforms import (
    Discount,
    Reward,
    Value,
    _apply_transform,
    _apply_value,
    _require_transform,
)


def effective_horizon(*, gamma: float) -> float:
    """Effective planning horizon ``1 / (1 - gamma)`` for ``gamma < 1``."""
    if gamma >= 1.0:
        return float("inf")
    if gamma <= 0.0:
        return 1.0
    return 1.0 / (1.0 - gamma)


def gamma_from_horizon(*, horizon: float) -> float:
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
        if gamma_start != gamma_deep:
            raise ValueError(
                f"num_backbone_layers=1 cannot interpolate between "
                f"gamma_start={gamma_start} and gamma_deep={gamma_deep}; "
                "pass the same value for both."
            )
        return [gamma_start]

    if gamma_start == gamma_deep:
        return [gamma_deep] * num_layers

    h_start = effective_horizon(gamma=gamma_start)
    h_deep = effective_horizon(gamma=gamma_deep)

    if math.isinf(h_start) or math.isinf(h_deep):
        raise ValueError(
            "gamma_start and gamma_deep must be below 1.0 for a finite horizon schedule."
        )

    return [
        gamma_from_horizon(horizon=h_start + (h_deep - h_start) * (layer_idx / (num_layers - 1)))
        for layer_idx in range(num_layers)
    ]


def _done_code_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """All ``(episode_done, task_done)`` pairs in ``{0,1,2}²``."""
    episode_done = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int64)
    task_done = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int64)
    return episode_done, task_done


def _probe_discount(
    *,
    discount: Discount | None,
    episode_done: int,
    task_done: int,
) -> float:
    """Evaluate ``discount`` on a single done-code pair."""
    if discount is None:
        return 1.0
    values = discount(
        episode_done=torch.tensor([episode_done], dtype=torch.int64),
        task_done=torch.tensor([task_done], dtype=torch.int64),
    )
    if not isinstance(values, torch.Tensor) or values.numel() != 1:
        raise ValueError(
            "discount must return a single value for a one-step done-code pair."
        )
    return float(values.reshape(-1)[0])


def _horizon_lerp(
    *,
    gamma_start: torch.Tensor,
    gamma_deep: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Horizon-linear interpolation of per-step discounts.

    ``t=0`` is ``gamma_start``; ``t=1`` is ``gamma_deep``. Equal values stay
    put (including ``γ >= 1``). Mixing a finite horizon with an infinite
    one raises, matching :func:`_build_layer_gamma_schedule`.
    """
    if t == 0.0:
        return gamma_start
    if t == 1.0:
        return gamma_deep
    same = gamma_start == gamma_deep
    infinite = (gamma_start >= 1.0) | (gamma_deep >= 1.0)
    if bool((infinite & ~same).any()):
        raise ValueError(
            "gamma_start and gamma_deep must be below 1.0 for a finite "
            "horizon schedule."
        )
    h_start = torch.where(
        gamma_start <= 0.0,
        torch.ones_like(gamma_start),
        torch.where(
            gamma_start >= 1.0,
            gamma_start.new_full(gamma_start.shape, float("inf")),
            1.0 / (1.0 - gamma_start),
        ),
    )
    h_deep = torch.where(
        gamma_deep <= 0.0,
        torch.ones_like(gamma_deep),
        torch.where(
            gamma_deep >= 1.0,
            gamma_deep.new_full(gamma_deep.shape, float("inf")),
            1.0 / (1.0 - gamma_deep),
        ),
    )
    horizon = h_start + (h_deep - h_start) * t
    finite = torch.isfinite(horizon) & (horizon > 1.0)
    lerped = torch.where(finite, 1.0 - 1.0 / horizon, torch.zeros_like(horizon))
    return torch.where(same, gamma_start, lerped)


class LayerwiseDqnObjective(Objective):
    """Bellman TD(λ) objective on every backbone layer.

    Reads the tensor for ``head`` on ``predictions`` and
    ``delayed_predictions`` with shape ``[P, L, A]``
    (one row per head-output token; ``objective_data["head_output_count"]`` maps
    rows to steps). Every head-output row of step ``i`` trains toward the same
    per-layer target; the bootstrap reads step ``i+1``'s last head-output row.
    Delayed Q comes from the delayed :class:`~mouse_core.models.base.Model`
    (``model.delayed_copy(heads=(head,))``) run on the
    same ``TokenBatch`` and is detached
    before the Bellman target, so the TD error does not backprop through it.
    Each layer uses its own discount. Layer ``0`` evaluates
    ``discount_start``; the deepest layer evaluates ``discount``. Intermediate
    layers horizon-lerp those two per-step outputs. A run is the same
    ``sequence_id`` and, when ``grouping_field`` is set and present, the same
    grouping column. Neighbor reads must stay in-run: out-of-run pairs are
    multiplied by ``0`` on every layer (all-zero weights → loss ``0``).
    ``action``,
    ``reward``, ``episode_done``, and ``task_done`` must be in the tokenizer
    ``objective_fields`` keep-list.

    Effective planning horizon is ``H(gamma) = 1 / (1 - gamma)``. Intermediate
    layers get **linearly increasing horizon** (linearly harder targets):

    ``H_l = H_start + (H_deep - H_start) * (l / (L - 1))``

    ``gamma_l = 1 - 1 / H_l``

    Example with ``num_backbone_layers=20``, ``discount_start`` returning
    ``0.0`` and ``discount`` returning ``0.99`` at a given done-code pair
    (``H_start=1``, ``H_deep=100``):

    +--------+---------------------------+----------+
    | Layer  | interpolated γ            | Horizon  |
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
    in-run greedy fraction.     ``temperature`` is the same SAC / soft-Q ``α``
    as :class:`~mouse_core.objectives.dqn.DqnObjective`: ``0`` is hard
    max-Q per layer; ``> 0`` bootstraps from ``α logsumexp(Q / α)``.
    ``metrics["entropy"]`` is the deepest layer's in-run mean of
    ``H[softmax(Q / α)]`` on online Q when ``α > 0``.

    Args:
        head: Layerwise Q head this objective trains. Must be the same
            instance passed to ``Model(heads=)``.
        num_backbone_layers: Number of transformer blocks (and Q heads).
        discount: Per-step γ at the deepest layer, from unpacked
            ``objective_data`` columns. ``boundary_discount`` is the
            standard ``gamma_step`` × extra lookup (``None`` skips the
            call and uses ``1``).
        discount_start: Per-step γ at layer 0. Intermediate layers
            horizon-lerp ``discount_start`` and ``discount``. With one
            layer both callables must agree on every done-code pair.
            ``None`` skips the call and uses ``1``.
        reward: Per-step reward from unpacked ``objective_data`` columns.
            ``affine_reward`` is the column affine; ``boundary_reward``
            applies episode / task scale and shift extras
            (``None`` skips the call);
            any ``reward(**objective_data) -> [N]`` is accepted.
        value: Per-step affine on online and delayed Q from unpacked
            ``objective_data`` columns plus ``value=``. ``affine_value``
            is the prediction affine; ``boundary_value`` applies episode
            / task scale and shift extras
            (``None`` skips the call);
            any ``value(value=..., **objective_data)`` returning the same
            shape is accepted. Same callable on both networks.
        action_key: Key in ``objective_data`` for the integer action.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        cql_weight: CQL penalty coefficient; ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor when scaling the CQL penalty.
        td_lambda: λ of the TD(λ) target in ``[0, 1]``. ``0.0`` (default) is
            the one-step target; ``1.0`` is the full in-run n-step return.
        watkins: Cut the λ-trace at non-greedy actions (Watkins's Q(λ)).
        grouping_field: Step column that isolates runs (typically
            ``task_index``). Required. Pass ``None`` only when the batch
            has no grouping isolation — omitting it is an error, not a
            silent skip.
        temperature: SAC / soft Q-learning ``α`` (``>= 0``). Required.
            ``0`` is hard max-Q. ``> 0`` bootstraps each layer from
            ``α logsumexp(Q / α)`` on delayed Q (after ``value``) and
            logs ``metrics["entropy"]``. Same units and meaning as
            ``get_action(temperature=)``.
    """

    def __init__(
        self,
        *,
        head: BaseHead,
        num_backbone_layers: int,
        discount: Discount | None,
        discount_start: Discount | None,
        reward: Reward | None,
        value: Value | None,
        temperature: float,
        action_key: str = "action",
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
        grouping_field: str | None,
        td_lambda: float = 0.0,
        watkins: bool = False,
    ) -> None:
        if not 0.0 <= float(td_lambda) <= 1.0:
            raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}.")
        self.head = require_head(head=head, what="head")
        self.temperature = _require_temperature(temperature)
        self.num_backbone_layers = int(num_backbone_layers)
        self.discount = _require_transform(discount, name="discount")
        self.discount_start = _require_transform(discount_start, name="discount_start")
        self.reward = _require_transform(reward, name="reward")
        self.value = _require_transform(value, name="value")
        self.action_key = action_key
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.grouping_field = grouping_field
        self.td_lambda = float(td_lambda)
        self.watkins = bool(watkins)

        n = self.num_backbone_layers
        grid_episode, grid_task = _done_code_grid()
        ones = torch.ones(grid_episode.shape, dtype=torch.float32)
        start_grid = (
            ones
            if self.discount_start is None
            else self.discount_start(
                episode_done=grid_episode, task_done=grid_task
            )
        )
        deep_grid = (
            ones
            if self.discount is None
            else self.discount(episode_done=grid_episode, task_done=grid_task)
        )
        if n == 1:
            if not torch.equal(start_grid, deep_grid):
                raise ValueError(
                    f"num_backbone_layers=1 cannot interpolate between "
                    f"discount_start and discount; pass callables that "
                    f"agree on every episode_done / task_done pair."
                )
        else:
            _horizon_lerp(gamma_start=start_grid, gamma_deep=deep_grid, t=0.5)

        build = _build_layer_gamma_schedule
        self.layer_gamma_step = build(
            num_layers=n,
            gamma_start=_probe_discount(
                discount=self.discount_start, episode_done=0, task_done=0
            ),
            gamma_deep=_probe_discount(
                discount=self.discount, episode_done=0, task_done=0
            ),
        )
        self.layer_gamma_episode_terminal = build(
            num_layers=n,
            gamma_start=_probe_discount(
                discount=self.discount_start, episode_done=1, task_done=0
            ),
            gamma_deep=_probe_discount(
                discount=self.discount, episode_done=1, task_done=0
            ),
        )

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        delayed_predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("LayerwiseDqnObjective requires delayed_predictions.")
        q: torch.Tensor = predictions_for(head=self.head, predictions=predictions, who="Layerwise DQN")
        q_target: torch.Tensor = predictions_for(
            head=self.head, predictions=delayed_predictions, who="Layerwise DQN delayed"
        ).detach()

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
        _require_action_ids(action, A)

        if N < 2:
            raise ValueError("Not enough valid q values in data.")

        reward = _apply_transform(
            transform=self.reward,
            name="reward",
            objective_data=objective_data,
            N=N,
            dtype=value_dtype,
            device=device,
            identity=objective_data["reward"],
        )

        _require_done_codes(
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
        q = _apply_value(
            transform=self.value,
            name="value",
            value=q,
            objective_data=objective_data,
            step_of=step_of,
            N=N,
        )
        q_target = _apply_value(
            transform=self.value,
            name="value",
            value=q_target,
            objective_data=objective_data,
            step_of=step_of,
            N=N,
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
        v_step_all = _soft_state_value(
            q_target[last_rows], temperature=self.temperature
        )  # [N, L]  V_l(s_i): max_a Q, or α logsumexp

        ones = torch.ones(N, dtype=value_dtype, device=device)
        gamma_start = _apply_transform(
            transform=self.discount_start,
            name="discount_start",
            objective_data=objective_data,
            N=N,
            dtype=value_dtype,
            device=device,
            identity=ones,
        )
        gamma_deep = _apply_transform(
            transform=self.discount,
            name="discount",
            objective_data=objective_data,
            N=N,
            dtype=value_dtype,
            device=device,
            identity=ones,
        )

        layer_losses: list[torch.Tensor] = []
        layer_curr_max_means: list[torch.Tensor] = []
        cql_penalties: list[torch.Tensor] = []
        deepest_curr_max_q: torch.Tensor | None = None
        deepest_greedy_from: torch.Tensor | None = None

        for layer_idx in range(L):
            t = 0.0 if L == 1 else layer_idx / (L - 1)
            discount_all = _horizon_lerp(
                gamma_start=gamma_start, gamma_deep=gamma_deep, t=t
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
        if self.temperature > 0.0:
            named["entropy"] = _weighted_mean(
                _boltzmann_entropy(
                    q[:, -1, :].detach(), temperature=self.temperature
                ),
                row_weight,
            )

        metrics: dict[str, float] = {
            key: (value.item() if value.numel() == 1 else float(value))
            for key, value in named.items()
        }
        return total_loss, metrics
