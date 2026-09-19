"""Fixed-horizon n-step DQN objective with a delayed target network."""

from __future__ import annotations

import torch

from mouse_core.models.heads.base import BaseHead, prediction_key
from mouse_core.objectives.base import Objective, predictions_for, require_head
from mouse_core.objectives.dqn import (
    _boltzmann_entropy,
    _head_output_layout,
    _in_run_stats,
    _pair_values_to_rows,
    _pair_weight,
    _require_action_ids,
    _require_done_codes,
    _require_temperature,
    _soft_state_value,
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


@torch.no_grad()
def _n_step_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    v_step: torch.Tensor,
    pair_weight: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """n-step target for every pair ``(t, t+1)``, shape ``[N-1]``.

    ``G_t^{(n)} = r_{t+1} + γ_{t+1} r_{t+2} + … + (∏_{k=1}^{n-1} γ_{t+k}) r_{t+n}
                  + (∏_{k=1}^{n} γ_{t+k}) V(s_{t+n})``

    The product of ``γ`` is the product of per-transition discounts along
    the window. The window truncates (bootstraps at the last in-run next
    state) at a run break or the end of the batch. Out-of-run first pairs
    return ``0`` (their rows carry weight ``0``). ``n=1`` is the one-step
    target ``r + γ V``. Episode / task boundaries are handled by ``γ``
    itself — a ``0`` gamma zeros the remaining rewards and the bootstrap.
    """
    dtype = v_step.dtype
    device = v_step.device
    N = int(reward.shape[0])
    T = N - 1
    r_all = reward.to(dtype=dtype)
    in_run = pair_weight > 0

    t = torch.arange(T, device=device)
    offsets = torch.arange(1, n + 1, device=device)
    idx = t.unsqueeze(1) + offsets.unsqueeze(0)  # [T, n]  t+k
    in_batch = idx < N
    idx_c = idx.clamp(max=N - 1)

    r_win = r_all[idx_c]
    g_win = discount_all[idx_c]
    v_win = v_step[idx_c]

    pair_idx = idx - 1  # t+k-1
    pair_ok = (pair_idx < (N - 1)) & in_run[pair_idx.clamp(max=N - 2)]
    usable = (in_batch & pair_ok).to(dtype=torch.int64).cumprod(dim=1).to(
        dtype=torch.bool
    )

    ones = g_win.new_ones(g_win.shape)
    g_for_prod = torch.where(usable, g_win, ones)
    gamma_after = g_for_prod.cumprod(dim=1)
    gamma_before = torch.cat([ones[:, :1], gamma_after[:, :-1]], dim=1)

    reward_term = (usable.to(dtype=dtype) * gamma_before * r_win).sum(dim=1)
    n_used = usable.to(dtype=torch.int64).sum(dim=1)
    last_idx = (n_used - 1).clamp(min=0).unsqueeze(1)
    last_gamma = gamma_after.gather(dim=1, index=last_idx).squeeze(1)
    last_v = v_win.gather(dim=1, index=last_idx).squeeze(1)
    bootstrap = (n_used > 0).to(dtype=dtype) * last_gamma * last_v
    return (reward_term + bootstrap) * in_run.to(dtype=dtype)


class NStepDqnObjective(Objective):
    """Bellman n-step objective with a delayed target network.

    Instantiate with hyperparameters, then call with
    ``objective_data=``, ``predictions=``, and ``delayed_predictions=``. Online Q is
    the tensor for ``head``; bootstrap Q is the same key on
    ``delayed_predictions`` from the delayed
    :class:`~mouse_core.models.base.Model`
    (``model.copy(heads=(head,))``) run on the same
    ``TokenBatch``. The delayed tensor is detached before the Bellman
    target, so the TD error does not backprop through it.

    One objective trains one Q head. To train several horizons, build
    one :class:`NStepDqnObjective` per head (each with its own ``n`` and
    ``head``) and add the returned losses. ``n=1`` is the
    one-step target ``r + γ V``; larger ``n`` sums that many discounted
    rewards and bootstraps from the delayed state value at ``s_{t+n}``.
    ``V`` is delayed max-Q when ``temperature=0``; a positive
    ``temperature`` (SAC / soft Q-learning ``α``) is
    ``α logsumexp(Q / α)``, the same bootstrap as
    :class:`~mouse_core.objectives.dqn.DqnObjective`. A window
    that hits a run break or the end of the batch truncates and bootstraps
    at the last in-run next state.

    Q rows are **per head-output token** (``[P, A]``), not per step: a step
    may own several head-output tokens (tokenizer input field flagged
    ``head_output=True`` emitting more than one token). The
    ``head_output_count`` column stamped by ``pack_token_batch`` maps rows
    to steps. Every head-output row of step ``i`` trains toward the *same*
    n-step target; the bootstrap reads the *last* head-output row of the
    bootstrap step.

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is
    set and present, the same grouping column (typically ``task_index``).
    Neighbor reads must stay in-run: an out-of-run pair still has a loss
    term, but it is multiplied by ``0``. If every weight is ``0`` the loss
    is ``0``. Episode resets inside a run (``episode_done`` 1/2, then a
    reset frame) are still in-run and may train. Gamma is the Bellman
    discount from the done codes at each lookahead step inside a same-run
    pair — it is not a run mask.

    ``reward(**objective_data)`` supplies the per-step reward used at
    each lookahead. ``affine_reward`` is the column affine;
    ``boundary_reward`` applies episode / task scale and shift extras.
    ``value(value=..., **objective_data)`` supplies the per-step affine
    on online and delayed Q. ``affine_value`` is the prediction affine;
    ``boundary_value`` applies episode / task scale and shift extras.
    ``discount(**objective_data)``
    supplies the per-step γ that multiplies later rewards and the
    bootstrap.
    ``boundary_discount`` is the standard ``gamma_step`` × episode-extra
    × task-extra lookup. A ``0`` gamma
    ends the remaining sum; a non-zero truncation gamma carries it
    (discounted) into the reset frame.

    Those columns arrive in ``objective_data`` only if they are listed in
    the tokenizer ``objective_fields`` keep-list (input fields are not
    auto-copied). ``task_done`` is an objective column only — do not add
    it as a tokenizer input field, or it will be fed
    to the transformer::

        tokenizer = Tokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "episode_done"},
                {"input_field": "task_done"},
            ],
        )

    Args:
        n: Backup horizon. ``1`` is one-step DQN; must be an ``int >= 1``.
        head: Q head this objective trains. Must be the same instance
            passed to ``Model(heads=)``.
        discount: Per-step γ from unpacked ``objective_data`` columns.
            ``boundary_discount`` is the standard ``gamma_step`` × extra
            lookup (``None`` skips the call and uses ``1``);
            any ``discount(**objective_data) -> [N]`` is accepted.
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
        action_key: Key in ``objective_data`` that holds the integer action.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        grouping_field: Step column that isolates runs (typically
            ``task_index``). Required. Pass ``None`` only when the batch
            has no grouping isolation — omitting it is an error, not a
            silent skip.
        cql_weight: Alpha coefficient for the Conservative Q-Learning
            penalty. ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.
        temperature: SAC / soft Q-learning ``α`` (``>= 0``). Required.
            ``0`` is hard max-Q. ``> 0`` bootstraps from
            ``α logsumexp(Q / α)`` on delayed Q (after ``value``) and
            logs ``metrics["entropy"]``. Same units and meaning as
            ``get_action(temperature=)``.
    """

    def __init__(
        self,
        *,
        n: int,
        head: BaseHead,
        discount: Discount | None,
        reward: Reward | None,
        value: Value | None,
        temperature: float,
        action_key: str = "action",
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        grouping_field: str | None,
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n must be an int >= 1, got {n!r}.")
        self.n = n
        self.head = require_head(head=head, what="head")
        self.prediction_key = prediction_key(head=self.head)
        self.discount = _require_transform(discount, name="discount")
        self.reward = _require_transform(reward, name="reward")
        self.value = _require_transform(value, name="value")
        self.action_key = action_key
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.grouping_field = grouping_field
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.temperature = _require_temperature(temperature)

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: dict[str, torch.Tensor],
        delayed_predictions: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("NStepDqnObjective requires delayed_predictions.")
        q: torch.Tensor = predictions_for(head=self.head, predictions=predictions, who="n-step DQN")
        q_target: torch.Tensor = predictions_for(
            head=self.head, predictions=delayed_predictions, who="n-step DQN delayed"
        ).detach()

        if q.ndim != 2:
            raise ValueError(
                f"n-step DQN expects {self.prediction_key} shape [P, A], "
                f"got {tuple(q.shape)}."
            )
        if q.dtype != torch.float32 or q_target.dtype != torch.float32:
            raise TypeError(
                "n-step DQN expects float32 Q (heads always run in fp32), got "
                f"online {q.dtype} and delayed {q_target.dtype}."
            )
        if q_target.shape != q.shape:
            raise ValueError(
                f"n-step DQN delayed {self.prediction_key} shape "
                f"{tuple(q_target.shape)} must match online shape {tuple(q.shape)}."
            )
        P, A = q.shape
        device = q.device
        value_dtype = q.dtype

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"n-step DQN objective expects action shape [N], "
                f"got {tuple(action.shape)}."
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
        row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]

        step_next = (step_of + 1).clamp(max=N - 1)
        next_actions = action[step_next]

        discount_all = _apply_transform(
            transform=self.discount,
            name="discount",
            objective_data=objective_data,
            N=N,
            dtype=value_dtype,
            device=device,
            identity=torch.ones(N, dtype=value_dtype, device=device),
        )

        q_values = q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)
        pair_target = _n_step_targets(
            reward=reward,
            discount_all=discount_all,
            v_step=_soft_state_value(
                q_target[last_rows], temperature=self.temperature
            ),
            pair_weight=pair_weight,
            n=self.n,
        )
        td_target = _pair_values_to_rows(pair_target, step_of)

        loss = (q_values - td_target) ** 2

        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty = torch.logsumexp(q, dim=-1) - q_values
            loss = loss + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = _weighted_mean(cql_penalty.detach(), row_weight)

        loss = _weighted_mean(loss, row_weight)

        curr_max_q = q.amax(dim=-1)
        q_mean, q_std, q_min, q_max = _in_run_stats(curr_max_q.detach(), row_weight)
        named: dict[str, torch.Tensor] = {
            "q_values_mean": q_mean,
            "q_values_std": q_std,
            "q_values_min": q_min,
            "q_values_max": q_max,
            "n_step": loss.detach(),
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean
        if self.temperature > 0.0:
            named["entropy"] = _weighted_mean(
                _boltzmann_entropy(q.detach(), temperature=self.temperature),
                row_weight,
            )

        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
