"""Retrace(λ) DQN objective with a delayed target network and a learned behavior head.

Munos, Stepleton, Harutyunyan, Bellemare. *Safe and efficient off-policy
reinforcement learning* (2016). https://arxiv.org/abs/1606.02647
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.models.heads.base import BaseHead
from mouse_core.objectives.base import Objective, predictions_for, require_head
from mouse_core.objectives.dqn import (
    _affine,
    _affine_scan_backward,
    _boltzmann_entropy,
    _boundary_discounts,
    _head_output_layout,
    _in_run_stats,
    _pair_values_to_rows,
    _pair_weight,
    _policy_entropy,
    _require_action_ids,
    _require_done_codes,
    _require_temperature,
    _shift_next,
    _weighted_mean,
)


def _require_aligned(
    predictions: TensorDict, *, head: BaseHead, shape: torch.Size, who: str
) -> torch.Tensor:
    """Validate a float32 ``[P, A]`` head output that must align with Q."""
    values = predictions_for(head=head, predictions=predictions, who=who)
    if values.dtype != torch.float32:
        raise TypeError(f"{who} expects float32 head outputs, got {values.dtype}.")
    if values.shape != shape:
        raise ValueError(
            f"{who} expects head shape {tuple(shape)} (same rows and actions as "
            f"the Q head), got {tuple(values.shape)}."
        )
    return values


def _softmax_policy(q: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Target policy over the last dim of ``q``, shape ``q.shape``.

    ``q`` is logits: ``softmax(q / temperature)``. Same convention as
    :meth:`~mouse_core.models.base.Model.get_action`.
    ``temperature = 0`` is the greedy policy with the argmax set sharing
    the mass evenly.
    """
    if float(temperature) == 0.0:
        is_max = q == q.amax(dim=-1, keepdim=True)
        return is_max.to(dtype=q.dtype) / is_max.sum(dim=-1, keepdim=True).to(
            dtype=q.dtype
        )
    return F.softmax(q / float(temperature), dim=-1)


@torch.no_grad()
def _retrace_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    q_step: torch.Tensor,
    pi_step: torch.Tensor,
    mu_step: torch.Tensor,
    action: torch.Tensor,
    pair_weight: torch.Tensor,
    td_lambda: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrace(λ) target for every pair ``(t, t+1)`` and its trace ratio.

    ``q_step`` / ``pi_step`` / ``mu_step`` are ``[N, A]`` per-step reads at
    the last head-output row: delayed Q, the softmax target policy over it,
    and the learned behavior distribution. Returns ``(G [N-1], ratio [N-1])``::

        G_t = r_t + γ_t * ( V_π(s_{t+1})
                            + c_{t+1} * (G_{t+1} - Q(s_{t+1}, a_{t+1})) )
        V_π = E_π Q + α H[π]
        c_{t+1} = λ * min(1, π(a_{t+1} | s_{t+1}) / μ(a_{t+1} | s_{t+1}))

    which is the paper's ``Q(x_t, a_t) + Σ_s γ^{s-t} (Π c_i) δ_s`` written
    as a backward recursion. ``ratio[t]`` is ``min(1, π/μ)`` for the action
    taken *from* ``s_t`` (before λ and before the continuation mask); it is
    reported as a metric. ``c_{t+1}`` is ``0`` when pair ``t+1`` does not
    exist or is out-of-run, so the trace never crosses a run break.
    Episode / task boundaries are handled by ``γ_t`` itself: a ``0`` gamma
    ends the trace and a non-zero truncation gamma carries it, discounted.
    ``λ = 0`` is the expected one-step target ``r + γ V_π``.
    ``temperature`` is the SAC ``α`` on that same ``π``; ``0`` is
    greedy ``π`` and ``V_π = E_π Q``.
    Out-of-run pairs return ``0`` (their rows carry weight ``0``).
    """
    dtype = q_step.dtype
    r = reward[1:].to(dtype=dtype)  # [N-1]  r_t (stored at t+1)
    g = discount_all[1:]  # [N-1]  γ_t from done codes at t+1
    a_taken = action[1:].unsqueeze(-1)  # [N-1, 1]  a_t (stored at t+1)

    # Per-pair reads at s_t for the action taken from it.
    pi_taken = pi_step[:-1].gather(dim=-1, index=a_taken).squeeze(-1)  # [N-1]
    mu_taken = mu_step[:-1].gather(dim=-1, index=a_taken).squeeze(-1)  # [N-1]
    q_taken = q_step[:-1].gather(dim=-1, index=a_taken).squeeze(-1)  # [N-1]
    # min(1, π/μ) as π / max(π, μ): no division by a vanishing μ, and 0 when π = 0.
    ratio = pi_taken / torch.maximum(pi_taken, mu_taken).clamp_min(
        torch.finfo(dtype).tiny
    )

    v_pi = (pi_step * q_step).sum(dim=-1)  # [N]  E_π Q(s_i, ·)
    if float(temperature) > 0.0:
        v_pi = v_pi + float(temperature) * _policy_entropy(pi_step)
    v_next = v_pi[1:]  # [N-1]  V_π(s_{t+1})

    in_run = pair_weight > 0
    cont = _shift_next(in_run.to(dtype=dtype))  # pair t+1 exists and is in-run
    c_next = float(td_lambda) * _shift_next(ratio) * cont  # [N-1]  c_{t+1}
    q_next_taken = _shift_next(q_taken)  # [N-1]  Q(s_{t+1}, a_{t+1})

    a = r + g * (v_next - c_next * q_next_taken)
    b = g * c_next
    if td_lambda == 0.0:
        returns = a
    else:
        returns = _affine_scan_backward(a, b)
    return returns * in_run.to(dtype=returns.dtype), ratio


class RetraceObjective(Objective):
    """Retrace(λ) objective with a delayed target network and a learned behavior head.

    Off-policy, return-based Q-learning from Munos et al. (2016). The TD
    target of every transition is the delayed one-step expected backup plus
    a trace of later TD errors, each scaled by the product of truncated
    importance ratios ``c_s = λ min(1, π(a_s|s_s) / μ(a_s|s_s))``. ``π`` is
    the target policy — ``softmax(Q / temperature)`` over the delayed
    Q (Q as logits), the same convention as
    :meth:`~mouse_core.models.base.Model.get_action`
    — and ``μ`` is the behavior policy that produced the data. Because the
    ratio is clipped at ``1``, near-on-policy transitions keep the full
    λ-return while strongly off-policy actions cut the trace, without
    importance-weight variance. A lower temperature is a greedier ``π``
    (the paper's increasingly-greedy sequence); ``temperature=0`` is the
    greedy policy, which is Watkins's Q(λ) with the greedy check on the
    delayed network.

    The dataset does not store ``μ``. It is **learned**: the model carries a
    second head, a :class:`~mouse_core.models.heads.ClassificationHead`
    (``behavior_head``) whose outputs are treated as
    **logits**: ``log_softmax`` of the row is ``log μ(· | s)``. The head is fit
    by negative log-likelihood of the action actually taken from each step,
    ``-log μ(a_t | s_t)``, over every head-output row of that step. The same
    distribution at the step's last head-output row is the ``μ(· | s_t)``
    the trace uses, read from the *online*
    predictions and detached, so the trace coefficients are the current best
    estimate of the data policy and receive no TD gradient. The returned
    loss is ``td_loss + behavior_weight * behavior_loss``; in-context, the
    behavior head sees the same history as the Q head and can track a
    behavior policy that changes along the run.

    Instantiate with hyperparameters, then call with
    ``objective_data=``, ``predictions=``, and ``delayed_predictions=``. Online Q is
    the tensor for ``head``; every target quantity — the soft
    bootstrap ``V_π(s') = E_π Q + temperature H[π]``, the corrected
    ``Q(s', a')``, and ``π`` itself — is read from
    the delayed tensor for ``head`` of the
    delayed :class:`~mouse_core.models.base.Model`
    (``model.delayed_copy(heads=(head,))``) run on the same
    ``TokenBatch``. The delayed tensor is detached, so the TD error does not
    backprop through it. The behavior head is not part of the delayed model:
    nothing reads its delayed values, so it is neither run there nor
    Polyak-interpolated — ``μ`` comes from the online head only.

    Head rows are **per head-output token** (``[P, A]``), not per step: a
    step may own several head-output tokens. The ``head_output_count``
    column stamped by ``pack_token_batch`` maps rows to steps; every
    head-output row of step ``i`` trains toward the same TD target and the
    same taken action, and the per-step reads (delayed Q, ``π``, ``μ``) use
    the *last* head-output row of each step.

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is set
    and present, the same grouping column (typically ``task_index``). The
    trace never crosses a run break; out-of-run pairs carry weight ``0`` in
    both losses. Episode resets inside a run are still in-run: the done-code
    gamma at ``i+1`` multiplies both the bootstrap and the continued trace,
    so a ``0`` gamma ends the trace there and a non-zero truncation gamma
    carries it (discounted) into the reset frame's return.

    The target along a run is::

        G_i = r_i + γ_i * ( V_π(s_{i+1})
                            + c_{i+1} * (G_{i+1} - Q(s_{i+1}, a_{i+1})) )

    with ``V_π = E_π Q + temperature H[π]``. ``temperature=0`` is the
    paper's greedy expected backup; a positive value is the SAC soft
    value on ``π = softmax(Q / temperature)``. So ``td_lambda=0`` is
    the expected one-step target and ``td_lambda=1``
    with an on-policy action (``π ≥ μ``) is the full in-run return. The
    paper's Atari runs use ``λ = 1`` with the exploration policy as ``π``;
    the clipped ratio does the trace cutting that ``Q*(λ)`` needs ``λ < 1``
    for. The λ-return is computed with a parallel scan on the device.
    ``π`` is ``softmax(Q / temperature)`` over the delayed head's raw
    Q (before ``q_scale`` / ``q_shift``), so ``temperature`` is in the
    units the head outputs and means the same thing here as in
    ``get_action(temperature=)``.

    Model construction pairs the two heads under caller-chosen keys, with
    ``get_action`` reading the Q head::

        model = Model(
            ...,
            heads={"action_value": q_head, "behavior": behavior_head},
            action_source=q_head,
        )
        delayed_model = model.delayed_copy(heads=(q_head,))

    Discounts follow the ``DqnObjective`` done-code table: the bootstrap and
    the continued trace are multiplied by the episode gamma
    (``episode_done`` ``0`` / ``1`` / ``2``), then by the task gamma
    (``task_done`` ``0`` uses ``1.0``). The objective columns are the
    ``DqnObjective`` ones (``action`` / ``reward`` / ``episode_done`` /
    ``task_done``); nothing about ``μ`` is stored.

    Args:
        td_lambda: λ of the trace in ``[0, 1]``. ``0.0`` is the expected
            one-step target; ``1.0`` cuts traces only through
            ``min(1, π/μ)``.
        temperature: Softmax temperature of the target policy
            ``π = softmax(Q / temperature)`` and the SAC ``α`` on the
            backup ``V_π = E_π Q + α H[π]``, ``>= 0``. Q is logits.
            ``0.0`` is greedy (Watkins's cut; argmax ties share the
            mass; ``V_π = E_π Q``); larger values flatten ``π`` toward
            uniform, cut fewer traces, and raise the soft value. Same
            units and meaning as ``get_action(temperature=)``.
        behavior_weight: Coefficient of the behavior head's NLL
            (``-log μ(a_t | s_t)``) in the returned loss. Must be ``>= 0``.
            ``0.0`` excludes the NLL from the returned loss; the head is
            still required and its detached softmax is still ``μ`` for
            the trace.
        gamma_step: Discount factor for running (non-terminal) transitions
            (``episode_done == 0``).
        gamma_episode_terminal: Discount applied when the episode terminates
            naturally (``episode_done == 1``). ``1.0`` bootstraps across
            episode boundaries (usual for multi-episode MOUSE tasks).
        gamma_episode_truncated: Discount applied when the episode is truncated
            (``episode_done == 2``). ``1.0`` bootstraps across episode
            boundaries.
        gamma_task_terminal: Extra discount when the task terminates
            (``task_done == 1``; ``EnvConfig.terminate_task``).
            Multiplies the episode discount. ``task_done == 0`` uses ``1.0``.
        gamma_task_truncated: Extra discount when the task is truncated
            (``task_done == 2``; last episode of ``max_task_episodes``).
            Multiplies the episode discount. ``0.0`` zeros the bootstrap.
        head: Q head this objective trains. Must be the same instance
            passed to ``Model(heads=)``.
        behavior_head: Behavior-policy head whose logits are ``μ``.
            Must be the same instance passed to ``Model(heads=)``.
        action_key: Key in ``objective_data`` that holds the integer action.
        reward_key: Key in ``objective_data`` that holds the per-step reward.
        reward_scale: Multiplier applied to ``reward`` before the target
            (default ``1.0``).
        reward_shift: Offset added after ``reward_scale`` (default ``0.0``).
        q_scale: Multiplier applied to online and delayed ``action_value``
            before the TD error (default ``1.0``). Same affine on both
            networks; ``π`` is taken over the raw Q and is unchanged by it.
        q_shift: Offset added after ``q_scale`` (default ``0.0``).
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        grouping_field: Step column that isolates runs (typically
            ``task_index``). Required. Pass ``None`` only when the batch
            has no grouping isolation.
        cql_weight: Alpha coefficient for the Conservative Q-Learning penalty
            on the Q head. ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.

    Metrics: ``retrace`` (the returned loss), ``td_loss`` (Q-head MSE
    including CQL), ``behavior_loss`` (behavior-head NLL, ``-log μ(a_t | s_t)``),
    ``behavior_prob_mean`` (in-run mean of ``μ(a_t | s_t)`` for the taken
    action — how well the behavior head predicts the data), ``q_values_*``
    (online max-Q over in-run rows), ``retrace_ratio_mean`` (in-run mean of
    ``min(1, π/μ)`` for the taken action — ``1`` is on-policy, near ``0``
    means the traces are cut everywhere), ``entropy`` when
    ``temperature > 0``, and ``cql_penalty`` when CQL is on.
    """

    def __init__(
        self,
        *,
        td_lambda: float,
        temperature: float,
        behavior_weight: float,
        gamma_step: float,
        gamma_episode_terminal: float,
        gamma_episode_truncated: float,
        gamma_task_terminal: float,
        gamma_task_truncated: float,
        head: BaseHead,
        behavior_head: BaseHead,
        action_key: str = "action",
        reward_key: str = "reward",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        q_scale: float = 1.0,
        q_shift: float = 0.0,
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        grouping_field: str | None,
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        if not 0.0 <= float(td_lambda) <= 1.0:
            raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}.")
        if not float(behavior_weight) >= 0.0:
            raise ValueError(
                f"behavior_weight must be >= 0, got {behavior_weight}."
            )
        self.td_lambda = float(td_lambda)
        self.temperature = _require_temperature(temperature)
        self.behavior_weight = float(behavior_weight)
        self.gamma_step = gamma_step
        self.gamma_episode_terminal = gamma_episode_terminal
        self.gamma_episode_truncated = gamma_episode_truncated
        self.gamma_task_terminal = gamma_task_terminal
        self.gamma_task_truncated = gamma_task_truncated
        self.head = require_head(head=head, what="head")
        self.behavior_head = require_head(head=behavior_head, what="behavior_head")
        self.action_key = action_key
        self.reward_key = reward_key
        self.reward_scale = float(reward_scale)
        self.reward_shift = float(reward_shift)
        self.q_scale = float(q_scale)
        self.q_shift = float(q_shift)
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.grouping_field = grouping_field
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps

    def __call__(
        self,
        *,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("RetraceObjective requires delayed_predictions.")
        q: torch.Tensor = predictions_for(head=self.head, predictions=predictions, who="Retrace")
        q_target: torch.Tensor = predictions_for(
            head=self.head, predictions=delayed_predictions, who="Retrace delayed"
        ).detach()

        if q.ndim != 2:
            raise ValueError(
                f"Retrace expects action_value shape [P, A], got {tuple(q.shape)}."
            )
        if q.dtype != torch.float32 or q_target.dtype != torch.float32:
            raise TypeError(
                "Retrace expects float32 action_value (heads always run in fp32), "
                f"got online {q.dtype} and delayed {q_target.dtype}."
            )
        if q_target.shape != q.shape:
            raise ValueError(
                f"Retrace delayed action_value shape {tuple(q_target.shape)} must "
                f"match online shape {tuple(q.shape)}."
            )
        behavior_logits = _require_aligned(
            predictions, head=self.behavior_head, shape=q.shape, who="Retrace"
        )
        q_target_raw = q_target  # π is taken over the head's own Q (get_action units)
        q = _affine(q, scale=self.q_scale, shift=self.q_shift)
        q_target = _affine(q_target, scale=self.q_scale, shift=self.q_shift)
        P, A = q.shape
        device = q.device
        value_dtype = q.dtype

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.ndim != 1:
            raise ValueError(
                f"Retrace objective expects action shape [N], got {tuple(action.shape)}."
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
                f"Retrace objective expects reward shape [{N}], got {tuple(reward.shape)}."
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
        row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]  # [P]

        # The action stored at i+1 is the one taken *from* obs_i.
        step_next = (step_of + 1).clamp(max=N - 1)  # [P]
        next_actions = action[step_next]  # [P]  a_i

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

        # Behavior head outputs logits; log_softmax is log μ(· | s). Every row of
        # step i is fit to the taken action a_i by NLL, and the same
        # (detached) distribution at the step's last row is the μ the trace uses.
        log_mu = F.log_softmax(behavior_logits, dim=-1)  # [P, A]
        behavior_nll = F.nll_loss(log_mu, next_actions, reduction="none")  # [P]  -log μ(a_i | s_i)
        behavior_loss = _weighted_mean(behavior_nll, row_weight)
        mu_step = log_mu.detach().exp()[last_rows]  # [N, A]  μ(· | s_i)

        q_values = q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [P]
        q_step = q_target[last_rows]  # [N, A]  delayed Q(s_i, ·)
        pi_step = _softmax_policy(
            q_target_raw[last_rows], temperature=self.temperature
        )  # [N, A]  π(· | s_i)
        pair_target, ratio = _retrace_targets(
            reward=reward,
            discount_all=discount_all,
            q_step=q_step,
            pi_step=pi_step,
            mu_step=mu_step,
            action=action,
            pair_weight=pair_weight,
            td_lambda=self.td_lambda,
            temperature=self.temperature,
        )
        td_target = _pair_values_to_rows(pair_target, step_of)  # [P]

        td_loss = (q_values - td_target) ** 2

        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty = torch.logsumexp(q, dim=-1) - q_values
            td_loss = td_loss + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = _weighted_mean(cql_penalty.detach(), row_weight)

        td_loss = _weighted_mean(td_loss, row_weight)
        loss = td_loss + self.behavior_weight * behavior_loss

        mu_taken = (
            mu_step[:-1].gather(dim=-1, index=action[1:].unsqueeze(-1)).squeeze(-1)
        )  # [N-1]  μ(a_t | s_t)
        curr_max_q = q.amax(dim=-1)  # [P]  max online Q at s_i
        q_mean, q_std, q_min, q_max = _in_run_stats(curr_max_q.detach(), row_weight)
        named: dict[str, torch.Tensor] = {
            "retrace": loss.detach(),
            "td_loss": td_loss.detach(),
            "behavior_loss": behavior_loss.detach(),
            "behavior_prob_mean": _weighted_mean(mu_taken, pair_weight),
            "q_values_mean": q_mean,
            "q_values_std": q_std,
            "q_values_min": q_min,
            "q_values_max": q_max,
            "retrace_ratio_mean": _weighted_mean(ratio, pair_weight),
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
