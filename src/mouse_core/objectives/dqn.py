"""TD(λ) DQN objective with a delayed target network."""

from __future__ import annotations

from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction
from mouse_core.objectives.transforms import (
    Discount,
    Gate,
    Reward,
    Value,
    _apply_transform,
    _apply_value,
    _as_flag,
    _require_transform,
)


def _require_done_flags(
    objective_data: dict[str, torch.Tensor],
    *,
    terminated_key: str,
    truncated_key: str,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and validate mouse-gym ``terminated`` / ``truncated`` columns."""
    flags: list[torch.Tensor] = []
    for key in (terminated_key, truncated_key):
        values = objective_data[key]
        if values.shape != torch.Size([N]):
            raise ValueError(
                f"objective expects {key} shape [{N}], got {tuple(values.shape)}."
            )
        flags.append(_as_flag(values, name=key).to(dtype=torch.int64))
    return flags[0], flags[1]


def _head_output_layout(
    objective_data: dict[str, torch.Tensor],
    *,
    N: int,
    P: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map head-output rows to steps: ``(step_of [P], last_rows [N])``.

    ``step_of[p]`` is the step of head-output row ``p``; ``last_rows[i]`` is
    the row of step ``i``'s last head-output token (the bootstrap read). Reads
    the ``head_output_count`` column stamped by ``pack_token_batch``; when the
    column is absent every step must have exactly one head-output token (``P == N``).
    """
    if "head_output_count" in objective_data.keys():
        counts = objective_data["head_output_count"]
        if counts.dtype != torch.int64:
            raise TypeError(f"head_output_count must be int64, got {counts.dtype}.")
        if counts.shape != torch.Size([N]):
            raise ValueError(
                f"head_output_count must have shape [{N}], got {tuple(counts.shape)}."
            )
        if bool((counts < 1).any()):
            raise ValueError("head_output_count entries must be >= 1.")
        if int(counts.sum()) != P:
            raise ValueError(
                f"head_output_count sums to {int(counts.sum())} but predictions "
                f"have {P} rows; predictions and objective_data are misaligned."
            )
        counts = counts.to(device=device)
    else:
        if P != N:
            raise ValueError(
                f"predictions have {P} rows for {N} steps but objective_data "
                "has no head_output_count column; pack with pack_token_batch "
                "or provide head_output_count."
            )
        counts = torch.ones(N, dtype=torch.int64, device=device)
    step_of = torch.repeat_interleave(
        torch.arange(N, dtype=torch.int64, device=device), counts
    )
    last_rows = counts.cumsum(dim=0) - 1
    return step_of, last_rows


def _pair_weight(
    *,
    group_id: torch.Tensor,
    N: int,
    device: torch.device | str | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[N-1]`` weights: ``1.0`` when ``(i, i+1)`` share a sample, else ``0.0``.

    A sample is one ``group_id``. The dataloader gives each sample it
    keeps in a batch a distinct id, so a group change is the sample
    boundary. ``group_id`` is int64 ``[N]``, one id per step, passed
    beside ``objective_data``. ``N < 2`` raises.
    """
    if device is None:
        device = torch.device("cpu")
    if N < 2:
        raise ValueError(f"pair weight needs at least 2 steps, got {N}.")
    if group_id.shape != torch.Size([N]):
        raise ValueError(
            f"group_id must have shape [{N}], got {tuple(group_id.shape)}."
        )
    same_run = group_id[1:] == group_id[:-1]
    return same_run.to(device=device, dtype=dtype)


def _require_action_ids(action: torch.Tensor, A: int) -> None:
    """Raise unless every action id is in ``[0, A)``."""
    if bool((action < 0).any() or (action >= A).any()):
        raise ValueError(
            f"action ids must be in [0, {A}), got min={int(action.min())} "
            f"max={int(action.max())}."
        )


def _require_step_aligned_predictions(
    objective_data: dict[str, torch.Tensor],
    *,
    n_pred: int,
    n_steps: int,
    who: str,
) -> None:
    """PPO / GRPO read one prediction row per step.

    DQN-family objectives expand multi-token steps via ``head_output_count``.
    Policy objectives do not: more than one head-output token per step is an
    alignment error, not a silent gather of the wrong rows.
    """
    if "head_output_count" not in objective_data.keys():
        return
    counts = objective_data["head_output_count"]
    total = int(counts.sum())
    if counts.shape != torch.Size([n_steps]) or total != n_pred or n_pred != n_steps:
        raise ValueError(
            f"{who} expects one prediction row per step "
            f"(got {n_pred} prediction rows, {n_steps} steps, "
            f"head_output_count sum {total}). Use a single head-output token "
            "per step, or a DQN-family objective."
        )


def _weighted_mean(values: torch.Tensor, pair_weight: torch.Tensor) -> torch.Tensor:
    """``(w * x).sum() / w.sum().clamp(min=1)``. All-zero weights yield ``0``."""
    weight = pair_weight.to(dtype=values.dtype)
    return (weight * values).sum() / weight.sum().clamp(min=1)


def _in_run_stats(
    values: torch.Tensor,
    pair_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean / std / min / max over in-run pairs. All-zero weights yield zeros."""
    in_run = pair_weight > 0
    zero = values.new_zeros(())
    if not bool(in_run.any()):
        return zero, zero, zero, zero
    selected = values[in_run]
    std = selected.std() if selected.numel() > 1 else zero
    return selected.mean(), std, selected.min(), selected.max()


def _require_temperature(temperature: float) -> float:
    """``temperature >= 0``. ``0`` is hard max / greedy."""
    if float(temperature) < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")
    return float(temperature)


def _policy_entropy(pi: torch.Tensor) -> torch.Tensor:
    """Per-row entropy of a categorical ``pi`` on the last dim."""
    log_pi = torch.log(pi.clamp_min(torch.finfo(pi.dtype).tiny))
    return -(pi * log_pi).sum(dim=-1)


def _soft_state_value(q: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """State value of ``q`` ``[..., A]``: hard max, or SAC / soft-Q ``α logsumexp``.

    ``temperature == 0`` is ``max_a Q``. Otherwise
    ``V = α log Σ_a exp(Q_a / α)``, equal to ``E_π[Q] + α H[π]`` for
    ``π = softmax(Q / α)``. Same ``α`` as
    :meth:`~mouse_core.models.base.Model.get_action`.
    """
    alpha = float(temperature)
    if alpha == 0.0:
        return q.amax(dim=-1)
    return alpha * torch.logsumexp(q / alpha, dim=-1)


def _double_state_value(
    *,
    q_online: torch.Tensor,
    q_target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Double DQN state value: choose with online Q, evaluate with delayed Q.

    ``temperature == 0`` is ``Q_target(s, argmax_a Q_online)``. Ties take
    the lowest index, the same rule as ``get_action(temperature=0)``.
    ``temperature > 0`` is ``E_π[Q_target] + α H[π]`` for
    ``π = softmax(Q_online / α)``. Online Q is detached, so the bootstrap
    stays a constant.
    """
    online = q_online.detach()
    alpha = float(temperature)
    if alpha == 0.0:
        chosen = online.argmax(dim=-1, keepdim=True)
        return q_target.gather(dim=-1, index=chosen).squeeze(-1)
    log_pi = F.log_softmax(online / alpha, dim=-1)
    pi = log_pi.exp()
    return (pi * q_target).sum(dim=-1) + alpha * _policy_entropy(pi)


def _boltzmann_entropy(q: torch.Tensor, *, temperature: float) -> torch.Tensor:
    """Per-row entropy of ``softmax(Q / α)``. ``α == 0`` is the greedy policy."""
    alpha = float(temperature)
    if alpha == 0.0:
        is_max = q == q.amax(dim=-1, keepdim=True)
        pi = is_max.to(dtype=q.dtype) / is_max.sum(dim=-1, keepdim=True).to(
            dtype=q.dtype
        )
        return _policy_entropy(pi)
    log_pi = F.log_softmax(q / alpha, dim=-1)
    return _policy_entropy(log_pi.exp())


def _shift_next(values: torch.Tensor) -> torch.Tensor:
    """``out[t] = values[t+1]``, zero at the end."""
    return torch.cat([values[1:], values.new_zeros(1)])


def _affine_scan_backward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve ``g_t = a_t + b_t * g_{t+1}`` with ``g_T = 0`` for every ``t``.

    Parallel (Hillis-Steele) scan over the affine maps ``x -> a + b * x``:
    composing ``(a1, b1)`` then ``(a2, b2)`` gives ``(a2 + b2 * a1, b2 * b1)``.
    ``ceil(log2 T)`` rounds of elementwise ops on the whole tensor, no
    host-device syncs, and no division, so zeros in ``b`` (run ends, trace
    cuts, terminal discounts) are exact.
    """
    A = a.flip(0)
    B = b.flip(0)
    T = int(A.shape[0])
    offset = 1
    while offset < T:
        A_prev = F.pad(A[:-offset], (offset, 0))
        B_prev = F.pad(B[:-offset], (offset, 0), value=1.0)
        A, B = A + B * A_prev, B * B_prev
        offset *= 2
    return A.flip(0)


# Rows of the gate matrix unrolled together. The workspace is this many
# rows by the remaining horizon, not a second ``[N, N]`` copy.
_CONTINUATION_ROWS = 256

CrossGroupBackups = Literal["bootstrap", "ignore", "fault"]


def _require_cross_group_backups(value: str) -> CrossGroupBackups:
    """``"bootstrap"``, ``"ignore"``, or ``"fault"``."""
    if value not in ("bootstrap", "ignore", "fault"):
        raise ValueError(
            "cross_group_backups must be 'bootstrap', 'ignore', or 'fault', "
            f"got {value!r}."
        )
    return cast(CrossGroupBackups, value)


def _raise_if_cross_group_backup(
    *,
    in_run: torch.Tensor,
    participate: torch.Tensor,
) -> None:
    """Raise when an in-run backup depends on a value past the group boundary."""
    crossed = in_run & (participate == 0)
    if bool(crossed.any()):
        step = int(crossed.nonzero()[0])
        raise ValueError(
            "cross-group backup at step "
            f"{step}: the target depends on a value past the group boundary."
        )


def _run_mask(*, pair_weight: torch.Tensor, N: int) -> torch.Tensor:
    """``[N]`` factor on a continuation stored at that step.

    ``mask[s]`` is 1 when the pair that starts at step ``s`` stays in the
    run. The first step and the last step are ``0``.
    """
    mask = torch.zeros(N, dtype=pair_weight.dtype, device=pair_weight.device)
    if N > 2:
        in_run = pair_weight > 0
        mask[1 : N - 1] = in_run[1:].to(dtype=pair_weight.dtype)
    return mask


def _zero_off_data_factor_scan(
    *,
    discount: torch.Tensor,
    in_sample: torch.Tensor,
    carries: torch.Tensor,
) -> torch.Tensor:
    """``1`` when every off-data value has factor ``0`` in this backup.

    ``in_sample`` is true when this transition's next-state value is
    inside the run. ``carries`` is true when a non-zero factor would pass
    a later off-data value back through this step. A ``0`` discount zeros
    the factor, which is why a true terminal stays. Reaching a cutoff is
    not enough: the step stays when that factor is ``0``.
    """
    open_discount = discount != 0
    direct = (open_discount & ~in_sample).to(dtype=discount.dtype)
    carry = (open_discount & carries).to(dtype=discount.dtype)
    reached = _affine_scan_backward(direct, carry)
    return (reached == 0).to(dtype=discount.dtype)


def _block_cutoff_factor_is_zero(
    *,
    step: torch.Tensor,
    alive: torch.Tensor,
    discount: torch.Tensor,
) -> torch.Tensor:
    """``1`` when no cutoff column has a non-zero factor on ``V``.

    ``step`` is the discounted in-run continuation, with columns before
    the start set to ``1``. A cutoff (``alive == 0``) contributes only
    when every earlier factor is non-zero and this column's discount is
    non-zero. Links are tested as non-zero, not as a float product, so a
    small λ still counts. A horizon that runs past the sample with a
    ``0`` link does not put the off-data value in the target.
    """
    rows, width = step.shape
    device = step.device
    if width == 1:
        reached = torch.ones(rows, 1, dtype=torch.bool, device=device)
    else:
        open_link = (step != 0).to(dtype=step.dtype)
        survived = torch.cumprod(open_link, dim=1)
        reached = torch.cat(
            [
                torch.ones(rows, 1, dtype=torch.bool, device=device),
                survived[:, :-1] != 0,
            ],
            dim=1,
        )
    row = torch.arange(rows, device=device).unsqueeze(1)
    col = torch.arange(width, device=device)
    hit = (
        reached
        & (col >= row)
        & (alive == 0).unsqueeze(0)
        & (discount != 0).unsqueeze(0)
    )
    return (~hit.any(dim=1)).to(dtype=step.dtype)


def _block_returns(
    *,
    continuation: torch.Tensor,
    col_mask: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    v_next: torch.Tensor,
    t0: int,
    rows: int,
    cross_group_backups: CrossGroupBackups,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns for starts ``t0 .. t0+rows``, and which starts stay.

    Reads columns ``t0+1:`` of those rows into one strip. Entries before
    each start are outside that start's return: their step is ``1`` so
    the cumprod is unchanged, and their reward term is ``0``.

    ``col_mask`` is ``0`` at a group boundary, where the continuation is
    outside the sampled run. ``"ignore"`` and ``"fault"`` leave that
    off-data ``V`` out of the target and mark a start when the factor on
    it is non-zero. A zero factor leaves the value out of the target, so
    the start stays. ``"bootstrap"`` fills ``V`` on the step before the
    boundary, and every start stays. A gate ``0`` on a step the mask
    still keeps bootstraps either way.

    The second tensor is ``1`` for a start that stays and ``0`` for a
    start that drops.
    """
    device = continuation.device
    dtype = reward.dtype
    end = t0 + rows
    width = int(col_mask.shape[0]) - t0
    # ``step`` is mutated in place below; ``clone`` keeps the gate's matrix
    # intact (a single-row strip can be a contiguous view of it).
    step = (
        continuation[t0:end, t0 + 1 :]
        .to(dtype=dtype)
        .clone(memory_format=torch.contiguous_format)
    )
    g = discount[t0:]
    alive = col_mask[t0:]
    # ``alive == 0`` is a group boundary whose continuation was not sampled.
    leave_out = cross_group_backups != "bootstrap"
    boot = (1 - step) * alive if leave_out else (1 - step * alive)
    step.mul_(alive)
    term = reward[t0:] + g * boot * v_next[t0:]
    step.mul_(g)
    if rows > 1:
        local_row = torch.arange(rows, device=device).unsqueeze(1)
        local_col = torch.arange(rows, device=device)
        before = local_col < local_row
        step[:, :rows] = torch.where(before, step.new_ones(()), step[:, :rows])
        # Out of place so a reward that depends on a parameter (the
        # centering constant) keeps a gradient. ``step`` does not.
        left = torch.where(before, term.new_zeros(()), term[:, :rows])
        term = torch.cat([left, term[:, rows:]], dim=1) if width > rows else left
    if leave_out:
        participate = _block_cutoff_factor_is_zero(step=step, alive=alive, discount=g)
    else:
        participate = step.new_ones(rows)
    if width > 1:
        torch.cumprod(step, dim=1, out=step)
        term = torch.cat([term[:, :1], term[:, 1:] * step[:, :-1]], dim=1)
    return term.sum(dim=1), participate


def _continuation_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    v_step: torch.Tensor,
    pair_weight: torch.Tensor,
    continuation: torch.Tensor,
    cross_group_backups: CrossGroupBackups,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return for every pair ``(t, t+1)``, and which pairs stay.

    ``G_t = r_{t+1} + γ_{t+1} ((1 - c) V_{t+1} + c G_{t+1})``.
    ``continuation`` is the gate matrix ``[N, N]``: row ``t``, column
    ``s`` is the continuation at absolute step ``s`` for the return that
    started at ``t``. Rows are cumprod'd in blocks of
    ``_CONTINUATION_ROWS`` and never read another start's return. The
    in-run mask zeros a continuation that would leave the run.
    ``"ignore"`` leaves that off-data ``V`` out of the target and drops
    the pair when the factor on it is non-zero, from the loss and from
    logged metrics. ``"fault"`` raises on that pair instead.
    ``"bootstrap"`` fills ``V`` on the step before that group boundary
    and keeps the pair. A zero factor leaves the value out of the
    target, so the pair stays and ``"fault"`` does not raise. A gate
    ``0`` on a step still inside the run still bootstraps. Out-of-run
    pairs return ``0``.

    Returns ``(returns [N-1], participate [N-1])``. ``participate`` is
    ``1`` for a pair that stays. ``"fault"`` raises when an in-run pair
    would be dropped.
    """
    cross_group_backups = _require_cross_group_backups(cross_group_backups)
    r = reward[1:].to(dtype=v_step.dtype)  # [N-1]  r_t (stored at t+1)
    g = discount_all[1:]  # [N-1]  γ_t
    v_next = v_step[1:]  # [N-1]  V(s_{t+1})
    N = int(reward.shape[0])
    if tuple(continuation.shape) != (N, N):
        raise ValueError(
            f"continuation must have shape [{N}, {N}], "
            f"got {tuple(continuation.shape)}."
        )
    T = N - 1
    in_run = pair_weight > 0
    mask = _run_mask(pair_weight=pair_weight.to(dtype=v_step.dtype), N=N)
    col_mask = mask[1:]
    returns = r.new_empty(T)
    participate = r.new_empty(T)
    for t0 in range(0, T, _CONTINUATION_ROWS):
        rows = min(_CONTINUATION_ROWS, T - t0)
        returns[t0 : t0 + rows], participate[t0 : t0 + rows] = _block_returns(
            continuation=continuation,
            col_mask=col_mask,
            reward=r,
            discount=g,
            v_next=v_next,
            t0=t0,
            rows=rows,
            cross_group_backups=cross_group_backups,
        )
    in_run_f = in_run.to(dtype=returns.dtype)
    if cross_group_backups == "fault":
        _raise_if_cross_group_backup(in_run=in_run, participate=participate)
    return returns * in_run_f, participate


def _read_gate(
    *,
    gate: object,
    objective_data: dict[str, torch.Tensor],
    q_step: torch.Tensor,
    q_delayed: torch.Tensor,
    N: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Call ``gate`` and check the continuation matrix. ``None`` is zeros.

    Type and shape are checked here. Values in ``[0, 1]`` are the gate's
    contract (:class:`~mouse_core.objectives.transforms.Gate`); reading that
    reduction back to the host would sync every forward. ``q_delayed`` is
    the detached per-step delayed Q, same layout as ``q_step``.
    """
    if gate is None:
        return torch.zeros(N, N, dtype=dtype, device=device)
    with torch.no_grad():
        values = gate(**objective_data, q=q_step, q_delayed=q_delayed)  # type: ignore[operator]
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"gate must return a Tensor, got {type(values)}.")
    values = values.to(dtype=dtype, device=device)
    if tuple(values.shape) != (N, N):
        raise ValueError(
            f"gate must return shape [{N}, {N}], got {tuple(values.shape)}."
        )
    return values


def _require_reward_center(
    center: torch.Tensor | None,
    *,
    name: str,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a 0-dim float32 centering constant, or ``None``."""
    if center is None:
        return None
    if not isinstance(center, torch.Tensor):
        raise TypeError(
            f"{name} must be a 0-dim float32 tensor or None, got "
            f"{type(center).__name__}."
        )
    if center.dtype != torch.float32 or center.ndim != 0:
        raise TypeError(
            f"{name} must be a 0-dim float32 tensor or None, got "
            f"shape {tuple(center.shape)} dtype {center.dtype}."
        )
    if center.device != device:
        raise ValueError(
            f"{name} device {center.device} must match predictions "
            f"device {device}."
        )
    return center


def _pair_values_to_rows(
    pair_values: torch.Tensor,
    step_of: torch.Tensor,
) -> torch.Tensor:
    """Broadcast ``[N-1]`` per-pair values onto head-output rows ``[P]``."""
    padded = torch.cat([pair_values, pair_values.new_zeros(1)])
    return padded[step_of]


class DqnObjective(Objective[...]):
    """Bellman TD(λ) objective with a delayed target network.

    Instantiate with hyperparameters, then call with
    ``objective_data=``, ``group_id=`` (int64 ``[N]``, one id per step,
    returned beside ``objective_data``), the online Q tensor as ``predictions=``, the
    delayed Q tensor as ``delayed_predictions=``, ``reward_center=``
    (a 0-dim float32 tensor, or ``None``), and
    ``delayed_reward_center=`` (the same, or ``None`` when
    ``reward_center`` is ``None``). Both Q tensors come
    from the matching head on
    :class:`~mouse_core.models.base.Model`
    (``model.copy(heads=True, backbone=True, reasoner=False)``) run on
    the same ``TokenBatch``. The delayed tensor is detached before
    the Bellman target, so the TD error does not backprop through it.

    Q rows are **per head-output token** (``[P, A]``), not per step: a step may
    own several head-output tokens (tokenizer input field flagged
    ``head_output=True`` emitting more than one token). The
    ``head_output_count`` column stamped by ``pack_token_batch`` maps rows to
    steps, so predictions and step fields can never misalign. Every
    head-output row of step ``i`` trains toward the *same* TD target; the
    bootstrap reads the *last* head-output row of step ``i+1`` (the most
    informed one).

    A **run** is one ``group_id`` (one dataloader sample). Neighbor
    reads (action / reward / done / next Q at ``i+1``) must stay in-run: an
    out-of-run pair still has a loss term, but it is multiplied by ``0`` so
    output ``i`` does not affect the scalar loss or the gradient. A group's
    backups are the same in a batch that also holds other groups as they
    are when that group is the whole batch: a backup never reads another
    group's reward or value. If every
    weight is ``0`` the loss is ``0``. A step whose ``terminated`` or
    ``truncated`` flag is set is still in-run when ``group_id`` does not
    change, and may train. Gamma is the Bellman discount from those flags
    at ``i+1`` inside a same-run pair — it is not a run mask.

    ``reward(**objective_data)`` supplies the per-step reward; the value
    stored at ``i+1`` is ``r_t``. ``affine_reward`` is the column
    affine; ``boundary_reward`` applies terminated / truncated scale and
    shift extras.
    ``value(value=..., **objective_data)`` supplies the per-step affine
    on online and delayed Q. ``affine_value`` is the prediction affine;
    ``boundary_value`` applies terminated / truncated scale and shift extras.
    ``discount(**objective_data)``
    supplies the per-step γ that multiplies the bootstrap and the
    continued return. ``boundary_discount`` is the standard
    ``gamma_step`` × episode-extra × task-extra lookup.
    ``gate(**objective_data, q=q, q_delayed=q_delayed)`` supplies the
    continuation. ``q`` is the detached per-step online Q.
    ``q_delayed`` is the detached per-step delayed Q. ``lambda_gate``
    is the λ-return to the end of the run. ``nstep_gate`` is the n-step
    return. ``watkins_gate`` cuts where the taken action is not an
    online argmax. ``value_gap_gate`` is the optimality-gap cut.
    ``policy_delayed`` picks ``a* = argmax``. ``value_delayed`` scores
    ``V = Q(s, a*)`` and the taken action. Either flag, or both, reads
    delayed Q. The signed gap is ``Q(s, a_taken) - V``. ``bias`` shifts
    it, then it is capped at ``0``. ``c = exp(beta * gap)``.
    ``normalize=True`` divides the signed gap by
    ``max_a Q - min_a Q + eps`` (``eps`` required). ``normalize=False``
    uses the raw signed gap. ``general_gate(gates=)``
    multiplies those matrices, so a step continues only where every
    gate continues. The callable returns
    ``[N, N]``. ``gate=None`` is the one-step target (a zero matrix).

    The target is
    ``G_t = r_{t+1} + γ_{t+1} ((1 - c) V(s_{t+1}) + c G_{t+1})`` with ``V``
    the delayed state value. The gate returns ``[N, N]``. Row ``t``,
    column ``s`` is the continuation at absolute step ``s`` for the
    return that started at ``t``, and that row is unrolled on its own.
    The objective then zeros a
    continuation that would leave the run. For any horizon, the last
    step of a task is not updated when its target depends on the next
    step's value. It is updated when the target does not.
    ``cross_group_backups="bootstrap"`` fills ``V`` on the step before
    that group boundary (the rest of the episode is not in the sample),
    so that step is updated from ``V`` and stays.
    ``cross_group_backups="ignore"`` leaves that off-data ``V`` out of
    the target and leaves the step out of the loss and out of logged
    metrics when the factor on it is non-zero.
    ``cross_group_backups="fault"`` raises instead. A terminated or
    truncated γ of
    ``0``, or a horizon that puts no
    weight on that value, does not depend on it, so the step stays. A
    gate cut on a later in-run step still bootstraps. ``V`` is delayed max-Q when
    ``temperature=0``. A
    positive ``temperature`` (SAC / soft Q-learning ``α``) replaces that
    with the soft value ``α log Σ_a exp(Q / α)``, equal to
    ``E_π[Q] + α H[π]`` for the Boltzmann policy
    ``π = softmax(Q / α)`` over the same (affine) delayed Q the TD error
    uses. ``α → 0`` recovers hard max. Pair a non-zero ``α`` with
    ``get_action(temperature=)`` so rollout samples that same policy.
    ``double=False`` reads that ``V`` from delayed Q alone.
    ``double=True`` is Double DQN (van Hasselt, Guez, and Silver, 2016):
    the action comes from online Q and the value from delayed Q, both
    after ``value``. ``temperature=0`` is
    ``Q_delayed(s, argmax_a Q_online)`` (lowest index on ties, same as
    ``get_action``). ``temperature > 0`` is
    ``E_π[Q_delayed] + α H[π]`` for ``π = softmax(Q_online / α)``.
    The online scores that choose the action are detached, so the TD
    error does not train them.
    The loss is the weighted mean of ``δ²`` on those rows, with
    ``δ = G - Q(s, a)`` and ``G`` the backup above. One-step,
    ``temperature=0``, and ``double=False`` make ``δ`` the residual
    ``r + γ max_a Q_delayed(s', a) - Q(s, a)``.
    ``reward_center`` is the online centering constant ``c``: a 0-dim
    float32 tensor (the output of
    :class:`~mouse_core.models.heads.constant.ConstantHead`,
    ``scale * value``), or
    ``None``. ``delayed_reward_center`` is the delayed constant
    ``c'``. Both are set, or both are ``None``. The backup uses
    ``c'`` for both potentials and detaches it, so one step is
    ``G = r - c' + γ c' + γ V``. The returned loss is a dict.
    ``action_value`` is the TD loss and trains Q. It does not train
    ``c`` or ``c'``: a constant shift of every action value is already
    a bias in the Q head. When the centers are set, ``reward_center``
    is ``-c · mean(δ)`` with ``δ`` detached, so gradient descent steps
    ``c`` by that mean. ``c'`` tracks ``c`` through the delayed copy.
    When ``c = c'`` the target is
    ``G - K c`` with ``K`` the same recursion on reward ``1 - γ`` and
    value ``0``, and the shift telescopes to exactly ``c`` on every
    action value — the policy ordering never changes. ``None`` keeps
    plain ``δ²``.
    The trace never crosses a run break. At a ``terminated`` or
    ``truncated`` step ``γ`` is ``discount`` from those flags and
    multiplies both the bootstrap and the continued return, so a ``0``
    discount ends the trace while a non-zero truncation gamma carries it
    (discounted) into the reset frame's return. A gate that cuts on the
    action reads detached Q. ``watkins_gate`` cuts wherever the taken
    action is not the online argmax (ties included).
    ``value_gap_gate(beta=, normalize=, policy_delayed=, value_delayed=, bias=, eps=)``
    softens that cut. ``policy_delayed`` picks ``a* = argmax``.
    ``value_delayed`` scores ``V = Q(s, a*)`` and the taken action.
    Either flag, or both, reads delayed Q. The signed gap is
    ``Q(s, a_taken) - V``. ``normalize=True`` divides by the value
    range (``eps`` required). ``bias`` is added and the gap is capped
    at ``0``. ``c = exp(beta * gap)``. ``normalize=False`` uses the raw
    signed gap and takes no ``eps``.
    A zero gap continues and a larger gap bootstraps the delayed
    state value. ``watkins_gate`` reads online Q.
    ``value_gap_gate`` reads online Q, delayed Q, or both. Both leave
    oracle columns such as ``info_q_star`` unread. ``metrics["entropy"]`` is the in-run mean of
    ``H[softmax(Q / α)]`` on online Q when ``temperature > 0``.
    ``metrics["backup"]`` is the per-row Bellman target ``G`` (``[P]``)
    the taken action is regressed to. When ``reward_center`` is set,
    ``G`` is the shaped target. ``metrics["backup_weight"]`` is
    the matching per-row weight (``0`` when the step is out of the loss
    and out of logged metrics). ``metrics["in_run_backup"]`` is
    ``backup`` on the rows with ``backup_weight > 0``.
    ``metrics["in_run_delta"]`` (present when ``reward_center`` is set)
    is that call's centered residual ``δ − K c`` on those same rows.
    All of these are detached. The
    continuation matrix stays the one ``[N, N]`` the gate returned. Rows
    are cumprod'd in blocks, and that read does not sync the host.

    Those columns arrive in ``objective_data`` only if they are listed in the
    tokenizer ``objective_fields`` keep-list (input fields are not auto-copied).
    ``terminated`` and ``truncated`` are objective columns only — do not add
    them as tokenizer input fields, or they will be fed to the transformer::

        tokenizer = Tokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "terminated"},
                {"input_field": "truncated"},
            ],
        )

    Args:
        discount: Per-step γ from unpacked ``objective_data`` columns.
            ``boundary_discount`` is the standard ``gamma_step`` × extra
            lookup (``None`` skips the call and uses ``1``);
            any ``discount(**objective_data) -> [N]`` is accepted.
        reward: Per-step reward from unpacked ``objective_data`` columns.
            ``affine_reward`` is the column affine; ``boundary_reward``
            applies terminated / truncated scale and shift extras
            (``None`` skips the call);
            any ``reward(**objective_data) -> [N]`` is accepted.
            Does not change ``objective_data``.
        value: Per-step affine on online and delayed Q from unpacked
            ``objective_data`` columns plus ``value=``. ``affine_value``
            is the prediction affine; ``boundary_value`` applies terminated
            / truncated scale and shift extras
            (``None`` skips the call);
            any ``value(value=..., **objective_data)`` returning the same
            shape is accepted. Same callable on both networks. Does not
            change the prediction tensors or eval ``argmax``.
        action_key: Key in ``objective_data`` that holds the integer action.
        terminated_key: Key in ``objective_data`` for mouse-gym ``terminated``.
        truncated_key: Key in ``objective_data`` for mouse-gym ``truncated``.
        cql_weight: Alpha coefficient for the Conservative Q-Learning penalty.
            ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.
        gate: Continuation from unpacked ``objective_data`` plus ``q=``
            and ``q_delayed=``.
            Required. ``None`` is the one-step target (a zero matrix).
            ``lambda_gate`` is the λ-return to the end of the run
            (``td_lambda=``). ``nstep_gate`` is the n-step return
            (``n=``). ``watkins_gate`` cuts where the taken action is
            not an online argmax. ``value_gap_gate`` is the optimality-gap cut.
            ``policy_delayed`` picks ``a* = argmax``. ``value_delayed``
            scores ``V = Q(s, a*)`` and the taken action. Either flag,
            or both, reads delayed Q. The signed gap is
            ``Q(s, a_taken) - V``. ``normalize=True`` divides by
            ``max_a Q - min_a Q + eps`` on the value Q (``eps`` required).
            ``bias`` shifts that gap, then it is capped at ``0``.
            ``c = exp(beta * gap)``. ``normalize=False``
            uses the raw signed gap and takes no ``eps``. ``general_gate(gates=)`` is the
            element-wise product of those matrices. A callable returning
            ``[N, N]`` is accepted. Row ``t``, column ``s`` is the continuation at
            absolute step ``s`` for the return that started at ``t``.
            See :class:`~mouse_core.objectives.transforms.Gate`.
        temperature: SAC / soft Q-learning ``α`` (``>= 0``). Required.
            ``0`` is hard max-Q. ``> 0`` bootstraps from
            ``α logsumexp(Q / α)`` on delayed Q (after ``value``) and
            logs ``metrics["entropy"]``. Same units and meaning as
            ``get_action(temperature=)``.
        double: Required. ``False`` bootstraps from delayed Q alone.
            ``True`` is Double DQN: online Q chooses the action and
            delayed Q evaluates it. ``temperature`` selects hard argmax
            or the online Boltzmann policy, as above.
        cross_group_backups: Required. ``"bootstrap"``, ``"ignore"``, or
            ``"fault"``. For any horizon, the last step of a task is not
            updated when its target depends on the next step's value. It
            is updated when the target does not. ``"bootstrap"`` fills
            ``V`` on the step before a group boundary (the end of the
            batch, or a ``group_id`` break: a chunk boundary, time
            limit, or truncation whose rest was not sampled), so that
            step is updated from ``V`` and stays. ``"ignore"`` leaves
            that off-data ``V`` out of the target and leaves the step
            out of the loss and out of logged metrics when the factor
            on it is non-zero. ``"fault"`` raises on that step instead.
            A terminated or truncated γ of ``0``, or a horizon that puts no weight on
            that value, does not depend on it, so the step stays and
            ``"fault"`` does not raise. An earlier step whose backup
            stays inside the task stays either way.
    """

    def __init__(
        self,
        *,
        discount: Discount | None,
        reward: Reward | None,
        value: Value | None,
        temperature: float,
        double: bool,
        cross_group_backups: CrossGroupBackups,
        action_key: str = "action",
        terminated_key: str = "terminated",
        truncated_key: str = "truncated",
        gate: Gate | None,
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        self.temperature = _require_temperature(temperature)
        self.double = bool(double)
        self.cross_group_backups: CrossGroupBackups = _require_cross_group_backups(
            cross_group_backups
        )
        self.discount = _require_transform(discount, name="discount")
        self.reward = _require_transform(reward, name="reward")
        self.value = _require_transform(value, name="value")
        self.action_key = action_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.gate = _require_transform(gate, name="gate")

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor,
        reward_center: torch.Tensor | None,
        delayed_reward_center: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor,
        reward_center: torch.Tensor | None,
        delayed_reward_center: torch.Tensor | None,
        value_predictions: None = None,
        targets: None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor | None = None,
        reward_center: torch.Tensor | None,
        delayed_reward_center: torch.Tensor | None,
        value_predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "DqnObjective",
            value_predictions=value_predictions,
            targets=targets,
        )
        q: torch.Tensor = predictions
        q_target: torch.Tensor = _require_prediction(
            delayed_predictions, owner="DqnObjective", name="delayed_predictions"
        ).detach()
        center = _require_reward_center(
            reward_center, name="reward_center", device=q.device
        )
        delayed_center = _require_reward_center(
            delayed_reward_center, name="delayed_reward_center", device=q.device
        )
        if (center is None) != (delayed_center is None):
            raise ValueError(
                "reward_center and delayed_reward_center must both be set "
                "or both be None."
            )
        if delayed_center is not None:
            delayed_center = delayed_center.detach()

        if q.ndim != 2:
            raise ValueError(
                f"DQN expects action_value shape [P, A], got {tuple(q.shape)}."
            )
        if q.dtype != torch.float32 or q_target.dtype != torch.float32:
            raise TypeError(
                "DQN expects float32 action_value (heads always run in fp32), got "
                f"online {q.dtype} and delayed {q_target.dtype}."
            )
        if q_target.shape != q.shape:
            raise ValueError(
                f"DQN delayed action_value shape {tuple(q_target.shape)} must "
                f"match online shape {tuple(q.shape)}."
            )
        P, A = q.shape
        device = q.device
        value_dtype = q.dtype

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

        reward = _apply_transform(
            transform=self.reward,
            name="reward",
            objective_data=objective_data,
            N=N,
            dtype=value_dtype,
            device=device,
            identity=objective_data["reward"],
        )

        _require_done_flags(
            objective_data,
            terminated_key=self.terminated_key,
            truncated_key=self.truncated_key,
            N=N,
        )

        # A step may own several head-output tokens; every row of step i trains
        # toward the same target, and the bootstrap reads step i+1's *last*
        # head-output row.
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
            group_id=group_id,
            N=N,
            device=device,
            dtype=value_dtype,
        )

        # Each token at position i encodes (obs_i, action_{i-1}, reward_{i-1},
        # terminated_{i-1}, truncated_{i-1}), i.e. the action, reward, and
        # flags stored at i are the ones that *produced* obs_i, not the
        # ones taken *from* obs_i.  The transition out of state i is therefore
        # described by the fields stored at i+1.
        step_next = (step_of + 1).clamp(max=N - 1)  # [P] (final step clamped, weight 0)
        next_actions = action[step_next]            # [P]  a_i (stored at i+1)

        discount_all = _apply_transform(
            transform=self.discount,
            name="discount",
            objective_data=objective_data,
            N=N,
            dtype=value_dtype,
            device=device,
            identity=torch.ones(N, dtype=value_dtype, device=device),
        )

        q_values = q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [P]
        q_next = q_target[last_rows]
        if self.double:
            v_step = _double_state_value(
                q_online=q[last_rows],
                q_target=q_next,
                temperature=self.temperature,
            )
        else:
            v_step = _soft_state_value(q_next, temperature=self.temperature)
        continuation = _read_gate(
            gate=self.gate,
            objective_data=objective_data,
            q_step=q[last_rows].detach(),
            q_delayed=q_target[last_rows].detach(),
            N=N,
            dtype=value_dtype,
            device=device,
        )
        if delayed_center is not None:
            # The backup uses the delayed center only. Each reward slot at
            # discount γ subtracts (1 - γ)·c'. c' is detached.
            reward = reward - (1.0 - discount_all) * delayed_center
        pair_target, participate = _continuation_targets(
            reward=reward,
            discount_all=discount_all,
            v_step=v_step,  # [N]  V(s_i)
            pair_weight=pair_weight,
            continuation=continuation,
            cross_group_backups=self.cross_group_backups,
        )
        # A non-zero factor on an off-data value drops the step from the loss
        # and from every logged metric. A zero factor leaves that value out
        # of the target, so the step keeps its in-run weight.
        pair_weight = pair_weight * participate
        row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]  # [P]
        td_target = _pair_values_to_rows(pair_target, step_of)  # [P]

        # δ = G - Q(s, a). G uses the delayed center. action_value trains Q.
        # metrics["in_run_delta"] is δ on in-run rows. reward_center is
        # -c · mean(δ) with δ detached.
        delta = td_target - q_values
        td_sq = delta ** 2

        per_row = td_sq
        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty = torch.logsumexp(q, dim=-1) - q_values
            per_row = per_row + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = _weighted_mean(cql_penalty.detach(), row_weight)

        td_loss = _weighted_mean(per_row, row_weight)
        td_loss_log = _weighted_mean(td_sq.detach(), row_weight)
        losses: dict[str, torch.Tensor] = {"action_value": td_loss}
        if center is not None:
            # δ is detached, so this term trains c and not Q.
            # Descent steps c by the weighted mean residual.
            losses["reward_center"] = -center * _weighted_mean(delta.detach(), row_weight)

        curr_max_q = q.amax(dim=-1)  # [P]  max online Q at s_i
        q_mean, q_std, q_min, q_max = _in_run_stats(curr_max_q.detach(), row_weight)
        named: dict[str, torch.Tensor] = {
            "q_values_mean":   q_mean,
            "q_values_std":    q_std,
            "q_values_min":    q_min,
            "q_values_max":    q_max,
            "action_value":    td_loss_log,
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean
        if center is not None and delayed_center is not None:
            # ``clone`` so the logged value is a snapshot of the parameter.
            named["reward_center"] = center.detach().clone()
            named["delayed_reward_center"] = delayed_center.detach().clone()
        if self.temperature > 0.0:
            named["entropy"] = _weighted_mean(
                _boltzmann_entropy(q.detach(), temperature=self.temperature),
                row_weight,
            )

        # Keep scalar stats as 0-dim tensors so the train step does not
        # host-sync via ``tolist()``; loggers can ``.item()`` at log time.
        metrics: dict[str, float | torch.Tensor] = dict(named)
        # Same G and row weight the loss used — callers log these
        # instead of rebuilding the backup. ``in_run_backup`` is the
        # rows with a positive weight, already selected.
        # ``in_run_delta`` is δ = G − Q on those rows. G uses the delayed
        # center. The reward_center loss trains c from that residual.
        metrics["backup"] = td_target.detach()
        metrics["backup_weight"] = row_weight.detach()
        in_run = metrics["backup_weight"] > 0
        metrics["in_run_backup"] = metrics["backup"][in_run]
        if center is not None:
            metrics["in_run_delta"] = delta.detach()[in_run]
        return losses, metrics
