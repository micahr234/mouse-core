"""TD(λ) DQN objective with a delayed target network."""

from __future__ import annotations

from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives._actions import canonical_actions, executed_actions

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
    _PreparedGate,
    _prepare_gate,
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
    the row of step ``i``'s last head-output token (the bootstrap read).
    A missing readout uses sentinel ``P``; ``_step_predictions`` supplies its
    zero placeholder, and return construction continues through that step. Reads
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
        if bool((counts < 0).any()):
            raise ValueError("head_output_count entries must be >= 0.")
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
        torch.arange(N, dtype=torch.int64, device=device), counts, output_size=P
    )
    last_rows = torch.where(counts > 0, counts.cumsum(dim=0) - 1, P)
    return step_of, last_rows


def _step_predictions(predictions: torch.Tensor, last_rows: torch.Tensor) -> torch.Tensor:
    """Scatter real readouts onto the step clock; absent values are placeholders."""
    return torch.cat([predictions, predictions.new_zeros((1,) + predictions.shape[1:])])[last_rows]


def _expand_policy_readouts(objective_data, predictions, *, who):
    counts = objective_data.get("head_output_count")
    if counts is None:
        return predictions, objective_data, torch.ones(predictions.shape[0], dtype=torch.bool, device=predictions.device)
    if counts.ndim != 1 or counts.dtype != torch.int64 or bool(((counts < 0) | (counts > 1)).any()):
        raise ValueError(f"{who} expects one prediction row per step with a readout (zero or one per step)")
    step_of, last_rows = _head_output_layout(objective_data, N=counts.numel(), P=predictions.shape[0], device=predictions.device)
    return (_step_predictions(predictions, last_rows),
            {**objective_data, "head_output_count": torch.ones_like(counts)}, counts.to(predictions.device) > 0)


def _validate_request_actions(readout, executed, group_id):
    if bool(((group_id[:-1] == group_id[1:]) & executed[1:] & ~readout[:-1]).any()):
        raise ValueError("An executed action requires a readout on its preceding step")


def _validate_missing_bootstrap(readout, discount, group_id, mode):
    if mode != "bootstrap":
        return
    last = torch.ones_like(readout)
    last[:-1] = group_id[:-1] != group_id[1:]
    if bool((last & ~readout & (discount != 0)).any()):
        raise ValueError("Cannot bootstrap from a final step without an action-request readout; use zero ending discount, ignore, or fault")


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
    if counts.dtype != torch.int64:
        raise TypeError(f"head_output_count must be int64, got {counts.dtype}.")
    total = int(counts.sum())
    if (counts.shape != torch.Size([n_steps]) or total != n_pred
            or n_pred != n_steps or bool((counts != 1).any())):
        raise ValueError(
            f"{who} expects one prediction row per step "
            f"(got {n_pred} prediction rows, {n_steps} steps, "
            f"head_output_count sum {total}). Use a single head-output token "
            "per step, or a DQN-family objective."
        )


def _weighted_mean(values: torch.Tensor, pair_weight: torch.Tensor) -> torch.Tensor:
    """``(w * x).sum() / w.sum().clamp(min=1)``. All-zero weights yield ``0``."""
    weight = pair_weight.to(dtype=values.dtype)
    return (weight * values).sum(dim=0) / weight.sum(dim=0).clamp(min=1)


def _in_run_stats(
    values: torch.Tensor,
    pair_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean / std / min / max over in-run pairs. All-zero weights yield zeros."""
    if values.shape[0] == 0:
        zero = values.sum(dim=0)
        return zero, zero, zero, zero
    in_run = pair_weight > 0
    count = in_run.sum(dim=0)
    selected = torch.where(in_run, values, 0)
    mean = selected.sum(dim=0) / count.clamp_min(1)
    centered = torch.where(in_run, values - mean, 0)
    std = (centered.square().sum(dim=0) / (count - 1).clamp_min(1)).sqrt()
    minimum = torch.where(in_run, values, torch.inf).amin(dim=0)
    maximum = torch.where(in_run, values, -torch.inf).amax(dim=0)
    return mean, std, torch.where(count > 0, minimum, 0), torch.where(count > 0, maximum, 0)


def _require_temperature(temperature: float) -> float:
    """``temperature >= 0``. ``0`` is hard max / greedy."""
    if float(temperature) < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")
    return float(temperature)


def _policy_entropy(pi: torch.Tensor) -> torch.Tensor:
    """Per-row entropy of a categorical ``pi`` on the last dim."""
    log_pi = torch.log(pi.clamp_min(torch.finfo(pi.dtype).tiny))
    return -(pi * log_pi).sum(dim=-1)


def _backup_state_value(
    *, q_actions: torch.Tensor, q_backup: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """Choose from action-head scores and evaluate with backup values.

    Zero temperature uses the lowest-index argmax. Positive temperature uses
    ``E_pi[Q_backup] + alpha H(pi)`` with ``pi = softmax(Q_actions / alpha)``.
    Both backup inputs are detached from the training graph.
    """
    actions, values = q_actions.detach(), q_backup.detach()
    alpha = float(temperature)
    if alpha == 0.0:
        chosen = actions.argmax(dim=-1, keepdim=True)
        return values.gather(dim=-1, index=chosen).squeeze(-1)
    log_pi = F.log_softmax(actions / alpha, dim=-1)
    pi = log_pi.exp()
    return (pi * values).sum(dim=-1) + alpha * _policy_entropy(pi)


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
        A_prev = torch.cat([A.new_zeros((offset,) + A.shape[1:]), A[:-offset]])
        B_prev = torch.cat([B.new_ones((offset,) + B.shape[1:]), B[:-offset]])
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
        step = int(crossed.nonzero()[0, 0])
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
    rows, width = step.shape[:2]
    device = step.device
    if width == 1:
        reached = torch.ones((rows, 1) + step.shape[2:], dtype=torch.bool, device=device)
    else:
        open_link = (step != 0).to(dtype=step.dtype)
        survived = torch.cumprod(open_link, dim=1)
        reached = torch.cat(
            [
                torch.ones((rows, 1) + step.shape[2:], dtype=torch.bool, device=device),
                survived[:, :-1] != 0,
            ],
            dim=1,
        )
    row = torch.arange(rows, device=device).unsqueeze(1)
    col = torch.arange(width, device=device)
    after = col >= row
    if step.ndim == 3:
        after = after.unsqueeze(-1)
    hit = (
        reached
        & after
        & (alive == 0).unsqueeze(0)
        & (discount != 0).unsqueeze(0)
    )
    return (~hit.any(dim=1)).to(dtype=step.dtype)


def _block_returns(
    *,
    continuation: torch.Tensor | _PreparedGate,
    col_mask: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    v_next: torch.Tensor,
    t0: int,
    rows: int,
    cross_group_backups: CrossGroupBackups,
    decision_count: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
    device = reward.device
    dtype = reward.dtype
    end = t0 + rows
    width = int(col_mask.shape[0]) - t0
    # ``step`` is mutated in place below; ``clone`` keeps the gate's matrix
    # intact (a single-row strip can be a contiguous view of it).
    strip = continuation.block(t0, end) if isinstance(continuation, _PreparedGate) else continuation[t0:end]
    if reward.ndim == 2 and strip.ndim == 2:
        strip = strip.unsqueeze(-1).expand(-1, -1, reward.shape[1])
    step = (
        strip[:, t0 + 1 :]
        .to(dtype=dtype)
        .clone(memory_format=torch.contiguous_format)
    )
    g = discount[t0:]
    alive = col_mask[t0:]
    if reward.ndim == 2:
        g = g.unsqueeze(-1) if g.ndim == 1 else g
        alive = alive.unsqueeze(-1)
    # ``alive == 0`` is a group boundary whose continuation was not sampled.
    leave_out = cross_group_backups != "bootstrap"
    boot = (1 - step) * alive if leave_out else (1 - step * alive)
    step.mul_(alive)
    term = reward[t0:] + g * boot * v_next[t0:]
    lengths = None
    if decision_count is not None:
        count = decision_count[t0:]
        if term.ndim == 3:
            count = count.unsqueeze(-1)
        lengths = count.unsqueeze(0).expand_as(term)
    step.mul_(g)
    if rows > 1:
        local_row = torch.arange(rows, device=device).unsqueeze(1)
        local_col = torch.arange(rows, device=device)
        before = local_col < local_row
        if step.ndim == 3:
            before = before.unsqueeze(-1)
        step[:, :rows] = torch.where(before, step.new_ones(()), step[:, :rows])
        # Out of place so a reward that requires grad keeps it. ``step`` does not.
        left = torch.where(before, term.new_zeros(()), term[:, :rows])
        term = torch.cat([left, term[:, rows:]], dim=1) if width > rows else left
        if lengths is not None:
            left_lengths = torch.where(before, 0, lengths[:, :rows])
            lengths = torch.cat([left_lengths, lengths[:, rows:]], dim=1) if width > rows else left_lengths
    if leave_out:
        participate = _block_cutoff_factor_is_zero(step=step, alive=alive, discount=g)
    else:
        participate = step.new_ones((rows,) + step.shape[2:])
    if width > 1:
        torch.cumprod(step, dim=1, out=step)
        term = torch.cat([term[:, :1], term[:, 1:] * step[:, :-1]], dim=1)
        if lengths is not None:
            lengths = torch.cat([lengths[:, :1], lengths[:, 1:] * step[:, :-1]], dim=1)
    return term.sum(dim=1), participate, None if lengths is None else lengths.sum(dim=1)


def _continuation_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    v_step: torch.Tensor,
    pair_weight: torch.Tensor,
    continuation: torch.Tensor | _PreparedGate,
    cross_group_backups: CrossGroupBackups,
    backup_lengths: torch.Tensor | None = None,
    decision_count: torch.Tensor | None = None,
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
    if not isinstance(continuation, _PreparedGate) and tuple(continuation.shape) not in ((N, N), (N, N) + v_step.shape[1:]):
        raise ValueError(
            f"continuation must have shape [{N}, {N}], "
            f"got {tuple(continuation.shape)}."
        )
    T = N - 1
    in_run = pair_weight > 0
    mask = _run_mask(pair_weight=pair_weight.to(dtype=v_step.dtype), N=N)
    col_mask = mask[1:]
    returns = r.new_empty((T,) + v_step.shape[1:])
    participate = torch.empty_like(returns)
    block_size = max(1, _CONTINUATION_ROWS // (v_step.shape[1] if v_step.ndim == 2 else 1))
    for t0 in range(0, T, block_size):
        rows = min(block_size, T - t0)
        block_return, block_keep, block_length = _block_returns(
            continuation=continuation,
            col_mask=col_mask,
            reward=r,
            discount=g,
            v_next=v_next,
            t0=t0,
            rows=rows,
            cross_group_backups=cross_group_backups,
            decision_count=decision_count,
        )
        returns[t0:t0 + rows] = block_return
        participate[t0:t0 + rows] = block_keep
        if backup_lengths is not None:
            assert block_length is not None
            backup_lengths[t0:t0 + rows] = block_length
    in_run_f = in_run.to(dtype=returns.dtype)
    if returns.ndim == 2:
        in_run = in_run.unsqueeze(-1)
        in_run_f = in_run_f.unsqueeze(-1)
    if backup_lengths is not None:
        backup_lengths.mul_(in_run_f)
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
    the detached all-action per-step delayed Q; ``q_step`` may instead
    contain one selected online value per step.
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


def _column_targets(
    *, reward: torch.Tensor, discount: torch.Tensor, v_step: torch.Tensor,
    pair_weight: torch.Tensor, columns: torch.Tensor,
    cross_group_backups: CrossGroupBackups,
    decision_count: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Linear-workspace targets for continuation independent of the start row."""
    n = reward.shape[0]
    alive = _run_mask(pair_weight=pair_weight, N=n)[1:]
    in_run = pair_weight
    g, c = discount[1:], columns[1:]
    if v_step.ndim == 2:
        alive, in_run = alive[:, None], in_run[:, None]
        g = g[:, None] if g.ndim == 1 else g
        c = c[:, None] if c.ndim == 1 else c
    link = g * c * alive
    boot = (1 - c) * alive if cross_group_backups != "bootstrap" else 1 - c * alive
    direct = reward[1:] + g * boot * v_step[1:]
    returns = _affine_scan_backward(direct, link)
    if cross_group_backups == "bootstrap":
        participate = torch.ones_like(returns)
    else:
        participate = _zero_off_data_factor_scan(
            discount=g, in_sample=alive > 0, carries=(c * alive) != 0,
        ).expand_as(returns)
    lengths = None
    if decision_count is not None:
        count = decision_count[:, None] if v_step.ndim == 2 else decision_count
        lengths = _affine_scan_backward(count.expand_as(returns), link) * in_run
    return returns * in_run, participate, lengths


def _pair_values_to_rows(
    pair_values: torch.Tensor,
    step_of: torch.Tensor,
) -> torch.Tensor:
    """Broadcast ``[N-1]`` per-pair values onto head-output rows ``[P]``."""
    padded = torch.cat([pair_values, pair_values.new_zeros((1,) + pair_values.shape[1:])])
    return padded[step_of]


class DqnObjective(Objective[...]):
    """Bellman TD(λ) objective with a delayed target network.

    Instantiate with hyperparameters, then call with
    ``objective_data=``, ``group_id=`` (int64 ``[N]``, one id per step,
    returned beside ``objective_data``), the online Q tensor as ``predictions=``, the
    backup value Q tensor as ``backup_value_predictions=``, and action-selection
    Q scores as ``backup_action_predictions=``. Backup tensors have shape ``[P, A]``.
    Online predictions may be all-action ``[P, A]`` or selected-action ``[P]``:
    each selected scalar is Q(s_i, action[i+1]), expanded over that step's
    head-output rows. Selected predictions avoid evaluating unobserved online
    actions. CQL and gates that read all online actions require dense predictions;
    delayed-only value-gap gates support either form. All tensors share a device.
    The two backup inputs are required and detached. Q statistics summarize the
    selected online values in selected mode, and online maxima in dense mode.
    Entropy uses the delayed action policy in selected mode.
    A delayed model can supply independent value and action heads, each
    Polyak-averaged from the same online value head with its own rate.

    Multiple heads use backups ``[P, K, A]`` and online predictions
    ``[P, K, A]`` or selected ``[P, K]``. Losses and scalar statistics then
    have shape ``[K]``; backup and backup_weight have shape ``[P, K]``.
    A reward transform may return shared ``[N]`` or head-specific ``[N, K]``.
    Gates may return shared ``[N, N]`` or head-specific ``[N, N, K]``.
    The objective does not average across heads; callers choose that reduction.
    ``include_statistics=False`` skips optional Q/entropy/loss diagnostics and
    the variable-size in_run_backup selection, which synchronizes CUDA.
    ``return_backup_lengths=True`` adds detached ``backup_length`` in the same
    shape as backup: the discounted expected number of executed decisions.
    This reuses the target's continuation and excludes automatic reset frames.

    Q rows are **per head-output token** (``[P, A]``), not per step: a step may
    own several head-output tokens (tokenizer input field flagged
    ``head_output=True`` emitting more than one token). The
    ``head_output_count`` column stamped by ``pack_token_batch`` maps rows to
    steps, so predictions and step fields can never misalign. Every
    head-output row of step ``i`` trains toward the *same* TD target; the
    bootstrap reads the *last* head-output row of step ``i+1`` (the most
    informed one).

    Sequential components of one environment action must use separate decision
    rows, each with its own incoming action ID. Internal rows carry zero reward
    and discount 1. ``boundary_discount(step_key=...)`` discounts only physical
    transitions; ``nstep_gate(step_key=...)`` counts those transitions. Ordinary
    one-step targets still bootstrap from the next component. Repeated output
    tokens within one row are readouts of the SAME decision, not components.

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
    ``gamma_step`` × episode-extra × episode-extra lookup.
    ``gate(**objective_data, q=q, q_delayed=q_delayed)`` supplies the
    continuation. ``q`` is the detached per-step online Q.
    ``q_delayed`` is the detached per-step delayed Q. ``lambda_gate``
    is the λ-return to the end of the run. ``nstep_gate`` is the n-step
    return. ``value_gap_gate`` is the optimality-gap cut.
    ``policy_delayed`` picks ``a* = argmax``. ``value_delayed`` scores
    ``V = Q(s, a*)`` and the taken action. Either flag, or both, reads
    delayed Q. The signed gap is ``Q(s, a_taken) - V``. ``bias`` is
    subtracted, then the gap is capped at ``0``. For positive beta,
    ``c = 2 ** (gap / beta)``. Zero beta gives ``1`` for zero gap
    and ``0`` for negative gap. With ``bias=0`` and both delayed
    flags ``False``, zero beta is the Watkins hard cut (ties continue).
    ``normalize=True`` divides the signed gap by
    ``|max_a Q| + eps`` (``eps`` required). ``normalize=False``
    uses the raw signed gap. A gap of ``0`` continues at ``c = 1``.
    For positive beta, a gap of ``-beta`` continues at ``1/2``. A larger
    ``beta`` is a softer cut. ``general_gate(gates=)`` multiplies those
    matrices; combining ``lambda_gate`` and ``value_gap_gate`` gives
    ``λ * c`` for both soft and hard cuts. The callable returns
    ``[N, N]``. ``gate=None`` is the one-step target (a zero matrix).

    The target is
    ``G_t = r_{t+1} + γ_{t+1} ((1 - c) V(s_{t+1}) + c G_{t+1})`` with ``V``
    the delayed state value. The gate returns ``[N, N]``. Row ``t``,
    column ``s`` is the continuation at absolute step ``s`` for the
    return that started at ``t``, and that row is unrolled on its own.
    The objective then zeros a
    continuation that would leave the run. For any horizon, the last
    step of an episode is not updated when its target depends on the next
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
    gate cut on a later in-run step still bootstraps. At zero temperature,
    ``V = Q_backup(s, argmax_a Q_actions(s, a))``; ties choose the lowest
    action index. At positive temperature ``alpha``, the action scores define
    ``pi = softmax(Q_actions / alpha)`` and
    ``V = E_pi[Q_backup] + alpha H(pi)``. ``value`` transforms all three Q
    tensors before target construction. The online predictions never select
    backup actions unless the caller explicitly passes those scores.
    The loss is the weighted mean of ``delta**2``, where
    ``delta = G - Q(s, a)``. Both backup inputs remain outside the gradient path.
    The returned loss is a dict. ``action_value`` is the TD loss and
    trains Q.
    Packing nullable ``action`` fields derives the batch mask
    ``objective_data["action_executed"]`` to mark real incoming actions. False
    marks an automatic transition (such as an inner reset): its arbitrary
    integer placeholder is never validated, fitted, or used to cut a
    value-gap trace. No action is selected from the preceding state's Q scores.
    Backups pass through that automatic transition using its reward and supplied
    discount, within the same group only. Its outgoing Q row has zero loss weight.
    Reset observations remain available as states for the next real decision.
    Built-in n-step gates count executed decisions unless given an explicit clock.
    The trace never crosses a run break. At a ``terminated`` or
    ``truncated`` step ``γ`` is ``discount`` from those flags and
    multiplies both the bootstrap and the continued return, so a ``0``
    discount ends the trace while a non-zero truncation gamma carries it
    (discounted) into the reset frame's return. A gate that cuts on the
    action reads detached Q.
    ``value_gap_gate(beta=, normalize=, policy_delayed=, value_delayed=, bias=, eps=)``
    controls that cut. ``policy_delayed`` picks ``a* = argmax``.
    ``value_delayed`` scores ``V = Q(s, a*)`` and the taken action.
    Either flag, or both, reads delayed Q. The signed gap is
    ``Q(s, a_taken) - V``. ``normalize=True`` divides by
    ``|max_a Q| + eps`` (``eps`` required). ``bias`` is subtracted and
    the gap is capped at ``0``. For positive beta, ``c = 2 ** (gap / beta)``.
    Zero beta gives ``1`` for zero gap and ``0`` for negative gap.
    ``normalize=False`` uses the raw signed gap and takes no ``eps``.
    A gap of ``0`` continues at ``c = 1``. For positive beta, a gap of
    ``-beta`` continues at ``1/2``. A larger ``beta`` is a softer cut.
    A negative gap moves the return toward the delayed state value.
    ``value_gap_gate`` reads online Q, delayed Q, or both. It leaves
    oracle columns such as ``q_star`` unread. ``metrics["entropy"]`` is the in-run mean of
    ``H[softmax(Q / α)]`` on online Q when ``temperature > 0``.
    ``metrics["backup"]`` is the per-row Bellman target ``G`` (``[P]``)
    the taken action is regressed to. ``metrics["backup_weight"]`` is
    the matching per-row weight (``0`` when the step is out of the loss
    and out of logged metrics). ``metrics["in_run_backup"]`` is
    ``backup`` on the rows with ``backup_weight > 0``.
    All of these are detached. Built-in column gates use a parallel affine
    scan with linear workspace. Start-dependent gates are evaluated in bounded
    strips, avoiding a dense per-head continuation matrix.

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
            shape is accepted. Same callable on all Q inputs. Does not
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
            (``n=``). ``value_gap_gate`` is the optimality-gap cut.
            ``policy_delayed`` picks ``a* = argmax``. ``value_delayed``
            scores ``V = Q(s, a*)`` and the taken action. Either flag,
            or both, reads delayed Q. The signed gap is
            ``Q(s, a_taken) - V``. ``normalize=True`` divides by
            ``|max_a Q| + eps`` on the value Q (``eps`` required).
            ``bias`` is subtracted, then the gap is capped at ``0``.
            For positive beta, ``c = 2 ** (gap / beta)``. Zero beta gives
            ``1`` for zero gap and ``0`` for negative gap. A gap of ``0``
            continues at ``c = 1``. For positive beta, a gap of ``-beta``
            continues at ``1/2``. A larger ``beta`` is a softer cut. ``normalize=False``
            uses the raw signed gap and takes no ``eps``. ``general_gate(gates=)`` is the
            element-wise product of those matrices. A callable returning
            ``[N, N]`` is accepted. Row ``t``, column ``s`` is the continuation at
            absolute step ``s`` for the return that started at ``t``.
            See :class:`~mouse_core.objectives.transforms.Gate`.
        temperature: SAC / soft Q-learning ``alpha`` (>= 0). Required.
            Zero selects the greedy action from ``backup_action_predictions``. Positive
            values use that head's Boltzmann policy and entropy bonus, evaluated
            with ``backup_value_predictions``. Same units as rollout temperature.
        cross_group_backups: Required. ``"bootstrap"``, ``"ignore"``, or
            ``"fault"``. For any horizon, the last step of an episode is not
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
            stays inside the episode stays either way.
    """

    def __init__(
        self,
        *,
        discount: Discount | None,
        reward: Reward | None,
        value: Value | None,
        temperature: float,
        cross_group_backups: CrossGroupBackups,
        action_key: str = "action",
        terminated_key: str = "terminated",
        truncated_key: str = "truncated",
        gate: Gate | None,
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        self.temperature = _require_temperature(temperature)
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
        backup_value_predictions: torch.Tensor,
        backup_action_predictions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        group_id: torch.Tensor,
        predictions: torch.Tensor,
        backup_value_predictions: torch.Tensor,
        backup_action_predictions: torch.Tensor,
        value_predictions: None = None,
        targets: None = None,
        include_statistics: bool = True,
        return_backup_lengths: bool = False,
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
        include_statistics: bool = True,
        return_backup_lengths: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "DqnObjective",
            value_predictions=value_predictions,
            targets=targets,
        )
        q: torch.Tensor = predictions
        q_target: torch.Tensor = _require_prediction(
            backup_value_predictions, owner="DqnObjective", name="backup_value_predictions"
        ).detach()
        q_actions = _require_prediction(
            backup_action_predictions, owner="DqnObjective", name="backup_action_predictions"
        ).detach()

        if q.ndim not in (1, 2, 3):
            raise ValueError("DQN predictions must have shape [P], [P, A], [P, K], or [P, K, A].")
        if q_target.ndim not in (2, 3) or q_target.shape[0] != q.shape[0]:
            raise ValueError("DQN backup_value_predictions must have shape [P, A] or [P, K, A].")
        selected = q.ndim == q_target.ndim - 1
        if q.shape != (q_target.shape[:-1] if selected else q_target.shape):
            raise ValueError("DQN predictions must match backup_value_predictions, optionally without the action dimension.")
        if selected and self.cql_weight > 0.0:
            raise ValueError("CQL requires all-action predictions [P, A] or [P, K, A].")
        if q.dtype != torch.float32:
            raise TypeError(f"DQN expects float32 predictions, got {q.dtype}.")
        for name, tensor in (("backup_value_predictions", q_target), ("backup_action_predictions", q_actions)):
            if tensor.dtype != torch.float32:
                raise TypeError(f"DQN expects float32 {name}, got {tensor.dtype}.")
            if tensor.shape != q_target.shape:
                raise ValueError(f"DQN {name} shape {tuple(tensor.shape)} must match expected shape {tuple(q_target.shape)}.")
            if tensor.device != q.device:
                raise ValueError(f"DQN {name} must share the predictions device.")
        P, A = q_target.shape[0], q_target.shape[-1]
        heads = q_target.shape[1] if q_target.ndim == 3 else None
        if heads is not None and heads < 1:
            raise ValueError("DQN needs at least one head.")
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
        executed = executed_actions(objective_data, action)
        action = canonical_actions(objective_data, action)
        _require_action_ids(action, A)
        objective_data = {**objective_data, self.action_key: action}

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
            heads=heads,
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

        q_actions = _apply_value(
            transform=self.value, name="value", value=q_actions,
            objective_data=objective_data, step_of=step_of, N=N,
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

        # No action is chosen from the state immediately before an automatic
        # reset. Keep the observation in the history, but never fit or maximize
        # its Q row. A backup passes through the automatic transition instead.
        readout = last_rows < P
        _validate_request_actions(readout, executed, group_id)
        _validate_missing_bootstrap(readout, discount_all, group_id, self.cross_group_backups)
        automatic = torch.cat([~executed[1:] & (pair_weight > 0), executed.new_zeros(1)]) | ~readout
        online_mask = automatic[step_of].reshape((P,) + (1,) * (q.ndim - 1))
        backup_mask = automatic[step_of].reshape((P,) + (1,) * (q_target.ndim - 1))
        q = q.masked_fill(online_mask, 0)
        q_target = q_target.masked_fill(backup_mask, 0)
        q_actions = q_actions.masked_fill(backup_mask, 0)
        gather_action = next_actions.reshape((P,) + (1,) * (q_target.ndim - 1))
        gather_action = gather_action.expand(*q_target.shape[:-1], 1)
        q_values = q if selected else q.gather(dim=-1, index=gather_action).squeeze(-1)
        v_step = _backup_state_value(
            q_actions=_step_predictions(q_actions, last_rows), q_backup=_step_predictions(q_target, last_rows),
            temperature=self.temperature,
        )
        if heads is not None and reward.ndim == 1:
            reward = reward[:, None].expand(-1, heads)
        with torch.no_grad():
            prepared = (_PreparedGate(n=N, columns=q.new_zeros(N)) if self.gate is None else
                        _prepare_gate(self.gate, **objective_data,
                                      q=_step_predictions(q, last_rows).detach(), q_delayed=_step_predictions(q_target, last_rows).detach()))
            if prepared.columns is not None:
                columns = prepared.columns.to(dtype=value_dtype, device=device)
                mask = automatic[:, None] if columns.ndim == 2 else automatic
                columns = torch.where(mask, 1.0, columns)
                prepared = _PreparedGate(n=N, columns=columns)
            else:
                original = prepared
                def block(start: int, end: int) -> torch.Tensor:
                    result = original.block(start, end).to(dtype=value_dtype, device=device)
                    mask = automatic[None, :, None] if result.ndim == 3 else automatic[None, :]
                    return torch.where(mask, 1.0, result)
                prepared = _PreparedGate(n=N, block=block)
        decision_count = (pair_weight * executed[1:]) if return_backup_lengths else None
        backup_lengths = None
        if prepared.columns is not None:
            pair_target, participate, backup_lengths = _column_targets(
                reward=reward, discount=discount_all, v_step=v_step,
                pair_weight=pair_weight, columns=prepared.columns,
                cross_group_backups=self.cross_group_backups,
                decision_count=decision_count,
            )
        else:
            if return_backup_lengths:
                backup_lengths = reward.new_empty((N - 1,) + v_step.shape[1:])
            pair_target, participate = _continuation_targets(
                reward=reward, discount_all=discount_all, v_step=v_step,
                pair_weight=pair_weight, continuation=prepared,
                cross_group_backups="ignore" if self.cross_group_backups == "fault" else self.cross_group_backups,
                backup_lengths=backup_lengths, decision_count=decision_count,
            )
        pair_weight = pair_weight * executed[1:].to(dtype=pair_weight.dtype)
        if heads is not None:
            pair_weight = pair_weight[:, None]
        if self.cross_group_backups == "fault":
            _raise_if_cross_group_backup(in_run=pair_weight > 0, participate=participate)
        pair_weight = pair_weight * participate
        row_weight = _pair_values_to_rows(pair_weight, step_of)
        td_target = _pair_values_to_rows(pair_target, step_of)

        # δ = G - Q(s, a). action_value trains Q.
        active = row_weight > 0
        delta = torch.where(active, td_target, 0) - torch.where(active, q_values, 0)
        td_sq = delta ** 2

        per_row = td_sq
        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (torch.where(active, td_target, 0).abs() + self.cql_scale_q_eps).detach()
            cql_q = q.masked_fill(~active[..., None], 0)
            cql_penalty = torch.logsumexp(cql_q, dim=-1) - torch.where(active, q_values, 0)
            per_row = per_row + self.cql_weight * q_scale * cql_penalty
            if include_statistics:
                cql_penalty_mean = _weighted_mean(cql_penalty.detach(), row_weight)

        td_loss = _weighted_mean(per_row, row_weight)
        losses: dict[str, torch.Tensor] = {"action_value": td_loss}

        metrics: dict[str, float | torch.Tensor] = {
            "backup": td_target.detach(), "backup_weight": row_weight.detach(),
        }
        if backup_lengths is not None:
            metrics["backup_length"] = _pair_values_to_rows(backup_lengths, step_of).detach()
        if include_statistics:
            metrics["in_run_backup"] = td_target.detach()[active]
            curr_max_q = q if selected else q.amax(dim=-1)
            q_mean, q_std, q_min, q_max = _in_run_stats(curr_max_q.detach(), row_weight)
            metrics.update({
                "q_values_mean": q_mean, "q_values_std": q_std,
                "q_values_min": q_min, "q_values_max": q_max,
                "action_value": _weighted_mean(td_sq.detach(), row_weight),
            })
            if cql_penalty_mean is not None:
                metrics["cql_penalty"] = cql_penalty_mean
            if self.temperature > 0.0:
                entropy_q = (q_actions if selected else q).detach().masked_fill(~active[..., None], 0)
                metrics["entropy"] = _weighted_mean(
                    _boltzmann_entropy(entropy_q, temperature=self.temperature), row_weight,
                )
        return losses, metrics
