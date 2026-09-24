"""TD(λ) DQN objective with a delayed target network."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective
from mouse_core.objectives.transforms import (
    Discount,
    Gate,
    Reward,
    Value,
    _apply_transform,
    _apply_value,
    _require_transform,
)


def _require_done_codes(
    objective_data: dict[str, torch.Tensor],
    *,
    episode_done_key: str,
    task_done_key: str,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and validate mouse-gym ``episode_done`` / ``task_done`` columns."""
    codes: list[torch.Tensor] = []
    for key in (episode_done_key, task_done_key):
        values = objective_data[key]
        if values.dtype != torch.int64:
            raise TypeError(f"{key} must be int64, got {values.dtype}.")
        if values.shape != torch.Size([N]):
            raise ValueError(
                f"objective expects {key} shape [{N}], got {tuple(values.shape)}."
            )
        if bool((values < 0).any() or (values > 2).any()):
            raise ValueError(f"{key} codes must be 0, 1, or 2.")
        codes.append(values)
    return codes[0], codes[1]


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
    objective_data: dict[str, torch.Tensor],
    N: int,
    device: torch.device | str | None,
    *,
    grouping_field: str | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[N-1]`` weights: ``1.0`` when ``(i, i+1)`` share a run, else ``0.0``.

    A run is the same ``sequence_id`` (when that column is present) and, when
    ``grouping_field`` is set, the same grouping column. ``grouping_field``
    set but missing from ``objective_data`` is an error — it must not silently
    train across task boundaries. ``N < 2`` raises.
    """
    if device is None:
        device = torch.device("cpu")
    if N < 2:
        raise ValueError(f"pair weight needs at least 2 steps, got {N}.")
    same_run = torch.ones(N - 1, dtype=torch.bool, device=device)
    if "sequence_id" in objective_data.keys():
        sequence_id = objective_data["sequence_id"]
        if sequence_id.shape != torch.Size([N]):
            raise ValueError(
                f"sequence_id must have shape [{N}], got {tuple(sequence_id.shape)}."
            )
        same_run &= sequence_id[1:] == sequence_id[:-1]
    if grouping_field is not None:
        if grouping_field not in objective_data.keys():
            raise KeyError(
                f"grouping_field={grouping_field!r} is not a column in "
                "objective_data; include it in tokenizer objective_fields."
            )
        grouping = objective_data[grouping_field]
        if grouping.shape != torch.Size([N]):
            raise ValueError(
                f"{grouping_field} must have shape [{N}], got {tuple(grouping.shape)}."
            )
        same_run &= grouping[1:] == grouping[:-1]
    return same_run.to(dtype=dtype)


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


def _block_returns(
    *,
    continuation: torch.Tensor,
    col_mask: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    v_next: torch.Tensor,
    t0: int,
    rows: int,
) -> torch.Tensor:
    """Returns for starts ``t0 .. t0+rows`` from the gate matrix.

    Reads columns ``t0+1:`` of those rows into one strip. Entries before
    each start are outside that start's return: their step is ``1`` so
    the cumprod is unchanged, and their reward term is ``0``.
    """
    device = continuation.device
    dtype = reward.dtype
    end = t0 + rows
    width = int(col_mask.shape[0]) - t0
    step = continuation[t0:end, t0 + 1 :].contiguous().to(dtype=dtype)
    g = discount[t0:]
    step.mul_(col_mask[t0:])
    term = reward[t0:] + g * (1 - step) * v_next[t0:]
    step.mul_(g)
    if rows > 1:
        local_row = torch.arange(rows, device=device).unsqueeze(1)
        local_col = torch.arange(rows, device=device)
        before = local_col < local_row
        step[:, :rows] = torch.where(before, step.new_ones(()), step[:, :rows])
        term[:, :rows] = torch.where(before, term.new_zeros(()), term[:, :rows])
    if width > 1:
        torch.cumprod(step, dim=1, out=step)
        term[:, 1:].mul_(step[:, :-1])
    return term.sum(dim=1)


@torch.no_grad()
def _continuation_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    v_step: torch.Tensor,
    pair_weight: torch.Tensor,
    continuation: torch.Tensor,
) -> torch.Tensor:
    """Return for every pair ``(t, t+1)``, shape ``[N-1]``.

    ``G_t = r_{t+1} + γ_{t+1} ((1 - c) V_{t+1} + c G_{t+1})``.
    ``continuation`` is the gate matrix ``[N, N]``: row ``t``, column
    ``s`` is the continuation at absolute step ``s`` for the return that
    started at ``t``. Rows are cumprod'd in blocks of
    ``_CONTINUATION_ROWS`` and never read another start's return. The
    in-run mask zeros a continuation that would leave the run.
    Out-of-run pairs return ``0``.
    """
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
    for t0 in range(0, T, _CONTINUATION_ROWS):
        rows = min(_CONTINUATION_ROWS, T - t0)
        returns[t0 : t0 + rows] = _block_returns(
            continuation=continuation,
            col_mask=col_mask,
            reward=r,
            discount=g,
            v_next=v_next,
            t0=t0,
            rows=rows,
        )
    return returns * in_run.to(dtype=returns.dtype)


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


def _require_w(
    w: torch.Tensor | None,
    *,
    P: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """One offset per head-output row. ``None`` is regular DQN.

    A one-output regression head returns ``[P, 1]``. ``[P]`` is the
    same row. The tensor stays in the graph so that head can learn.
    """
    if w is None:
        return None
    if not isinstance(w, torch.Tensor):
        raise TypeError(f"w must be a Tensor or None, got {type(w)}.")
    if w.dtype != dtype:
        raise TypeError(f"w must be {dtype}, got {w.dtype}.")
    if w.device != device:
        raise ValueError(f"w must be on {device}, got {w.device}.")
    if tuple(w.shape) == (P, 1):
        return w.squeeze(-1)
    if tuple(w.shape) != (P,):
        raise ValueError(
            f"w must have shape [{P}] or [{P}, 1], got {tuple(w.shape)}."
        )
    return w


def _pair_values_to_rows(
    pair_values: torch.Tensor,
    step_of: torch.Tensor,
) -> torch.Tensor:
    """Broadcast ``[N-1]`` per-pair values onto head-output rows ``[P]``."""
    padded = torch.cat([pair_values, pair_values.new_zeros(1)])
    return padded[step_of]


class DqnObjective(Objective):
    """Bellman TD(λ) objective with a delayed target network.

    Instantiate with hyperparameters, then call with
    ``objective_data=``, the online Q tensor as ``predictions=``, the
    delayed Q tensor as ``delayed_predictions=``, and the offset as
    ``w=``. Both Q tensors come
    from the Q head on
    :class:`~mouse_core.models.base.Model`
    (``model.copy(heads=(q_head,), backbone=True, reasoner=False)``) run on
    the same ``TokenBatch``. The delayed tensor is detached before
    the Bellman target, so the TD error does not backprop through it.
    The offset head is not in that copy.

    Q rows are **per head-output token** (``[P, A]``), not per step: a step may
    own several head-output tokens (tokenizer input field flagged
    ``head_output=True`` emitting more than one token). The
    ``head_output_count`` column stamped by ``pack_token_batch`` maps rows to
    steps, so predictions and step fields can never misalign. Every
    head-output row of step ``i`` trains toward the *same* TD target; the
    bootstrap reads the *last* head-output row of step ``i+1`` (the most
    informed one).

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is set
    and present, the same grouping column (typically ``task_index``). Neighbor
    reads (action / reward / done / next Q at ``i+1``) must stay in-run: an
    out-of-run pair still has a loss term, but it is multiplied by ``0`` so
    output ``i`` does not affect the scalar loss or the gradient. If every
    weight is ``0`` the loss is ``0``. Episode resets inside a run
    (``episode_done`` 1/2, then a reset frame) are still in-run and may train.
    Gamma is the Bellman discount from the done codes at ``i+1`` inside a
    same-run pair — it is not a run mask.

    ``reward(**objective_data)`` supplies the per-step reward; the value
    stored at ``i+1`` is ``r_t``. ``affine_reward`` is the column
    affine; ``boundary_reward`` applies episode / task scale and shift extras.
    ``value(value=..., **objective_data)`` supplies the per-step affine
    on online and delayed Q. ``affine_value`` is the prediction affine;
    ``boundary_value`` applies episode / task scale and shift extras.
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
    continuation that would leave the run. ``V`` is delayed max-Q when
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
    ``δ = G - Q(s, a)`` and ``G`` the backup above. ``w=None`` is that
    loss. A tensor is the online one-output regression head on the
    history ``h``, one scalar per head-output row (``[P]`` or ``[P, 1]``).
    The loss is then the weighted mean of ``(δ - w)²``. Leave that head
    out of :meth:`~mouse_core.models.base.Model.copy`: the delayed model
    does not run it, and Polyak does not average it. Gradient descent
    trains it, and that head's learning rate sets how fast the offset
    moves. ``metrics["td_offset"]`` is the in-run mean of ``w`` when
    ``w`` is a tensor. One-step,
    ``temperature=0``, and ``double=False`` make ``δ`` the residual
    ``r + γ max_a Q_delayed(s', a) - Q(s, a)``.
    The trace never crosses a run break. At an episode /
    task boundary ``γ`` is ``discount`` at the done codes stored there and
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
    ``H[softmax(Q / α)]`` on online Q when ``temperature > 0``. The
    continuation matrix stays the one ``[N, N]`` the gate returned. Rows
    are cumprod'd in blocks, and that read does not sync the host.

    Those columns arrive in ``objective_data`` only if they are listed in the
    tokenizer ``objective_fields`` keep-list (input fields are not auto-copied).
    ``task_done`` is an objective column only — do not add it as a tokenizer
    input field, or it will be fed to the transformer::

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
        discount: Per-step γ from unpacked ``objective_data`` columns.
            ``boundary_discount`` is the standard ``gamma_step`` × extra
            lookup (``None`` skips the call and uses ``1``);
            any ``discount(**objective_data) -> [N]`` is accepted.
        reward: Per-step reward from unpacked ``objective_data`` columns.
            ``affine_reward`` is the column affine; ``boundary_reward``
            applies episode / task scale and shift extras
            (``None`` skips the call);
            any ``reward(**objective_data) -> [N]`` is accepted.
            Does not change ``objective_data``.
        value: Per-step affine on online and delayed Q from unpacked
            ``objective_data`` columns plus ``value=``. ``affine_value``
            is the prediction affine; ``boundary_value`` applies episode
            / task scale and shift extras
            (``None`` skips the call);
            any ``value(value=..., **objective_data)`` returning the same
            shape is accepted. Same callable on both networks. Does not
            change the prediction tensors or eval ``argmax``.
        action_key: Key in ``objective_data`` that holds the integer action.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        grouping_field: Step column that isolates runs (typically
            ``task_index``). Required. Pass ``None`` only when the batch
            has no grouping isolation — omitting it is an error, not a
            silent skip.
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
    """

    def __init__(
        self,
        *,
        discount: Discount | None,
        reward: Reward | None,
        value: Value | None,
        temperature: float,
        double: bool,
        action_key: str = "action",
        episode_done_key: str = "episode_done",
        task_done_key: str = "task_done",
        grouping_field: str | None,
        gate: Gate | None,
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        self.temperature = _require_temperature(temperature)
        self.double = bool(double)
        self.discount = _require_transform(discount, name="discount")
        self.reward = _require_transform(reward, name="reward")
        self.value = _require_transform(value, name="value")
        self.action_key = action_key
        self.episode_done_key = episode_done_key
        self.task_done_key = task_done_key
        self.grouping_field = grouping_field
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.gate = _require_transform(gate, name="gate")

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor,
        w: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q: torch.Tensor = predictions
        q_target: torch.Tensor = delayed_predictions.detach()

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

        _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
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
            objective_data,
            N,
            device,
            grouping_field=self.grouping_field,
            dtype=value_dtype,
        )
        # Row weight = the (i, i+1) pair weight of the row's step; rows of the
        # final step have no next step and get weight 0.
        row_weight = torch.cat([pair_weight, pair_weight.new_zeros(1)])[step_of]  # [P]

        # Each token at position i encodes (obs_i, action_{i-1}, reward_{i-1},
        # episode_done_{i-1}, task_done_{i-1}), i.e. the action, reward, and
        # done codes stored at i are the ones that *produced* obs_i, not the
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
        pair_target = _continuation_targets(
            reward=reward,
            discount_all=discount_all,
            v_step=v_step,  # [N]  V(s_i)
            pair_weight=pair_weight,
            continuation=continuation,
        )
        td_target = _pair_values_to_rows(pair_target, step_of)  # [P]

        # δ = G - Q. w is a one-output head on the history. None squares δ.
        # w stays in the graph so that head's learning rate moves the offset.
        delta = td_target - q_values
        offset = _require_w(w, P=P, dtype=value_dtype, device=device)
        loss = delta ** 2 if offset is None else (delta - offset) ** 2

        cql_penalty_mean: torch.Tensor | None = None
        if self.cql_weight > 0.0:
            q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
            cql_penalty = torch.logsumexp(q, dim=-1) - q_values
            loss = loss + self.cql_weight * q_scale * cql_penalty
            cql_penalty_mean = _weighted_mean(cql_penalty.detach(), row_weight)

        loss = _weighted_mean(loss, row_weight)

        curr_max_q = q.amax(dim=-1)  # [P]  max online Q at s_i
        q_mean, q_std, q_min, q_max = _in_run_stats(curr_max_q.detach(), row_weight)
        named: dict[str, torch.Tensor] = {
            "q_values_mean":   q_mean,
            "q_values_std":    q_std,
            "q_values_min":    q_min,
            "q_values_max":    q_max,
            "action_value":    loss.detach(),
        }
        if offset is not None:
            named["td_offset"] = _weighted_mean(offset.detach(), row_weight)
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean
        if self.temperature > 0.0:
            named["entropy"] = _weighted_mean(
                _boltzmann_entropy(q.detach(), temperature=self.temperature),
                row_weight,
            )

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
