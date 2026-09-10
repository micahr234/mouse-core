"""TD(λ) DQN objective with a delayed target network."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


def _require_done_codes(
    objective_data: TensorDict,
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


def _boundary_discounts(
    *,
    episode_done: torch.Tensor,
    task_done: torch.Tensor,
    gamma_step: float,
    gamma_episode_terminal: float,
    gamma_episode_truncated: float,
    gamma_task_terminal: float,
    gamma_task_truncated: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Per-transition discount from mouse-gym ``episode_done`` / ``task_done``.

    Both fields use codes ``0`` / ``1`` / ``2``. The bootstrap is multiplied by
    the episode gamma, then by the task gamma. ``task_done == 0`` uses ``1.0``
    so a mid-task step is unchanged by the task factor. When a task ends both
    fire (e.g. ``episode_done=1`` and ``task_done=2``) and the product is used;
    a task gamma of ``0.0`` zeros the whole bootstrap.
    """
    episode_gammas = torch.tensor(
        [gamma_step, gamma_episode_terminal, gamma_episode_truncated],
        dtype=dtype,
        device=device,
    )
    task_gammas = torch.tensor(
        [1.0, gamma_task_terminal, gamma_task_truncated],
        dtype=dtype,
        device=device,
    )
    return episode_gammas[episode_done] * task_gammas[task_done]


def _head_output_layout(
    objective_data: TensorDict,
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
    objective_data: TensorDict,
    N: int,
    device: torch.device | str | None,
    *,
    grouping_field: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[N-1]`` weights: ``1.0`` when ``(i, i+1)`` share a run, else ``0.0``.

    A run is the same ``sequence_id`` and, when ``grouping_field`` is set and
    present, the same grouping column. Batches without those columns skip the
    corresponding check (every pair stays weight ``1``).
    """
    if device is None:
        device = torch.device("cpu")
    if N < 2:
        return torch.zeros(0, dtype=dtype, device=device)
    same_run = torch.ones(N - 1, dtype=torch.bool, device=device)
    if "sequence_id" in objective_data.keys():
        sequence_id = objective_data["sequence_id"]
        if sequence_id.shape != torch.Size([N]):
            raise ValueError(
                f"sequence_id must have shape [{N}], got {tuple(sequence_id.shape)}."
            )
        same_run &= sequence_id[1:] == sequence_id[:-1]
    if grouping_field is not None and grouping_field in objective_data.keys():
        grouping = objective_data[grouping_field]
        if grouping.shape != torch.Size([N]):
            raise ValueError(
                f"{grouping_field} must have shape [{N}], got {tuple(grouping.shape)}."
            )
        same_run &= grouping[1:] == grouping[:-1]
    return same_run.to(dtype=dtype)


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


def _greedy_from_online_q(
    *,
    q: torch.Tensor,
    action: torch.Tensor,
    last_rows: torch.Tensor,
) -> torch.Tensor:
    """``[N-1]``: 1 if the action taken from step ``i`` matches online Q.

    Compares the taken action (stored at ``i+1``) to the online network's
    scores at step ``i`` (its last head-output row). Ties count as a match
    if the taken action is among the max scores. Never reads oracle columns
    such as ``info_q_star``.
    """
    scores = q.detach()[last_rows][:-1]  # [N-1, A]  Q(s_i)
    taken = action[1:].unsqueeze(-1)  # [N-1, 1]  a_i
    is_max = scores == scores.amax(dim=-1, keepdim=True)
    return is_max.gather(dim=-1, index=taken).squeeze(-1).to(dtype=scores.dtype)


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


@torch.no_grad()
def _td_lambda_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    v_step: torch.Tensor,
    pair_weight: torch.Tensor,
    td_lambda: float,
    greedy_from: torch.Tensor | None,
) -> torch.Tensor:
    """TD(λ) target for every pair ``(t, t+1)``, shape ``[N-1]``.

    ``G_t = r_{t+1} + γ_{t+1} * ((1 - λ c_t) * V_{t+1} + λ c_t * G_{t+1})``
    where ``V`` is the delayed max-Q and ``c_t`` says whether the trace
    continues through ``s_{t+1}``: pair ``t+1`` must exist and be in-run, and
    with Watkins the action taken from ``s_{t+1}`` must be online-greedy
    (``greedy_from``). Episode / task boundaries are handled by ``γ_{t+1}``
    itself — the done-code discount from ``_boundary_discounts`` multiplies
    both the bootstrap and the continued return, so a ``0`` gamma ends the
    trace and a non-zero truncation gamma carries it through, discounted.
    ``λ = 0`` or ``c_t = 0`` is exactly the one-step target ``r + γ V``.
    Out-of-run pairs return ``0`` (their rows carry weight ``0``).
    """
    r = reward[1:].to(dtype=v_step.dtype)  # [N-1]  r_t (stored at t+1)
    g = discount_all[1:]  # [N-1]  γ_t from done codes at t+1
    v_next = v_step[1:]  # [N-1]  V(s_{t+1})
    in_run = pair_weight > 0
    cont = _shift_next(in_run.to(dtype=v_step.dtype))
    if greedy_from is not None:
        cont = cont * _shift_next(greedy_from)
    mix = float(td_lambda) * cont
    a = r + g * (1.0 - mix) * v_next
    if td_lambda == 0.0:
        returns = a
    else:
        returns = _affine_scan_backward(a, g * mix)
    return returns * in_run.to(dtype=returns.dtype)


def _affine(
    values: torch.Tensor, *, scale: float, shift: float
) -> torch.Tensor:
    """``scale * values + shift``. Identity is ``1`` / ``0``.

    Used for the TD reward and for online / delayed Q. Does not mutate
    ``objective_data`` or the prediction tensors.
    """
    if float(scale) == 1.0 and float(shift) == 0.0:
        return values
    return values * float(scale) + float(shift)


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
    ``(objective_data, predictions, delayed_predictions)``. Online Q is
    ``predictions["action_value"]``; bootstrap Q is
    ``delayed_predictions["action_value"]`` from the delayed
    :class:`~mouse_core.models.base.Model` (``model.delayed_copy()``) run on
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

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is set
    and present, the same grouping column (typically ``task_index``). Neighbor
    reads (action / reward / done / next Q at ``i+1``) must stay in-run: an
    out-of-run pair still has a loss term, but it is multiplied by ``0`` so
    output ``i`` does not affect the scalar loss or the gradient. If every
    weight is ``0`` the loss is ``0``. Episode resets inside a run
    (``episode_done`` 1/2, then a reset frame) are still in-run and may train.
    Gamma is the Bellman discount from the done codes at ``i+1`` inside a
    same-run pair — it is not a run mask.

    The ``episode_done`` / ``task_done`` codes stored at ``i+1`` determine the
    discount applied to the bootstrap value. Both factors always multiply:
    ``V ← episode_gamma * task_gamma * V``. ``task_done == 0`` uses task
    factor ``1.0``.

    The target is the TD(λ) return along the run,
    ``G_i = r + γ * ((1 - λ) * V(s_{i+1}) + λ * G_{i+1})`` with ``V`` the
    delayed max-Q, so ``td_lambda=0`` (default) is the plain one-step target
    ``r + γ V`` and ``td_lambda=1`` is the full n-step return to the end of
    the run. The trace never crosses a run break. At an episode / task
    boundary ``γ`` is the corresponding done-code gamma and multiplies both
    the bootstrap and the continued return, so ``gamma_*_terminal = 0`` ends
    the trace there while a non-zero truncation gamma carries it (discounted)
    into the reset frame's return. Off-policy behavior is not
    corrected unless ``watkins=True`` (Watkins's Q(λ)), which also cuts the
    trace wherever the taken action is not the online argmax of
    ``predictions["action_value"]`` (ties included; never compared against
    oracle columns such as ``info_q_star``). ``metrics["watkins_greedy_frac"]``
    then reports the in-run fraction of taken actions that were greedy —
    near ``0`` means the traces are cut everywhere and the target is one-step.
    The λ-return is computed with a parallel scan on the device (no host syncs).

    Those columns arrive in ``objective_data`` only if they are listed in the
    tokenizer ``objective_fields`` keep-list (input fields are not auto-copied).
    ``task_done`` is an objective column only — do not add it as a tokenizer
    input field or embedder modality, or it will be fed to the transformer::

        tokenizer = NumericTokenizer(
            ...,
            objective_fields=[
                {"input_field": "action"},
                {"input_field": "reward"},
                {"input_field": "episode_done"},
                {"input_field": "task_done"},
            ],
        )

    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | episode_done | task_done | Meaning                          | Discount                                      |
    +==============+===========+==================================+===============================================+
    | 0            | 0         | Running                          | ``gamma_step``                                |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 1            | 0         | Episode terminated (mid-task)    | ``gamma_episode_terminal``                    |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 2            | 0         | Episode truncated (mid-task)     | ``gamma_episode_truncated``                   |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 1            | 2         | Last episode terminated          | ``gamma_episode_terminal * gamma_task_truncated`` |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 2            | 2         | Last episode truncated           | ``gamma_episode_truncated * gamma_task_truncated`` |
    +--------------+-----------+----------------------------------+-----------------------------------------------+
    | 1 or 2       | 1         | Task terminated (reserved)       | episode gamma ``* gamma_task_terminal``       |
    +--------------+-----------+----------------------------------+-----------------------------------------------+

    Args:
        gamma_step: Discount factor for running (non-terminal) transitions
            (``episode_done == 0``).
        gamma_episode_terminal: Discount applied when the episode terminates
            naturally (``episode_done == 1``). Set to ``1.0`` to bootstrap
            across episode boundaries (recommended for multi-episode MOUSE
            tasks).
        gamma_episode_truncated: Discount applied when the episode is truncated
            (``episode_done == 2``). Set to ``1.0`` to bootstrap across
            episode boundaries.
        gamma_task_terminal: Extra discount when the task terminates
            (``task_done == 1``; reserved, unused by mouse-gym today).
            Multiplies the episode discount. ``task_done == 0`` uses ``1.0``.
        gamma_task_truncated: Extra discount when the task is truncated
            (``task_done == 2``; last episode of ``episodes_per_task``).
            Multiplies the episode discount. ``0.0`` zeros the bootstrap.
        action_key: Key in ``objective_data`` that holds the integer action.
        reward_key: Key in ``objective_data`` that holds the per-step reward.
        reward_scale: Multiplier applied to ``reward`` before the TD target
            (default ``1.0``). Does not change ``objective_data`` or the
            tokenizer.
        reward_shift: Offset added after ``reward_scale`` (default ``0.0``).
            The TD reward is ``reward_scale * r + reward_shift``.
        q_scale: Multiplier applied to online and delayed ``action_value``
            before the TD error (default ``1.0``). Same affine on both
            networks. Does not change the prediction tensors or eval
            ``argmax``.
        q_shift: Offset added after ``q_scale`` (default ``0.0``). The TD
            Q is ``q_scale * Q + q_shift``.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        cql_weight: Alpha coefficient for the Conservative Q-Learning penalty.
            ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.
        td_lambda: λ of the TD(λ) target in ``[0, 1]``. ``0.0`` (default) is
            the one-step target; ``1.0`` is the full in-run n-step return.
        watkins: Cut the λ-trace at non-greedy actions (Watkins's Q(λ)).
    """

    def __init__(
        self,
        *,
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
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
        td_lambda: float = 0.0,
        watkins: bool = False,
    ) -> None:
        if not 0.0 <= float(td_lambda) <= 1.0:
            raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}.")
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
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps
        self.td_lambda = float(td_lambda)
        self.watkins = bool(watkins)

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("DqnObjective requires delayed_predictions.")
        q: torch.Tensor = predictions["action_value"]
        q_target: torch.Tensor = delayed_predictions["action_value"].detach()

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
                f"DQN objective expects action shape [N], got {tuple(action.shape)}."
            )
        N = int(action.shape[0])

        if N < 2:
            raise ValueError("Not enough valid q values in data.")

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([N]):
            raise ValueError(f"DQN objective expects reward shape [{N}], got {tuple(reward.shape)}.")
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
        # toward the same target, and the bootstrap reads step i+1's *last*
        # head-output row.
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

        # Each token at position i encodes (obs_i, action_{i-1}, reward_{i-1},
        # episode_done_{i-1}, task_done_{i-1}), i.e. the action, reward, and
        # done codes stored at i are the ones that *produced* obs_i, not the
        # ones taken *from* obs_i.  The transition out of state i is therefore
        # described by the fields stored at i+1.
        step_next = (step_of + 1).clamp(max=N - 1)  # [P] (final step clamped, weight 0)
        next_actions = action[step_next]            # [P]  a_i (stored at i+1)

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

        q_values = q.gather(dim=-1, index=next_actions.unsqueeze(-1)).squeeze(-1)  # [P]
        greedy_from = (
            _greedy_from_online_q(q=q, action=action, last_rows=last_rows)
            if self.watkins
            else None
        )
        pair_target = _td_lambda_targets(
            reward=reward,
            discount_all=discount_all,
            v_step=q_target[last_rows].amax(dim=-1),  # [N]  V(s_i) = max_a Q_target
            pair_weight=pair_weight,
            td_lambda=self.td_lambda,
            greedy_from=greedy_from,
        )
        td_target = _pair_values_to_rows(pair_target, step_of)  # [P]

        loss = (q_values - td_target) ** 2

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
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean
        if greedy_from is not None:
            named["watkins_greedy_frac"] = _weighted_mean(greedy_from, pair_weight)

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
