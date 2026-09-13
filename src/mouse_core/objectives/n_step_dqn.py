"""n-step return DQN objective with a delayed target network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _affine,
    _boundary_discounts,
    _head_output_layout,
    _in_run_stats,
    _pair_values_to_rows,
    _pair_weight,
    _require_action_ids,
    _require_done_codes,
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


def _shift_pair(values: torch.Tensor, offset: int) -> torch.Tensor:
    """``out[t] = values[t + offset]``, filled with ``0`` / ``False`` at the tail."""
    t = int(values.shape[0])
    if offset == 0:
        return values
    if offset >= t:
        return values.new_zeros(t)
    return torch.cat([values[offset:], values.new_zeros(offset)])


def _bootstrap_at(bootstrap: torch.Tensor, n: int, p: int) -> torch.Tensor:
    """``out[t] = bootstrap[t + n]`` for ``t in [0, p)``, ``0`` if past the end."""
    n_steps = int(bootstrap.shape[0])
    out = bootstrap.new_zeros(p)
    take = min(p, max(0, n_steps - n))
    if take > 0:
        out[:take] = bootstrap[n : n + take]
    return out


@torch.no_grad()
def _n_step_horizon_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    bootstrap: torch.Tensor,
    pair_weight: torch.Tensor,
    horizons: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete n-step returns and validity for each horizon, shapes ``[N-1, H]``.

    Pair ``t`` is the transition out of step ``t`` (reward / gamma stored at
    ``t+1``). Horizon ``n`` is the product-of-gammas return
    ``r_{t+1} + γ_{t+1} r_{t+2} + … + (∏_{k=1}^{n} γ_{t+k}) V_{t+n}``.
    ``discount`` in the loop is that running product: it starts at ``1``
    and multiplies each per-transition ``γ`` so a later ``γ == 0`` zeros
    every remaining reward and the bootstrap. Incomplete windows stay
    ``-inf`` / ``False`` — they are never silently shortened. ``horizons``
    must be sorted ascending. A zero gamma includes that reward, then
    fills every remaining ``n >= k+1`` with the stopped sum and needs no
    later endpoint.
    """
    r = reward[1:].to(dtype=bootstrap.dtype)
    g = discount_all[1:].to(dtype=bootstrap.dtype)
    in_run = pair_weight > 0
    p = int(r.shape[0])
    h_count = len(horizons)
    horizon_index = {n: i for i, n in enumerate(horizons)}
    max_h = horizons[-1]

    acc = r.new_zeros(p)
    discount = r.new_ones(p)
    alive = in_run.clone()
    stopped = torch.zeros(p, dtype=torch.bool, device=r.device)
    candidates = r.new_full((p, h_count), float("-inf"))
    valid = torch.zeros(p, h_count, dtype=torch.bool, device=r.device)

    for k in range(max_h):
        n = k + 1
        r_k = _shift_pair(r, k)
        g_k = _shift_pair(g, k)
        run_k = _shift_pair(in_run, k)
        usable = alive & ~stopped & run_k
        # ``discount`` is ∏ γ already seen; ``discount * g_k`` is the product
        # through this transition (a 0 here zeros the rest of the return).
        acc = acc + discount * r_k * usable.to(dtype=acc.dtype)
        zero_g = usable & (g_k == 0)

        if n in horizon_index:
            h = horizon_index[n]
            boot = usable & ~zero_g
            b_end = _bootstrap_at(bootstrap, n, p)
            bootstrapped = acc + discount * g_k * b_end
            candidates[:, h] = torch.where(boot, bootstrapped, candidates[:, h])
            candidates[:, h] = torch.where(zero_g, acc, candidates[:, h])
            valid[:, h] = valid[:, h] | boot | zero_g

        if n < max_h and bool(zero_g.any()):
            for m in horizons:
                if m > n:
                    hm = horizon_index[m]
                    candidates[:, hm] = torch.where(zero_g, acc, candidates[:, hm])
                    valid[:, hm] = valid[:, hm] | zero_g

        stopped = stopped | zero_g
        alive = alive & (usable | stopped)
        discount = torch.where(usable, discount * g_k, discount)

    return candidates, valid


@torch.no_grad()
def _n_step_return_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    bootstrap: torch.Tensor,
    pair_weight: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete n-step return and validity for every pair ``(t, t+1)``."""
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount_all,
        bootstrap=bootstrap,
        pair_weight=pair_weight,
        horizons=(int(n),),
    )
    target = candidates[:, 0]
    ok = valid[:, 0]
    return torch.where(ok, target, target.new_zeros(target.shape)), ok


def _pair_valid_to_rows(
    pair_valid: torch.Tensor,
    step_of: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast ``[N-1]`` pair validity onto head-output rows ``[P]``."""
    return torch.cat(
        [pair_valid.to(dtype=dtype), pair_valid.new_zeros(1, dtype=dtype)]
    )[step_of]


@dataclass(frozen=True)
class _NStepBatch:
    """Step tensors shared by n-step and max-over-n-step."""

    action: torch.Tensor
    reward: torch.Tensor
    discount_all: torch.Tensor
    pair_weight: torch.Tensor
    step_of: torch.Tensor
    last_rows: torch.Tensor
    next_actions: torch.Tensor
    n_steps: int


def _n_step_batch(
    objective_data: TensorDict,
    *,
    n_pred: int,
    n_actions: int,
    device: torch.device,
    dtype: torch.dtype,
    action_key: str,
    reward_key: str,
    reward_scale: float,
    reward_shift: float,
    episode_done_key: str,
    task_done_key: str,
    grouping_field: str | None,
    gamma_step: float,
    gamma_episode_terminal: float,
    gamma_episode_truncated: float,
    gamma_task_terminal: float,
    gamma_task_truncated: float,
) -> _NStepBatch:
    """Read the transition columns both n-step objectives share."""
    action = objective_data[action_key]
    if action.dtype != torch.int64:
        raise TypeError(f"action must be int64, got {action.dtype}.")
    if action.ndim != 1:
        raise ValueError(
            f"DQN objective expects action shape [N], got {tuple(action.shape)}."
        )
    n_steps = int(action.shape[0])
    _require_action_ids(action, n_actions)
    if n_steps < 2:
        raise ValueError("Not enough valid q values in data.")

    reward = objective_data[reward_key]
    if reward.dtype != torch.float32:
        raise TypeError(f"reward must be float32, got {reward.dtype}.")
    if reward.shape != torch.Size([n_steps]):
        raise ValueError(
            f"DQN objective expects reward shape [{n_steps}], "
            f"got {tuple(reward.shape)}."
        )
    reward = _affine(reward, scale=reward_scale, shift=reward_shift)

    episode_done, task_done = _require_done_codes(
        objective_data,
        episode_done_key=episode_done_key,
        task_done_key=task_done_key,
        N=n_steps,
    )
    step_of, last_rows = _head_output_layout(
        objective_data, N=n_steps, P=n_pred, device=device
    )
    pair_weight = _pair_weight(
        objective_data,
        n_steps,
        device,
        grouping_field=grouping_field,
        dtype=dtype,
    )
    discount_all = _boundary_discounts(
        episode_done=episode_done,
        task_done=task_done,
        gamma_step=gamma_step,
        gamma_episode_terminal=gamma_episode_terminal,
        gamma_episode_truncated=gamma_episode_truncated,
        gamma_task_terminal=gamma_task_terminal,
        gamma_task_truncated=gamma_task_truncated,
        dtype=dtype,
        device=device,
    )
    step_next = (step_of + 1).clamp(max=n_steps - 1)
    next_actions = action[step_next]
    return _NStepBatch(
        action=action,
        reward=reward,
        discount_all=discount_all,
        pair_weight=pair_weight,
        step_of=step_of,
        last_rows=last_rows,
        next_actions=next_actions,
        n_steps=n_steps,
    )


class NStepDqnObjective(Objective):
    """Bellman n-step return objective with a delayed target network.

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
    head-output row of step ``i`` trains toward the *same* n-step return;
    the bootstrap reads the *last* head-output row of the state ``n`` steps
    ahead. A start that does not have ``n`` in-run steps ahead is
    masked (weight ``0``), the same way the last step of one-step TD
    has no next pair. A ``γ == 0`` stop includes that reward and
    completes the return with no later endpoint.

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
    discount applied to the rest of the return. Both factors always multiply:
    ``V ← episode_gamma * task_gamma * V``. ``task_done == 0`` uses task
    factor ``1.0``.

    The target is the complete n-step return along the run,
    ``G_i^{(n)} = r_{i+1} + γ_{i+1} r_{i+2} + … + (∏_{k=1}^{n} γ_{i+k}) V(s_{i+n})``
    with ``V`` the delayed max-Q and each ``γ`` the done-code discount
    of that transition (not a single ``gamma_step ** n``). ``n=1`` is
    the one-step target ``r + γ V``. A ``γ == 0`` in the product zeros
    every later reward and the bootstrap; the window is still complete.
    Incomplete windows are masked, never shortened.
    :class:`~mouse_core.objectives.max_n_step_dqn.MaxNStepDqnObjective`
    uses this same return; it only adds the max-over-horizons head and
    the selector that picks the bootstrap action.
    ``metrics["n_step_valid_frac"]`` is the in-run fraction of starts
    that have a complete n-step window. The return never
    crosses a run break, is not mixed with intermediate delayed Q (no
    TD(λ)), and is not cut when the taken action is off the online
    greedy policy (no Watkins). At an episode / task boundary ``γ``
    is the corresponding done-code gamma and multiplies both the
    bootstrap and the continued return, so ``gamma_*_terminal = 0``
    ends the return there while a non-zero truncation gamma carries
    it (discounted) into the reset frame — but only while that next
    pair is still in the sample.

    Those columns arrive in ``objective_data`` only if they are listed in the
    tokenizer ``objective_fields`` keep-list (input fields are not auto-copied).
    ``task_done`` is an objective column only — do not add it as a tokenizer
    input field or embedder modality, or it will be fed to the transformer::

        tokenizer = Tokenizer(
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
        n: Return length. ``1`` is one-step TD; each larger integer adds
            one more observed reward before the delayed max-Q bootstrap.
            Starts without that many in-run steps ahead do not train.
        gamma_step: Discount factor for running (non-terminal) transitions
            (``episode_done == 0``).
        gamma_episode_terminal: Discount applied when the episode terminates
            naturally (``episode_done == 1``). ``1.0`` bootstraps across
            episode boundaries (usual for multi-episode MOUSE tasks).
        gamma_episode_truncated: Discount applied when the episode is truncated
            (``episode_done == 2``). ``1.0`` bootstraps across episode
            boundaries.
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
        grouping_field: Step column that isolates runs. ``None`` skips the
            grouping check. When set, the column must be present.
        cql_weight: Alpha coefficient for the Conservative Q-Learning penalty.
            ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor used when scaling the CQL penalty.
    """

    def __init__(
        self,
        *,
        n: int,
        gamma_step: float,
        gamma_episode_terminal: float,
        gamma_episode_truncated: float,
        gamma_task_terminal: float,
        gamma_task_truncated: float,
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
    ) -> None:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"n must be an integer >= 1, got {n}.")
        self.n = n
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

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("NStepDqnObjective requires delayed_predictions.")
        q = _require_q(predictions, "action_value")
        q_target = _require_q(delayed_predictions, "action_value").detach()
        if q_target.shape != q.shape:
            raise ValueError(
                f"DQN delayed action_value shape {tuple(q_target.shape)} must "
                f"match online shape {tuple(q.shape)}."
            )
        q = _affine(q, scale=self.q_scale, shift=self.q_shift)
        q_target = _affine(q_target, scale=self.q_scale, shift=self.q_shift)
        p_rows, n_actions = q.shape
        device = q.device
        value_dtype = q.dtype

        batch = _n_step_batch(
            objective_data,
            n_pred=p_rows,
            n_actions=n_actions,
            device=device,
            dtype=value_dtype,
            action_key=self.action_key,
            reward_key=self.reward_key,
            reward_scale=self.reward_scale,
            reward_shift=self.reward_shift,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            grouping_field=self.grouping_field,
            gamma_step=self.gamma_step,
            gamma_episode_terminal=self.gamma_episode_terminal,
            gamma_episode_truncated=self.gamma_episode_truncated,
            gamma_task_terminal=self.gamma_task_terminal,
            gamma_task_truncated=self.gamma_task_truncated,
        )
        q_values = q.gather(
            dim=-1, index=batch.next_actions.unsqueeze(-1)
        ).squeeze(-1)
        pair_target, pair_valid = _n_step_return_targets(
            reward=batch.reward,
            discount_all=batch.discount_all,
            bootstrap=q_target[batch.last_rows].amax(dim=-1),
            pair_weight=batch.pair_weight,
            n=self.n,
        )
        td_target = _pair_values_to_rows(pair_target, batch.step_of)
        row_weight = _pair_valid_to_rows(
            pair_valid, batch.step_of, dtype=value_dtype
        )

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
            "n_step_valid_frac": _weighted_mean(
                pair_valid.to(dtype=value_dtype), batch.pair_weight
            ),
        }
        if cql_penalty_mean is not None:
            named["cql_penalty"] = cql_penalty_mean

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
