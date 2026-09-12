"""Max-over-n-step DQN: selector head plus a maximum-return head."""

from __future__ import annotations

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


def _require_horizons(horizons: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Sorted unique horizons; every entry ``>= 1`` and ``1`` must be present."""
    if not isinstance(horizons, (tuple, list)):
        raise TypeError(
            f"horizons must be a tuple or list of ints, got {type(horizons).__name__}."
        )
    if len(horizons) == 0:
        raise ValueError("horizons must be non-empty.")
    out: list[int] = []
    for n in horizons:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"horizons must be integers >= 1, got {n}.")
        out.append(n)
    if len(set(out)) != len(out):
        raise ValueError(f"horizons must be unique, got {tuple(horizons)}.")
    if 1 not in out:
        raise ValueError(f"horizons must contain 1, got {tuple(horizons)}.")
    return tuple(sorted(out))


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
def _max_n_step_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    bootstrap: torch.Tensor,
    pair_weight: torch.Tensor,
    horizons: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-horizon n-step targets and validity, shapes ``[N-1, H]``.

    Pair ``t`` is the transition out of step ``t`` (reward / gamma stored at
    ``t+1``). Horizon ``n`` uses the next ``n`` recorded rewards and, unless
    a ``gamma == 0`` stop fires first, bootstraps ``bootstrap[t + n]``.
    Incomplete horizons stay ``-inf`` / ``False`` — they are never silently
    shortened. A zero gamma includes that reward, then fills every remaining
    ``n >= k+1`` with the stopped sum and needs no later endpoint.
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


class MaxNStepDqnObjective(Objective):
    """Two-head max-over-n-step Bellman objective with a delayed target network.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions, delayed_predictions)``. Online Q is
    ``predictions["selector"]`` (deployed policy) and
    ``predictions["max_return"]``. Bootstrap values come from delayed
    ``max_return`` only, gathered at the action the *online* selector
    chooses at each endpoint. Delayed tensors are detached before the
    targets, so the TD error does not backprop through them.

    For each in-run start ``i`` and each horizon ``n`` in ``horizons``, the
    candidate is the recorded n-step return
    ``Y_i^{(n)} = r + γ r' + … + D_n Q̄_max(s_{i+n}, a*)`` with
    ``a* = argmax_a Q_selector(s_{i+n})``. Incomplete horizons are masked,
    never shortened. ``γ == 0`` includes that reward, then every remaining
    longer horizon shares that stopped sum and needs no later Q. The
    max-return head trains toward ``Y^{(M)} = max valid Y^{(n)}``; the
    selector trains toward ``Y^{(1)}`` (one reward, then the same delayed
    max-return bootstrap). Ties pick the shortest horizon.

    Q rows are **per head-output token** (``[P, A]``). Every head-output row
    of step ``i`` trains toward that step's targets; selector / bootstrap
    reads use the last head-output row of the endpoint step.

    A **run** is the same ``sequence_id`` and, when ``grouping_field`` is set
    and present, the same grouping column (typically ``task_index``). A
    missing next pair is truncation: that horizon is invalid. If every
    weight is ``0`` the corresponding loss is ``0``. Gamma is the Bellman
    discount from the done codes at ``i+1`` inside a same-run pair.

    Those columns arrive in ``objective_data`` only if they are listed in the
    tokenizer ``objective_fields`` keep-list::

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
        horizons: Return lengths. Must contain ``1``, every entry ``>= 1``,
            no duplicates. Stored sorted.
        max_return_weight: Multiplier on the max-return loss. Combined loss
            is ``L_selector + max_return_weight * L_max_return``.
        gamma_step: Discount for running transitions (``episode_done == 0``).
        gamma_episode_terminal: Discount when the episode terminates
            (``episode_done == 1``).
        gamma_episode_truncated: Discount when the episode is truncated
            (``episode_done == 2``).
        gamma_task_terminal: Extra discount when the task terminates
            (``task_done == 1``). Multiplies the episode discount.
        gamma_task_truncated: Extra discount when the task is truncated
            (``task_done == 2``). Multiplies the episode discount.
        action_key: Key in ``objective_data`` for the integer action.
        reward_key: Key in ``objective_data`` for per-step reward.
        reward_scale: Multiplier applied to ``reward`` before the TD target.
        reward_shift: Offset added after ``reward_scale``.
        q_scale: Multiplier applied to online and delayed Q before the TD
            error. Same affine on both networks and both heads.
        q_shift: Offset added after ``q_scale``.
        episode_done_key: Key in ``objective_data`` for the episode-done code.
        task_done_key: Key in ``objective_data`` for the task-done code.
        grouping_field: Step column that isolates runs. ``None`` skips the
            grouping check. When set, the column must be present.
    """

    def __init__(
        self,
        *,
        horizons: tuple[int, ...] | list[int],
        max_return_weight: float,
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
    ) -> None:
        self.horizons = _require_horizons(horizons)
        self.max_return_weight = float(max_return_weight)
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

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
        delayed_predictions: TensorDict | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if delayed_predictions is None:
            raise ValueError("MaxNStepDqnObjective requires delayed_predictions.")
        q_sel = _require_q(predictions, "selector")
        q_max = _require_q(predictions, "max_return")
        q_max_delayed = _require_q(delayed_predictions, "max_return").detach()
        if q_sel.shape != q_max.shape:
            raise ValueError(
                f"selector shape {tuple(q_sel.shape)} must match "
                f"max_return shape {tuple(q_max.shape)}."
            )
        if q_max_delayed.shape != q_max.shape:
            raise ValueError(
                "DQN delayed max_return shape must match the online head."
            )

        q_sel = _affine(q_sel, scale=self.q_scale, shift=self.q_shift)
        q_max = _affine(q_max, scale=self.q_scale, shift=self.q_shift)
        q_max_delayed = _affine(q_max_delayed, scale=self.q_scale, shift=self.q_shift)
        p_rows, n_actions = q_sel.shape
        device = q_sel.device
        value_dtype = q_sel.dtype

        action = objective_data[self.action_key]
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

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([n_steps]):
            raise ValueError(
                f"DQN objective expects reward shape [{n_steps}], "
                f"got {tuple(reward.shape)}."
            )
        reward = _affine(reward, scale=self.reward_scale, shift=self.reward_shift)

        episode_done, task_done = _require_done_codes(
            objective_data,
            episode_done_key=self.episode_done_key,
            task_done_key=self.task_done_key,
            N=n_steps,
        )

        step_of, last_rows = _head_output_layout(
            objective_data, N=n_steps, P=p_rows, device=device
        )
        pair_weight = _pair_weight(
            objective_data,
            n_steps,
            device,
            grouping_field=self.grouping_field,
            dtype=value_dtype,
        )

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

        # a* from the online selector at each step's last head-output row;
        # bootstrap value from delayed max-return at that action.
        sel_step = q_sel.detach()[last_rows]
        a_star = sel_step.argmax(dim=-1)
        bootstrap = q_max_delayed[last_rows].gather(
            dim=-1, index=a_star.unsqueeze(-1)
        ).squeeze(-1)

        candidates, valid = _max_n_step_targets(
            reward=reward,
            discount_all=discount_all,
            bootstrap=bootstrap,
            pair_weight=pair_weight,
            horizons=self.horizons,
        )
        valid_max = valid.any(dim=-1)
        y_max = candidates.max(dim=-1).values
        y_one = candidates[:, 0]
        valid_one = valid[:, 0]
        zero = y_max.new_zeros(y_max.shape)
        y_one_safe = torch.where(valid_one, y_one, zero)
        y_max_safe = torch.where(valid_max, y_max, zero)

        step_next = (step_of + 1).clamp(max=n_steps - 1)
        next_actions = action[step_next]
        q_sel_taken = q_sel.gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)
        q_max_taken = q_max.gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)

        td_one = _pair_values_to_rows(y_one_safe, step_of)
        td_max = _pair_values_to_rows(y_max_safe, step_of)
        row_w_one = torch.cat(
            [valid_one.to(dtype=value_dtype), pair_weight.new_zeros(1)]
        )[step_of]
        row_w_max = torch.cat(
            [valid_max.to(dtype=value_dtype), pair_weight.new_zeros(1)]
        )[step_of]
        row_w_run = torch.cat(
            [pair_weight, pair_weight.new_zeros(1)]
        )[step_of]

        loss_sel = _weighted_mean((q_sel_taken - td_one) ** 2, row_w_one)
        loss_max = _weighted_mean((q_max_taken - td_max) ** 2, row_w_max)
        loss = loss_sel + self.max_return_weight * loss_max

        sel_mean, sel_std, sel_min, sel_max = _in_run_stats(
            q_sel.amax(dim=-1).detach(), row_w_run
        )
        max_mean, max_std, max_min, max_hi = _in_run_stats(
            q_max.amax(dim=-1).detach(), row_w_run
        )
        selected = candidates.argmax(dim=-1)
        delta = torch.where(valid_one, y_max_safe - y_one_safe, zero)
        named: dict[str, torch.Tensor] = {
            "q_selector_mean": sel_mean,
            "q_selector_std": sel_std,
            "q_selector_min": sel_min,
            "q_selector_max": sel_max,
            "q_max_return_mean": max_mean,
            "q_max_return_std": max_std,
            "q_max_return_min": max_min,
            "q_max_return_max": max_hi,
            "selector": loss_sel.detach(),
            "max_return": loss_max.detach(),
            "y_max_minus_y_one": _weighted_mean(delta, valid_one.to(dtype=value_dtype)),
        }
        for h, n in enumerate(self.horizons):
            named[f"horizon_{n}_valid_frac"] = _weighted_mean(
                valid[:, h].to(dtype=value_dtype), pair_weight
            )
            named[f"horizon_{n}_selected_frac"] = _weighted_mean(
                (selected == h).to(dtype=value_dtype),
                valid_max.to(dtype=value_dtype),
            )

        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
