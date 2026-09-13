"""Max-over-n-step DQN: selector head plus a maximum-return head."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _affine,
    _in_run_stats,
    _pair_values_to_rows,
    _weighted_mean,
)
from mouse_core.objectives.n_step_dqn import (
    _n_step_batch,
    _n_step_horizon_targets,
    _pair_valid_to_rows,
    _require_q,
)


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


class MaxNStepDqnObjective(Objective):
    """Two-head max-over-n-step Bellman objective with a delayed target network.

    Same complete n-step returns as :class:`NStepDqnObjective` (incomplete
    windows masked, ``γ == 0`` completes early). The extras unique to
    maximizing over n-steps are the two heads and the max / actor
    selection: bootstrap values come from delayed ``max_return`` gathered
    at the action the *online* ``selector`` chooses at each endpoint, the
    max-return head trains toward the max valid candidate, and the
    selector (deployed policy) trains toward the one-step candidate.

    Instantiate with hyperparameters, then call with
    ``(objective_data, predictions, delayed_predictions)``. Online Q is
    ``predictions["selector"]`` and ``predictions["max_return"]``.
    Delayed tensors are detached before the targets, so the TD error does
    not backprop through them.

    For each in-run start ``i`` and each horizon ``n`` in ``horizons``, the
    candidate is the recorded n-step return
    ``Y_i^{(n)} = r_{i+1} + γ_{i+1} r_{i+2} + … + (∏_{k=1}^{n} γ_{i+k}) Q̄_max(s_{i+n}, a*)``
    with ``a* = argmax_a Q_selector(s_{i+n})``. Each ``γ`` is that
    transition's done-code discount; a ``0`` in the product zeros the
    rest. Incomplete horizons are masked,
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

        # a* from the online selector at each step's last head-output row;
        # bootstrap value from delayed max-return at that action.
        sel_step = q_sel.detach()[batch.last_rows]
        a_star = sel_step.argmax(dim=-1)
        bootstrap = q_max_delayed[batch.last_rows].gather(
            dim=-1, index=a_star.unsqueeze(-1)
        ).squeeze(-1)

        candidates, valid = _n_step_horizon_targets(
            reward=batch.reward,
            discount_all=batch.discount_all,
            bootstrap=bootstrap,
            pair_weight=batch.pair_weight,
            horizons=self.horizons,
        )
        valid_max = valid.any(dim=-1)
        y_max = candidates.max(dim=-1).values
        y_one = candidates[:, 0]
        valid_one = valid[:, 0]
        zero = y_max.new_zeros(y_max.shape)
        y_one_safe = torch.where(valid_one, y_one, zero)
        y_max_safe = torch.where(valid_max, y_max, zero)

        q_sel_taken = q_sel.gather(
            dim=-1, index=batch.next_actions.unsqueeze(-1)
        ).squeeze(-1)
        q_max_taken = q_max.gather(
            dim=-1, index=batch.next_actions.unsqueeze(-1)
        ).squeeze(-1)

        td_one = _pair_values_to_rows(y_one_safe, batch.step_of)
        td_max = _pair_values_to_rows(y_max_safe, batch.step_of)
        row_w_one = _pair_valid_to_rows(valid_one, batch.step_of, dtype=value_dtype)
        row_w_max = _pair_valid_to_rows(valid_max, batch.step_of, dtype=value_dtype)
        row_w_run = _pair_valid_to_rows(
            batch.pair_weight > 0, batch.step_of, dtype=value_dtype
        )

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
                valid[:, h].to(dtype=value_dtype), batch.pair_weight
            )
            named[f"horizon_{n}_selected_frac"] = _weighted_mean(
                (selected == h).to(dtype=value_dtype),
                valid_max.to(dtype=value_dtype),
            )

        metrics: dict[str, float] = dict(
            zip(named, torch.stack(list(named.values())).tolist())
        )
        return loss, metrics
