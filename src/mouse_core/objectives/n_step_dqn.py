"""n-step return DQN objective with a delayed target network."""

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
    _shift_next,
    _weighted_mean,
)


@torch.no_grad()
def _n_step_return_targets(
    *,
    reward: torch.Tensor,
    discount_all: torch.Tensor,
    v_step: torch.Tensor,
    pair_weight: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """n-step return for every pair ``(t, t+1)``, shape ``[N-1]``.

    ``G_t^{(1)} = r_{t+1} + γ_{t+1} V_{t+1}``
    ``G_t^{(k+1)} = r_{t+1} + γ_{t+1} * (c_t G_{t+1}^{(k)} + (1 - c_t) V_{t+1})``
    where ``V`` is the delayed max-Q and ``c_t`` says whether the return
    continues through ``s_{t+1}``: pair ``t+1`` must exist and be in-run.
    Episode / task boundaries are handled by ``γ_{t+1}`` itself — the
    done-code discount from ``_boundary_discounts`` multiplies both the
    bootstrap and the continued return, so a ``0`` gamma ends the return
    and a non-zero truncation gamma carries it through, discounted.
    ``c_t = 0`` is the one-step target ``r + γ V``. Out-of-run pairs
    return ``0`` (their rows carry weight ``0``).
    """
    r = reward[1:].to(dtype=v_step.dtype)  # [N-1]  r_t (stored at t+1)
    g = discount_all[1:]  # [N-1]  γ_t from done codes at t+1
    v_next = v_step[1:]  # [N-1]  V(s_{t+1})
    in_run = pair_weight > 0
    cont = _shift_next(in_run.to(dtype=v_step.dtype))
    returns = r + g * v_next
    for _ in range(int(n) - 1):
        returns = r + g * (cont * _shift_next(returns) + (1.0 - cont) * v_next)
    return returns * in_run.to(dtype=returns.dtype)


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
    ahead (or the last in-run next state, if that comes first).

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

    The target is the n-step return along the run,
    ``G_i^{(1)} = r + γ V(s_{i+1})`` and
    ``G_i^{(k+1)} = r + γ G_{i+1}^{(k)}`` while the next pair is in-run,
    with ``V`` the delayed max-Q. ``n=1`` is the one-step target
    ``r + γ V``. The return never crosses a run break, is not mixed with
    intermediate delayed Q (no TD(λ)), and is not cut when the taken
    action is off the online greedy policy (no Watkins). At an episode /
    task boundary ``γ`` is the corresponding done-code gamma and
    multiplies both the bootstrap and the continued return, so
    ``gamma_*_terminal = 0`` ends the return there while a non-zero
    truncation gamma carries it (discounted) into the reset frame.

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
        _require_action_ids(action, A)

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
        # toward the same target, and the bootstrap reads the n-step-ahead
        # (or last in-run) next step's *last* head-output row.
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
        pair_target = _n_step_return_targets(
            reward=reward,
            discount_all=discount_all,
            v_step=q_target[last_rows].amax(dim=-1),  # [N]  V(s_i) = max_a Q_target
            pair_weight=pair_weight,
            n=self.n,
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

        metrics: dict[str, float] = dict(zip(named, torch.stack(list(named.values())).tolist()))
        return loss, metrics
