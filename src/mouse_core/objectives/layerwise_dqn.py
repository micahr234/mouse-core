"""Layerwise DQN objective with per-layer discount schedules."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import _valid_transitions


def effective_horizon(gamma: float) -> float:
    """Effective planning horizon ``1 / (1 - gamma)`` for ``gamma < 1``."""
    if gamma >= 1.0:
        return float("inf")
    if gamma <= 0.0:
        return 1.0
    return 1.0 / (1.0 - gamma)


def gamma_from_horizon(horizon: float) -> float:
    """Discount factor with effective horizon ``horizon >= 1``."""
    if not math.isfinite(horizon) or horizon <= 1.0:
        return 0.0
    return 1.0 - 1.0 / horizon


def _build_layer_gamma_schedule(
    num_layers: int,
    *,
    gamma_start: float,
    gamma_deep: float,
) -> list[float]:
    """Build per-layer gammas with linearly increasing effective horizon.

    Layer ``0`` is exactly ``gamma_start``; layer ``L - 1`` is exactly
    ``gamma_deep``. Intermediate layers linearly interpolate horizon:

    ``H_l = H_start + (H_deep - H_start) * (l / (L - 1))``

    ``gamma_l = 1 - 1 / H_l``
    """
    if num_layers < 1:
        raise ValueError(f"num_backbone_layers must be >= 1, got {num_layers}.")
    if num_layers == 1:
        return [gamma_deep]

    if gamma_start == gamma_deep:
        return [gamma_deep] * num_layers

    h_start = effective_horizon(gamma_start)
    h_deep = effective_horizon(gamma_deep)

    if math.isinf(h_start) or math.isinf(h_deep):
        raise ValueError(
            "gamma_start and gamma_deep must be below 1.0 for a finite horizon schedule."
        )

    return [
        gamma_from_horizon(h_start + (h_deep - h_start) * (layer_idx / (num_layers - 1)))
        for layer_idx in range(num_layers)
    ]


class LayerwiseDqnObjective(Objective):
    """One-step Bellman TD objective on every backbone layer.

    Reads ``predictions["action_value_layerwise"]`` and
    ``predictions["action_value_layerwise_target"]`` with shape ``[B, S, L, A]``.
    Each layer and each done-code uses its own discount, built at construction
    from explicit shallow/deep endpoint pairs. Row pairs that straddle a
    packed-segment seam (``is_seam`` from ``DataLoader(pack=True)``) are
    excluded from every layer's loss.

    Effective planning horizon is ``H(gamma) = 1 / (1 - gamma)``. Layer ``0`` uses
    each ``gamma_*_start``; the deepest layer uses the deep value
    (``gamma_step``, ``gamma_episode_terminal``, …). Intermediate layers get
    **linearly increasing horizon** (linearly harder targets):

    ``H_l = H_start + (H_deep - H_start) * (l / (L - 1))``

    ``gamma_l = 1 - 1 / H_l``

    Example with ``num_backbone_layers=20``, ``gamma_episode_terminal_start=0.0``,
    ``gamma_episode_terminal=0.99`` (``H_start=1``, ``H_deep=100``):

    +--------+---------------------------+----------+
    | Layer  | gamma_episode_terminal    | Horizon  |
    +========+===========================+==========+
    | 0      | 0.0                       | 1        |
    | 5      | ~0.963                    | ~27      |
    | 10     | ~0.981                    | ~53      |
    | 19     | 0.99                      | 100      |
    +--------+---------------------------+----------+

    ``get_action`` on a model with this head uses the deepest layer's Q-values.

    Args:
        num_backbone_layers: Number of transformer blocks (and Q heads).
        gamma_step_start: Step discount at layer 0 (``done == 0``).
        gamma_step: Step discount at the deepest layer.
        gamma_episode_terminal_start: Episode-terminal discount at layer 0.
        gamma_episode_terminal: Episode-terminal discount at the deepest layer.
        gamma_episode_truncated_start: Episode-truncated discount at layer 0.
        gamma_episode_truncated: Episode-truncated discount at the deepest layer.
        gamma_task_terminal_start: Task-terminal discount at layer 0.
        gamma_task_terminal: Task-terminal discount at the deepest layer.
        gamma_task_truncated_start: Task-truncated discount at layer 0.
        gamma_task_truncated: Task-truncated discount at the deepest layer.
        tau: Polyak coefficient for target-network updates.
        action_key: Key in ``objective_data`` for the integer action.
        reward_key: Key in ``objective_data`` for per-step reward.
        done_key: Key in ``objective_data`` for the integer done code.
        cql_weight: CQL penalty coefficient; ``0.0`` disables CQL.
        cql_scale_q_eps: Additive floor when scaling the CQL penalty.
    """

    effective_horizon = staticmethod(effective_horizon)
    gamma_from_horizon = staticmethod(gamma_from_horizon)

    def __init__(
        self,
        *,
        num_backbone_layers: int,
        gamma_step_start: float,
        gamma_step: float = 0.99,
        gamma_episode_terminal_start: float = 0.0,
        gamma_episode_terminal: float = 0.0,
        gamma_episode_truncated_start: float = 0.0,
        gamma_episode_truncated: float = 0.0,
        gamma_task_terminal_start: float = 0.0,
        gamma_task_terminal: float = 0.0,
        gamma_task_truncated_start: float = 0.0,
        gamma_task_truncated: float = 0.0,
        tau: float = 0.01,
        action_key: str = "action",
        reward_key: str = "reward",
        done_key: str = "done",
        cql_weight: float = 0.0,
        cql_scale_q_eps: float = 1.0,
    ) -> None:
        self.num_backbone_layers = int(num_backbone_layers)
        self.gamma_step_start = float(gamma_step_start)
        self.gamma_step = float(gamma_step)
        self.gamma_episode_terminal_start = float(gamma_episode_terminal_start)
        self.gamma_episode_terminal = float(gamma_episode_terminal)
        self.gamma_episode_truncated_start = float(gamma_episode_truncated_start)
        self.gamma_episode_truncated = float(gamma_episode_truncated)
        self.gamma_task_terminal_start = float(gamma_task_terminal_start)
        self.gamma_task_terminal = float(gamma_task_terminal)
        self.gamma_task_truncated_start = float(gamma_task_truncated_start)
        self.gamma_task_truncated = float(gamma_task_truncated)
        self.tau = tau
        self.action_key = action_key
        self.reward_key = reward_key
        self.done_key = done_key
        self.cql_weight = cql_weight
        self.cql_scale_q_eps = cql_scale_q_eps

        build = _build_layer_gamma_schedule
        n = self.num_backbone_layers
        self.layer_gamma_step = build(
            n, gamma_start=self.gamma_step_start, gamma_deep=self.gamma_step
        )
        self.layer_gamma_episode_terminal = build(
            n,
            gamma_start=self.gamma_episode_terminal_start,
            gamma_deep=self.gamma_episode_terminal,
        )
        self.layer_gamma_episode_truncated = build(
            n,
            gamma_start=self.gamma_episode_truncated_start,
            gamma_deep=self.gamma_episode_truncated,
        )
        self.layer_gamma_task_terminal = build(
            n,
            gamma_start=self.gamma_task_terminal_start,
            gamma_deep=self.gamma_task_terminal,
        )
        self.layer_gamma_task_truncated = build(
            n,
            gamma_start=self.gamma_task_truncated_start,
            gamma_deep=self.gamma_task_truncated,
        )
        self.layer_done_gammas: list[list[float]] = [
            [
                self.layer_gamma_step[layer_idx],
                self.layer_gamma_episode_terminal[layer_idx],
                self.layer_gamma_episode_truncated[layer_idx],
                self.layer_gamma_task_terminal[layer_idx],
                self.layer_gamma_task_truncated[layer_idx],
            ]
            for layer_idx in range(n)
        ]

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        q: torch.Tensor = predictions["action_value_layerwise"]
        q_target: torch.Tensor = predictions["action_value_layerwise_target"]

        B, S, L, A = q.shape
        device = q.device
        value_dtype = q.dtype

        if L != self.num_backbone_layers:
            raise ValueError(
                f"Layerwise DQN objective expects {self.num_backbone_layers} Q layers "
                f"but predictions have {L}."
            )

        if S < 2:
            raise ValueError("Not enough valid q values in data.")

        action = objective_data[self.action_key]
        if action.dtype != torch.int64:
            raise TypeError(f"action must be int64, got {action.dtype}.")
        if action.shape != torch.Size([B, S]):
            raise ValueError(
                f"Layerwise DQN objective expects action shape [{B}, {S}], got {tuple(action.shape)}."
            )

        reward = objective_data[self.reward_key]
        if reward.dtype != torch.float32:
            raise TypeError(f"reward must be float32, got {reward.dtype}.")
        if reward.shape != torch.Size([B, S]):
            raise ValueError(
                f"Layerwise DQN objective expects reward shape [{B}, {S}], got {tuple(reward.shape)}."
            )

        done = objective_data[self.done_key]
        if done.dtype != torch.int64:
            raise TypeError(f"done must be int64, got {done.dtype}.")
        if done.shape != torch.Size([B, S]):
            raise ValueError(
                f"Layerwise DQN objective expects done shape [{B}, {S}], got {tuple(done.shape)}."
            )

        valid = _valid_transitions(objective_data, B, S, device)

        curr_q = q[:, :-1, :, :]              # [B, S-1, L, A]
        next_actions = action[:, 1:]        # [B, S-1]
        next_rewards = reward[:, 1:]        # [B, S-1]
        next_done = done[:, 1:]               # [B, S-1]
        next_q_target = q_target[:, 1:, :, :]  # [B, S-1, L, A]

        layer_losses: list[torch.Tensor] = []
        layer_q_means: list[torch.Tensor] = []
        cql_penalties: list[torch.Tensor] = []

        for layer_idx in range(L):
            gammas = torch.tensor(
                self.layer_done_gammas[layer_idx],
                dtype=value_dtype,
                device=device,
            )
            discount = gammas[next_done]  # [B, S-1]

            curr_q_layer = curr_q[:, :, layer_idx, :]
            next_q_target_layer = next_q_target[:, :, layer_idx, :]

            q_values = curr_q_layer.gather(
                dim=-1, index=next_actions.unsqueeze(-1)
            ).squeeze(-1)
            next_max_q_target = next_q_target_layer.amax(dim=-1)
            td_target = next_rewards + discount * next_max_q_target

            loss = (q_values - td_target.detach()) ** 2

            if self.cql_weight > 0.0:
                q_scale = (td_target.abs() + self.cql_scale_q_eps).detach()
                cql_penalty = torch.logsumexp(curr_q_layer, dim=-1) - q_values
                loss = loss + self.cql_weight * q_scale * cql_penalty
                cql_penalties.append(cql_penalty.detach()[valid].mean())

            layer_losses.append(loss[valid].mean())
            layer_q_means.append(q_values.detach()[valid].mean())

        total_loss = torch.stack(layer_losses).mean()

        deepest_q = q[:, :-1, -1, :].gather(
            dim=-1, index=next_actions.unsqueeze(-1)
        ).squeeze(-1)
        q_det = deepest_q.detach()[valid]
        q_std = (
            q_det.std()
            if q_det.numel() > 1
            else torch.zeros((), device=device, dtype=value_dtype)
        )

        named: dict[str, torch.Tensor] = {
            "q_values_mean": q_det.mean(),
            "q_values_std": q_std,
            "q_values_min": q_det.min(),
            "q_values_max": q_det.max(),
            "action_value_layerwise": total_loss.detach(),
        }
        for layer_idx, gamma in enumerate(self.layer_gamma_step):
            named[f"layer_{layer_idx}_gamma_step"] = torch.tensor(
                gamma, device=device, dtype=value_dtype
            )
            named[f"layer_{layer_idx}_loss"] = layer_losses[layer_idx].detach()
            named[f"layer_{layer_idx}_q_mean"] = layer_q_means[layer_idx]
        if cql_penalties:
            named["cql_penalty"] = torch.stack(cql_penalties).mean()

        metrics: dict[str, float] = {
            key: (value.item() if value.numel() == 1 else float(value))
            for key, value in named.items()
        }
        return total_loss, metrics
