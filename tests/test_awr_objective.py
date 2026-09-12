"""Tests for AWR objective on synthetic tensors."""

from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from mouse_core.objectives import AwrObjective


def _awr(
    *,
    advantage_temperature: float = 1.0,
    weight_clip: float = 20.0,
    policy_coef: float = 1.0,
    **kwargs,
) -> AwrObjective:
    return AwrObjective(
        advantage_temperature=advantage_temperature,
        weight_clip=weight_clip,
        policy_coef=policy_coef,
        **kwargs,
    )


def _batch(
    *,
    action: torch.Tensor,
    reward: torch.Tensor,
    episode_done: torch.Tensor | None = None,
    task_done: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    q: torch.Tensor,
    logits: torch.Tensor,
    requires_grad: bool = False,
) -> tuple[TensorDict, TensorDict]:
    n = int(action.shape[0])
    if episode_done is None:
        episode_done = torch.zeros(n, dtype=torch.int64)
    if task_done is None:
        task_done = torch.zeros(n, dtype=torch.int64)
    data: dict[str, torch.Tensor] = {
        "action": action,
        "reward": reward,
        "episode_done": episode_done,
        "task_done": task_done,
    }
    if sequence_id is not None:
        data["sequence_id"] = sequence_id
    objective_data = TensorDict(data, batch_size=[n])
    if requires_grad:
        q = q.detach().requires_grad_(True)
        logits = logits.detach().requires_grad_(True)
    predictions = TensorDict({"action_value": q, "action": logits}, batch_size=[n])
    return objective_data, predictions


def test_awr_requires_advantage_hparams() -> None:
    with pytest.raises(TypeError):
        AwrObjective()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AwrObjective(advantage_temperature=1.0, weight_clip=20.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AwrObjective(advantage_temperature=1.0, policy_coef=1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AwrObjective(weight_clip=20.0, policy_coef=1.0)  # type: ignore[call-arg]


def test_awr_rejects_non_positive_temperature_and_clip() -> None:
    with pytest.raises(ValueError, match="advantage_temperature"):
        AwrObjective(advantage_temperature=0.0, weight_clip=20.0, policy_coef=1.0)
    with pytest.raises(ValueError, match="weight_clip"):
        AwrObjective(advantage_temperature=1.0, weight_clip=0.0, policy_coef=1.0)
    with pytest.raises(ValueError, match="policy_coef"):
        AwrObjective(advantage_temperature=1.0, weight_clip=20.0, policy_coef=-1.0)


def test_awr_requires_action_value_head() -> None:
    objective_data, predictions = _batch(
        action=torch.tensor([0, 1]),
        reward=torch.tensor([0.0, 1.0]),
        q=torch.zeros(2, 2),
        logits=torch.zeros(2, 2),
    )
    del predictions["action_value"]
    with pytest.raises(KeyError, match="action_value"):
        _awr()(objective_data, predictions)


def test_awr_requires_policy_logits() -> None:
    objective_data, predictions = _batch(
        action=torch.tensor([0, 1]),
        reward=torch.tensor([0.0, 1.0]),
        q=torch.zeros(2, 2),
        logits=torch.zeros(2, 2),
    )
    del predictions["action"]
    with pytest.raises(KeyError, match="action"):
        _awr()(objective_data, predictions)


def test_awr_mc_return_is_discounted_sum_when_terminal_gamma_zero() -> None:
    """s0 → s1 running, s1 → s2 terminates: G_0 = r1 + γ r2, G_1 = r2."""
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0, 1]),
        reward=torch.tensor([0.0, 1.0, 4.0]),
        episode_done=torch.tensor([0, 0, 1]),
        q=torch.zeros(3, 2),
        logits=torch.zeros(3, 2),
    )
    loss, metrics = _awr(policy_coef=0.0, gamma_step=0.5)(
        objective_data, predictions
    )
    # G = [3, 4]; Q = 0 → MSE (9 + 16) / 2 = 12.5
    assert abs(loss.item() - 12.5) < 1e-5
    assert abs(metrics["return_mean"] - 3.5) < 1e-5
    assert abs(metrics["q_loss"] - 12.5) < 1e-5


def test_awr_g_stops_at_episode_boundary_when_episode_gamma_zero() -> None:
    """Default gamma_episode_* = 0: G does not include rewards after done 1 or 2."""
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0, 1, 0]),
        reward=torch.tensor([0.0, 1.0, 10.0, 4.0]),
        episode_done=torch.tensor([0, 0, 1, 0]),
        q=torch.zeros(4, 2),
        logits=torch.zeros(4, 2),
    )
    _, metrics = _awr(policy_coef=0.0, gamma_step=0.5)(objective_data, predictions)
    # G_0 = 1 + 0.5 * 10 = 6; G_1 = 10 (cut); G_2 = 4 → mean 20/3
    assert abs(metrics["return_mean"] - (6.0 + 10.0 + 4.0) / 3) < 1e-5

    objective_data["episode_done"] = torch.tensor([0, 0, 2, 0])
    _, truncated = _awr(policy_coef=0.0, gamma_step=0.5)(objective_data, predictions)
    assert abs(truncated["return_mean"] - metrics["return_mean"]) < 1e-5


def test_awr_g_stops_at_task_boundary_when_task_gamma_zero() -> None:
    """Default gamma_task_* = 0: task_done 1 or 2 cuts G while the episode runs."""
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0, 1, 0]),
        reward=torch.tensor([0.0, 1.0, 10.0, 4.0]),
        episode_done=torch.tensor([0, 0, 0, 0]),
        task_done=torch.tensor([0, 0, 2, 0]),
        q=torch.zeros(4, 2),
        logits=torch.zeros(4, 2),
    )
    _, metrics = _awr(policy_coef=0.0, gamma_step=0.5)(objective_data, predictions)
    assert abs(metrics["return_mean"] - (6.0 + 10.0 + 4.0) / 3) < 1e-5


def test_awr_mc_return_respects_episode_and_task_gammas() -> None:
    """Non-zero boundary gammas carry later rewards through, discounted."""
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0, 1, 0]),
        reward=torch.tensor([0.0, 1.0, 10.0, 4.0]),
        episode_done=torch.tensor([0, 0, 1, 0]),
        q=torch.zeros(4, 2),
        logits=torch.zeros(4, 2),
    )
    # G_2 = 4; G_1 = 10 + 0.5 * 4 = 12; G_0 = 1 + 1.0 * 12 = 13
    _, metrics = _awr(
        policy_coef=0.0,
        gamma_step=1.0,
        gamma_episode_terminal=0.5,
    )(objective_data, predictions)
    assert abs(metrics["return_mean"] - (13.0 + 12.0 + 4.0) / 3) < 1e-5

    objective_data["episode_done"] = torch.tensor([0, 0, 2, 0])
    _, truncated = _awr(
        policy_coef=0.0,
        gamma_step=1.0,
        gamma_episode_truncated=0.5,
    )(objective_data, predictions)
    assert abs(truncated["return_mean"] - metrics["return_mean"]) < 1e-5

    objective_data["episode_done"] = torch.zeros(4, dtype=torch.int64)
    objective_data["task_done"] = torch.tensor([0, 0, 2, 0])
    # Same chain with γ = gamma_step * gamma_task_truncated at the task cut
    _, task_metrics = _awr(
        policy_coef=0.0,
        gamma_step=1.0,
        gamma_task_truncated=0.5,
    )(objective_data, predictions)
    assert abs(task_metrics["return_mean"] - metrics["return_mean"]) < 1e-5


def test_awr_window_edge_is_pure_monte_carlo() -> None:
    """Last in-run pair never bootstraps Q: G = r even when max Q(s') is large."""
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0]),
        reward=torch.tensor([0.0, 2.0]),
        q=torch.tensor([[0.0, 0.0], [7.0, 1.0]]),
        logits=torch.zeros(2, 2),
    )
    loss, metrics = _awr(policy_coef=0.0, gamma_step=1.0)(objective_data, predictions)
    # G_0 = 2; Q(s0, a)=0 → MSE 4
    assert abs(metrics["return_mean"] - 2.0) < 1e-5
    assert abs(loss.item() - 4.0) < 1e-5


def test_awr_weight_is_clipped_exp_advantage() -> None:
    """One terminal transition: G = 10, V = 0, β = 1, ω_max = 5 → w = 5."""
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0]),
        reward=torch.tensor([0.0, 10.0]),
        episode_done=torch.tensor([0, 1]),
        q=torch.zeros(2, 2),
        logits=torch.zeros(2, 2),
    )
    _, metrics = _awr(
        advantage_temperature=1.0, weight_clip=5.0, policy_coef=1.0, gamma_step=1.0
    )(objective_data, predictions)
    assert abs(metrics["advantage_mean"] - 10.0) < 1e-5
    assert abs(metrics["weight_mean"] - 5.0) < 1e-5
    # Uniform 2-way logits: log π = -log(2); L_π = 5 * log(2)
    assert abs(metrics["policy_loss"] - 5.0 * math.log(2.0)) < 1e-5


def test_awr_out_of_run_pairs_do_not_contribute() -> None:
    objective_data, predictions = _batch(
        action=torch.tensor([0, 1]),
        reward=torch.tensor([0.0, 99.0]),
        sequence_id=torch.tensor([0, 1]),
        q=torch.zeros(2, 2),
        logits=torch.zeros(2, 2),
    )
    loss, metrics = _awr()(objective_data, predictions)
    assert abs(loss.item()) < 1e-6
    assert abs(metrics["return_mean"]) < 1e-6
    assert abs(metrics["q_loss"]) < 1e-6


def test_awr_policy_weights_do_not_carry_q_gradients() -> None:
    """Q already equals G so L_Q = 0; detached A means Q.grad stays 0."""
    q = torch.tensor([[10.0, 0.0], [0.0, 0.0]], requires_grad=True)
    logits = torch.zeros(2, 2, requires_grad=True)
    objective_data, predictions = _batch(
        action=torch.tensor([0, 0]),
        reward=torch.tensor([0.0, 10.0]),
        episode_done=torch.tensor([0, 1]),
        q=q,
        logits=logits,
    )
    predictions["action_value"] = q
    predictions["action"] = logits
    loss, metrics = _awr(
        advantage_temperature=1.0, weight_clip=20.0, policy_coef=1.0
    )(objective_data, predictions)
    assert abs(metrics["q_loss"]) < 1e-5
    loss.backward()
    assert q.grad is not None
    assert torch.allclose(q.grad, torch.zeros_like(q.grad))
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_awr_requires_min_sequence() -> None:
    objective_data, predictions = _batch(
        action=torch.zeros(1, dtype=torch.int64),
        reward=torch.zeros(1),
        q=torch.zeros(1, 2),
        logits=torch.zeros(1, 2),
    )
    with pytest.raises(ValueError, match="N >= 2"):
        _awr()(objective_data, predictions)
