"""Tests for layerwise DQN objective and discount schedule."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mouse_core.objectives import LayerwiseDqnObjective, effective_horizon


def test_effective_horizon() -> None:
    assert effective_horizon(0.0) == 1.0
    assert effective_horizon(0.99) == pytest.approx(100.0)


def test_layerwise_dqn_objective_anchors_endpoints() -> None:
    objective = LayerwiseDqnObjective(
        num_backbone_layers=4,
        gamma_step_start=1.0,
        gamma_step=1.0,
        gamma_episode_terminal_start=0.0,
        gamma_episode_terminal=0.99,
    )
    assert objective.layer_gamma_step == [1.0, 1.0, 1.0, 1.0]
    assert objective.layer_gamma_episode_terminal[0] == 0.0
    assert objective.layer_gamma_episode_terminal[-1] == 0.99


def test_layerwise_dqn_objective_linear_horizon() -> None:
    objective = LayerwiseDqnObjective(
        num_backbone_layers=4,
        gamma_step_start=0.0,
        gamma_step=0.99,
    )
    horizons = [effective_horizon(g) for g in objective.layer_gamma_step]
    assert horizons == pytest.approx([1.0, 34.0, 67.0, 100.0])
    assert objective.layer_gamma_step[0] == 0.0
    assert objective.layer_gamma_step[-1] == 0.99


def test_layerwise_dqn_objective_runs() -> None:
    b, s, layers, a = 2, 4, 3, 3
    step_stream = TensorDict(
        {
            "action": torch.randint(0, a, (b, s)),
            "reward": torch.randn(b, s),
            "done": torch.zeros(b, s, dtype=torch.long),
        },
        batch_size=(b, s),
    )
    out = TensorDict(
        {
            "action_value_layerwise": torch.randn(b, s, layers, a),
            "action_value_layerwise_target": torch.randn(b, s, layers, a),
        },
        batch_size=(b, s),
    )
    objective = LayerwiseDqnObjective(
        num_backbone_layers=layers,
        gamma_step_start=0.0,
        gamma_step=0.99,
    )
    loss, metrics = objective(step_stream, out)
    assert loss.ndim == 0
    assert "action_value_layerwise" in metrics
    assert metrics["layer_0_gamma_step"] < metrics["layer_2_gamma_step"]
    assert metrics["action_value_layerwise"] >= 0.0
    assert metrics["layer_0_loss"] >= 0.0


def test_layerwise_dqn_objective_skips_transitions_across_pack_seams() -> None:
    """Corrupting data at a seam-masked pair must leave the loss unchanged."""
    b, s, layers, a = 1, 5, 2, 3
    torch.manual_seed(0)
    step_stream = TensorDict(
        {
            "action": torch.randint(0, a, (b, s)),
            "reward": torch.randn(b, s),
            "done": torch.zeros(b, s, dtype=torch.long),
            "is_seam": torch.tensor([[0, 0, 1, 0, 0]]),
        },
        batch_size=(b, s),
    )
    out = TensorDict(
        {
            "action_value_layerwise": torch.randn(b, s, layers, a),
            "action_value_layerwise_target": torch.randn(b, s, layers, a),
        },
        batch_size=(b, s),
    )
    objective = LayerwiseDqnObjective(
        num_backbone_layers=layers,
        gamma_step_start=0.0,
        gamma_step=0.99,
    )

    loss_before, _ = objective(step_stream, out)

    corrupted = step_stream.clone()
    corrupted["reward"][0, 2] = 1.0e6  # reward entering the seam row (pair t=1)
    loss_after, _ = objective(corrupted, out)

    assert torch.allclose(loss_before, loss_after)


def test_layerwise_dqn_objective_rejects_layer_mismatch() -> None:
    step_stream = TensorDict(
        {
            "action": torch.zeros(1, 3, dtype=torch.long),
            "reward": torch.zeros(1, 3),
            "done": torch.zeros(1, 3, dtype=torch.long),
        },
        batch_size=(1, 3),
    )
    out = TensorDict(
        {
            "action_value_layerwise": torch.zeros(1, 3, 2, 2),
            "action_value_layerwise_target": torch.zeros(1, 3, 2, 2),
        },
        batch_size=(1, 3),
    )
    objective = LayerwiseDqnObjective(
        num_backbone_layers=3,
        gamma_step_start=0.0,
        gamma_step=0.99,
    )
    try:
        objective(step_stream, out)
    except ValueError as exc:
        assert "expects 3 Q layers" in str(exc)
    else:
        raise AssertionError("expected ValueError for layer count mismatch")
