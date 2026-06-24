"""Tests for DQN objective on synthetic tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import DqnObjectiveConfig, dqn_objective


def test_dqn_objective_runs() -> None:
    b, s, a = 2, 4, 3
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
            "action_value": torch.randn(b, s, a),
            "action_value_target": torch.randn(b, s, a),
        },
        batch_size=(b, s),
    )
    cfg = DqnObjectiveConfig(weight=1.0, gamma=0.99)
    loss, metrics = dqn_objective(step_stream, out, cfg)
    assert loss.ndim == 0
    assert "action_value" in metrics
    assert metrics["action_value"] >= 0.0


def test_dqn_objective_accepts_trailing_singleton_action_dim() -> None:
    b, s, a = 2, 4, 3
    step_stream = TensorDict(
        {
            "action": torch.randint(0, a, (b, s, 1)),
            "reward": torch.randn(b, s),
            "done": torch.zeros(b, s, dtype=torch.long),
        },
        batch_size=(b, s),
    )
    out = TensorDict(
        {
            "action_value": torch.randn(b, s, a),
            "action_value_target": torch.randn(b, s, a),
        },
        batch_size=(b, s),
    )
    loss, metrics = dqn_objective(step_stream, out, DqnObjectiveConfig(weight=1.0, gamma=0.99))
    assert loss.ndim == 0
    assert "action_value" in metrics


def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = TensorDict(
        {
            "action": torch.zeros(1, 1, dtype=torch.long),
            "reward": torch.zeros(1, 1),
            "done": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1, 1),
    )
    out = TensorDict(
        {"action_value": torch.zeros(1, 1, 2), "action_value_target": torch.zeros(1, 1, 2)},
        batch_size=(1, 1),
    )
    try:
        dqn_objective(step_stream, out, DqnObjectiveConfig(weight=1.0))
    except ValueError as e:
        assert "Not enough" in str(e)
    else:
        raise AssertionError("expected ValueError for S < 2")
