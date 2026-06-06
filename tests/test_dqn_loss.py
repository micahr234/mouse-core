"""Tests for DQN loss on synthetic tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.losses import DqnLossConfig, dqn_loss


def test_dqn_loss_runs() -> None:
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
            "dqn": torch.randn(b, s, a),
            "dqn_target": torch.randn(b, s, a),
        },
        batch_size=(b, s),
    )
    cfg = DqnLossConfig(weight=1.0, gamma=0.99)
    loss, metrics = dqn_loss(step_stream, out, cfg)
    assert loss.ndim == 0
    assert "dqn_loss" in metrics
    assert metrics["dqn_loss"] >= 0.0


def test_dqn_loss_requires_min_sequence() -> None:
    step_stream = TensorDict(
        {
            "action": torch.zeros(1, 1, dtype=torch.long),
            "reward": torch.zeros(1, 1),
            "done": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1, 1),
    )
    out = TensorDict(
        {"dqn": torch.zeros(1, 1, 2), "dqn_target": torch.zeros(1, 1, 2)},
        batch_size=(1, 1),
    )
    try:
        dqn_loss(step_stream, out, DqnLossConfig(weight=1.0))
    except ValueError as e:
        assert "Not enough" in str(e)
    else:
        raise AssertionError("expected ValueError for S < 2")
