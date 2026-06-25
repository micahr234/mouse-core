"""Tests for DQN objective on synthetic tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import DqnObjective


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
    objective = DqnObjective(gamma_step=0.99)
    loss, metrics = objective(step_stream, out)
    assert loss.ndim == 0
    assert "action_value" in metrics
    assert metrics["action_value"] >= 0.0


def test_dqn_objective_rejects_wrong_action_shape() -> None:
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
    try:
        DqnObjective(gamma_step=0.99)(step_stream, out)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for action shape [B, S, 1]")


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
        DqnObjective()(step_stream, out)
    except ValueError as e:
        assert "Not enough" in str(e)
    else:
        raise AssertionError("expected ValueError for S < 2")


def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    # Sequence: s0 (running) → s1 (episode terminal, done=1) → s2 (reset).
    # The transition out of s1 (stored at t+2) should still be trained.
    # With gamma_episode_terminal=0.0 the td_target for that transition is
    # just reward[2], so the loss on it is (q(s1, a2) - reward[2])^2.
    step_stream = TensorDict(
        {
            "action": torch.tensor([[0, 1, 0]]),
            "reward": torch.tensor([[0.0, 1.0, 5.0]]),
            "done": torch.tensor([[0, 1, 0]]),
        },
        batch_size=(1, 3),
    )
    out = TensorDict(
        {
            "action_value": torch.tensor([[[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]]),
            "action_value_target": torch.zeros(1, 3, 2),
        },
        batch_size=(1, 3),
    )

    loss, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0)(step_stream, out)

    # Two transitions:
    #   t=0: q(s0, a=1)=2.0, td_target=reward[1]+0=1.0  → (2-1)^2 = 1.0
    #   t=1: q(s1, a=0)=3.0, td_target=reward[2]+0=5.0  → (3-5)^2 = 4.0
    # mean = 2.5
    assert abs(loss.item() - 2.5) < 1e-5
