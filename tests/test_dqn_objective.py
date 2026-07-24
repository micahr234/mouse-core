"""Tests for DQN objective on synthetic tensors."""
from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import DqnObjective

def test_dqn_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}, batch_size=[n])
    out = TensorDict({'action_value': torch.randn(n, a), 'action_value_target': torch.randn(n, a)}, batch_size=[n])
    objective = DqnObjective(gamma_step=0.99)
    loss, metrics = objective(step_stream, out)
    assert loss.ndim == 0
    assert 'action_value' in metrics
    assert metrics['action_value'] >= 0.0

def test_dqn_objective_rejects_wrong_action_shape() -> None:
    n, a = (4, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n, 1)), 'reward': torch.randn(n), 'done': torch.zeros(n, dtype=torch.long)}, batch_size=[n])
    out = TensorDict({'action_value': torch.randn(n, a), 'action_value_target': torch.randn(n, a)}, batch_size=[n])
    try:
        DqnObjective(gamma_step=0.99)(step_stream, out)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for action shape [N, 1]')

def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = TensorDict({'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'done': torch.zeros(1, dtype=torch.long)}, batch_size=[1])
    out = TensorDict({'action_value': torch.zeros(1, 2), 'action_value_target': torch.zeros(1, 2)}, batch_size=[1])
    try:
        DqnObjective()(step_stream, out)
    except ValueError as e:
        assert 'Not enough' in str(e)
    else:
        raise AssertionError('expected ValueError for N < 2')

def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'done': torch.tensor([0, 1, 0])}, batch_size=[3])
    out = TensorDict({'action_value': torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), 'action_value_target': torch.zeros(3, 2)}, batch_size=[3])
    loss, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0)(step_stream, out)
    assert abs(loss.item() - 2.5) < 1e-05

def _sequence_fixture(sequence_id: list[int]) -> tuple[TensorDict, TensorDict]:
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor(sequence_id)}, batch_size=[3])
    out = TensorDict({'action_value': torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), 'action_value_target': torch.zeros(3, 2)}, batch_size=[3])
    return (step_stream, out)

def test_dqn_objective_skips_transitions_across_sequences() -> None:
    """A pair whose steps belong to different sequences is not a transition."""
    step_stream, out = _sequence_fixture([0, 1, 1])
    loss, metrics = DqnObjective(gamma_step=0.0)(step_stream, out)
    assert abs(loss.item() - 4.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 3.0) < 1e-05

def test_dqn_objective_without_sequence_breaks_trains_all_pairs() -> None:
    step_stream, out = _sequence_fixture([0, 0, 0])
    loss, _ = DqnObjective(gamma_step=0.0)(step_stream, out)
    assert abs(loss.item() - 2.5) < 1e-05

def test_dqn_objective_raises_when_all_pairs_cross_sequences() -> None:
    step_stream, out = _sequence_fixture([0, 1, 2])
    try:
        DqnObjective(gamma_step=0.0)(step_stream, out)
    except ValueError as e:
        assert 'sequence boundary' in str(e)
    else:
        raise AssertionError('expected ValueError when every pair crosses a sequence boundary')
