"""Tests for DQN objective on synthetic tensors."""
from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import DqnObjective
from mouse_core.objectives.dqn import _pair_weight, _weighted_mean

def test_dqn_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}, batch_size=[n])
    out = TensorDict({'action_value': torch.randn(n, a), 'action_value_target': torch.randn(n, a)}, batch_size=[n])
    objective = DqnObjective(gamma_step=0.99)
    loss, metrics = objective(step_stream, out)
    assert loss.ndim == 0
    assert 'action_value' in metrics
    assert metrics['action_value'] >= 0.0

def test_dqn_objective_rejects_wrong_action_shape() -> None:
    n, a = (4, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n, 1)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long)}, batch_size=[n])
    out = TensorDict({'action_value': torch.randn(n, a), 'action_value_target': torch.randn(n, a)}, batch_size=[n])
    try:
        DqnObjective(gamma_step=0.99)(step_stream, out)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for action shape [N, 1]')

def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = TensorDict({'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'episode_done': torch.zeros(1, dtype=torch.long), 'task_done': torch.zeros(1, dtype=torch.long)}, batch_size=[1])
    out = TensorDict({'action_value': torch.zeros(1, 2), 'action_value_target': torch.zeros(1, 2)}, batch_size=[1])
    try:
        DqnObjective()(step_stream, out)
    except ValueError as e:
        assert 'Not enough' in str(e)
    else:
        raise AssertionError('expected ValueError for N < 2')

def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 1, 0]), 'task_done': torch.tensor([0, 0, 0])}, batch_size=[3])
    out = TensorDict({'action_value': torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), 'action_value_target': torch.zeros(3, 2)}, batch_size=[3])
    loss, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0)(step_stream, out)
    assert abs(loss.item() - 2.5) < 1e-05

def _sequence_fixture(sequence_id: list[int]) -> tuple[TensorDict, TensorDict]:
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor(sequence_id)}, batch_size=[3])
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

def test_dqn_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream, out = _sequence_fixture([0, 1, 2])
    loss, metrics = DqnObjective(gamma_step=0.0)(step_stream, out)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics['q_values_mean']) < 1e-05

def test_dqn_objective_skips_transitions_across_tasks() -> None:
    """A pair whose steps belong to different tasks is not a transition."""
    step_stream = TensorDict(
        {
            'action': torch.tensor([0, 1, 0]),
            'reward': torch.tensor([0.0, 1.0, 5.0]),
            'episode_done': torch.tensor([0, 1, 0]),
            'task_done': torch.tensor([0, 2, 0]),
            'sequence_id': torch.tensor([0, 0, 0]),
            'grouping_id': torch.tensor([0, 0, 1]),
        },
        batch_size=[3],
    )
    out = TensorDict(
        {
            'action_value': torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]),
            'action_value_target': torch.zeros(3, 2),
        },
        batch_size=[3],
    )
    # Only pair (0,1) is valid (same task); pair (1,2) crosses grouping_id.
    loss, metrics = DqnObjective(gamma_step=0.0, grouping_field="grouping_id")(step_stream, out)
    assert abs(loss.item() - 1.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 2.0) < 1e-05


def test_pair_weight_zeros_window_cut_and_task_change() -> None:
    """Last output of a window or grouping run has weight 0; in-run reset stays 1."""
    window = TensorDict(
        {
            'sequence_id': torch.tensor([0, 0, 1, 1]),
            'task_index': torch.tensor([0, 0, 0, 0]),
        },
        batch_size=[4],
    )
    assert torch.equal(
        _pair_weight(window, 4, 'cpu', grouping_field='task_index'),
        torch.tensor([1.0, 0.0, 1.0]),
    )

    task_change = TensorDict(
        {
            'sequence_id': torch.tensor([0, 0, 0]),
            'task_index': torch.tensor([0, 0, 1]),
        },
        batch_size=[3],
    )
    assert torch.equal(
        _pair_weight(task_change, 3, 'cpu', grouping_field='task_index'),
        torch.tensor([1.0, 0.0]),
    )

    reset = TensorDict(
        {
            'sequence_id': torch.tensor([0, 0, 0]),
            'task_index': torch.tensor([0, 0, 0]),
            'episode_done': torch.tensor([0, 1, 0]),
        },
        batch_size=[3],
    )
    assert torch.equal(
        _pair_weight(reset, 3, 'cpu', grouping_field='task_index'),
        torch.tensor([1.0, 1.0]),
    )


def test_weighted_mean_all_zero_is_zero() -> None:
    assert abs(_weighted_mean(torch.tensor([3.0, 4.0]), torch.zeros(2)).item()) < 1e-05


def test_dqn_same_run_episode_reset_trains_both_pairs() -> None:
    """live→terminal→reset with the same sequence_id and task_index still trains."""
    step_stream = TensorDict(
        {
            'action': torch.tensor([0, 1, 0]),
            'reward': torch.tensor([0.0, 1.0, 5.0]),
            'episode_done': torch.tensor([0, 1, 0]),
            'task_done': torch.tensor([0, 0, 0]),
            'sequence_id': torch.tensor([0, 0, 0]),
            'task_index': torch.tensor([0, 0, 0]),
        },
        batch_size=[3],
    )
    out = TensorDict(
        {
            'action_value': torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]),
            'action_value_target': torch.zeros(3, 2),
        },
        batch_size=[3],
    )
    loss, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, grouping_field='task_index'
    )(step_stream, out)
    assert abs(loss.item() - 2.5) < 1e-05


def test_dqn_objective_multiplies_episode_and_task_gammas() -> None:
    """Last episode of a task applies episode gamma, then task gamma."""
    step_stream = TensorDict(
        {
            'action': torch.tensor([0, 1]),
            'reward': torch.tensor([0.0, 1.0]),
            'episode_done': torch.tensor([0, 1]),
            'task_done': torch.tensor([0, 2]),
        },
        batch_size=[2],
    )
    out = TensorDict(
        {
            'action_value': torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
            'action_value_target': torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
        },
        batch_size=[2],
    )
    # Q(s0, a=1) = 2; next max Q = 10; r = 1.
    # Product 0.5 * 0.4 ⇒ target = 1 + 2.0 = 3; loss (2-3)^2 = 1.
    # Episode-only would be 1 + 5 = 6 (loss 16); task-only 1 + 4 = 5 (loss 9).
    loss, _ = DqnObjective(
        gamma_step=0.0,
        gamma_episode_terminal=0.5,
        gamma_task_truncated=0.4,
    )(step_stream, out)
    assert abs(loss.item() - 1.0) < 1e-05
