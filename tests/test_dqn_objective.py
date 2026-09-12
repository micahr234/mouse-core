"""Tests for DQN objective on synthetic tensors."""
from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import DqnObjective
import pytest
from mouse_core.objectives.dqn import (
    _affine_scan_backward,
    _pair_weight,
    _weighted_mean,
)


def _q(online: torch.Tensor, delayed: torch.Tensor) -> tuple[TensorDict, TensorDict]:
    n = online.shape[0]
    return (
        TensorDict({"action_value": online}, batch_size=[n]),
        TensorDict({"action_value": delayed}, batch_size=[n]),
    )


def test_dqn_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}, batch_size=[n])
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    objective = DqnObjective(gamma_step=0.99, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)
    loss, metrics = objective(step_stream, predictions, delayed)
    assert loss.ndim == 0
    assert 'action_value' in metrics
    assert metrics['action_value'] >= 0.0
    assert 'watkins_greedy_frac' not in metrics

def test_dqn_objective_rejects_wrong_action_shape() -> None:
    n, a = (4, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n, 1)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long)}, batch_size=[n])
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    with pytest.raises(ValueError, match="action shape"):
        DqnObjective(gamma_step=0.99, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)

def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = TensorDict({'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'episode_done': torch.zeros(1, dtype=torch.long), 'task_done': torch.zeros(1, dtype=torch.long)}, batch_size=[1])
    predictions, delayed = _q(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Not enough"):
        DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)

def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 1, 0]), 'task_done': torch.tensor([0, 0, 0])}, batch_size=[3])
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - 2.5) < 1e-05


def test_dqn_objective_reward_affine_is_identity_by_default() -> None:
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        },
        batch_size=[3],
    )
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    plain, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(
        step_stream, predictions, delayed
    )
    affine, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, reward_scale=1.0, reward_shift=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(plain.item() - affine.item()) < 1e-05


def test_dqn_objective_reward_scale_and_shift() -> None:
    """gamma=0 so targets are the affine rewards stored at i+1."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        },
        batch_size=[3],
    )
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    # Unscaled targets 1 and 5 → MSE 2.5. Scale 2: targets 2 and 10 → (2-2)^2, (3-10)^2.
    scaled, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, reward_scale=2.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(scaled.item() - 24.5) < 1e-05
    # Shift 1: targets 2 and 6 → (2-2)^2, (3-6)^2.
    shifted, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, reward_shift=1.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(shifted.item() - 4.5) < 1e-05
    # Leaves objective_data reward unchanged.
    assert torch.equal(step_stream["reward"], torch.tensor([0.0, 1.0, 5.0]))


def test_dqn_objective_q_affine_is_identity_by_default() -> None:
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        },
        batch_size=[3],
    )
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    plain, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(
        step_stream, predictions, delayed
    )
    affine, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, q_scale=1.0, q_shift=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(plain.item() - affine.item()) < 1e-05


def test_dqn_objective_q_scale_and_shift() -> None:
    """gamma=0 so bootstrap is unused; affine applies to online Q only."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        },
        batch_size=[3],
    )
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    predictions, delayed = _q(online, torch.zeros(3, 2))
    # Taken Q 2 and 3, targets 1 and 5. Scale 2: 4 and 6 → (4-1)^2, (6-5)^2.
    scaled, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, q_scale=2.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(scaled.item() - 5.0) < 1e-05
    # Shift 1: 3 and 4 → (3-1)^2, (4-5)^2.
    shifted, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, q_shift=1.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(shifted.item() - 2.5) < 1e-05
    assert torch.equal(predictions["action_value"], online)


def _sequence_fixture(sequence_id: list[int]) -> tuple[TensorDict, TensorDict, TensorDict]:
    step_stream = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor(sequence_id)}, batch_size=[3])
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    return (step_stream, predictions, delayed)

def test_dqn_objective_skips_transitions_across_sequences() -> None:
    """A pair whose steps belong to different sequences is not a transition."""
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 1])
    loss, metrics = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - 4.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 3.0) < 1e-05

def test_dqn_objective_without_sequence_breaks_trains_all_pairs() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 0, 0])
    loss, _ = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - 2.5) < 1e-05

def test_dqn_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 2])
    loss, metrics = DqnObjective(gamma_step=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
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
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    # Only pair (0,1) is valid (same task); pair (1,2) crosses grouping_id.
    loss, metrics = DqnObjective(gamma_step=0.0, grouping_field="grouping_id", gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
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


def test_pair_weight_missing_grouping_field_raises() -> None:
    data = TensorDict({'sequence_id': torch.tensor([0, 0])}, batch_size=[2])
    with pytest.raises(KeyError, match="grouping_field"):
        _pair_weight(data, 2, 'cpu', grouping_field='task_index')


def test_dqn_objective_rejects_out_of_range_action() -> None:
    step_stream = TensorDict(
        {
            'action': torch.tensor([0, 9]),
            'reward': torch.tensor([0.0, 1.0]),
            'episode_done': torch.zeros(2, dtype=torch.long),
            'task_done': torch.zeros(2, dtype=torch.long),
        },
        batch_size=[2],
    )
    predictions, delayed = _q(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="action ids must be in"):
        DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)


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
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective(
        gamma_step=0.0, gamma_episode_terminal=0.0, grouping_field='task_index',
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
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
    predictions, delayed = _q(
        torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
    )
    # Q(s0, a=1) = 2; next max Q = 10; r = 1.
    # Product 0.5 * 0.4 ⇒ target = 1 + 2.0 = 3; loss (2-3)^2 = 1.
    # Episode-only would be 1 + 5 = 6 (loss 16); task-only 1 + 4 = 5 (loss 9).
    loss, _ = DqnObjective(
        gamma_step=0.0,
        gamma_episode_terminal=0.5,
        gamma_task_truncated=0.4,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - 1.0) < 1e-05


def test_dqn_objective_does_not_backprop_through_delayed_q() -> None:
    """Bootstrap Q is a constant: delayed Q must not receive a gradient."""
    n, a = 4, 2
    step_stream = TensorDict(
        {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
        },
        batch_size=[n],
    )
    online = torch.randn(n, a, requires_grad=True)
    delayed = torch.randn(n, a, requires_grad=True)
    predictions, delayed_td = _q(online, delayed)
    loss, _ = DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed_td)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None


def _lambda_fixture() -> tuple[TensorDict, TensorDict, TensorDict]:
    """Three in-run steps so λ can mix a two-step backup from s0.

    Action from s0 is 0; from s1 is 1. Delayed max-Q is 3 at s1 and 100 at s2.
    Rewards out of s0 / s1 are 1 and 10. One-step target at s0 is 4; the
    λ=1 (full n-step) target at s0 is 111.
    """
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.zeros(3, dtype=torch.int64),
            "info_q_star": torch.tensor(
                [[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]
            ),
        },
        batch_size=[3],
    )
    # Online Q(s0, a=0) = 5; Q(s1, a=1) = 0.
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    return step_stream, *_q(online, delayed)


# One-step MSEs: (5-4)^2 = 1 and (0-110)^2 = 12100 → mean 6050.5.
_ONE_STEP = 6050.5
# λ=1 target at s0 is 1 + 10 + 100 = 111 → (5-111)^2 = 11236; s1 stays 12100 → mean 11668.
_FULL_RETURN = 11668.0


def test_td_lambda_zero_is_the_one_step_target() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, metrics = DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - _ONE_STEP) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_dqn_objective_q_affine_applies_to_online_and_delayed() -> None:
    """Same affine on online Q and delayed bootstrap (γ=1 one-step)."""
    step_stream, predictions, delayed = _lambda_fixture()
    # Taken Q 10 and 0; delayed max 6 and 200; targets 7 and 210.
    loss, _ = DqnObjective(gamma_step=1.0, q_scale=2.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(
        step_stream, predictions, delayed
    )
    expected = (9.0 + 44100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_td_lambda_one_is_the_full_n_step_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = DqnObjective(gamma_step=1.0, td_lambda=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - _FULL_RETURN) < 1e-03


def test_td_lambda_half_mixes_bootstrap_and_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    # G_0 = 1 + (0.5 * 3 + 0.5 * 110) = 57.5 → (5 - 57.5)^2 = 2756.25; s1 stays 12100.
    expected = (2756.25 + 12100.0) / 2
    loss, _ = DqnObjective(gamma_step=1.0, td_lambda=0.5, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - expected) < 1e-03


def test_td_lambda_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DqnObjective(td_lambda=1.5, gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)


def test_td_lambda_terminal_gamma_zero_ends_the_trace() -> None:
    """s0 → s1 terminates the episode (code stored at s1): G_0 = r only."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["episode_done"] = torch.tensor([0, 1, 0])
    loss, _ = DqnObjective(
        gamma_step=1.0, gamma_episode_terminal=0.0, td_lambda=1.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    # s0: (5 - 1)^2 = 16 — neither V(s1) nor the next episode's return; s1: 12100.
    assert abs(loss.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_td_lambda_truncation_gamma_carries_the_trace_discounted() -> None:
    """A non-zero truncation gamma bootstraps through the reset, so the trace does too."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["episode_done"] = torch.tensor([0, 2, 0])
    loss, _ = DqnObjective(
        gamma_step=1.0, gamma_episode_truncated=0.5, td_lambda=1.0,
        gamma_episode_terminal=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    # s0: 1 + 0.5 * G_1 = 1 + 0.5 * 110 = 56 → (5 - 56)^2 = 2601; s1: 12100.
    assert abs(loss.item() - (2601.0 + 12100.0) / 2) < 1e-03


def test_td_lambda_task_gamma_scales_the_trace() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["task_done"] = torch.tensor([0, 2, 0])
    full, _ = DqnObjective(
        gamma_step=1.0, gamma_task_truncated=1.0, td_lambda=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0)(step_stream, predictions, delayed)
    assert abs(full.item() - _FULL_RETURN) < 1e-03
    cut, _ = DqnObjective(
        gamma_step=1.0, gamma_task_truncated=0.0, td_lambda=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0)(step_stream, predictions, delayed)
    assert abs(cut.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_watkins_cuts_when_taken_action_is_not_online_greedy() -> None:
    """Online Q at s1 prefers 0; taken action is 1 → cut. Oracle would continue."""
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.clone()
    q = predictions["action_value"].clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions["action_value"] = q
    # A separate action-head would prefer the taken action; Watkins must ignore it.
    predictions["action"] = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    loss, metrics = DqnObjective(gamma_step=1.0, td_lambda=1.0, watkins=True, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(
        step_stream, predictions, delayed
    )
    one_step, _ = DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04
    # a_0 = 0 is greedy at s0 (Q = [5, 0]); a_1 = 1 is not at s1 → 1 of 2 in-run pairs.
    assert abs(metrics["watkins_greedy_frac"] - 0.5) < 1e-06


def test_watkins_continues_when_taken_action_matches_online_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.clone()
    q = predictions["action_value"].clone()
    q[1] = torch.tensor([-1.0, 0.0])  # greedy at s1 is the taken action; Q(s1,a=1) stays 0
    predictions["action_value"] = q
    loss, metrics = DqnObjective(gamma_step=1.0, td_lambda=1.0, watkins=True, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(
        step_stream, predictions, delayed
    )
    assert abs(loss.item() - _FULL_RETURN) < 1e-03
    assert abs(metrics["watkins_greedy_frac"] - 1.0) < 1e-06


def test_lambda_returns_do_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["sequence_id"] = torch.tensor([0, 0, 1])
    loss, _ = DqnObjective(gamma_step=1.0, td_lambda=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    one_step, _ = DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04


def test_td_lambda_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's λ-return; bootstrap reads the last row."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["head_output_count"] = torch.tensor([2, 1, 2])
    online = torch.tensor([[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]])
    predictions, delayed = _q(online, delayed_q)
    loss, _ = DqnObjective(gamma_step=1.0, td_lambda=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
    # s0 rows: (5-111)^2 = 11236, (7-111)^2 = 10816; s1 row: (0-110)^2 = 12100; s2 rows weight 0.
    assert abs(loss.item() - (11236.0 + 10816.0 + 12100.0) / 3) < 1e-02


def test_affine_scan_matches_sequential_recursion() -> None:
    torch.manual_seed(0)
    T = 1000
    a = torch.randn(T)
    b = torch.rand(T) * 0.99
    b[torch.rand(T) < 0.1] = 0.0  # run ends / trace cuts / terminal discounts
    expected = torch.zeros(T)
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = a[t].item() + b[t].item() * g
        expected[t] = g
    assert torch.allclose(_affine_scan_backward(a, b), expected, atol=1e-04, rtol=1e-05)


def test_dqn_objective_rejects_non_fp32_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.clone()
    predictions["action_value"] = predictions["action_value"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        DqnObjective(gamma_step=1.0, gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)(step_stream, predictions, delayed)
