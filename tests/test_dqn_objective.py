"""Tests for DQN objective on synthetic tensors."""
from __future__ import annotations
import torch
from mouse_core.objectives import DqnObjective, affine_reward, affine_value, boundary_discount, boundary_reward, boundary_value
from tests._bound_head import BoundHead
import pytest
from mouse_core.objectives.dqn import (
    _affine_scan_backward,
    _pair_weight,
    _weighted_mean,
)


def _disc(**overrides: float | None):
    kwargs: dict[str, float | None] = dict(
        gamma_step=None,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    kwargs.update(overrides)
    return boundary_discount(**kwargs)


def _rew(**overrides: object):
    kwargs: dict[str, object] = dict(scale=None, shift=None)
    kwargs.update(overrides)
    return affine_reward(**kwargs)  # type: ignore[arg-type]


def _val(**overrides: object):
    kwargs: dict[str, object] = dict(scale=None, shift=None)
    kwargs.update(overrides)
    return affine_value(**kwargs)  # type: ignore[arg-type]


_Q = BoundHead("action_value")


def test_boundary_discount_always_multiplies_gamma_step() -> None:
    discount = boundary_discount(
        gamma_step=0.5,
        gamma_episode_terminal=0.4,
        gamma_episode_truncated=0.2,
        gamma_task_terminal=0.8,
        gamma_task_truncated=0.25,
    )
    got = discount(
        episode_done=torch.tensor([0, 1, 2, 1, 2]),
        task_done=torch.tensor([0, 0, 0, 2, 1]),
    )
    want = torch.tensor([0.5, 0.5 * 0.4, 0.5 * 0.2, 0.5 * 0.4 * 0.25, 0.5 * 0.2 * 0.8])
    assert torch.allclose(got, want)


def test_boundary_reward_applies_episode_and_task_affine() -> None:
    reward = boundary_reward(
        scale=2.0,
        shift=1.0,
        reward_episode_terminal_scale=3.0,
        reward_episode_terminal_shift=10.0,
        reward_episode_truncated_scale=0.5,
        reward_episode_truncated_shift=4.0,
        reward_task_terminal_scale=2.0,
        reward_task_terminal_shift=3.0,
        reward_task_truncated_scale=0.25,
        reward_task_truncated_shift=0.5,
    )
    got = reward(
        reward=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
        episode_done=torch.tensor([0, 1, 2, 1, 2]),
        task_done=torch.tensor([0, 0, 0, 2, 1]),
    )
    want = torch.tensor([
        2.0 * 1.0 + 1.0,
        2.0 * 3.0 * 1.0 + 1.0 + 10.0,
        2.0 * 0.5 * 1.0 + 1.0 + 4.0,
        2.0 * 3.0 * 0.25 * 1.0 + 1.0 + 10.0 + 0.5,
        2.0 * 0.5 * 2.0 * 1.0 + 1.0 + 4.0 + 3.0,
    ])
    assert torch.allclose(got, want)


def test_boundary_value_applies_episode_and_task_affine() -> None:
    value = boundary_value(
        scale=2.0,
        shift=1.0,
        value_episode_terminal_scale=3.0,
        value_episode_terminal_shift=10.0,
        value_episode_truncated_scale=0.5,
        value_episode_truncated_shift=4.0,
        value_task_terminal_scale=2.0,
        value_task_terminal_shift=3.0,
        value_task_truncated_scale=0.25,
        value_task_truncated_shift=0.5,
    )
    q = torch.ones(5, 2)
    got = value(
        value=q,
        episode_done=torch.tensor([0, 1, 2, 1, 2]),
        task_done=torch.tensor([0, 0, 0, 2, 1]),
    )
    want_rows = torch.tensor([
        2.0 * 1.0 + 1.0,
        2.0 * 3.0 * 1.0 + 1.0 + 10.0,
        2.0 * 0.5 * 1.0 + 1.0 + 4.0,
        2.0 * 3.0 * 0.25 * 1.0 + 1.0 + 10.0 + 0.5,
        2.0 * 0.5 * 2.0 * 1.0 + 1.0 + 4.0 + 3.0,
    ])
    assert got.shape == q.shape
    assert torch.allclose(got, want_rows.unsqueeze(-1).expand_as(q))


def test_factory_none_args_are_identity() -> None:
    reward_col = torch.tensor([0.5, -1.0, 3.0])
    q = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    episode_done = torch.tensor([0, 1, 2])
    task_done = torch.tensor([1, 2, 0])
    reward = boundary_reward(
        scale=None,
        shift=None,
        reward_episode_terminal_scale=None,
        reward_episode_terminal_shift=None,
        reward_episode_truncated_scale=None,
        reward_episode_truncated_shift=None,
        reward_task_terminal_scale=None,
        reward_task_terminal_shift=None,
        reward_task_truncated_scale=None,
        reward_task_truncated_shift=None,
    )
    value = boundary_value(
        scale=None,
        shift=None,
        value_episode_terminal_scale=None,
        value_episode_terminal_shift=None,
        value_episode_truncated_scale=None,
        value_episode_truncated_shift=None,
        value_task_terminal_scale=None,
        value_task_terminal_shift=None,
        value_task_truncated_scale=None,
        value_task_truncated_shift=None,
    )
    discount = boundary_discount(
        gamma_step=None,
        gamma_episode_terminal=None,
        gamma_episode_truncated=None,
        gamma_task_terminal=None,
        gamma_task_truncated=None,
    )
    got_reward = reward(reward=reward_col, episode_done=episode_done, task_done=task_done)
    got_value = value(value=q, episode_done=torch.tensor([0, 1]), task_done=torch.tensor([2, 0]))
    got_discount = discount(episode_done=episode_done, task_done=task_done)
    assert got_reward is reward_col
    assert got_value is q
    assert torch.equal(got_discount, torch.ones(3))
    assert affine_reward(scale=None, shift=None)(reward=reward_col) is reward_col
    assert affine_value(scale=None, shift=None)(value=q) is q


def test_factory_args_are_required() -> None:
    with pytest.raises(TypeError):
        boundary_discount()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        boundary_reward()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        boundary_value()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        affine_reward()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        affine_value()  # type: ignore[call-arg]


def test_dqn_requires_temperature_argument() -> None:
    with pytest.raises(TypeError, match="temperature"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            grouping_field=None,
        )


def test_dqn_requires_grouping_field_argument() -> None:
    with pytest.raises(TypeError, match="grouping_field"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            temperature=0.0,
        )


def _q(online: torch.Tensor, delayed: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n = online.shape[0]
    return (
        {"action_value": online},
        {"action_value": delayed},
    )


def test_dqn_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = {'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    objective = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.99), grouping_field=None, temperature=0.0)
    loss, metrics = objective(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert loss.ndim == 0
    assert 'action_value' in metrics
    assert metrics['action_value'] >= 0.0
    assert 'watkins_greedy_frac' not in metrics
    assert 'entropy' not in metrics

def test_dqn_objective_rejects_wrong_action_shape() -> None:
    n, a = (4, 3)
    step_stream = {'action': torch.randint(0, a, (n, 1)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long)}
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    with pytest.raises(ValueError, match="action shape"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.99), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)

def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = {'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'episode_done': torch.zeros(1, dtype=torch.long), 'task_done': torch.zeros(1, dtype=torch.long)}
    predictions, delayed = _q(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Not enough"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)

def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 1, 0]), 'task_done': torch.tensor([0, 0, 0])}
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 2.5) < 1e-05


def test_dqn_requires_reward_argument() -> None:
    with pytest.raises(TypeError, match="reward"):
        DqnObjective(  # type: ignore[call-arg]
            head=_Q,
            discount=_disc(gamma_step=0.99),
            value=_val(),
            grouping_field=None,
            temperature=0.0,
        )


def test_dqn_requires_value_argument() -> None:
    with pytest.raises(TypeError, match="value"):
        DqnObjective(  # type: ignore[call-arg]
            head=_Q,
            reward=_rew(),
            discount=_disc(gamma_step=0.99),
            grouping_field=None,
            temperature=0.0,
        )


def test_affine_reward_reads_column() -> None:
    reward = affine_reward(scale=2.0, shift=1.0)
    got = reward(
        reward=torch.tensor([0.0, 1.0, 5.0]),
        bonus=torch.tensor([9.0, 8.0, 7.0]),
    )
    assert torch.allclose(got, torch.tensor([1.0, 3.0, 11.0]))


def test_dqn_objective_reward_scale_and_shift() -> None:
    """gamma=0 so targets are the affine rewards stored at i+1."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    # Unscaled targets 1 and 5 → MSE 2.5. Scale 2: targets 2 and 10 → (2-2)^2, (3-10)^2.
    scaled, _ = DqnObjective(head=_Q, reward=_rew(scale=2.0), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(scaled.item() - 24.5) < 1e-05
    # Shift 1: targets 2 and 6 → (2-2)^2, (3-6)^2.
    shifted, _ = DqnObjective(head=_Q, reward=_rew(shift=1.0), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(shifted.item() - 4.5) < 1e-05
    # Leaves objective_data reward unchanged.
    assert torch.equal(step_stream["reward"], torch.tensor([0.0, 1.0, 5.0]))


def test_dqn_objective_custom_reward_reads_objective_data() -> None:
    """A custom reward can combine columns; the raw ``reward`` column is unused."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "bonus": torch.tensor([0.0, 1.0, 1.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))

    def reward(*, reward: torch.Tensor, bonus: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return reward + bonus

    # Targets 2 and 6 → (2-2)^2, (3-6)^2.
    loss, _ = DqnObjective(head=_Q, reward=reward, value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 4.5) < 1e-05


def test_dqn_objective_q_affine_is_identity_by_default() -> None:
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    plain, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    affine, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(scale=1.0, shift=0.0), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(plain.item() - affine.item()) < 1e-05


def test_dqn_objective_q_scale_and_shift() -> None:
    """gamma=0 so bootstrap is unused; affine applies to online Q only."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    predictions, delayed = _q(online, torch.zeros(3, 2))
    # Taken Q 2 and 3, targets 1 and 5. Scale 2: 4 and 6 → (4-1)^2, (6-5)^2.
    scaled, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(scale=2.0), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(scaled.item() - 5.0) < 1e-05
    # Shift 1: 3 and 4 → (3-1)^2, (4-5)^2.
    shifted, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(shift=1.0), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(shifted.item() - 2.5) < 1e-05
    assert torch.equal(predictions["action_value"], online)


def _sequence_fixture(sequence_id: list[int]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor(sequence_id)}
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    return (step_stream, predictions, delayed)

def test_dqn_objective_skips_transitions_across_sequences() -> None:
    """A pair whose steps belong to different sequences is not a transition."""
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 1])
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 4.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 3.0) < 1e-05

def test_dqn_objective_without_sequence_breaks_trains_all_pairs() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 0, 0])
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 2.5) < 1e-05

def test_dqn_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 2])
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics['q_values_mean']) < 1e-05

def test_dqn_objective_skips_transitions_across_tasks() -> None:
    """A pair whose steps belong to different tasks is not a transition."""
    step_stream = {
            'action': torch.tensor([0, 1, 0]),
            'reward': torch.tensor([0.0, 1.0, 5.0]),
            'episode_done': torch.tensor([0, 1, 0]),
            'task_done': torch.tensor([0, 2, 0]),
            'sequence_id': torch.tensor([0, 0, 0]),
            'grouping_id': torch.tensor([0, 0, 1]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    # Only pair (0,1) is valid (same task); pair (1,2) crosses grouping_id.
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field="grouping_id", temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 1.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 2.0) < 1e-05


def test_pair_weight_zeros_window_cut_and_task_change() -> None:
    """Last output of a window or grouping run has weight 0; in-run reset stays 1."""
    window = {
            'sequence_id': torch.tensor([0, 0, 1, 1]),
            'task_index': torch.tensor([0, 0, 0, 0]),
        }
    assert torch.equal(
        _pair_weight(window, 4, 'cpu', grouping_field='task_index'),
        torch.tensor([1.0, 0.0, 1.0]),
    )

    task_change = {
            'sequence_id': torch.tensor([0, 0, 0]),
            'task_index': torch.tensor([0, 0, 1]),
        }
    assert torch.equal(
        _pair_weight(task_change, 3, 'cpu', grouping_field='task_index'),
        torch.tensor([1.0, 0.0]),
    )

    reset = {
            'sequence_id': torch.tensor([0, 0, 0]),
            'task_index': torch.tensor([0, 0, 0]),
            'episode_done': torch.tensor([0, 1, 0]),
        }
    assert torch.equal(
        _pair_weight(reset, 3, 'cpu', grouping_field='task_index'),
        torch.tensor([1.0, 1.0]),
    )


def test_pair_weight_missing_grouping_field_raises() -> None:
    data = {'sequence_id': torch.tensor([0, 0])}
    with pytest.raises(KeyError, match="grouping_field"):
        _pair_weight(data, 2, 'cpu', grouping_field='task_index')


def test_dqn_objective_rejects_out_of_range_action() -> None:
    step_stream = {
            'action': torch.tensor([0, 9]),
            'reward': torch.tensor([0.0, 1.0]),
            'episode_done': torch.zeros(2, dtype=torch.long),
            'task_done': torch.zeros(2, dtype=torch.long),
        }
    predictions, delayed = _q(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="action ids must be in"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)


def test_weighted_mean_all_zero_is_zero() -> None:
    assert abs(_weighted_mean(torch.tensor([3.0, 4.0]), torch.zeros(2)).item()) < 1e-05


def test_dqn_same_run_episode_reset_trains_both_pairs() -> None:
    """live→terminal→reset with the same sequence_id and task_index still trains."""
    step_stream = {
            'action': torch.tensor([0, 1, 0]),
            'reward': torch.tensor([0.0, 1.0, 5.0]),
            'episode_done': torch.tensor([0, 1, 0]),
            'task_done': torch.tensor([0, 0, 0]),
            'sequence_id': torch.tensor([0, 0, 0]),
            'task_index': torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field='task_index', temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 2.5) < 1e-05


def test_dqn_objective_multiplies_step_episode_and_task_gammas() -> None:
    """``gamma_step`` always multiplies the episode extra and the task extra."""
    step_stream = {
            'action': torch.tensor([0, 1]),
            'reward': torch.tensor([0.0, 1.0]),
            'episode_done': torch.tensor([0, 1]),
            'task_done': torch.tensor([0, 2]),
        }
    predictions, delayed = _q(
        torch.tensor([[0.0, 2.0], [4.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
    )
    # Q(s0, a=1) = 2; next max Q = 10; r = 1.
    # Product 0.5 * 1.0 * 0.4 ⇒ target = 1 + 2.0 = 3; loss (2-3)^2 = 1.
    # Without gamma_step the product would be 0.4 (target 5, loss 9).
    loss, _ = DqnObjective(
        head=_Q,
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.5, gamma_episode_terminal=1.0, gamma_task_truncated=0.4),
        grouping_field=None,
        temperature=0.0,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 1.0) < 1e-05
    # gamma_step=0 zeros the bootstrap even when both extras are 1.
    zero_step, _ = DqnObjective(
        head=_Q,
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.0, gamma_episode_terminal=1.0, gamma_task_truncated=1.0),
        grouping_field=None,
        temperature=0.0,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(zero_step.item() - 1.0) < 1e-05


def test_dqn_objective_does_not_backprop_through_delayed_q() -> None:
    """Bootstrap Q is a constant: delayed Q must not receive a gradient."""
    n, a = 4, 2
    step_stream = {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
        }
    online = torch.randn(n, a, requires_grad=True)
    delayed = torch.randn(n, a, requires_grad=True)
    predictions, delayed_td = _q(online, delayed)
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_td)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None


def _lambda_fixture() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Three in-run steps so λ can mix a two-step backup from s0.

    Action from s0 is 0; from s1 is 1. Delayed max-Q is 3 at s1 and 100 at s2.
    Rewards out of s0 / s1 are 1 and 10. One-step target at s0 is 4; the
    λ=1 (full n-step) target at s0 is 111.
    """
    step_stream = {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.zeros(3, dtype=torch.int64),
            "info_q_star": torch.tensor(
                [[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]
            ),
        }
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
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - _ONE_STEP) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_dqn_objective_q_affine_applies_to_online_and_delayed() -> None:
    """Same affine on online Q and delayed bootstrap (γ=1 one-step)."""
    step_stream, predictions, delayed = _lambda_fixture()
    # Taken Q 10 and 0; delayed max 6 and 200; targets 7 and 210.
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(scale=2.0), discount=_disc(), grouping_field=None, temperature=0.0)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    expected = (9.0 + 44100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_td_lambda_one_is_the_full_n_step_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=1.0, grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - _FULL_RETURN) < 1e-03


def test_td_lambda_half_mixes_bootstrap_and_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    # G_0 = 1 + (0.5 * 3 + 0.5 * 110) = 57.5 → (5 - 57.5)^2 = 2756.25; s1 stays 12100.
    expected = (2756.25 + 12100.0) / 2
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=0.5, grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - expected) < 1e-03


def test_td_lambda_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), td_lambda=1.5, discount=_disc(), grouping_field=None, temperature=0.0)


def test_td_lambda_terminal_gamma_zero_ends_the_trace() -> None:
    """s0 → s1 terminates the episode (code stored at s1): G_0 = r only."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 1, 0])
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=1.0,
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # s0: (5 - 1)^2 = 16 — neither V(s1) nor the next episode's return; s1: 12100.
    assert abs(loss.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_td_lambda_truncation_gamma_carries_the_trace_discounted() -> None:
    """A non-zero truncation gamma bootstraps through the reset, so the trace does too."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 2, 0])
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_episode_truncated=0.5), td_lambda=1.0,
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # s0: 1 + 0.5 * G_1 = 1 + 0.5 * 110 = 56 → (5 - 56)^2 = 2601; s1: 12100.
    assert abs(loss.item() - (2601.0 + 12100.0) / 2) < 1e-03


def test_td_lambda_task_gamma_scales_the_trace() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["task_done"] = torch.tensor([0, 2, 0])
    full, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_task_truncated=1.0), td_lambda=1.0,
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(full.item() - _FULL_RETURN) < 1e-03
    cut, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_task_truncated=0.0), td_lambda=1.0,
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(cut.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_watkins_cuts_when_taken_action_is_not_online_greedy() -> None:
    """Online Q at s1 prefers 0; taken action is 1 → cut. Oracle would continue."""
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = {key: value.clone() for key, value in predictions.items()}
    q = predictions["action_value"].clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions["action_value"] = q
    # A separate action-head would prefer the taken action; Watkins must ignore it.
    predictions["action"] = torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=1.0, watkins=True, grouping_field=None, temperature=0.0)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    one_step, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04
    # a_0 = 0 is greedy at s0 (Q = [5, 0]); a_1 = 1 is not at s1 → 1 of 2 in-run pairs.
    assert abs(metrics["watkins_greedy_frac"] - 0.5) < 1e-06


def test_watkins_continues_when_taken_action_matches_online_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = {key: value.clone() for key, value in predictions.items()}
    q = predictions["action_value"].clone()
    q[1] = torch.tensor([-1.0, 0.0])  # greedy at s1 is the taken action; Q(s1,a=1) stays 0
    predictions["action_value"] = q
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=1.0, watkins=True, grouping_field=None, temperature=0.0)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    assert abs(loss.item() - _FULL_RETURN) < 1e-03
    assert abs(metrics["watkins_greedy_frac"] - 1.0) < 1e-06


def test_lambda_returns_do_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["sequence_id"] = torch.tensor([0, 0, 1])
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=1.0, grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    one_step, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04


def test_td_lambda_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's λ-return; bootstrap reads the last row."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["head_output_count"] = torch.tensor([2, 1, 2])
    online = torch.tensor([[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]])
    predictions, delayed = _q(online, delayed_q)
    loss, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), td_lambda=1.0, grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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
    predictions = {key: value.clone() for key, value in predictions.items()}
    predictions["action_value"] = predictions["action_value"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)


def _entropy_kw() -> dict[str, object]:
    return dict(
        discount=_disc(),
        grouping_field=None,
        temperature=0.0,
    )


def test_temperature_zero_matches_hard_max() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    hard, hard_m = DqnObjective(head=_Q, reward=_rew(), value=_val(), **_entropy_kw())(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    soft0, soft0_m = DqnObjective(head=_Q, reward=_rew(), value=_val(), **_entropy_kw())(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    assert abs(hard.item() - soft0.item()) < 1e-06
    assert "entropy" not in hard_m
    assert "entropy" not in soft0_m


def test_temperature_soft_value_is_logsumexp() -> None:
    """Two equal delayed Qs: V = α log 2 instead of max = 0."""
    step_stream = {
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 0.0]),
            "episode_done": torch.zeros(2, dtype=torch.int64),
            "task_done": torch.zeros(2, dtype=torch.int64),
        }
    online = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    predictions, delayed_td = _q(online, delayed)
    alpha = 1.0
    loss, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), **{**_entropy_kw(), 'temperature': alpha})(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_td
    )
    # Q(s0, a=0) = 0; target = 0 + 1 * log(2); one in-run pair.
    expected = float(torch.log(torch.tensor(2.0)) ** 2)
    assert abs(loss.item() - expected) < 1e-05
    assert abs(metrics["entropy"] - float(torch.log(torch.tensor(2.0)))) < 1e-05


def test_temperature_matches_expected_q_plus_entropy() -> None:
    """α logsumexp(Q/α) = E_π[Q] + α H[π] for π = softmax(Q/α)."""
    from mouse_core.objectives.dqn import _boltzmann_entropy, _soft_state_value

    q = torch.tensor([[1.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    alpha = 0.5
    pi = torch.softmax(q / alpha, dim=-1)
    expected = (pi * q).sum(dim=-1) + alpha * _boltzmann_entropy(q, temperature=alpha)
    assert torch.allclose(_soft_state_value(q, temperature=alpha), expected, atol=1e-06)


def test_temperature_rejects_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), **{**_entropy_kw(), 'temperature': -0.1})


def test_temperature_does_not_change_gamma_zero_target() -> None:
    """γ=0 so V is unused; soft and hard losses match."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(
        torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2)
    )
    hard, _ = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None,
        temperature=0.0,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    soft, metrics = DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None,
        temperature=2.0,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(hard.item() - soft.item()) < 1e-05
    assert "entropy" in metrics


def test_dqn_requires_discount_argument() -> None:
    with pytest.raises(TypeError, match="discount"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), grouping_field=None, temperature=0.0)  # type: ignore[call-arg]


def test_dqn_custom_discount_function() -> None:
    """A caller-supplied discount is used instead of the done-code lookup."""
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "episode_done": torch.tensor([0, 0]),
            "task_done": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(
        torch.tensor([[0.0, 2.0], [0.0, 0.0]]),
        torch.tensor([[0.0, 0.0], [10.0, 0.0]]),
    )

    def half(*, episode_done: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.full(episode_done.shape, 0.5, dtype=torch.float32)

    # Q(s0, a=1)=2; r=1; V=10; γ=0.5 → target 6; loss (2-6)^2 = 16.
    loss, _ = DqnObjective(
        head=_Q, reward=_rew(), value=_val(), discount=half, grouping_field=None, temperature=0.0
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 16.0) < 1e-05


def test_dqn_discount_must_return_per_step_tensor() -> None:
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "episode_done": torch.tensor([0, 0]),
            "task_done": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))

    def bad(*, episode_done: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.5])

    with pytest.raises(ValueError, match="discount must return shape"):
        DqnObjective(head=_Q, reward=_rew(), value=_val(), discount=bad, grouping_field=None, temperature=0.0)(
            objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
        )


def test_dqn_reward_must_return_per_step_tensor() -> None:
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "episode_done": torch.tensor([0, 0]),
            "task_done": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))

    def bad(*, reward: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.5])

    with pytest.raises(ValueError, match="reward must return shape"):
        DqnObjective(head=_Q, reward=bad, value=_val(), discount=_disc(), grouping_field=None, temperature=0.0)(
            objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
        )


def test_dqn_value_must_return_same_shape() -> None:
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "episode_done": torch.tensor([0, 0]),
            "task_done": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))

    def bad(*, value: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return value[:, 0]

    with pytest.raises(ValueError, match="value must return shape"):
        DqnObjective(head=_Q, reward=_rew(), value=bad, discount=_disc(), grouping_field=None, temperature=0.0)(
            objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
        )
