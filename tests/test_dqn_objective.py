"""Tests for DQN objective on synthetic tensors."""
from __future__ import annotations
import math
import torch

from mouse_core.objectives import DqnObjective, affine_reward, affine_value, boundary_discount, boundary_reward, boundary_value, value_gap_gate, general_gate, lambda_gate, nstep_gate, watkins_gate
import pytest
from mouse_core.objectives.dqn import (
    _affine_scan_backward,
    _continuation_targets,
    _pair_weight,
    _weighted_mean,
)


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    kwargs.update(overrides)
    return boundary_discount(**kwargs)


def _rew(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_reward(**kwargs)  # type: ignore[arg-type]


def _val(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_value(**kwargs)  # type: ignore[arg-type]




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


def test_objective_none_transforms_are_identity() -> None:
    """``None`` skips the callable: reward column, raw Q, discount ``1``."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(
        torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]),
        torch.zeros(3, 2),
    )
    skipped, _ = DqnObjective(
        reward=None, value=None, discount=None,
        grouping_field=None, temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    explicit, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(gamma_step=1.0, gamma_episode_terminal=1.0, gamma_episode_truncated=1.0, gamma_task_terminal=1.0, gamma_task_truncated=1.0),
        grouping_field=None, temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(skipped.item() - explicit.item()) < 1e-05


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
        DqnObjective(reward=_rew(), value=_val(), # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            grouping_field=None,
            double=False, gate=None,
        )


def test_dqn_requires_double_argument() -> None:
    with pytest.raises(TypeError, match="double"):
        DqnObjective(reward=_rew(), value=_val(), # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            grouping_field=None,
            temperature=0.0,
            gate=None,
        )


def _gate(*, td_lambda: float):
    return lambda_gate(td_lambda=td_lambda)


def test_lambda_gate_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        lambda_gate(td_lambda=1.5)
    with pytest.raises(ValueError, match="int >= 1"):
        nstep_gate(n=0)


def test_dqn_requires_gate() -> None:
    with pytest.raises(TypeError, match="gate"):
        DqnObjective(reward=_rew(), value=_val(),  # type: ignore[call-arg]
            discount=_disc(),
            grouping_field=None,
            temperature=0.0, double=False,
        )


def test_dqn_requires_grouping_field_argument() -> None:
    with pytest.raises(TypeError, match="grouping_field"):
        DqnObjective(reward=_rew(), value=_val(), # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            temperature=0.0, double=False, gate=None,
        )


def _q(online: torch.Tensor, delayed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return online, delayed


def test_dqn_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = {'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    objective = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.99), grouping_field=None, temperature=0.0, double=False, gate=None)
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
        DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.99), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)

def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = {'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'episode_done': torch.zeros(1, dtype=torch.long), 'task_done': torch.zeros(1, dtype=torch.long)}
    predictions, delayed = _q(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Not enough"):
        DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)

def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 1, 0]), 'task_done': torch.tensor([0, 0, 0])}
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 2.5) < 1e-05


def test_dqn_requires_reward_argument() -> None:
    with pytest.raises(TypeError, match="reward"):
        DqnObjective(  # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            value=_val(),
            grouping_field=None,
            temperature=0.0, double=False, gate=None,
        )


def test_dqn_requires_value_argument() -> None:
    with pytest.raises(TypeError, match="value"):
        DqnObjective(  # type: ignore[call-arg]
            reward=_rew(),
            discount=_disc(gamma_step=0.99),
            grouping_field=None,
            temperature=0.0, double=False, gate=None,
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
    scaled, _ = DqnObjective(reward=_rew(scale=2.0), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(scaled.item() - 24.5) < 1e-05
    # Shift 1: targets 2 and 6 → (2-2)^2, (3-6)^2.
    shifted, _ = DqnObjective(reward=_rew(shift=1.0), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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
    loss, _ = DqnObjective(reward=reward, value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 4.5) < 1e-05


def test_dqn_objective_q_affine_is_identity_by_default() -> None:
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    plain, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0, double=False, gate=None)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    affine, _ = DqnObjective(reward=_rew(), value=_val(scale=1.0, shift=0.0), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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
    scaled, _ = DqnObjective(reward=_rew(), value=_val(scale=2.0), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(scaled.item() - 5.0) < 1e-05
    # Shift 1: 3 and 4 → (3-1)^2, (4-5)^2.
    shifted, _ = DqnObjective(reward=_rew(), value=_val(shift=1.0), discount=_disc(gamma_step=0.0),
        grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(shifted.item() - 2.5) < 1e-05
    assert torch.equal(predictions, online)


def _sequence_fixture(sequence_id: list[int]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor(sequence_id)}
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    return (step_stream, predictions, delayed)

def test_dqn_objective_skips_transitions_across_sequences() -> None:
    """A pair whose steps belong to different sequences is not a transition."""
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 1])
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 4.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 3.0) < 1e-05

def test_dqn_objective_without_sequence_breaks_trains_all_pairs() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 0, 0])
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 2.5) < 1e-05

def test_dqn_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 2])
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field="grouping_id", temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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
        DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)


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
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), grouping_field='task_index', temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.5, gamma_episode_terminal=1.0, gamma_task_truncated=0.4),
        grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - 1.0) < 1e-05
    # gamma_step=0 zeros the bootstrap even when both extras are 1.
    zero_step, _ = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.0, gamma_episode_terminal=1.0, gamma_task_truncated=1.0),
        grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(zero_step.item() - 1.0) < 1e-05


def _double_step() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """One transition where online and delayed argmax disagree.

    Taken action from s0 is 0 with Q = 2. Delayed Q at s1 is ``[9, 1]``
    (max 9 at action 0). Online Q at s1 is ``[0, 5]`` (argmax 1).
    """
    step_stream = {
        "action": torch.tensor([0, 0]),
        "reward": torch.tensor([0.0, 0.0]),
        "episode_done": torch.zeros(2, dtype=torch.int64),
        "task_done": torch.zeros(2, dtype=torch.int64),
    }
    online = torch.tensor([[2.0, 0.0], [0.0, 5.0]])
    delayed = torch.tensor([[0.0, 0.0], [9.0, 1.0]])
    return step_stream, online, delayed


def test_double_reads_delayed_q_at_online_argmax() -> None:
    """Double DQN target is 1, not the delayed max 9. Loss is (2 - 1)^2."""
    step_stream, online, delayed = _double_step()
    predictions, delayed_q = _q(online, delayed)
    vanilla, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    doubled, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    assert abs(vanilla.item() - 49.0) < 1e-05
    assert abs(doubled.item() - 1.0) < 1e-05


def test_double_matches_max_when_online_argmax_agrees() -> None:
    step_stream, online, delayed = _double_step()
    online = online.clone()
    online[1] = torch.tensor([5.0, 0.0])
    predictions, delayed_q = _q(online, delayed)
    vanilla, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    doubled, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    assert abs(vanilla.item() - doubled.item()) < 1e-05
    assert abs(doubled.item() - 49.0) < 1e-05


def test_double_tie_takes_lowest_index() -> None:
    """Tied online scores take action 0, the same rule as get_action."""
    step_stream, online, delayed = _double_step()
    online = online.clone()
    online[0] = torch.tensor([0.0, 0.0])
    online[1] = torch.tensor([3.0, 3.0])
    delayed = delayed.clone()
    delayed[1] = torch.tensor([4.0, 8.0])
    predictions, delayed_q = _q(online, delayed)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    assert abs(loss.item() - 16.0) < 1e-05


def test_double_does_not_backprop_through_selector_or_delayed_q() -> None:
    step_stream, _, _ = _double_step()
    online = torch.tensor([[2.0, 0.0], [0.0, 5.0]], requires_grad=True)
    delayed = torch.tensor([[0.0, 0.0], [9.0, 1.0]], requires_grad=True)
    predictions, delayed_q = _q(online, delayed)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None
    assert abs(online.grad[0, 0].item() - 2.0) < 1e-05
    assert torch.equal(online.grad[1], torch.zeros(2))


def test_double_soft_value_uses_online_policy_on_delayed_q() -> None:
    """π from online Q; V = E_π[Q_delayed] + α H[π]. Equal Q matches logsumexp."""
    step_stream = {
        "action": torch.tensor([0, 0]),
        "reward": torch.tensor([0.0, 0.0]),
        "episode_done": torch.zeros(2, dtype=torch.int64),
        "task_done": torch.zeros(2, dtype=torch.int64),
    }
    online = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    predictions, delayed_q = _q(online, delayed)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=1.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    value = 1.0 + float(torch.log(torch.tensor(2.0)))
    assert abs(loss.item() - value ** 2) < 1e-05

    same = torch.zeros(2, 2)
    predictions, delayed_q = _q(same, same.clone())
    vanilla, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=1.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    doubled, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=1.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    assert abs(vanilla.item() - doubled.item()) < 1e-05


def test_double_enters_the_lambda_return() -> None:
    """λ = 0.5 mixes V(s1) from the online argmax into the return."""
    step_stream = {
        "action": torch.tensor([0, 0, 0]),
        "reward": torch.tensor([0.0, 1.0, 0.0]),
        "episode_done": torch.zeros(3, dtype=torch.int64),
        "task_done": torch.zeros(3, dtype=torch.int64),
    }
    online = torch.tensor([[5.0, 0.0], [0.0, 9.0], [0.0, 3.0]])
    delayed = torch.tensor([[0.0, 0.0], [4.0, 1.0], [0.0, 8.0]])
    predictions, delayed_q = _q(online, delayed)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=0.5),
        grouping_field=None, temperature=0.0, double=True,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    assert abs(loss.item() - (0.25 + 64.0) / 2) < 1e-04


def test_double_bootstrap_reads_the_last_head_output_row() -> None:
    step_stream, online, delayed = _double_step()
    step_stream = {**step_stream, "head_output_count": torch.tensor([1, 2])}
    online = torch.tensor([[1.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    delayed = torch.tensor([[0.0, 0.0], [9.0, 0.0], [9.0, 1.0]])
    predictions, delayed_q = _q(online, delayed)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    assert abs(loss.item() - 0.0) < 1e-05


def test_double_uses_affine_q() -> None:
    """The online argmax and the delayed read both see ``value``."""
    step_stream, online, delayed = _double_step()
    predictions, delayed_q = _q(online, delayed)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(scale=2.0), discount=_disc(), grouping_field=None,
        temperature=0.0, double=True, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_q)
    # Taken Q 4; delayed at online argmax 2; target 2; loss 4.
    assert abs(loss.item() - 4.0) < 1e-05


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
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_td)
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
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - _ONE_STEP) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_dqn_objective_q_affine_applies_to_online_and_delayed() -> None:
    """Same affine on online Q and delayed bootstrap (γ=1 one-step)."""
    step_stream, predictions, delayed = _lambda_fixture()
    # Taken Q 10 and 0; delayed max 6 and 200; targets 7 and 210.
    loss, _ = DqnObjective(reward=_rew(), value=_val(scale=2.0), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    expected = (9.0 + 44100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_td_lambda_one_is_the_full_n_step_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - _FULL_RETURN) < 1e-03


def test_td_lambda_half_mixes_bootstrap_and_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    # G_0 = 1 + (0.5 * 3 + 0.5 * 110) = 57.5 → (5 - 57.5)^2 = 2756.25; s1 stays 12100.
    expected = (2756.25 + 12100.0) / 2
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=0.5), grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - expected) < 1e-03


def test_td_lambda_terminal_gamma_zero_ends_the_trace() -> None:
    """s0 → s1 terminates the episode (code stored at s1): G_0 = r only."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 1, 0])
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0),
        grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # s0: (5 - 1)^2 = 16 — neither V(s1) nor the next episode's return; s1: 12100.
    assert abs(loss.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_td_lambda_truncation_gamma_carries_the_trace_discounted() -> None:
    """A non-zero truncation gamma bootstraps through the reset, so the trace does too."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 2, 0])
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_episode_truncated=0.5), gate=_gate(td_lambda=1.0),
        grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    # s0: 1 + 0.5 * G_1 = 1 + 0.5 * 110 = 56 → (5 - 56)^2 = 2601; s1: 12100.
    assert abs(loss.item() - (2601.0 + 12100.0) / 2) < 1e-03


def test_td_lambda_task_gamma_scales_the_trace() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["task_done"] = torch.tensor([0, 2, 0])
    full, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_task_truncated=1.0), gate=_gate(td_lambda=1.0),
        grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(full.item() - _FULL_RETURN) < 1e-03
    cut, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_task_truncated=0.0), gate=_gate(td_lambda=1.0),
        grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(cut.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_watkins_cuts_when_taken_action_is_not_online_greedy() -> None:
    """Online Q at s1 prefers 0; taken action is 1 → cut. Oracle would continue."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions = q
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=watkins_gate(), grouping_field=None, temperature=0.0, double=False)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    one_step, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04
    assert "watkins_greedy_frac" not in metrics


def test_gates_reject_a_short_batch() -> None:
    q = torch.zeros(1, 2)
    action = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="at least 2"):
        value_gap_gate(beta=1.0, normalize=True, eps=1.0)(q=q, action=action)
    with pytest.raises(ValueError, match="at least 2"):
        lambda_gate(td_lambda=1.0)(q=q, action=action)
    with pytest.raises(ValueError, match="at least 2"):
        watkins_gate()(q=q, action=action)
    with pytest.raises(ValueError, match="at least 2"):
        nstep_gate(n=2)(q=q, action=action)
    with pytest.raises(ValueError, match="at least 2"):
        general_gate(gates=(lambda_gate(td_lambda=1.0),))(q=q, action=action)
    with pytest.raises(ValueError, match="action must have shape"):
        value_gap_gate(beta=1.0, normalize=True, eps=1.0)(q=torch.zeros(2, 2), action=torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="at least 2"):
        _pair_weight(
            {"action": action}, 1, "cpu", grouping_field=None
        )


def test_gate_rejects_a_vector() -> None:
    step_stream, predictions, delayed = _lambda_fixture()

    def gate(*, q: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)

    with pytest.raises(ValueError, match=r"\[3, 3\]"):
        DqnObjective(
            reward=_rew(), value=_val(), discount=_disc(), gate=gate,
            grouping_field=None, temperature=0.0, double=False,
        )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    for beta in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="beta"):
            value_gap_gate(beta=beta, normalize=True, eps=1.0)
    for beta in (True, "1"):
        with pytest.raises(TypeError, match="beta"):
            value_gap_gate(beta=beta, normalize=True, eps=1.0)  # type: ignore[arg-type]
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eps"):
            value_gap_gate(beta=1.0, normalize=True, eps=eps)
    for eps in (True, "1"):
        with pytest.raises(TypeError, match="eps"):
            value_gap_gate(beta=1.0, normalize=True, eps=eps)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="normalize"):
        value_gap_gate(beta=1.0, normalize=1, eps=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eps is required"):
        value_gap_gate(beta=1.0, normalize=True)
    with pytest.raises(ValueError, match="only when normalize"):
        value_gap_gate(beta=1.0, normalize=False, eps=1.0)


def test_general_gate_multiplies_continuation_matrices() -> None:
    """The product continues only where every gate continues."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    lam = lambda_gate(td_lambda=0.5)(q=q, action=action)
    horizon = nstep_gate(n=2)(q=q, action=action)
    greedy = watkins_gate()(q=q, action=action)
    gap = value_gap_gate(beta=0.5, normalize=True, eps=1.0)(q=q, action=action)
    got = general_gate(gates=(
        lambda_gate(td_lambda=0.5),
        nstep_gate(n=2),
        watkins_gate(),
        value_gap_gate(beta=0.5, normalize=True, eps=1.0),
    ))(q=q, action=action)
    assert torch.equal(got, lam * horizon * greedy * gap)
    one = general_gate(gates=(lambda_gate(td_lambda=0.5),))(q=q, action=action)
    assert torch.equal(one, lam)


def test_general_gate_matches_the_truncated_lambda_return() -> None:
    """λ times an n-step horizon is the truncated λ-return."""
    torch.manual_seed(0)
    T = 40
    N = T + 1
    reward = torch.randn(N)
    discount = torch.rand(N)
    v = torch.randn(N)
    pair_weight = torch.randint(0, 2, (T,)).to(dtype=torch.float32)
    greedy = torch.randint(0, 2, (T,)).to(dtype=torch.float32)
    q, action = _q_for_greedy(greedy=greedy)
    lam = 0.4
    horizon = 6
    in_run = pair_weight > 0
    run_mask = torch.cat([in_run[1:].to(dtype=torch.float32), in_run.new_zeros(1)])
    greedy_cont = run_mask * torch.cat([greedy[1:], greedy.new_zeros(1)])
    got = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=general_gate(gates=(
            lambda_gate(td_lambda=lam), nstep_gate(n=horizon),
        ))(q=q, action=action),
    )
    ref = _reference_lambda(
        r=reward[1:], g=discount[1:], v_next=v[1:], cont=run_mask,
        td_lambda=lam, n=horizon,
    )
    ref = ref * in_run.to(dtype=ref.dtype)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    watkins = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=general_gate(gates=(
            lambda_gate(td_lambda=lam), watkins_gate(),
        ))(q=q, action=action),
    )
    watkins_ref = _reference_lambda(
        r=reward[1:], g=discount[1:], v_next=v[1:], cont=greedy_cont,
        td_lambda=lam, n=T,
    )
    watkins_ref = watkins_ref * in_run.to(dtype=watkins_ref.dtype)
    assert torch.allclose(watkins, watkins_ref, atol=1e-5, rtol=1e-5)


def test_general_gate_rejects_a_bad_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        general_gate(gates=())
    with pytest.raises(TypeError, match="sequence"):
        general_gate(gates=lambda_gate(td_lambda=1.0))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        general_gate(gates=(lambda_gate(td_lambda=1.0), 1))  # type: ignore[list-item]

    def vector(*, q: torch.Tensor, action: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.zeros(q.shape[0])

    q = torch.zeros(3, 2)
    action = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\[3, 3\]"):
        general_gate(gates=(vector, lambda_gate(td_lambda=1.0)))(q=q, action=action)


def test_value_gap_gate_stores_exp_neg_beta_gap_at_the_taken_step() -> None:
    """Column ``s`` is the normalized gap; the diagonal and the ends are 0."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    eps = 1.0
    c = value_gap_gate(beta=beta, normalize=True, eps=eps)(q=q, action=action)
    # Action from s1 is stored at index 2 and is 1: Q = [5, 1].
    # (max - Q) / (max - min + eps) = 4 / 5.
    # That continuation is column 1, for the return that started at 0.
    assert c.shape == (3, 3)
    assert float(c[0, 0]) == 0.0
    assert float(c[0, 2]) == 0.0
    assert float(c[1, 2]) == 0.0
    assert torch.equal(torch.tril(c), torch.zeros(3, 3))
    assert abs(float(c[0, 1]) - math.exp(-beta * 4.0 / 5.0)) < 1e-6


def test_value_gap_gate_normalize_false_uses_the_raw_gap() -> None:
    """``normalize=False`` skips the range and uses ``exp(-beta * (max - Q))``."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    c = value_gap_gate(beta=beta, normalize=False)(q=q, action=action)
    assert abs(float(c[0, 1]) - math.exp(-beta * 4.0)) < 1e-6


def test_value_gap_gate_normalize_false_soft_cuts_by_the_raw_gap() -> None:
    """Taken action at s1 is 2 below the online max; c = exp(-beta * 2)."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([2.0, 0.0])
    predictions = q
    beta = 0.5
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=beta, normalize=False),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    c = math.exp(-beta * 2.0)
    g0 = 1.0 + (1.0 - c) * 3.0 + c * 110.0
    expected = ((5.0 - g0) ** 2 + 12100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_value_gap_gate_normalizes_a_mid_range_action() -> None:
    """A mid-range action uses (max - Q) / (max - min + eps)."""
    q = torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 10.0], [0.0, 0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 2.0
    eps = 1.0
    c = value_gap_gate(beta=beta, normalize=True, eps=eps)(q=q, action=action)
    assert abs(float(c[0, 1]) - math.exp(-beta * 7.0 / 11.0)) < 1e-6


def test_value_gap_gate_flat_q_is_a_full_continuation() -> None:
    """max = min leaves a zero numerator, so c = 1."""
    q = torch.zeros(3, 4)
    action = torch.tensor([1, 2, 3])
    c = value_gap_gate(beta=4.0, normalize=True, eps=1e-3)(q=q, action=action)
    assert abs(float(c[0, 1]) - 1.0) < 1e-6


def test_value_gap_gate_zero_gap_is_the_full_return() -> None:
    """Tied online Q at s1 has gap 0, so the trace continues with c = 1."""
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=1.0, normalize=True, eps=1.0),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - _FULL_RETURN) < 1e-03


def test_value_gap_gate_soft_cuts_by_the_online_optimality_gap() -> None:
    """Taken action at s1 is the online min; range is 2, eps is 1."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([2.0, 0.0])
    predictions = q
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=beta, normalize=True, eps=eps),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    c = math.exp(-beta * 2.0 / 3.0)
    g0 = 1.0 + (1.0 - c) * 3.0 + c * 110.0
    expected = ((5.0 - g0) ** 2 + 12100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_value_gap_gate_uses_q_after_value() -> None:
    """The gap is on the same affine Q the TD error trains."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([2.0, 0.0])
    predictions = q
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(scale=2.0), discount=_disc(), gate=value_gap_gate(beta=beta, normalize=True, eps=eps),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    c = math.exp(-beta * 4.0 / 5.0)
    g0 = 1.0 + (1.0 - c) * 6.0 + c * 210.0
    expected = ((10.0 - g0) ** 2 + (210.0 ** 2)) / 2
    assert abs(loss.item() - expected) < 1e-02


def test_value_gap_gate_ignores_the_gap_of_the_action_being_trained() -> None:
    """The return at s0 continues on the action taken from s1, not from s0."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[0] = torch.tensor([5.0, 100.0])
    predictions = q
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=10.0, normalize=True, eps=1.0),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - _FULL_RETURN) < 1e-03


def test_value_gap_gate_large_beta_matches_a_hard_cut() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions = q
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=50.0, normalize=True, eps=1e-6),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    one_step, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04


def test_gap_does_not_backprop_into_the_online_max() -> None:
    """The gap is a constant: only the trained Q(s, a_taken) gets a gradient."""
    step_stream, _, delayed = _lambda_fixture()
    online = torch.tensor([[5.0, 0.0], [2.0, 0.0], [0.0, 0.0]], requires_grad=True)
    loss, _ = DqnObjective(
        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=1.0, normalize=True, eps=1.0),
        grouping_field=None, temperature=0.0, double=False,
    )(objective_data=step_stream, predictions=online, delayed_predictions=delayed)
    loss.backward()
    assert online.grad is not None
    assert online.grad[1, 0].item() == 0.0
    assert online.grad[0, 0].item() != 0.0


def test_watkins_continues_when_taken_action_matches_online_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([-1.0, 0.0])  # greedy at s1 is the taken action; Q(s1,a=1) stays 0
    predictions = q
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=watkins_gate(), grouping_field=None, temperature=0.0, double=False)(
        objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
    )
    assert abs(loss.item() - _FULL_RETURN) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_lambda_returns_do_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["sequence_id"] = torch.tensor([0, 0, 1])
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    one_step, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item() - one_step.item()) < 1e-04


def test_td_lambda_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's λ-return; bootstrap reads the last row."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["head_output_count"] = torch.tensor([2, 1, 2])
    online = torch.tensor([[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]])
    predictions, delayed = _q(online, delayed_q)
    loss, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), grouping_field=None, temperature=0.0, double=False)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
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


def _reference_lambda(
    *,
    r: torch.Tensor,
    g: torch.Tensor,
    v_next: torch.Tensor,
    cont: torch.Tensor,
    td_lambda: float,
    n: int,
) -> torch.Tensor:
    """Per-start truncation of the λ-return. ``n`` may exceed ``T``."""
    T = int(r.shape[0])
    out = torch.zeros(T)
    for t in range(T):
        scale = 1.0
        acc = 0.0
        for k in range(n):
            p = t + k
            if p >= T:
                break
            c = 0.0 if k == n - 1 else td_lambda * float(cont[p])
            acc += scale * (float(r[p]) + float(g[p]) * (1.0 - c) * float(v_next[p]))
            scale *= float(g[p]) * c
        out[t] = acc
    return out


def _q_for_greedy(*, greedy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Online Q and actions whose greedy flags match ``greedy`` ``[T]``.

    ``greedy[i]`` is whether the action taken from step ``i`` is an argmax.
    The action from step ``i`` is stored at index ``i + 1`` and is always 0.
    """
    T = int(greedy.shape[0])
    N = T + 1
    q = torch.zeros(N, 2)
    action = torch.zeros(N, dtype=torch.long)
    for i in range(T):
        if float(greedy[i]) > 0.0:
            q[i, 0] = 1.0
        else:
            q[i, 1] = 1.0
    return q, action


def test_nstep_and_lambda_match_per_start_recursion() -> None:
    """n-step matches a horizon of 1s. λ matches the same recursion run to the end."""
    torch.manual_seed(0)
    T = 40
    N = T + 1
    reward = torch.randn(N)
    discount = torch.rand(N)
    v = torch.randn(N)
    pair_weight = torch.randint(0, 2, (T,)).to(dtype=torch.float32)
    greedy = torch.randint(0, 2, (T,)).to(dtype=torch.float32)
    q, action = _q_for_greedy(greedy=greedy)
    lam = 0.4
    in_run = pair_weight > 0
    run_mask = torch.cat([in_run[1:].to(dtype=torch.float32), in_run.new_zeros(1)])
    greedy_cont = run_mask * torch.cat([greedy[1:], greedy.new_zeros(1)])
    r = reward[1:]
    g = discount[1:]
    v_next = v[1:]

    def returns(continuation: torch.Tensor) -> torch.Tensor:
        return _continuation_targets(
            reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
            continuation=continuation,
        )

    for n in (2, 6, 15):
        got = returns(nstep_gate(n=n)(q=torch.zeros(N, 1), action=torch.zeros(N, dtype=torch.long)))
        ref = _reference_lambda(r=r, g=g, v_next=v_next, cont=run_mask, td_lambda=1.0, n=n)
        ref = ref * in_run.to(dtype=ref.dtype)
        assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5), n
    lam_got = returns(lambda_gate(td_lambda=lam)(q=q, action=action))
    lam_ref = _reference_lambda(
        r=r, g=g, v_next=v_next, cont=run_mask, td_lambda=lam, n=T
    )
    lam_ref = lam_ref * in_run.to(dtype=lam_ref.dtype)
    assert torch.allclose(lam_got, lam_ref, atol=1e-5, rtol=1e-5)
    watkins_got = returns(watkins_gate()(q=q, action=action))
    watkins_ref = _reference_lambda(
        r=r, g=g, v_next=v_next, cont=greedy_cont, td_lambda=1.0, n=T
    )
    watkins_ref = watkins_ref * in_run.to(dtype=watkins_ref.dtype)
    assert torch.allclose(watkins_got, watkins_ref, atol=1e-5, rtol=1e-5)
    past = returns(nstep_gate(n=T + 5)(q=torch.zeros(N, 1), action=torch.zeros(N, dtype=torch.long)))
    full = returns(lambda_gate(td_lambda=1.0)(q=torch.zeros(N, 1), action=torch.zeros(N, dtype=torch.long)))
    assert torch.allclose(past, full, atol=1e-5, rtol=1e-5)
    ones = torch.ones(T)
    open_q = torch.zeros(N, 1)
    open_action = torch.zeros(N, dtype=torch.long)
    short = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=ones,
        continuation=nstep_gate(n=3)(q=open_q, action=open_action),
    )
    opened = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=ones,
        continuation=lambda_gate(td_lambda=1.0)(q=open_q, action=open_action),
    )
    assert not torch.allclose(short, opened)


def test_dqn_objective_rejects_non_fp32_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        DqnObjective(reward=_rew(), value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)


def _entropy_kw() -> dict[str, object]:
    return dict(
        discount=_disc(),
        grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )


def test_temperature_zero_matches_hard_max() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    hard, hard_m = DqnObjective(reward=_rew(), value=_val(), **_entropy_kw())(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    soft0, soft0_m = DqnObjective(reward=_rew(), value=_val(), **_entropy_kw())(
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
    loss, metrics = DqnObjective(reward=_rew(), value=_val(), **{**_entropy_kw(), 'temperature': alpha})(
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
        DqnObjective(reward=_rew(), value=_val(), **{**_entropy_kw(), 'temperature': -0.1})


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
    hard, _ = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None,
        temperature=0.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    soft, metrics = DqnObjective(reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0),
        grouping_field=None,
        temperature=2.0, double=False, gate=None,
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(hard.item() - soft.item()) < 1e-05
    assert "entropy" in metrics


def test_dqn_requires_discount_argument() -> None:
    with pytest.raises(TypeError, match="discount"):
        DqnObjective(reward=_rew(), value=_val(), grouping_field=None, temperature=0.0, double=False, gate=None)  # type: ignore[call-arg]


def test_dqn_requires_reward_argument() -> None:
    with pytest.raises(TypeError, match="reward"):
        DqnObjective(value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)  # type: ignore[call-arg]


def test_dqn_requires_value_argument() -> None:
    with pytest.raises(TypeError, match="value"):
        DqnObjective(reward=_rew(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)  # type: ignore[call-arg]


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
        reward=_rew(), value=_val(), discount=half, grouping_field=None, temperature=0.0, double=False, gate=None
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
        DqnObjective(reward=_rew(), value=_val(), discount=bad, grouping_field=None, temperature=0.0, double=False, gate=None)(
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
        DqnObjective(reward=bad, value=_val(), discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(
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
        DqnObjective(reward=_rew(), value=bad, discount=_disc(), grouping_field=None, temperature=0.0, double=False, gate=None)(
            objective_data=step_stream, predictions=predictions, delayed_predictions=delayed
        )
