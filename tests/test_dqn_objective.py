"""Tests for DQN objective on synthetic tensors."""
from __future__ import annotations

from tests._prediction_layout import fixture_head_steps
from typing import TypedDict

import torch

from mouse_core.objectives import (
    Discount,
    DqnObjective,
    affine_reward,
    affine_value,
    boundary_discount,
    boundary_reward,
    boundary_value,
    value_gap_gate,
    general_gate,
    lambda_gate,
    nstep_gate,
)
import pytest
from mouse_core.objectives.dqn import (
    _affine_scan_backward,
    _continuation_targets,
    _pair_weight,
    _weighted_mean,
)


def _group_id(data: dict[str, torch.Tensor]) -> torch.Tensor:
    if "group_id" in data:
        return data["group_id"]
    n = int(next(iter(data.values())).shape[0])
    return torch.zeros(n, dtype=torch.int64)


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_terminated=0.0,
        gamma_truncated=0.0,
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
        gamma_terminated=0.4,
        gamma_truncated=0.2,
    )
    got = discount(
        terminated=torch.tensor([0, 1, 0, 1, 1]),
        truncated=torch.tensor([0, 0, 1, 1, 1]),
    )
    want = torch.tensor([
        0.5,
        0.5 * 0.4,
        0.5 * 0.2,
        0.5 * 0.4 * 0.2,
        0.5 * 0.4 * 0.2,
    ])
    assert torch.allclose(got, want)


def test_boundary_reward_applies_terminated_and_truncated_affine() -> None:
    reward = boundary_reward(
        scale=2.0,
        shift=1.0,
        reward_terminated_scale=3.0,
        reward_terminated_shift=10.0,
        reward_truncated_scale=0.5,
        reward_truncated_shift=4.0,
    )
    got = reward(
        reward=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        terminated=torch.tensor([0, 1, 0, 1]),
        truncated=torch.tensor([0, 0, 1, 1]),
    )
    want = torch.tensor([
        2.0 * 1.0 + 1.0,
        2.0 * 3.0 * 1.0 + 1.0 + 10.0,
        2.0 * 0.5 * 1.0 + 1.0 + 4.0,
        2.0 * 3.0 * 0.5 * 1.0 + 1.0 + 10.0 + 4.0,
    ])
    assert torch.allclose(got, want)


def test_boundary_value_applies_terminated_and_truncated_affine() -> None:
    value = boundary_value(
        scale=2.0,
        shift=1.0,
        value_terminated_scale=3.0,
        value_terminated_shift=10.0,
        value_truncated_scale=0.5,
        value_truncated_shift=4.0,
    )
    q = torch.ones(4, 2)
    got = value(
        value=q,
        terminated=torch.tensor([0, 1, 0, 1]),
        truncated=torch.tensor([0, 0, 1, 1]),
    )
    want_rows = torch.tensor([
        2.0 * 1.0 + 1.0,
        2.0 * 3.0 * 1.0 + 1.0 + 10.0,
        2.0 * 0.5 * 1.0 + 1.0 + 4.0,
        2.0 * 3.0 * 0.5 * 1.0 + 1.0 + 10.0 + 4.0,
    ])
    assert got.shape == q.shape
    assert torch.allclose(got, want_rows.unsqueeze(-1).expand_as(q))


def test_objective_none_transforms_are_identity() -> None:
    data, online, delayed = _lambda_fixture()
    outputs = []
    for reward, value in [(None, None), (_rew(), _val())]:
        losses, _ = DqnObjective(reward=reward, value=value, discount=_disc(), temperature=0., gate=None)(
            objective_data=data, group_id=_group_id(data), head_step_ids=fixture_head_steps(data),  predictions=online,
            backup_value_predictions=delayed, backup_action_predictions=delayed)
        outputs.append(losses['action_value'])
    torch.testing.assert_close(*outputs)
    # Identity discount stays one even on a terminal row: the unfinished return must fault.
    with pytest.raises(ValueError, match='cross-group backup'):
        DqnObjective(reward=None, value=None, discount=None, temperature=0., gate=None)(
            objective_data=data, group_id=_group_id(data), head_step_ids=fixture_head_steps(data),  predictions=online,
            backup_value_predictions=delayed, backup_action_predictions=delayed)


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
        DqnObjective( reward=_rew(), value=_val(), # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            gate=None)


def test_dqn_rejects_removed_double_argument() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'double'"):
        DqnObjective(discount=None, reward=None, value=None, gate=None,
                     temperature=0.0,  double=True)  # type: ignore[call-arg]


def _gate(*, td_lambda: float):
    return lambda_gate(td_lambda=td_lambda)


def test_lambda_gate_rejects_bad_lambda() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        lambda_gate(td_lambda=1.5)
    with pytest.raises(ValueError, match="int >= 1"):
        nstep_gate(n=0)


def test_dqn_requires_gate() -> None:
    with pytest.raises(TypeError, match="gate"):
        DqnObjective( reward=_rew(), value=_val(),  # type: ignore[call-arg]
            discount=_disc(),
            temperature=0.0)


def _q(online: torch.Tensor, delayed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return online, delayed


def test_dqn_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = {'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'terminated': torch.zeros(n, dtype=torch.long), 'truncated': torch.zeros(n, dtype=torch.long), 'group_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    step_stream["terminated"][-1] = True
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    step_stream["terminated"][3] = True
    objective = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.99), temperature=0.0, gate=None)
    loss, metrics = objective(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert loss["action_value"].ndim == 0
    assert 'action_value' in metrics
    assert isinstance(metrics['action_value'], torch.Tensor)
    assert metrics['action_value'].ndim == 0
    assert metrics['action_value'] >= 0.0
    assert isinstance(metrics['q_values_mean'], torch.Tensor)
    assert metrics['q_values_mean'].ndim == 0
    assert 'watkins_greedy_frac' not in metrics
    assert 'entropy' not in metrics
    backup = metrics['backup']
    backup_weight = metrics['backup_weight']
    assert isinstance(backup, torch.Tensor)
    assert isinstance(backup_weight, torch.Tensor)
    assert backup.shape == (n,)
    assert backup_weight.shape == (n,)
    assert not backup.requires_grad
    assert not backup_weight.requires_grad
    in_run_backup = metrics["in_run_backup"]
    assert isinstance(in_run_backup, torch.Tensor)
    assert torch.equal(in_run_backup, backup[backup_weight > 0])

def test_dqn_metrics_backup_matches_loss_target() -> None:
    data, online, delayed = _lambda_fixture()
    loss, metrics = DqnObjective(reward=None, value=None, discount=_disc(),
                                 temperature=0., gate=None)(
        objective_data=data, group_id=_group_id(data), head_step_ids=fixture_head_steps(data),  predictions=online,
        backup_value_predictions=delayed, backup_action_predictions=delayed)
    torch.testing.assert_close(metrics['backup'], torch.tensor([4., 10., 0.]))
    torch.testing.assert_close(metrics['backup_weight'], torch.tensor([1., 1., 0.]))
    torch.testing.assert_close(loss['action_value'], ((online[:2, [0, 1]].diagonal() - metrics['backup'][:2]) ** 2).mean())

def test_dqn_objective_rejects_wrong_action_shape() -> None:
    n, a = (4, 3)
    step_stream = {'action': torch.randint(0, a, (n, 1)), 'reward': torch.randn(n), 'terminated': torch.zeros(n, dtype=torch.long), 'truncated': torch.zeros(n, dtype=torch.long)}
    step_stream["terminated"][-1] = True
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    with pytest.raises(ValueError, match="action shape"):
        DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.99), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)

def test_dqn_objective_requires_min_sequence() -> None:
    step_stream = {'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'terminated': torch.zeros(1, dtype=torch.long), 'truncated': torch.zeros(1, dtype=torch.long)}
    predictions, delayed = _q(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Not enough"):
        DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)

def test_dqn_objective_trains_on_terminal_transitions() -> None:
    """Transitions *from* terminal states must contribute to the loss."""
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'terminated': torch.tensor([0, 1, 0]), 'truncated': torch.tensor([0, 0, 0])}
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - 2.5) < 1e-05


def test_dqn_requires_reward_argument() -> None:
    with pytest.raises(TypeError, match="reward"):
        DqnObjective(  # type: ignore[call-arg]
            discount=_disc(gamma_step=0.99),
            value=_val(),
            temperature=0.0, gate=None)


def test_dqn_requires_value_argument() -> None:
    with pytest.raises(TypeError, match="value"):
        DqnObjective(  # type: ignore[call-arg]
            reward=_rew(),
            discount=_disc(gamma_step=0.99),
            temperature=0.0, gate=None)


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
            "terminated": torch.tensor([0, 1, 0]),
            "truncated": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    # Unscaled targets 1 and 5 → MSE 2.5. Scale 2: targets 2 and 10 → (2-2)^2, (3-10)^2.
    scaled, _ = DqnObjective( reward=_rew(scale=2.0), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(scaled["action_value"].item() - 24.5) < 1e-05
    # Shift 1: targets 2 and 6 → (2-2)^2, (3-6)^2.
    shifted, _ = DqnObjective( reward=_rew(shift=1.0), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(shifted["action_value"].item() - 4.5) < 1e-05
    # Leaves objective_data reward unchanged.
    assert torch.equal(step_stream["reward"], torch.tensor([0.0, 1.0, 5.0]))


def test_dqn_objective_custom_reward_reads_objective_data() -> None:
    """A custom reward can combine columns; the raw ``reward`` column is unused."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "bonus": torch.tensor([0.0, 1.0, 1.0]),
            "terminated": torch.tensor([0, 1, 0]),
            "truncated": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))

    def reward(*, reward: torch.Tensor, bonus: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return reward + bonus

    # Targets 2 and 6 → (2-2)^2, (3-6)^2.
    loss, _ = DqnObjective( reward=reward, value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - 4.5) < 1e-05


def test_dqn_objective_q_affine_is_identity_by_default() -> None:
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "terminated": torch.tensor([0, 1, 0]),
            "truncated": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    plain, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(
        objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
    )
    affine, _ = DqnObjective( reward=_rew(), value=_val(scale=1.0, shift=0.0), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(plain["action_value"].item() - affine["action_value"].item()) < 1e-05


def test_dqn_objective_q_scale_and_shift() -> None:
    """gamma=0 so bootstrap is unused; affine applies to online Q only."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "terminated": torch.tensor([0, 1, 0]),
            "truncated": torch.tensor([0, 0, 0]),
        }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    predictions, delayed = _q(online, torch.zeros(3, 2))
    # Taken Q 2 and 3, targets 1 and 5. Scale 2: 4 and 6 → (4-1)^2, (6-5)^2.
    scaled, _ = DqnObjective( reward=_rew(), value=_val(scale=2.0), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(scaled["action_value"].item() - 5.0) < 1e-05
    # Shift 1: 3 and 4 → (3-1)^2, (4-5)^2.
    shifted, _ = DqnObjective( reward=_rew(), value=_val(shift=1.0), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(shifted["action_value"].item() - 2.5) < 1e-05
    assert torch.equal(predictions, online)


def _sequence_fixture(group_id: list[int]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    step_stream = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'terminated': torch.tensor([0, 0, 0]), 'truncated': torch.tensor([0, 0, 0]), 'group_id': torch.tensor(group_id)}
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    return (step_stream, predictions, delayed)

def test_dqn_objective_skips_transitions_across_sequences() -> None:
    """A pair whose steps belong to different sequences is not a transition."""
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 1])
    loss, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - 4.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 3.0) < 1e-05

def test_dqn_objective_without_sequence_breaks_trains_all_pairs() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 0, 0])
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - 2.5) < 1e-05

def test_dqn_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream, predictions, delayed = _sequence_fixture([0, 1, 2])
    loss, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item()) < 1e-05
    assert abs(metrics['q_values_mean']) < 1e-05

def test_dqn_objective_skips_transitions_across_episodes() -> None:
    """A pair whose steps belong to different samples is not a transition."""
    step_stream = {
            'action': torch.tensor([0, 1, 0]),
            'reward': torch.tensor([0.0, 1.0, 5.0]),
            'terminated': torch.tensor([0, 1, 0]),
            'truncated': torch.tensor([0, 1, 0]),
            'group_id': torch.tensor([0, 0, 1]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    # Only pair (0,1) is valid (same sample); pair (1,2) crosses group_id.
    loss, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - 1.0) < 1e-05
    assert abs(metrics['q_values_mean'] - 2.0) < 1e-05


def test_pair_weight_zeros_window_cut_and_episode_change() -> None:
    """Last output of a sample has weight 0; an in-sample reset stays 1."""
    window = {
            'group_id': torch.tensor([0, 0, 1, 1]),
        }
    assert torch.equal(
        _pair_weight(group_id=window["group_id"], N=4, device="cpu"),
        torch.tensor([1.0, 0.0, 1.0]),
    )

    episode_change = {
            'group_id': torch.tensor([0, 0, 1]),
        }
    assert torch.equal(
        _pair_weight(group_id=episode_change["group_id"], N=3, device="cpu"),
        torch.tensor([1.0, 0.0]),
    )

    reset = {
            'group_id': torch.tensor([0, 0, 0]),
            'episode_index': torch.tensor([0, 0, 0]),
            'terminated': torch.tensor([0, 1, 0]),
        }
    assert torch.equal(
        _pair_weight(group_id=reset["group_id"], N=3, device="cpu"),
        torch.tensor([1.0, 1.0]),
    )


def test_dqn_objective_rejects_out_of_range_action() -> None:
    step_stream = {
            'action': torch.tensor([0, 9]),
            'reward': torch.tensor([0.0, 1.0]),
            'terminated': torch.zeros(2, dtype=torch.long),
            'truncated': torch.zeros(2, dtype=torch.long),
        }
    predictions, delayed = _q(torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="action ids must be in"):
        DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)


def test_weighted_mean_all_zero_is_zero() -> None:
    assert abs(_weighted_mean(torch.tensor([3.0, 4.0]), torch.zeros(2)).item()) < 1e-05


def test_dqn_same_run_episode_reset_trains_both_pairs() -> None:
    """live→terminal→reset with the same group_id and episode_index still trains."""
    step_stream = {
            'action': torch.tensor([0, 1, 0]),
            'reward': torch.tensor([0.0, 1.0, 5.0]),
            'terminated': torch.tensor([0, 1, 0]),
            'truncated': torch.tensor([0, 0, 0]),
            'group_id': torch.tensor([0, 0, 0]),
            'episode_index': torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - 2.5) < 1e-05


def test_nonzero_ending_discount_requires_an_in_group_continuation() -> None:
    data, online, delayed = _lambda_fixture()
    data['truncated'][-1] = True
    with pytest.raises(ValueError, match='cross-group backup'):
        DqnObjective(reward=None, value=None, discount=_disc(
            gamma_step=.5, gamma_terminated=1., gamma_truncated=.4), temperature=0., gate=None)(
                objective_data=data, group_id=_group_id(data), head_step_ids=fixture_head_steps(data),  predictions=online,
                backup_value_predictions=delayed, backup_action_predictions=delayed)


def test_dqn_objective_does_not_backprop_through_delayed_q() -> None:
    """Bootstrap Q is a constant: delayed Q must not receive a gradient."""
    n, a = 4, 2
    step_stream = {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "terminated": torch.zeros(n, dtype=torch.long),
            "truncated": torch.zeros(n, dtype=torch.long),
        }
    step_stream["terminated"][-1] = True
    online = torch.randn(n, a, requires_grad=True)
    delayed = torch.randn(n, a, requires_grad=True)
    predictions, delayed_td = _q(online, delayed)
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed_td, backup_action_predictions=delayed_td)
    loss["action_value"].backward()
    assert online.grad is not None
    assert delayed.grad is None


def _lambda_fixture() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Three in-run steps so λ can mix a two-step backup from s0.

    Action from s0 is 0; from s1 is 1. Delayed max-Q is 3 at s1 and 100 at s2.
    Rewards out of s0 / s1 are 1 and 10. One-step target at s0 is 4; the
    λ=1 (full n-step) target at s0 is 111.
    """
    step_stream = {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "terminated": torch.tensor([0, 0, 1]),
            "truncated": torch.zeros(3, dtype=torch.int64),
            "group_id": torch.zeros(3, dtype=torch.int64),
            "q_star": torch.tensor(
                [[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]]
            ),
        }
    # Online Q(s0, a=0) = 5; Q(s1, a=1) = 0.
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    return step_stream, *_q(online, delayed)


# One-step MSEs: (5-4)^2 = 1 and (0-110)^2 = 12100 → mean 50.5.
_ONE_STEP = 50.5
# λ=1 target at s0 is 1 + 10 + 100 = 111 → (5-111)^2 = 11236; s1 stays 12100 → mean 11668.
_FULL_RETURN = 68.0


def test_td_lambda_zero_is_the_one_step_target() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - _ONE_STEP) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_dqn_objective_q_affine_applies_to_online_and_delayed() -> None:
    """Same affine on online Q and delayed bootstrap (γ=1 one-step)."""
    step_stream, predictions, delayed = _lambda_fixture()
    # Taken Q 10 and 0; delayed max 6 and 200; targets 7 and 210.
    loss, _ = DqnObjective( reward=_rew(), value=_val(scale=2.0), discount=_disc(), temperature=0.0, gate=None)(
        objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
    )
    expected = (9.0 + 100.0) / 2
    assert abs(loss["action_value"].item() - expected) < 1e-03


def test_td_lambda_one_is_the_full_n_step_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - _FULL_RETURN) < 1e-03


def test_td_lambda_half_mixes_bootstrap_and_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    # G_0 = 1 + (0.5 * 3 + 0.5 * 110) = 57.5 → (5 - 57.5)^2 = 6.25; s1 stays 12100.
    expected = (6.25 + 100.0) / 2
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=0.5), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - expected) < 1e-03


def test_td_lambda_terminal_gamma_zero_ends_the_trace() -> None:
    """s0 → s1 terminates the episode (code stored at s1): G_0 = r only."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["terminated"] = torch.tensor([0, 1, 1])
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    # s0: (5 - 1)^2 = 16 — neither V(s1) nor the next episode's return; s1: 12100.
    assert abs(loss["action_value"].item() - (16.0 + 100.0) / 2) < 1e-03


def test_td_lambda_truncation_gamma_carries_the_trace_discounted() -> None:
    """A non-zero truncation gamma bootstraps through the reset, so the trace does too."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["truncated"] = torch.tensor([0, 1, 0])
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_truncated=0.5), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    # s0: 1 + 0.5 * G_1 = 1 + 0.5 * 110 = 56 → (5 - 56)^2 = 2601; s1: 12100.
    assert abs(loss["action_value"].item() - (1.0 + 100.0) / 2) < 1e-03


def test_td_lambda_episode_gamma_scales_the_trace() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["truncated"] = torch.tensor([0, 1, 0])
    full, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_truncated=1.0), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(full["action_value"].item() - _FULL_RETURN) < 1e-03
    cut, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_truncated=0.0), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(cut["action_value"].item() - (16.0 + 100.0) / 2) < 1e-03


def test_value_gap_zero_beta_cuts_when_taken_action_is_not_online_greedy() -> None:
    """Online Q at s1 prefers 0; taken action is 1 → cut. Oracle would continue."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions = q
    loss, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=0.0, normalize=False, policy_delayed=False, value_delayed=False, bias=0.0), temperature=0.0)(
        objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
    )
    one_step, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - one_step["action_value"].item()) < 1e-04
    assert "watkins_greedy_frac" not in metrics


def test_gates_reject_a_short_batch() -> None:
    q = torch.zeros(1, 2)
    action = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="at least 2"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0)(q=q, action=action, q_delayed=q)
    with pytest.raises(ValueError, match="at least 2"):
        lambda_gate(td_lambda=1.0)(q=q, action=action)
    with pytest.raises(ValueError, match="at least 2"):
        value_gap_gate(beta=0.0, normalize=False, policy_delayed=False, value_delayed=False, bias=0.0)(q=q, action=action, q_delayed=q)
    with pytest.raises(ValueError, match="at least 2"):
        nstep_gate(n=2)(q=q, action=action)
    with pytest.raises(ValueError, match="at least 2"):
        general_gate(gates=(lambda_gate(td_lambda=1.0),))(q=q, action=action)
    with pytest.raises(ValueError, match="action must have shape"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0)(q=torch.zeros(2, 2), action=torch.zeros(1, dtype=torch.long), q_delayed=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="at least 2"):
        _pair_weight(
            group_id=torch.zeros(1, dtype=torch.int64), N=1, device="cpu"
        )


def test_gate_rejects_a_vector() -> None:
    step_stream, predictions, delayed = _lambda_fixture()

    def gate(*, q: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)

    with pytest.raises(ValueError, match=r"\[3, 3\]"):
        DqnObjective(            reward=_rew(), value=_val(), discount=_disc(), gate=gate, temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    for beta in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="beta"):
            value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0)
    for beta in (True, "1"):
        with pytest.raises(TypeError, match="beta"):
            value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0)  # type: ignore[arg-type]
    for eps in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eps"):
            value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True, eps=eps)
    for eps in (True, "1"):
        with pytest.raises(TypeError, match="eps"):
            value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True, eps=eps)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="normalize"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=1, eps=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy_delayed"):
        value_gap_gate(bias=0.0, beta=1.0, value_delayed=False, normalize=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="value_delayed"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, normalize=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="policy_delayed"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=1, value_delayed=False, normalize=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="value_delayed"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=1, normalize=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eps is required"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True)
    with pytest.raises(ValueError, match="only when normalize"):
        value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=False, eps=1.0)
    with pytest.raises(TypeError, match="bias"):
        value_gap_gate(beta=1.0, policy_delayed=False, value_delayed=False, normalize=False)  # type: ignore[call-arg]
    for bias in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="bias"):
            value_gap_gate(beta=1.0, policy_delayed=False, value_delayed=False, normalize=False, bias=bias)
    for bias in (True, "0"):
        with pytest.raises(TypeError, match="bias"):
            value_gap_gate(beta=1.0, policy_delayed=False, value_delayed=False, normalize=False, bias=bias)  # type: ignore[arg-type]


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_general_gate_multiplies_continuation_matrices(beta: float) -> None:
    """The product continues only where every gate continues."""
    q = torch.tensor([[1.0, 4.0], [1.0, 5.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1, 1])
    lam = lambda_gate(td_lambda=0.5)(q=q, action=action)
    horizon = nstep_gate(n=2)(q=q, action=action)
    gap = value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0)(q=q, action=action, q_delayed=q)
    got = general_gate(gates=(
        lambda_gate(td_lambda=0.5),
        nstep_gate(n=2),
        value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0),
    ))(q=q, action=action, q_delayed=q)
    assert torch.equal(got, lam * horizon * gap)
    assert got[0, 1].item() == 0.5
    assert got[1, 2].item() == pytest.approx(0.0 if beta == 0 else 0.5 * 2 ** (-4 / 3))
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
    got, _ = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=general_gate(gates=(
            lambda_gate(td_lambda=lam), nstep_gate(n=horizon),
        ))(q=q, action=action))
    ref = _reference_lambda(
        r=reward[1:], g=discount[1:], v_next=v[1:] * run_mask, cont=run_mask,
        td_lambda=lam, n=horizon,
    )
    ref = ref * in_run.to(dtype=ref.dtype)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)
    hard_gap, _ = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=general_gate(gates=(
            lambda_gate(td_lambda=lam), value_gap_gate(beta=0.0, normalize=False, policy_delayed=False, value_delayed=False, bias=0.0),
        ))(q=q, action=action, q_delayed=q))
    hard_gap_ref = _reference_lambda(
        r=reward[1:], g=discount[1:], v_next=v[1:] * run_mask, cont=greedy_cont,
        td_lambda=lam, n=T,
    )
    hard_gap_ref = hard_gap_ref * in_run.to(dtype=hard_gap_ref.dtype)
    assert torch.allclose(hard_gap, hard_gap_ref, atol=1e-5, rtol=1e-5)


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


def test_value_gap_gate_stores_two_pow_gap_over_beta_at_the_taken_step() -> None:
    """Column ``s`` is ``2 ** (gap / beta)``; the diagonal and the ends are 0."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    eps = 1.0
    c = value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps)(q=q, action=action, q_delayed=q)
    # Action from s1 is stored at index 2 and is 1: Q = [5, 1].
    # (Q - max) / (|max| + eps) = -4 / 6.
    # That continuation is column 1, for the return that started at 0.
    assert c.shape == (3, 3)
    assert float(c[0, 0]) == 0.0
    assert float(c[0, 2]) == 0.0
    assert float(c[1, 2]) == 0.0
    assert torch.equal(torch.tril(c), torch.zeros(3, 3))
    assert abs(float(c[0, 1]) - 2.0 ** ((-4.0 / 6.0) / beta)) < 1e-6


def test_value_gap_gate_bias_shifts_then_caps_at_zero() -> None:
    """``bias`` is subtracted from the signed gap, and the result stays at or below 0."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    eps = 1.0
    shifted = value_gap_gate(
        bias=0.2, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps,
    )(q=q, action=action, q_delayed=q)
    assert abs(float(shifted[0, 1]) - 2.0 ** ((-4.0 / 6.0 - 0.2) / beta)) < 1e-6
    capped = value_gap_gate(
        bias=-2.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps,
    )(q=q, action=action, q_delayed=q)
    assert abs(float(capped[0, 1]) - 1.0) < 1e-6
    flat = torch.zeros(3, 2)
    cut = value_gap_gate(
        bias=0.5, beta=beta, policy_delayed=False, value_delayed=False, normalize=False,
    )(q=flat, action=torch.tensor([0, 1, 0]), q_delayed=flat)
    assert abs(float(cut[0, 1]) - 2.0 ** (-0.5 / beta)) < 1e-6


def test_value_gap_gate_mixes_policy_and_value_networks() -> None:
    """Each flag selects one net. A negative raw gap continues at ``c = 1``."""
    q = torch.tensor([[0.0, 0.0, 0.0], [6.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    q_delayed = torch.tensor([[0.0, 0.0, 0.0], [5.0, 1.0, 8.0], [0.0, 0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    eps = 1.0

    def column(*, policy_delayed: bool, value_delayed: bool) -> float:
        c = value_gap_gate(
            bias=0.0,
            beta=beta,
            normalize=True,
            policy_delayed=policy_delayed,
            value_delayed=value_delayed,
            eps=eps,
        )(q=q, action=action, q_delayed=q_delayed)
        return float(c[0, 1])

    # Online argmax is 0, delayed argmax is 2, taken action is 1.
    assert abs(column(policy_delayed=False, value_delayed=False) - 2.0 ** ((-4.0 / 7.0) / beta)) < 1e-6
    assert abs(column(policy_delayed=True, value_delayed=True) - 2.0 ** ((-7.0 / 9.0) / beta)) < 1e-6
    assert abs(column(policy_delayed=True, value_delayed=False) - 2.0 ** ((-2.0 / 7.0) / beta)) < 1e-6
    assert abs(column(policy_delayed=False, value_delayed=True) - 2.0 ** ((-4.0 / 9.0) / beta)) < 1e-6
    # Value net prefers the taken action over the policy action.
    above = torch.tensor([[0.0, 0.0], [1.0, 4.0], [0.0, 0.0]])
    c = value_gap_gate(
        bias=0.0, beta=beta, normalize=False, policy_delayed=False, value_delayed=True,
    )(q=q[:, :2], action=action, q_delayed=above)
    assert abs(float(c[0, 1]) - 1.0) < 1e-6


def test_value_gap_gate_delayed_reads_q_delayed() -> None:
    """Both flags read ``q_delayed``. Both clear keeps online Q."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    q_delayed = torch.tensor([[0.0, 0.0], [9.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    eps = 1.0
    online = value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps)(
        q=q, action=action, q_delayed=torch.zeros(3, 1),
    )
    delayed = value_gap_gate(bias=0.0, beta=beta, policy_delayed=True, value_delayed=True, normalize=True, eps=eps)(
        q=q, action=action, q_delayed=q_delayed,
    )
    forwarded = general_gate(gates=(
        value_gap_gate(bias=0.0, beta=beta, policy_delayed=True, value_delayed=True, normalize=True, eps=eps),
    ))(q=q, action=action, q_delayed=q_delayed)
    # Online at s1 is [5, 1], taken action 1: (1 - 5) / (|5| + 1) = -4 / 6.
    assert abs(float(online[0, 1]) - 2.0 ** ((-4.0 / 6.0) / beta)) < 1e-6
    # Delayed at s1 is [9, 1], taken action 1: (1 - 9) / (|9| + 1) = -8 / 10.
    assert abs(float(delayed[0, 1]) - 2.0 ** ((-8.0 / 10.0) / beta)) < 1e-6
    assert torch.equal(forwarded, delayed)


def test_value_gap_gate_delayed_soft_cuts_by_the_delayed_gap() -> None:
    """Online Q at s1 is flat. The cut follows delayed Q [3, 0], action 1."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions = q
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(),
        gate=value_gap_gate(bias=0.0, beta=beta, policy_delayed=True, value_delayed=True, normalize=True, eps=eps), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    c = 2.0 ** ((-3.0 / 4.0) / beta)
    g0 = 1.0 + (1.0 - c) * 3.0 + c * 10.0
    expected = ((5.0 - g0) ** 2 + 100.0) / 2
    assert abs(loss["action_value"].item() - expected) < 1e-03


def test_value_gap_gate_delayed_uses_q_after_value() -> None:
    """The delayed gap sees the same affine as the delayed bootstrap."""
    step_stream, predictions, delayed = _lambda_fixture()
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(        reward=_rew(), value=_val(scale=2.0), discount=_disc(),
        gate=value_gap_gate(bias=0.0, beta=beta, policy_delayed=True, value_delayed=True, normalize=True, eps=eps), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    c = 2.0 ** ((-6.0 / 7.0) / beta)
    g0 = 1.0 + (1.0 - c) * 6.0 + c * 10.0
    expected = ((10.0 - g0) ** 2 + (10.0 ** 2)) / 2
    assert abs(loss["action_value"].item() - expected) < 1e-02


def test_value_gap_gate_delayed_reads_the_last_head_output_row() -> None:
    """Two rows at s1: the gap uses the last delayed row, ``[3, 0]``."""
    step_stream, _, _ = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["readout_counts"] = torch.tensor([1, 2, 1])
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 10.0], [3.0, 0.0], [0.0, 100.0]])
    predictions, delayed = _q(online, delayed_q)
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(),
        gate=value_gap_gate(bias=0.0, beta=beta, policy_delayed=True, value_delayed=True, normalize=True, eps=eps), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    c = 2.0 ** ((-3.0 / 4.0) / beta)
    g0 = 1.0 + (1.0 - c) * 3.0 + c * 10.0
    # s0 row trains Q=5. Both s1 rows train Q(s1, a=1)=0 toward 110. s2 has weight 0.
    expected = ((5.0 - g0) ** 2 + 100.0 + 100.0) / 3
    assert abs(loss["action_value"].item() - expected) < 1e-03


def test_value_gap_gate_normalize_false_uses_the_raw_gap() -> None:
    """``normalize=False`` keeps the raw gap and continues at ``2 ** ((Q - max) / beta)``."""
    q = torch.tensor([[1.0, 4.0], [5.0, 1.0], [0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 0.5
    c = value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=False)(q=q, action=action, q_delayed=q)
    assert abs(float(c[0, 1]) - 2.0 ** (-4.0 / beta)) < 1e-6


def test_value_gap_gate_normalize_false_soft_cuts_by_the_raw_gap() -> None:
    """Taken action at s1 is 2 below the online max; c = 2 ** (-2 / beta)."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([2.0, 0.0])
    predictions = q
    beta = 0.5
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=False), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    c = 2.0 ** (-2.0 / beta)
    g0 = 1.0 + (1.0 - c) * 3.0 + c * 10.0
    expected = ((5.0 - g0) ** 2 + 100.0) / 2
    assert abs(loss["action_value"].item() - expected) < 1e-03


def test_value_gap_gate_normalizes_a_mid_range_action() -> None:
    """A mid-range action uses (Q - max) / (|max| + eps)."""
    q = torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 10.0], [0.0, 0.0, 0.0]])
    action = torch.tensor([0, 0, 1])
    beta = 2.0
    eps = 1.0
    c = value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps)(q=q, action=action, q_delayed=q)
    assert abs(float(c[0, 1]) - 2.0 ** ((-7.0 / 11.0) / beta)) < 1e-6


def test_value_gap_gate_flat_q_is_a_full_continuation() -> None:
    """A flat Q has signed gap 0, so c = 1."""
    q = torch.zeros(3, 4)
    action = torch.tensor([1, 2, 3])
    c = value_gap_gate(bias=0.0, beta=4.0, policy_delayed=False, value_delayed=False, normalize=True, eps=1e-3)(q=q, action=action, q_delayed=q)
    assert abs(float(c[0, 1]) - 1.0) < 1e-6


def test_value_gap_gate_zero_gap_is_the_full_return() -> None:
    """Tied online Q at s1 has gap 0, so the trace continues with c = 1."""
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - _FULL_RETURN) < 1e-03


def test_value_gap_gate_soft_cuts_by_the_online_optimality_gap() -> None:
    """Taken action at s1 is 2 below the online max; |max| is 2 and eps is 1."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([2.0, 0.0])
    predictions = q
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    c = 2.0 ** ((-2.0 / 3.0) / beta)
    g0 = 1.0 + (1.0 - c) * 3.0 + c * 10.0
    expected = ((5.0 - g0) ** 2 + 100.0) / 2
    assert abs(loss["action_value"].item() - expected) < 1e-03


def test_value_gap_gate_uses_q_after_value() -> None:
    """The gap is on the same affine Q the TD error trains."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([2.0, 0.0])
    predictions = q
    beta = 0.5
    eps = 1.0
    loss, _ = DqnObjective(        reward=_rew(), value=_val(scale=2.0), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=beta, policy_delayed=False, value_delayed=False, normalize=True, eps=eps), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    c = 2.0 ** ((-4.0 / 5.0) / beta)
    g0 = 1.0 + (1.0 - c) * 6.0 + c * 10.0
    expected = ((10.0 - g0) ** 2 + (10.0 ** 2)) / 2
    assert abs(loss["action_value"].item() - expected) < 1e-02


def test_value_gap_gate_ignores_the_gap_of_the_action_being_trained() -> None:
    """The return at s0 continues on the action taken from s1, not from s0."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[0] = torch.tensor([5.0, 100.0])
    predictions = q
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=10.0, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - _FULL_RETURN) < 1e-03


def test_value_gap_gate_small_beta_matches_a_hard_cut() -> None:
    """A small beta sends a gap near ``-1`` to the one-step backup."""
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions = q
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=0.02, policy_delayed=False, value_delayed=False, normalize=True, eps=1e-6), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    one_step, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(),
        temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - one_step["action_value"].item()) < 1e-04


def test_gap_does_not_backprop_into_the_online_max() -> None:
    """The gap is a constant: only the trained Q(s, a_taken) gets a gradient."""
    step_stream, _, delayed = _lambda_fixture()
    online = torch.tensor([[5.0, 0.0], [2.0, 0.0], [0.0, 0.0]], requires_grad=True)
    loss, _ = DqnObjective(        reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(bias=0.0, beta=1.0, policy_delayed=False, value_delayed=False, normalize=True, eps=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=online,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    loss["action_value"].backward()
    assert online.grad is not None
    assert online.grad[1, 0].item() == 0.0
    assert online.grad[0, 0].item() != 0.0


def test_value_gap_zero_beta_continues_when_taken_action_matches_online_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    q = predictions.clone()
    q[1] = torch.tensor([-1.0, 0.0])  # greedy at s1 is the taken action; Q(s1,a=1) stays 0
    predictions = q
    loss, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=value_gap_gate(beta=0.0, normalize=False, policy_delayed=False, value_delayed=False, bias=0.0), temperature=0.0)(
        objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
    )
    assert abs(loss["action_value"].item() - _FULL_RETURN) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_lambda_returns_do_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["group_id"] = torch.tensor([0, 0, 1])
    step_stream["terminated"][1] = True
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    one_step, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(loss["action_value"].item() - one_step["action_value"].item()) < 1e-04


def test_td_lambda_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's λ-return; bootstrap reads the last row."""
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["readout_counts"] = torch.tensor([2, 1, 2])
    online = torch.tensor([[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]])
    predictions, delayed = _q(online, delayed_q)
    loss, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(), gate=_gate(td_lambda=1.0), temperature=0.0)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    # s0 rows: (5-111)^2 = 11236, (7-111)^2 = 10816; s1 row: (0-110)^2 = 12100; s2 rows weight 0.
    assert abs(loss["action_value"].item() - (36.0 + 16.0 + 100.0) / 3) < 1e-02


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
    v_next = v[1:] * run_mask

    def returns(continuation: torch.Tensor) -> torch.Tensor:
        values, _ = _continuation_targets(
            reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
            continuation=continuation)
        return values

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
    hard_gap_got = returns(value_gap_gate(beta=0.0, normalize=False, policy_delayed=False, value_delayed=False, bias=0.0)(q=q, action=action, q_delayed=q))
    hard_gap_ref = _reference_lambda(
        r=r, g=g, v_next=v_next, cont=greedy_cont, td_lambda=1.0, n=T
    )
    hard_gap_ref = hard_gap_ref * in_run.to(dtype=hard_gap_ref.dtype)
    assert torch.allclose(hard_gap_got, hard_gap_ref, atol=1e-5, rtol=1e-5)
    past = returns(nstep_gate(n=T + 5)(q=torch.zeros(N, 1), action=torch.zeros(N, dtype=torch.long)))
    full = returns(lambda_gate(td_lambda=1.0)(q=torch.zeros(N, 1), action=torch.zeros(N, dtype=torch.long)))
    assert torch.allclose(past, full, atol=1e-5, rtol=1e-5)
    ones = torch.ones(T)
    open_q = torch.zeros(N, 1)
    open_action = torch.zeros(N, dtype=torch.long)
    short, _ = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=ones,
        continuation=nstep_gate(n=3)(q=open_q, action=open_action))
    opened, _ = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=ones,
        continuation=lambda_gate(td_lambda=1.0)(q=open_q, action=open_action))
    assert not torch.allclose(short, opened)


def test_dqn_objective_rejects_non_fp32_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        DqnObjective( reward=_rew(), value=_val(), discount=_disc(), temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)


class _EntropyKw(TypedDict):
    discount: Discount
    temperature: float

    gate: None


def _entropy_kw(*, temperature: float = 0.0) -> _EntropyKw:
    return {
        "discount": _disc(),
        "temperature": temperature,

        "gate": None,
    }


def test_temperature_zero_matches_hard_max() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    hard, hard_m = DqnObjective(reward=_rew(), value=_val(), **_entropy_kw())(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    soft0, soft0_m = DqnObjective(reward=_rew(), value=_val(), **_entropy_kw())(
        objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
    )
    assert abs(hard["action_value"].item() - soft0["action_value"].item()) < 1e-06
    assert "entropy" not in hard_m
    assert "entropy" not in soft0_m


def test_temperature_soft_value_is_logsumexp() -> None:
    data = dict(action=torch.zeros(3, dtype=torch.long), reward=torch.zeros(3),
                terminated=torch.tensor([0, 0, 1]), truncated=torch.zeros(3, dtype=torch.long))
    q = torch.zeros(3, 2)
    loss, metrics = DqnObjective(reward=None, value=None, **_entropy_kw(temperature=1.))(
        objective_data=data, group_id=_group_id(data), head_step_ids=fixture_head_steps(data),  predictions=q,
        backup_value_predictions=q, backup_action_predictions=q)
    expected = torch.log(torch.tensor(2.))
    torch.testing.assert_close(metrics['backup'], torch.tensor([expected, 0., 0.]))
    assert loss['action_value'].item() == pytest.approx(expected.square().item() / 2)
    assert metrics['entropy'].item() == pytest.approx(expected.item())


def test_temperature_matches_expected_q_plus_entropy() -> None:
    """α logsumexp(Q/α) = E_π[Q] + α H[π] for π = softmax(Q/α)."""
    from mouse_core.objectives.dqn import _boltzmann_entropy, _backup_state_value

    q = torch.tensor([[1.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    alpha = 0.5
    pi = torch.softmax(q / alpha, dim=-1)
    expected = (pi * q).sum(dim=-1) + alpha * _boltzmann_entropy(q, temperature=alpha)
    assert torch.allclose(_backup_state_value(q_actions=q, q_backup=q, temperature=alpha), expected, atol=1e-06)


def test_temperature_rejects_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        DqnObjective(reward=_rew(), value=_val(), **_entropy_kw(temperature=-0.1))


def test_temperature_does_not_change_gamma_zero_target() -> None:
    """γ=0 so V is unused; soft and hard losses match."""
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "terminated": torch.tensor([0, 1, 0]),
            "truncated": torch.tensor([0, 0, 0]),
        }
    predictions, delayed = _q(
        torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2)
    )
    hard, _ = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0),
        temperature=0.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    soft, metrics = DqnObjective( reward=_rew(), value=_val(), discount=_disc(gamma_step=0.0),
        temperature=2.0, gate=None)(objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed)
    assert abs(hard["action_value"].item() - soft["action_value"].item()) < 1e-05
    assert "entropy" in metrics


def test_dqn_requires_discount_argument() -> None:
    with pytest.raises(TypeError, match="discount"):
        DqnObjective( reward=_rew(), value=_val(), temperature=0.0, gate=None)  # type: ignore[call-arg]


def test_dqn_custom_discount_function() -> None:
    data = dict(action=torch.tensor([0, 1, 0]), reward=torch.tensor([0., 1., 0.]),
                terminated=torch.tensor([0, 0, 1]), truncated=torch.zeros(3, dtype=torch.long))
    online = torch.tensor([[0., 2.], [0., 0.], [0., 0.]])
    delayed = torch.tensor([[0., 0.], [10., 0.], [999., 999.]])
    def half(*, terminated, **_):
        return .5 * (1 - terminated.float())
    loss, metrics = DqnObjective(reward=None, value=None, discount=half, temperature=0., gate=None)(
        objective_data=data, group_id=_group_id(data), head_step_ids=fixture_head_steps(data),  predictions=online,
        backup_value_predictions=delayed, backup_action_predictions=delayed)
    torch.testing.assert_close(metrics['backup'], torch.tensor([6., 0., 0.]))
    assert loss['action_value'].item() == pytest.approx(((2 - 6) ** 2 + 0) / 2)


def test_dqn_discount_must_return_per_step_tensor() -> None:
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "terminated": torch.tensor([0, 0]),
            "truncated": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))

    def bad(*, terminated: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.5])

    with pytest.raises(ValueError, match="discount must return shape"):
        DqnObjective( reward=_rew(), value=_val(), discount=bad, temperature=0.0, gate=None)(
            objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
        )


def test_dqn_reward_must_return_per_step_tensor() -> None:
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "terminated": torch.tensor([0, 0]),
            "truncated": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))

    def bad(*, reward: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.5])

    with pytest.raises(ValueError, match="reward must return shape"):
        DqnObjective( reward=bad, value=_val(), discount=_disc(), temperature=0.0, gate=None)(
            objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
        )


def test_dqn_value_must_return_same_shape() -> None:
    step_stream = {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "terminated": torch.tensor([0, 0]),
            "truncated": torch.tensor([0, 0]),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))

    def bad(*, value: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return value[:, 0]

    with pytest.raises(ValueError, match="value must return shape"):
        DqnObjective( reward=_rew(), value=bad, discount=_disc(), temperature=0.0, gate=None)(
            objective_data=step_stream, group_id=_group_id(step_stream), head_step_ids=fixture_head_steps(step_stream),  predictions=predictions,  backup_value_predictions=delayed, backup_action_predictions=delayed
        )


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("delayed", [False, True])
def test_value_gap_zero_beta_matches_hard_greedy_cut(normalize, delayed):
    q = torch.tensor([[0., 0.], [1., 1.], [0., -1e-7], [2., 1.], [0., 0.]])
    q_delayed = q.flip(-1)
    action = torch.tensor([0, 0, 1, 1, 0])
    kwargs = {"eps": 1e-6} if normalize else {}
    c = value_gap_gate(
        beta=0., normalize=normalize, policy_delayed=delayed,
        value_delayed=delayed, bias=0., **kwargs,
    )(q=q, q_delayed=q_delayed, action=action)
    # Column 1 is an exact tie. Column 2 differs by only 1e-7;
    # column 3 switches its greedy action between online and delayed Q.
    expected = torch.tensor([
        [0., 1., float(delayed), float(not delayed), 0.],
        [0., 0., float(delayed), float(not delayed), 0.],
        [0., 0., 0., float(not delayed), 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
    ])
    torch.testing.assert_close(c, expected)
    assert torch.isfinite(c).all()


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("bias,expected", [(0., 0.), (0.1, 0.), (-0.5, 1.)])
def test_value_gap_zero_beta_applies_bias_before_cut(normalize, bias, expected):
    q = torch.tensor([[0., 0.], [0.5, 0.25], [0., 0.]])
    kwargs = {"eps": 1e-6} if normalize else {}
    c = value_gap_gate(
        beta=0., normalize=normalize, policy_delayed=False,
        value_delayed=False, bias=bias, **kwargs,
    )(q=q, q_delayed=q, action=torch.tensor([0, 0, 1]))
    assert c[0, 1].item() == expected


def test_value_gap_zero_beta_positive_bias_cuts_ties():
    q = torch.zeros(3, 2)
    c = value_gap_gate(
        beta=0., normalize=False, policy_delayed=False,
        value_delayed=False, bias=0.1,
    )(q=q, q_delayed=q, action=torch.zeros(3, dtype=torch.long))
    assert not c.any()


def test_value_gap_zero_beta_mixed_networks_preserves_positive_gap_cap():
    q = torch.tensor([[0., 0.], [2., 1.], [0., 0.]])
    c = value_gap_gate(
        beta=0., normalize=False, policy_delayed=False,
        value_delayed=True, bias=0.,
    )(q=q, q_delayed=q.flip(-1), action=torch.tensor([0, 0, 1]))
    assert c[0, 1].item() == 1.
