"""Tests for GRPO objective and group-relative advantages."""
from __future__ import annotations
import pytest
import torch
from mouse_core.objectives import GrpoObjective, group_relative_advantages

def _group_id(data: dict[str, torch.Tensor]) -> torch.Tensor:
    if "group_id" in data:
        return data["group_id"]
    n = int(next(iter(data.values())).shape[0])
    return torch.zeros(n, dtype=torch.int64)




def test_group_relative_advantages_zscore() -> None:
    rewards = torch.tensor([1.0, 3.0, 5.0])
    adv = group_relative_advantages(rewards=rewards)
    assert adv.shape == (3,)
    assert abs(adv.mean().item()) < 1e-05
    assert abs(adv[0].item() + adv[2].item()) < 1e-05
    assert adv[2].item() > 0 and adv[0].item() < 0

def test_group_relative_advantages_identical_is_zero() -> None:
    adv = group_relative_advantages(rewards=torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(adv, torch.zeros(3))

def test_group_relative_advantages_singleton_is_zero() -> None:
    adv = group_relative_advantages(rewards=torch.tensor([7.0]))
    assert torch.allclose(adv, torch.zeros(1))

def test_grpo_objective_runs() -> None:
    n, a = (8, 3)
    objective_data = {'action': torch.randint(0, a, (n,)), 'old_log_prob': torch.randn(n), 'advantage': torch.randn(n), 'group_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    predictions = torch.randn(n, a)
    loss, metrics = GrpoObjective()(objective_data=objective_data, group_id=_group_id(objective_data), predictions=predictions)
    assert loss["grpo"].ndim == 0
    assert 'grpo' in metrics
    assert 'policy_loss' in metrics
    assert 'entropy' in metrics

def test_grpo_objective_requires_advantage() -> None:
    objective_data = {'action': torch.zeros(2, dtype=torch.long), 'old_log_prob': torch.zeros(2)}
    predictions = torch.zeros(2, 2)
    try:
        GrpoObjective()(objective_data=objective_data, group_id=_group_id(objective_data), predictions=predictions)
    except KeyError as e:
        assert 'advantage' in str(e)
    else:
        raise AssertionError('expected KeyError when advantage is missing')

def test_grpo_objective_closed_form() -> None:
    """ratio≈1, advantage=3 → policy_loss = -3; ent_coef=0 → loss = -3."""
    objective_data = {'action': torch.tensor([0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0]), 'advantage': torch.tensor([0.0, 3.0])}
    predictions = torch.tensor([[20.0, -20.0], [20.0, -20.0]])
    loss, metrics = GrpoObjective(ent_coef=0.0)(objective_data=objective_data, group_id=_group_id(objective_data), predictions=predictions)
    assert abs(loss["grpo"].item() - -3.0) < 0.001
    assert abs(metrics['policy_loss'] - -3.0) < 0.001

def test_grpo_objective_skips_sequence_boundaries() -> None:
    objective_data = {'action': torch.tensor([0, 0, 0]), 'old_log_prob': torch.zeros(3), 'advantage': torch.tensor([9.0, 9.0, 2.0]), 'group_id': torch.tensor([0, 1, 1])}
    predictions = torch.tensor([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]])
    loss, _ = GrpoObjective(ent_coef=0.0)(objective_data=objective_data, group_id=_group_id(objective_data), predictions=predictions)
    assert abs(loss["grpo"].item() - -2.0) < 0.001


def test_grpo_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    objective_data = {'action': torch.tensor([0, 1, 0]), 'old_log_prob': torch.zeros(3), 'advantage': torch.tensor([9.0, 9.0, 2.0]), 'group_id': torch.tensor([0, 1, 2])}
    predictions = torch.zeros(3, 2)
    loss, metrics = GrpoObjective(ent_coef=0.0)(objective_data=objective_data, group_id=_group_id(objective_data), predictions=predictions)
    assert abs(loss["grpo"].item()) < 1e-05
    assert abs(metrics['advantage_mean']) < 1e-05


def test_ignored_reset_action_has_no_policy_gradient_or_metrics():
    data = {
        'action': torch.tensor([0, 1, 0, 1]),
        'action_executed': torch.tensor([False, True, False, True]),
        'old_log_prob': torch.tensor([0., -.5, -3., -.8]),
        'advantage': torch.tensor([0., 2., 999., -1.]),
    }
    predictions = torch.tensor([[.2, .4], [1., -2.], [.5, .1], [8., 9.]], requires_grad=True)
    objective = GrpoObjective(ent_coef=.1)
    loss, metrics = objective(objective_data=data, group_id=torch.zeros(4, dtype=torch.int64), predictions=predictions)
    loss['grpo'].backward()
    assert predictions.grad is not None
    torch.testing.assert_close(predictions.grad[1], torch.zeros(2))
    assert predictions.grad[0].abs().sum() > 0
    assert predictions.grad[2].abs().sum() > 0
    # Removing the unexecuted transition gives the same result, including the
    # denominator and all statistics. The reset observation still predicts the
    # next real action.
    real_data = {key: value[[0, 1, 2, 3]] for key, value in data.items() if key != 'action_executed'}
    expected, expected_metrics = objective(
        objective_data=real_data, group_id=torch.tensor([0, 0, 1, 1]), predictions=predictions.detach(),
    )
    torch.testing.assert_close(loss['grpo'], expected['grpo'])
    assert metrics == pytest.approx(expected_metrics)
    changed = {**data, 'action': torch.tensor([1, 1, 1, 1]),
               'old_log_prob': torch.tensor([999., -.5, -6., -.8]),
               'advantage': torch.tensor([999., 2., -999., -1.])}
    changed_loss, changed_metrics = objective(
        objective_data=changed, group_id=torch.zeros(4, dtype=torch.int64), predictions=predictions.detach(),
    )
    torch.testing.assert_close(changed_loss['grpo'], loss['grpo'])
    assert changed_metrics == pytest.approx(metrics)


@pytest.mark.parametrize('executed', [torch.ones(3), torch.ones(2, dtype=torch.bool)])
def test_action_executed_requires_aligned_boolean_column(executed):
    data = {'action': torch.zeros(3, dtype=torch.int64), 'advantage': torch.ones(3),
            'action_executed': executed}
    with pytest.raises(ValueError, match='action_executed must be bool'):
        GrpoObjective()(objective_data=data, group_id=torch.zeros(3, dtype=torch.int64), predictions=torch.zeros(3, 2))
