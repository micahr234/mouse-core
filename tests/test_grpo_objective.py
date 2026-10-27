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


@pytest.mark.parametrize('ratio,advantage,expected_loss,clipped', [
    (0.8, -2.0, 1.8, True),
    (0.8, 2.0, -1.6, False),
    (1.6, -2.0, 3.2, False),
    (0.95, -2.0, 1.9, False),
    (1.3, 2.0, -2.6, False),
    (1.6, 2.0, -2.8, True),
])
def test_asymmetric_policy_ratio_clipping(ratio, advantage, expected_loss, clipped):
    import math
    data = {
        'action': torch.tensor([0, 0]),
        'old_log_prob': torch.tensor([0.0, math.log(0.5 / ratio)]),
        'advantage': torch.tensor([0.0, advantage]),
    }
    logits = torch.zeros(2, 2, requires_grad=True)
    objective = GrpoObjective(clip_ratio_min=0.9, clip_ratio_max=1.4, ent_coef=0.0)
    losses, metrics = objective(objective_data=data, group_id=_group_id(data), predictions=logits)
    assert losses['grpo'].item() == pytest.approx(expected_loss)
    assert metrics['clipfrac'] == float(clipped)
    losses['grpo'].backward()
    assert bool(logits.grad.abs().sum() > 0) == (not clipped)


@pytest.mark.parametrize('lower,upper', [(0, 1.2), (-0.1, 1.2), (1.1, 1.2),
                                       (0.8, 0.9), (1.2, 0.8),
                                       (float('nan'), 1.2), (0.8, float('inf'))])
def test_invalid_policy_ratio_bounds(lower, upper):
    with pytest.raises(ValueError, match='clip ratios'):
        GrpoObjective(clip_ratio_min=lower, clip_ratio_max=upper)

@pytest.mark.parametrize('scaling,expected', [
    ('group', [-2./2.0001, 2./2.0001]), ('none', [-2., 2.]),
    ('batch', [-2./(5**.5 + 1e-4), 2./(5**.5 + 1e-4)]),
])
def test_reward_scaling_keeps_group_baseline(scaling, expected):
    from mouse_core.objectives import group_relative_advantages
    actual = group_relative_advantages(rewards=torch.tensor([1., 5.]), scaling=scaling,
                                       batch_rewards=torch.tensor([1., 5., 3., 7.]))
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_batch_scaling_requires_collection_rewards():
    from mouse_core.objectives import group_relative_advantages
    with pytest.raises(ValueError, match='batch_rewards'):
        group_relative_advantages(rewards=torch.tensor([1., 2.]), scaling='batch')

@pytest.mark.parametrize('rewards,expected', [
    ([0., 0., 0., 1.], [-.25/.2501, -.25/.2501, -.25/.2501, .75/.2501]),
    ([0., 0.], [0., 0.]),
    ([2., 2.], [0., 0.]),
    ([1., 5.], [-2./3.0001, 2./3.0001]),
])
def test_maxrl_reward_scaling(rewards, expected):
    actual = group_relative_advantages(rewards=torch.tensor(rewards), scaling='maxrl')
    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize('rewards', [[-1., 2.], [float('nan'), 1.], [float('inf'), 1.]])
def test_maxrl_rejects_invalid_rewards(rewards):
    with pytest.raises(ValueError, match='finite nonnegative'):
        group_relative_advantages(rewards=torch.tensor(rewards), scaling='maxrl')


@pytest.mark.parametrize('level,expected', [('action', -2.9/3), ('episode', -3.2/3)])
def test_policy_importance_sampling_level_separates_branches_and_skips_resets(level, expected):
    import math
    # Two executed actions in branch zero, then one in branch one; reset log-probs are invalid.
    data = {'action': torch.zeros(6, dtype=torch.long),
            'action_executed': torch.tensor([False, True, False, True, False, True]),
            'advantage': torch.ones(6),
            'old_log_prob': torch.tensor([float('nan'), math.log(1.), float('nan'),
                                          math.log(.25), float('nan'), math.log(.125)])}
    logits = torch.zeros(6, 2, requires_grad=True)
    objective = GrpoObjective(ent_coef=0., importance_sampling_level=level)
    losses, metrics = objective(objective_data=data, group_id=torch.tensor([0,0,0,0,1,1]), predictions=logits)
    assert losses['grpo'].item() == pytest.approx(expected)
    assert metrics['clipfrac'] == pytest.approx(2./3. if level == 'action' else 1./3.)
    losses['grpo'].backward()
    if level == 'episode':
        assert logits.grad[0,0].item() == pytest.approx(-1./6.)
        assert logits.grad[2,0].item() == pytest.approx(-1./6.)
    assert logits.grad[4].abs().sum() == 0
    assert logits.grad[[1,3,5]].abs().sum() == 0


def test_invalid_policy_importance_sampling_level():
    with pytest.raises(ValueError, match='importance_sampling_level'):
        GrpoObjective(importance_sampling_level='invalid')


@pytest.mark.parametrize("scaling", ["group", "batch", "maxrl"])
def test_small_rewards_use_stabilizer(scaling):
    rewards = torch.tensor([0., 2e-6], dtype=torch.float64)
    actual = group_relative_advantages(rewards=rewards, scaling=scaling, batch_rewards=rewards)
    torch.testing.assert_close(actual, torch.tensor([-1e-6 / 1.01e-4, 1e-6 / 1.01e-4], dtype=torch.float64))


def test_zero_advantage_does_not_count_as_clipped():
    data = {"action": torch.zeros(2, dtype=torch.long), "old_log_prob": torch.zeros(2),
            "advantage": torch.zeros(2)}
    _, metrics = GrpoObjective()(objective_data=data, group_id=_group_id(data), predictions=torch.zeros(2, 2))
    assert metrics["clipfrac"] == 0.
