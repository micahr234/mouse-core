"""Tests for GRPO objective and group-relative advantages."""
from __future__ import annotations
import pytest
import torch
from mouse_core.objectives import GrpoObjective, group_relative_advantages
from tests._bound_head import BoundHead

_POLICY = BoundHead("action")


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

def test_grpo_requires_grouping_field_argument() -> None:
    with pytest.raises(TypeError, match="grouping_field"):
        GrpoObjective(head=_POLICY, )  # type: ignore[call-arg]


def test_grpo_objective_runs() -> None:
    n, a = (8, 3)
    objective_data = {'action': torch.randint(0, a, (n,)), 'old_log_prob': torch.randn(n), 'advantage': torch.randn(n), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    predictions = {'action': torch.randn(n, a)}
    loss, metrics = GrpoObjective(head=_POLICY, grouping_field=None)(objective_data=objective_data, predictions=predictions)
    assert loss.ndim == 0
    assert 'grpo' in metrics
    assert 'policy_loss' in metrics
    assert 'entropy' in metrics

def test_grpo_objective_requires_advantage() -> None:
    objective_data = {'action': torch.zeros(2, dtype=torch.long), 'old_log_prob': torch.zeros(2)}
    predictions = {'action': torch.zeros(2, 2)}
    try:
        GrpoObjective(head=_POLICY, grouping_field=None)(objective_data=objective_data, predictions=predictions)
    except KeyError as e:
        assert 'advantage' in str(e)
    else:
        raise AssertionError('expected KeyError when advantage is missing')

def test_grpo_objective_closed_form() -> None:
    """ratio≈1, advantage=3 → policy_loss = -3; ent_coef=0 → loss = -3."""
    objective_data = {'action': torch.tensor([0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0]), 'advantage': torch.tensor([0.0, 3.0])}
    predictions = {'action': torch.tensor([[20.0, -20.0], [20.0, -20.0]])}
    loss, metrics = GrpoObjective(head=_POLICY, ent_coef=0.0, grouping_field=None)(objective_data=objective_data, predictions=predictions)
    assert abs(loss.item() - -3.0) < 0.001
    assert abs(metrics['policy_loss'] - -3.0) < 0.001

def test_grpo_masks_cross_task_pairs_only_when_grouping_field_set() -> None:
    """Same sequence_id + a task change: grouping_field is the only cut."""
    objective_data = {
            "action": torch.tensor([0, 0]),
            "old_log_prob": torch.tensor([0.0, 0.0]),
            "advantage": torch.tensor([0.0, 3.0]),
            "sequence_id": torch.tensor([0, 0]),
            "task_index": torch.tensor([0, 1]),
        }
    predictions = {"action": torch.tensor([[20.0, -20.0], [20.0, -20.0]])}
    loss_cut, _ = GrpoObjective(head=_POLICY, ent_coef=0.0, grouping_field="task_index")(
        objective_data=objective_data, predictions=predictions
    )
    assert abs(loss_cut.item()) < 1e-5
    loss_leak, _ = GrpoObjective(head=_POLICY, ent_coef=0.0, grouping_field=None)(objective_data=objective_data, predictions=predictions)
    assert abs(loss_leak.item() - -3.0) < 0.001


def test_grpo_objective_skips_sequence_boundaries() -> None:
    objective_data = {'action': torch.tensor([0, 0, 0]), 'old_log_prob': torch.zeros(3), 'advantage': torch.tensor([9.0, 9.0, 2.0]), 'sequence_id': torch.tensor([0, 1, 1])}
    predictions = {'action': torch.tensor([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]])}
    loss, _ = GrpoObjective(head=_POLICY, ent_coef=0.0, grouping_field=None)(objective_data=objective_data, predictions=predictions)
    assert abs(loss.item() - -2.0) < 0.001


def test_grpo_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    objective_data = {'action': torch.tensor([0, 1, 0]), 'old_log_prob': torch.zeros(3), 'advantage': torch.tensor([9.0, 9.0, 2.0]), 'sequence_id': torch.tensor([0, 1, 2])}
    predictions = {'action': torch.zeros(3, 2)}
    loss, metrics = GrpoObjective(head=_POLICY, ent_coef=0.0, grouping_field=None)(objective_data=objective_data, predictions=predictions)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics['advantage_mean']) < 1e-05
