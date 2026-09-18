"""Tests for layerwise DQN objective and discount schedule."""
from __future__ import annotations
import pytest
import torch
from tensordict import TensorDict
from mouse_core.objectives import LayerwiseDqnObjective, effective_horizon
from tests._bound_head import BoundHead

_LW = BoundHead("action_value_layerwise")


def test_effective_horizon() -> None:
    assert effective_horizon(gamma=0.0) == 1.0
    assert effective_horizon(gamma=0.99) == pytest.approx(100.0)

def test_layerwise_dqn_objective_anchors_endpoints() -> None:
    objective = LayerwiseDqnObjective(head=_LW, num_backbone_layers=4, gamma_step_start=1.0, gamma_step=1.0, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.99, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)
    assert objective.layer_gamma_step == [1.0, 1.0, 1.0, 1.0]
    assert objective.layer_gamma_episode_terminal[0] == 0.0
    assert objective.layer_gamma_episode_terminal[-1] == 0.99

def test_layerwise_dqn_objective_linear_horizon() -> None:
    objective = LayerwiseDqnObjective(head=_LW, num_backbone_layers=4, gamma_step_start=0.0, gamma_step=0.99, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)
    horizons = [effective_horizon(gamma=g) for g in objective.layer_gamma_step]
    assert horizons == pytest.approx([1.0, 34.0, 67.0, 100.0])
    assert objective.layer_gamma_step[0] == 0.0
    assert objective.layer_gamma_step[-1] == 0.99

def test_layerwise_dqn_objective_runs() -> None:
    n, layers, a = (8, 3, 3)
    step_stream = TensorDict({'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}, batch_size=[n])
    predictions = TensorDict({'action_value_layerwise': torch.randn(n, layers, a)}, batch_size=[n])
    delayed = TensorDict({'action_value_layerwise': torch.randn(n, layers, a)}, batch_size=[n])
    objective = LayerwiseDqnObjective(head=_LW, num_backbone_layers=layers, gamma_step_start=0.0, gamma_step=0.99, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)
    loss, metrics = objective(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert loss.ndim == 0
    assert 'action_value_layerwise' in metrics
    assert metrics['layer_0_gamma_step'] < metrics['layer_2_gamma_step']
    assert metrics['action_value_layerwise'] >= 0.0
    assert metrics['layer_0_loss'] >= 0.0
    assert 'entropy' not in metrics

def test_layerwise_dqn_objective_skips_transitions_across_sequences() -> None:
    """Corrupting data at a sequence-boundary pair must leave the loss unchanged."""
    n, layers, a = (5, 2, 3)
    torch.manual_seed(0)
    step_stream = TensorDict({'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 1, 1, 1])}, batch_size=[n])
    predictions = TensorDict({'action_value_layerwise': torch.randn(n, layers, a)}, batch_size=[n])
    delayed = TensorDict({'action_value_layerwise': torch.randn(n, layers, a)}, batch_size=[n])
    objective = LayerwiseDqnObjective(head=_LW, num_backbone_layers=layers, gamma_step_start=0.0, gamma_step=0.99, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)
    loss_before, _ = objective(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    corrupted = step_stream.clone()
    corrupted['reward'][2] = 1000000.0
    loss_after, _ = objective(objective_data=corrupted, predictions=predictions, delayed_predictions=delayed)
    assert torch.allclose(loss_before, loss_after)

def test_layerwise_dqn_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream = TensorDict(
        {
            'action': torch.zeros(3, dtype=torch.long),
            'reward': torch.zeros(3),
            'episode_done': torch.zeros(3, dtype=torch.long),
            'task_done': torch.zeros(3, dtype=torch.long),
            'sequence_id': torch.tensor([0, 1, 2]),
        },
        batch_size=[3],
    )
    predictions = TensorDict({'action_value_layerwise': torch.randn(3, 2, 2)}, batch_size=[3])
    delayed = TensorDict({'action_value_layerwise': torch.randn(3, 2, 2)}, batch_size=[3])
    loss, metrics = LayerwiseDqnObjective(head=_LW, 
        num_backbone_layers=2, gamma_step_start=0.0, gamma_step=0.99,
        gamma_episode_terminal_start=0.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated_start=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal_start=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated_start=0.0,
        gamma_task_truncated=0.0,
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics['q_values_mean']) < 1e-05


def test_layerwise_dqn_objective_rejects_layer_mismatch() -> None:
    step_stream = TensorDict({'action': torch.zeros(3, dtype=torch.long), 'reward': torch.zeros(3), 'episode_done': torch.zeros(3, dtype=torch.long), 'task_done': torch.zeros(3, dtype=torch.long)}, batch_size=[3])
    predictions = TensorDict({'action_value_layerwise': torch.zeros(3, 2, 2)}, batch_size=[3])
    delayed = TensorDict({'action_value_layerwise': torch.zeros(3, 2, 2)}, batch_size=[3])
    objective = LayerwiseDqnObjective(head=_LW, num_backbone_layers=3, gamma_step_start=0.0, gamma_step=0.99, gamma_episode_terminal_start=0.0, gamma_episode_terminal=0.0, gamma_episode_truncated_start=0.0, gamma_episode_truncated=0.0, gamma_task_terminal_start=0.0, gamma_task_terminal=0.0, gamma_task_truncated_start=0.0, gamma_task_truncated=0.0, grouping_field=None, temperature=0.0)
    try:
        objective(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    except ValueError as exc:
        assert 'expects 3 Q layers' in str(exc)
    else:
        raise AssertionError('expected ValueError for layer count mismatch')


def test_layerwise_dqn_objective_does_not_backprop_through_delayed_q() -> None:
    """Bootstrap Q is a constant: delayed Q must not receive a gradient."""
    n, layers, a = 4, 2, 2
    step_stream = TensorDict(
        {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
        },
        batch_size=[n],
    )
    online = torch.randn(n, layers, a, requires_grad=True)
    delayed = torch.randn(n, layers, a, requires_grad=True)
    predictions = TensorDict({"action_value_layerwise": online}, batch_size=[n])
    delayed_td = TensorDict({"action_value_layerwise": delayed}, batch_size=[n])
    loss, _ = LayerwiseDqnObjective(head=_LW, 
        num_backbone_layers=layers, gamma_step_start=1.0, gamma_step=1.0,
        gamma_episode_terminal_start=0.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated_start=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal_start=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated_start=0.0,
        gamma_task_truncated=0.0,
        grouping_field=None, temperature=0.0)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_td)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None


def test_single_layer_rejects_mismatched_start_and_deep_gamma() -> None:
    with pytest.raises(ValueError, match="num_backbone_layers=1"):
        LayerwiseDqnObjective(head=_LW, 
            num_backbone_layers=1,
            gamma_step_start=0.0,
            gamma_step=0.99,
            gamma_episode_terminal_start=0.0,
            gamma_episode_terminal=0.0,
            gamma_episode_truncated_start=0.0,
            gamma_episode_truncated=0.0,
            gamma_task_terminal_start=0.0,
            gamma_task_terminal=0.0,
            gamma_task_truncated_start=0.0,
            gamma_task_truncated=0.0,
            grouping_field=None, temperature=0.0)
    objective = LayerwiseDqnObjective(head=_LW, 
        num_backbone_layers=1,
        gamma_step_start=0.5,
        gamma_step=0.5,
        gamma_episode_terminal_start=0.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated_start=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal_start=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated_start=0.0,
        gamma_task_truncated=0.0,
        grouping_field=None, temperature=0.0)
    assert objective.layer_gamma_step == [0.5]


def _layerwise(**overrides: object) -> LayerwiseDqnObjective:
    kwargs: dict[str, object] = dict(
        head=_LW,
        num_backbone_layers=1,
        gamma_step_start=1.0,
        gamma_step=1.0,
        gamma_episode_terminal_start=0.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated_start=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal_start=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated_start=0.0,
        gamma_task_truncated=0.0,
        grouping_field=None, temperature=0.0,
    )
    kwargs.update(overrides)
    return LayerwiseDqnObjective(**kwargs)  # type: ignore[arg-type]


def test_layerwise_temperature_matches_dqn_one_layer() -> None:
    """One layer with the same α is DqnObjective's soft backup."""
    from mouse_core.objectives import DqnObjective

    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 0.0]),
            "episode_done": torch.zeros(2, dtype=torch.int64),
            "task_done": torch.zeros(2, dtype=torch.int64),
        },
        batch_size=[2],
    )
    online = torch.zeros(2, 2)
    delayed = torch.zeros(2, 2)
    layerwise_pred = TensorDict(
        {"action_value_layerwise": online.unsqueeze(1)}, batch_size=[2]
    )
    layerwise_del = TensorDict(
        {"action_value_layerwise": delayed.unsqueeze(1)}, batch_size=[2]
    )
    dqn_pred = TensorDict({"action_value": online}, batch_size=[2])
    dqn_del = TensorDict({"action_value": delayed}, batch_size=[2])
    lw_loss, lw_m = _layerwise(temperature=1.0)(
        objective_data=step_stream, predictions=layerwise_pred, delayed_predictions=layerwise_del
    )
    dqn_loss, dqn_m = DqnObjective(head=BoundHead("action_value"), 
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
        grouping_field=None,
        temperature=1.0,
    )(objective_data=step_stream, predictions=dqn_pred, delayed_predictions=dqn_del)
    assert abs(lw_loss.item() - dqn_loss.item()) < 1e-06
    assert abs(lw_m["entropy"] - dqn_m["entropy"]) < 1e-06


def test_layerwise_temperature_rejects_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _layerwise(temperature=-0.1)
