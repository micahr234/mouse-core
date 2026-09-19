"""Tests for layerwise DQN objective and discount schedule."""
from __future__ import annotations
import pytest
import torch
from mouse_core.objectives import LayerwiseDqnObjective, affine_reward, affine_value, boundary_discount, effective_horizon
from tests._bound_head import BoundHead

_LW = BoundHead("action_value_layerwise")


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


def test_effective_horizon() -> None:
    assert effective_horizon(gamma=0.0) == 1.0
    assert effective_horizon(gamma=0.99) == pytest.approx(100.0)

def test_layerwise_dqn_objective_anchors_endpoints() -> None:
    objective = LayerwiseDqnObjective(
        head=_LW,
        num_backbone_layers=4,
        discount_start=_disc(gamma_episode_terminal=0.0),
        discount=_disc(gamma_episode_terminal=0.99),
        grouping_field=None,
        temperature=0.0,
        reward=_rew(),
        value=_val(),
    )
    assert objective.layer_gamma_step == [1.0, 1.0, 1.0, 1.0]
    assert objective.layer_gamma_episode_terminal[0] == 0.0
    assert objective.layer_gamma_episode_terminal[-1] == pytest.approx(0.99)

def test_layerwise_dqn_objective_linear_horizon() -> None:
    objective = LayerwiseDqnObjective(
        head=_LW,
        num_backbone_layers=4,
        discount_start=_disc(gamma_step=0.0),
        discount=_disc(gamma_step=0.99),
        grouping_field=None,
        temperature=0.0,
        reward=_rew(),
        value=_val(),
    )
    horizons = [effective_horizon(gamma=g) for g in objective.layer_gamma_step]
    assert horizons == pytest.approx([1.0, 34.0, 67.0, 100.0])
    assert objective.layer_gamma_step[0] == 0.0
    assert objective.layer_gamma_step[-1] == pytest.approx(0.99)

def _lw(*, num_backbone_layers: int, **overrides: object) -> LayerwiseDqnObjective:
    kwargs: dict[str, object] = dict(
        head=_LW,
        num_backbone_layers=num_backbone_layers,
        discount_start=_disc(gamma_step=0.0),
        discount=_disc(gamma_step=0.99),
        grouping_field=None,
        temperature=0.0,
        reward=_rew(),
        value=_val(),
    )
    kwargs.update(overrides)
    return LayerwiseDqnObjective(**kwargs)  # type: ignore[arg-type]


def test_layerwise_dqn_objective_runs() -> None:
    n, layers, a = (8, 3, 3)
    step_stream = {'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])}
    predictions = {'action_value_layerwise': torch.randn(n, layers, a)}
    delayed = {'action_value_layerwise': torch.randn(n, layers, a)}
    objective = _lw(num_backbone_layers=layers)
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
    step_stream = {'action': torch.randint(0, a, (n,)), 'reward': torch.randn(n), 'episode_done': torch.zeros(n, dtype=torch.long), 'task_done': torch.zeros(n, dtype=torch.long), 'sequence_id': torch.tensor([0, 0, 1, 1, 1])}
    predictions = {'action_value_layerwise': torch.randn(n, layers, a)}
    delayed = {'action_value_layerwise': torch.randn(n, layers, a)}
    objective = _lw(num_backbone_layers=layers)
    loss_before, _ = objective(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    corrupted = {key: value.clone() for key, value in step_stream.items()}
    corrupted['reward'][2] = 1000000.0
    loss_after, _ = objective(objective_data=corrupted, predictions=predictions, delayed_predictions=delayed)
    assert torch.allclose(loss_before, loss_after)

def test_layerwise_dqn_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream = {
            'action': torch.zeros(3, dtype=torch.long),
            'reward': torch.zeros(3),
            'episode_done': torch.zeros(3, dtype=torch.long),
            'task_done': torch.zeros(3, dtype=torch.long),
            'sequence_id': torch.tensor([0, 1, 2]),
        }
    predictions = {'action_value_layerwise': torch.randn(3, 2, 2)}
    delayed = {'action_value_layerwise': torch.randn(3, 2, 2)}
    loss, metrics = _lw(num_backbone_layers=2)(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics['q_values_mean']) < 1e-05


def test_layerwise_dqn_objective_rejects_layer_mismatch() -> None:
    step_stream = {'action': torch.zeros(3, dtype=torch.long), 'reward': torch.zeros(3), 'episode_done': torch.zeros(3, dtype=torch.long), 'task_done': torch.zeros(3, dtype=torch.long)}
    predictions = {'action_value_layerwise': torch.zeros(3, 2, 2)}
    delayed = {'action_value_layerwise': torch.zeros(3, 2, 2)}
    objective = _lw(num_backbone_layers=3)
    try:
        objective(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed)
    except ValueError as exc:
        assert 'expects 3 Q layers' in str(exc)
    else:
        raise AssertionError('expected ValueError for layer count mismatch')


def test_layerwise_dqn_objective_does_not_backprop_through_delayed_q() -> None:
    """Bootstrap Q is a constant: delayed Q must not receive a gradient."""
    n, layers, a = 4, 2, 2
    step_stream = {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
        }
    online = torch.randn(n, layers, a, requires_grad=True)
    delayed = torch.randn(n, layers, a, requires_grad=True)
    predictions = {"action_value_layerwise": online}
    delayed_td = {"action_value_layerwise": delayed}
    same = _disc()
    loss, _ = _lw(
        num_backbone_layers=layers, discount_start=same, discount=same
    )(objective_data=step_stream, predictions=predictions, delayed_predictions=delayed_td)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None


def test_single_layer_rejects_mismatched_start_and_deep_gamma() -> None:
    with pytest.raises(ValueError, match="num_backbone_layers=1"):
        _lw(num_backbone_layers=1)
    same = _disc(gamma_step=0.5)
    objective = _lw(num_backbone_layers=1, discount_start=same, discount=same)
    assert objective.layer_gamma_step == [0.5]


def _layerwise(**overrides: object) -> LayerwiseDqnObjective:
    same = _disc()
    kwargs: dict[str, object] = dict(
        head=_LW,
        num_backbone_layers=1,
        discount_start=same,
        discount=same,
        grouping_field=None,
        temperature=0.0,
        reward=_rew(),
        value=_val(),
    )
    kwargs.update(overrides)
    return LayerwiseDqnObjective(**kwargs)  # type: ignore[arg-type]


def test_layerwise_temperature_matches_dqn_one_layer() -> None:
    """One layer with the same α is DqnObjective's soft backup."""
    from mouse_core.objectives import DqnObjective

    step_stream = {
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 0.0]),
            "episode_done": torch.zeros(2, dtype=torch.int64),
            "task_done": torch.zeros(2, dtype=torch.int64),
        }
    online = torch.zeros(2, 2)
    delayed = torch.zeros(2, 2)
    layerwise_pred = {"action_value_layerwise": online.unsqueeze(1)}
    layerwise_del = {"action_value_layerwise": delayed.unsqueeze(1)}
    dqn_pred = {"action_value": online}
    dqn_del = {"action_value": delayed}
    lw_loss, lw_m = _layerwise(temperature=1.0)(
        objective_data=step_stream, predictions=layerwise_pred, delayed_predictions=layerwise_del
    )
    dqn_loss, dqn_m = DqnObjective(head=BoundHead("action_value"), 
        discount=_disc(),
        reward=_rew(),
        value=_val(),
        grouping_field=None,
        temperature=1.0,
    )(objective_data=step_stream, predictions=dqn_pred, delayed_predictions=dqn_del)
    assert abs(lw_loss.item() - dqn_loss.item()) < 1e-06
    assert abs(lw_m["entropy"] - dqn_m["entropy"]) < 1e-06


def test_layerwise_temperature_rejects_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _layerwise(temperature=-0.1)
