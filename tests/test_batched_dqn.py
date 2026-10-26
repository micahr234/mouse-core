"""Batched head objectives preserve scalar losses, targets, and derivatives."""

import pytest
import torch

from mouse_core.objectives import DqnObjective, boundary_discount, general_gate, lambda_gate, nstep_gate, value_gap_gate


def _objective(gate, reward=None, *, cql=0.0):
    return DqnObjective(
        discount=boundary_discount(gamma_step=0.9, gamma_terminated=0.0, gamma_truncated=0.4),
        reward=reward, value=None, temperature=0.25, cross_group_backups="ignore",
        gate=gate, cql_weight=cql,
    )


def _data():
    return dict(
        action=torch.tensor([0, 1, -200, 2, 0, 1, 2]),
        action_executed=torch.tensor([False, True, False, True, True, True, True]),
        reward=torch.tensor([0.0, 1.2, -0.1, 0.4, 0.0, 0.8, 1.0]),
        terminated=torch.tensor([False, False, False, True, False, False, True]),
        truncated=torch.tensor([False, False, False, False, False, True, False]),
        head_output_count=torch.tensor([1, 2, 1, 3, 1, 2, 1]),
    )


@pytest.mark.parametrize("gate_name", ["none", "lambda", "nstep", "gap", "combined", "custom"])
@pytest.mark.parametrize("selected", [False, True])
def test_batched_loss_and_gradients_equal_independent_heads(gate_name, selected):
    torch.manual_seed(41)
    data = _data()
    groups = torch.tensor([0, 0, 0, 0, 2, 2, 2])
    gap = value_gap_gate(beta=0.7, normalize=False, policy_delayed=True, value_delayed=True, bias=0.1)
    gates = {"none": None, "lambda": lambda_gate(td_lambda=0.8), "nstep": nstep_gate(n=3),
             "gap": gap, "combined": general_gate(gates=[nstep_gate(n=3), gap]),
             "custom": lambda **kw: kw["q"].new_full((7, 7), 0.4)}
    gate = gates[gate_name]
    scales = torch.tensor([0.3, 1.0, 2.0])
    q = torch.randn(11, 3, 4, requires_grad=True)
    if selected:
        q = q[..., 0].detach().requires_grad_()
    backup = torch.randn(11, 3, 4, requires_grad=True)
    actions = torch.randn_like(backup, requires_grad=True)
    objective = _objective(gate, lambda reward, **_: reward[:, None] * scales,
                           cql=0.0 if selected else 0.2)
    batched_loss, batched_metrics = objective(
        objective_data=data, group_id=groups, predictions=q,
        backup_value_predictions=backup, backup_action_predictions=actions,
        return_backup_lengths=True,
    )
    batched_loss["action_value"].sum().backward()
    assert batched_loss["action_value"].shape == (3,)
    assert backup.grad is None and actions.grad is None
    assert q.grad is not None
    for head in range(3):
        single = q.detach()[:, head].requires_grad_()
        scalar = _objective(gate, lambda reward, **_: reward * scales[head],
                            cql=0.0 if selected else 0.2)
        losses, metrics = scalar(
            objective_data=data, group_id=groups, predictions=single,
            backup_value_predictions=backup[:, head], backup_action_predictions=actions[:, head],
            return_backup_lengths=True,
        )
        losses["action_value"].backward()
        torch.testing.assert_close(batched_loss["action_value"][head], losses["action_value"])
        torch.testing.assert_close(q.grad[:, head], single.grad)
        for name, expected in metrics.items():
            if name == "in_run_backup":
                continue  # This optional diagnostic flattens participating heads.
            actual = batched_metrics[name]
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual[..., head], expected)


def test_shared_rewards_and_fixed_size_training_diagnostics():
    q = torch.randn(11, 5, 4, requires_grad=True)
    losses, metrics = _objective(None)(
        objective_data=_data(), group_id=torch.zeros(7, dtype=torch.long),
        predictions=q, backup_value_predictions=q.detach(), backup_action_predictions=q.detach(),
        include_statistics=False, return_backup_lengths=True,
    )
    assert losses["action_value"].shape == (5,)
    assert set(metrics) == {"backup", "backup_weight", "backup_length"}
    assert all(isinstance(value, torch.Tensor) and value.shape == (11, 5) for value in metrics.values())


def test_batched_custom_gate_validates_head_dimension():
    q = torch.randn(11, 5, 4)
    with pytest.raises(ValueError, match="gate must return shape"):
        _objective(lambda **_: torch.zeros(7, 7, 2))(
            objective_data=_data(), group_id=torch.zeros(7, dtype=torch.long),
            predictions=q, backup_value_predictions=q, backup_action_predictions=q,
        )


@pytest.mark.parametrize("shape", [(7, 4), (7, 3, 1), (1, 3)])
def test_batched_reward_rejects_wrong_shape(shape):
    q = torch.randn(11, 3, 4)
    with pytest.raises(ValueError, match="reward must return shape"):
        _objective(None, lambda **_: torch.zeros(shape))(
            objective_data=_data(), group_id=torch.zeros(7, dtype=torch.long),
            predictions=q, backup_value_predictions=q, backup_action_predictions=q,
        )


@pytest.mark.parametrize("shape", [(), (2, 3, 4, 5)])
def test_invalid_prediction_rank_is_rejected(shape):
    with pytest.raises(ValueError, match="predictions must have shape"):
        _objective(None)(
            objective_data={}, group_id=torch.zeros(2, dtype=torch.long),
            predictions=torch.zeros(shape), backup_value_predictions=torch.ones(2, 3),
            backup_action_predictions=torch.ones(2, 3),
        )
