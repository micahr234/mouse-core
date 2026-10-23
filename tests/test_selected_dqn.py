"""Selected-action online Q must preserve dense DQN targets and gradients."""
import pytest
import torch
from mouse_core.objectives import (
    DqnObjective, boundary_discount, boundary_value, general_gate,
    lambda_gate, nstep_gate, value_gap_gate,
)


def objective(gate=None, temperature=0.0, cql_weight=0.0):
    return DqnObjective(
        reward=None,
        value=boundary_value(scale=1.3, shift=-0.2, value_terminated_scale=0.7,
                             value_terminated_shift=0.1, value_truncated_scale=1.0,
                             value_truncated_shift=0.0),
        discount=boundary_discount(gamma_step=0.9, gamma_terminated=0.0, gamma_truncated=0.0),
        gate=gate, temperature=temperature, cross_group_backups="bootstrap", cql_weight=cql_weight,
    )


@pytest.mark.parametrize("multiple_readouts", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 0.3])
@pytest.mark.parametrize("gate_name", ["none", "lambda", "nstep", "gap"])
def test_selected_matches_dense_targets_loss_and_gradients(multiple_readouts, temperature, gate_name):
    torch.manual_seed(12)
    counts = torch.tensor([2, 1, 3, 1, 2]) if multiple_readouts else torch.ones(5, dtype=torch.long)
    data = dict(action=torch.tensor([0, 1, -999, 0, 2]), reward=torch.randn(5),
                terminated=torch.tensor([False, False, False, False, True]),
                truncated=torch.zeros(5, dtype=torch.bool),
                action_executed=torch.tensor([False, True, False, True, True]),
                head_output_count=counts)
    groups = torch.tensor([0, 0, 0, 1, 1])
    step_of = torch.repeat_interleave(torch.arange(5), counts)
    actions = torch.tensor([1, 0, 0, 2, 2])[step_of]
    q = torch.randn(int(counts.sum()), 3, requires_grad=True)
    selected = q.detach().gather(-1, actions[:, None]).squeeze(-1).requires_grad_()
    delayed = torch.randn_like(q, requires_grad=True)
    gates = dict(none=None, **{"lambda": lambda_gate(td_lambda=0.6)}, nstep=nstep_gate(n=2),
                 gap=general_gate(gates=[lambda_gate(td_lambda=0.8), value_gap_gate(
                     beta=0.4, normalize=True, eps=1e-8, bias=0.0,
                     policy_delayed=True, value_delayed=True)]))
    loss = objective(gates[gate_name], temperature=temperature)
    kwargs = dict(objective_data=data, group_id=groups,
                  backup_value_predictions=delayed, backup_action_predictions=delayed)
    dense_losses, dense_metrics = loss(predictions=q, **kwargs)
    losses, metrics = loss(predictions=selected, **kwargs)
    torch.testing.assert_close(losses["action_value"], dense_losses["action_value"])
    for key in ("backup", "backup_weight", "in_run_backup"):
        torch.testing.assert_close(metrics[key], dense_metrics[key])
    losses["action_value"].backward()
    dense_losses["action_value"].backward()
    torch.testing.assert_close(selected.grad, q.grad.gather(-1, actions[:, None]).squeeze(-1))
    assert delayed.grad is None


def test_selected_uses_maximum_of_delayed_actions():
    q = torch.tensor([2., 3., 0.], requires_grad=True)
    delayed = torch.tensor([[1., 2.], [7., 5.], [6., 8.]], requires_grad=True)
    data = dict(action=torch.tensor([0, 1, 0]), reward=torch.tensor([0., 1., 4.]),
                terminated=torch.tensor([False, False, True]), truncated=torch.zeros(3, dtype=torch.bool))
    loss = DqnObjective(reward=None, value=None, gate=None, temperature=0.0,
        discount=boundary_discount(gamma_step=0.5, gamma_terminated=0.0, gamma_truncated=0.0),
        cross_group_backups="fault")
    losses, metrics = loss(objective_data=data, group_id=torch.zeros(3, dtype=torch.long),
                          predictions=q, backup_value_predictions=delayed, backup_action_predictions=delayed)
    torch.testing.assert_close(metrics["backup"][:2], torch.tensor([4.5, 4.]))
    torch.testing.assert_close(losses["action_value"], torch.tensor((2.5**2 + 1) / 2))
    losses["action_value"].backward()
    assert delayed.grad is None and q.grad[-1] == 0


@pytest.mark.parametrize("policy_delayed,value_delayed", [(False, False), (True, False), (False, True)])
def test_selected_rejects_gate_that_needs_online_action_comparison(policy_delayed, value_delayed):
    gate = value_gap_gate(beta=1.0, normalize=False, bias=0.0,
                          policy_delayed=policy_delayed, value_delayed=value_delayed)
    with pytest.raises(ValueError, match="delayed-only"):
        gate(q=torch.ones(3), q_delayed=torch.ones(3, 2), action=torch.zeros(3, dtype=torch.long))


def test_selected_rejects_cql():
    with pytest.raises(ValueError, match="CQL requires all-action"):
        objective(cql_weight=0.5)(objective_data={}, group_id=torch.zeros(3, dtype=torch.long),
            predictions=torch.ones(3), backup_value_predictions=torch.ones(3, 2),
            backup_action_predictions=torch.ones(3, 2))
