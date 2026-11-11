"""Independent backup action/value inputs, including the soft policy target."""

from tests._prediction_layout import fixture_head_steps
import pytest
import torch

from mouse_core.objectives import DqnObjective, boundary_discount


def _objective(temperature=0.):
    return DqnObjective(discount=boundary_discount(gamma_step=1., gamma_terminated=0., gamma_truncated=0.), reward=None, value=None, gate=None,
                        temperature=temperature)


def _data():
    return {"action": torch.zeros(2, dtype=torch.int64), "reward": torch.zeros(2),
            "terminated": torch.zeros(2, dtype=torch.bool), "truncated": torch.zeros(2, dtype=torch.bool)}


@pytest.mark.parametrize("temperature", [0., 1.5])
def test_backup_heads_choose_and_evaluate_independently_of_online_predictions(temperature):
    online = torch.tensor([[2., 0., 0.], [8., 0., 0.]], requires_grad=True)
    values = torch.tensor([[0., 0., 0.], [9., 3., 1.]], requires_grad=True)
    actions = torch.tensor([[0., 0., 0.], [0., 6., 0.]], requires_grad=True)
    online = torch.cat([online.detach(), torch.zeros(1, 3)]).requires_grad_()
    values = torch.cat([values.detach(), torch.zeros(1, 3)]).requires_grad_()
    actions = torch.cat([actions.detach(), torch.zeros(1, 3)]).requires_grad_()
    data = {"action": torch.tensor([float("nan"), 0., 0.]), "reward": torch.zeros(3),
            "terminated": torch.tensor([False, False, True]), "truncated": torch.zeros(3, dtype=torch.bool)}
    losses, metrics = _objective(temperature)(
        objective_data=data, sequence_ids=torch.zeros(3, dtype=torch.int64), step_ids=torch.arange(3),
        predictions=online, backup_value_predictions=values,
        backup_action_predictions=actions,
    )
    if temperature == 0:
        expected = torch.tensor(3.)
    else:
        probabilities = torch.softmax(actions[1].detach() / temperature, dim=0)
        expected = (probabilities * values[1].detach()).sum() - temperature * (probabilities * probabilities.log()).sum()
    torch.testing.assert_close(losses["action_value"], ((expected - 2).square() + 8**2)/2)
    backup = metrics["backup"]
    assert isinstance(backup, torch.Tensor)
    torch.testing.assert_close(backup[0], expected)
    losses["action_value"].backward()
    assert values.grad is None and actions.grad is None
    assert online.grad is not None and online.grad[2].count_nonzero() == 0


@pytest.mark.parametrize("name", ["backup_value_predictions", "backup_action_predictions"])
@pytest.mark.parametrize("bad", [torch.zeros(2), torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.float64)])
def test_backup_inputs_validate_shape_and_dtype(name, bad):
    inputs = {"backup_value_predictions": torch.zeros(2, 2), "backup_action_predictions": torch.zeros(2, 2)}
    inputs[name] = bad
    with pytest.raises((TypeError, ValueError), match=name):
        _objective()(objective_data=_data(), sequence_ids=torch.zeros(2, dtype=torch.int64), step_ids=fixture_head_steps(_data()),  predictions=torch.zeros(2, 2), **inputs)
