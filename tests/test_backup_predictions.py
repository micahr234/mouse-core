"""Independent backup action/value inputs, including the soft policy target."""
import pytest
import torch

from mouse_core.objectives import DqnObjective


def _objective(temperature=0.):
    return DqnObjective(discount=None, reward=None, value=None, gate=None,
                        temperature=temperature, cross_group_backups="bootstrap")


def _data():
    return {"action": torch.zeros(2, dtype=torch.int64), "reward": torch.zeros(2),
            "terminated": torch.zeros(2, dtype=torch.bool), "truncated": torch.zeros(2, dtype=torch.bool)}


@pytest.mark.parametrize("temperature", [0., 1.5])
def test_backup_heads_choose_and_evaluate_independently_of_online_predictions(temperature):
    online = torch.tensor([[2., 0., 0.], [8., 0., 0.]], requires_grad=True)
    values = torch.tensor([[0., 0., 0.], [9., 3., 1.]], requires_grad=True)
    actions = torch.tensor([[0., 0., 0.], [0., 6., 0.]], requires_grad=True)
    losses, metrics = _objective(temperature)(
        objective_data=_data(), group_id=torch.zeros(2, dtype=torch.int64),
        predictions=online, backup_value_predictions=values,
        backup_action_predictions=actions,
    )
    if temperature == 0:
        expected = torch.tensor(3.)
    else:
        probabilities = torch.softmax(actions[1].detach() / temperature, dim=0)
        expected = (probabilities * values[1].detach()).sum() - temperature * (probabilities * probabilities.log()).sum()
    torch.testing.assert_close(losses["action_value"], (expected - 2).square())
    backup = metrics["backup"]
    assert isinstance(backup, torch.Tensor)
    torch.testing.assert_close(backup[0], expected)
    losses["action_value"].backward()
    assert values.grad is None and actions.grad is None
    assert online.grad is not None and online.grad[1].count_nonzero() == 0


@pytest.mark.parametrize("name", ["backup_value_predictions", "backup_action_predictions"])
@pytest.mark.parametrize("bad", [torch.zeros(2), torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.float64)])
def test_backup_inputs_validate_shape_and_dtype(name, bad):
    inputs = {"backup_value_predictions": torch.zeros(2, 2), "backup_action_predictions": torch.zeros(2, 2)}
    inputs[name] = bad
    with pytest.raises((TypeError, ValueError), match=name):
        _objective()(objective_data=_data(), group_id=torch.zeros(2, dtype=torch.int64), predictions=torch.zeros(2, 2), **inputs)
