"""Prediction-to-step alignment uses the TokenBatch map, never duplicate counts."""
import pytest
import torch
from mouse_core.objectives import DqnObjective, boundary_discount
from mouse_core.objectives.dqn import _head_output_layout


def test_same_number_of_predictions_and_steps_does_not_imply_identity():
    # Two readouts at the initial decision, one at the next, none at terminal.
    steps = torch.tensor([0, 0, 1])
    data = dict(action=torch.tensor([float("nan"), 1., 0.]),
                reward=torch.tensor([0., 2., 3.]),
                terminated=torch.tensor([False, False, True]),
                truncated=torch.zeros(3, dtype=torch.bool))
    predictions = torch.tensor([[0., 1.], [0., 4.], [2., 0.]], requires_grad=True)
    backup = torch.tensor([[99., 99.], [99., 99.], [6., 8.]])
    objective = DqnObjective(reward=None, value=None, gate=None, temperature=0.,
                            discount=boundary_discount(gamma_step=.5, gamma_terminated=0., gamma_truncated=0.))
    losses, metrics = objective(objective_data=data, group_id=torch.zeros(3, dtype=torch.long),
                                head_step_ids=steps, predictions=predictions,
                                backup_value_predictions=backup, backup_action_predictions=backup)
    # Both initial readouts target 2 + .5*8 = 6; next readout targets terminal reward 3.
    torch.testing.assert_close(metrics["backup"], torch.tensor([6., 6., 3.]))
    torch.testing.assert_close(losses["action_value"], torch.tensor((25.+4.+1.)/3))
    losses["action_value"].backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[0., -10./3], [0., -4./3], [-2./3, 0.]]))
    assert set(data.keys()) == {"action", "reward", "terminated", "truncated"}


@pytest.mark.parametrize("mapping,exception,message", [
    (torch.tensor([0., 1.]), TypeError, "int64"),
    (torch.tensor([[0, 1]]), ValueError, "shape"),
    (torch.tensor([0]), ValueError, "shape"),
    (torch.tensor([-1, 1]), ValueError, "identify steps"),
    (torch.tensor([0, 3]), ValueError, "identify steps"),
    (torch.tensor([2, 1]), ValueError, "step order"),
])
def test_invalid_prediction_mapping_is_rejected(mapping, exception, message):
    with pytest.raises(exception, match=message):
        _head_output_layout(mapping, N=3, P=2, device="cpu")


def test_empty_mapping_preserves_steps_without_predictions():
    steps, last = _head_output_layout(torch.empty(0, dtype=torch.long), N=3, P=0, device="cpu")
    assert steps.numel() == 0
    assert last.tolist() == [0, 0, 0]
