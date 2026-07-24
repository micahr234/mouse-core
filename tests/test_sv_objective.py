from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import SvObjective

def test_sv_objective_default_targets_key() -> None:
    objective_data = TensorDict({'info_q_star': torch.tensor([[[1.0, 2.0]]])}, batch_size=(1, 1))
    predictions = TensorDict({'value': torch.tensor([[[1.0, 2.0]]])}, batch_size=(1, 1))
    loss, metrics = SvObjective(loss_type='mse')(objective_data, predictions)
    assert loss.item() == 0.0
    assert metrics['value'] == 0.0

def test_sv_objective_custom_targets_key() -> None:
    objective_data = TensorDict({'teacher_q': torch.tensor([[[0.0, 1.0]]])}, batch_size=(1, 1))
    predictions = TensorDict({'value': torch.tensor([[[0.5, 0.5]]])}, batch_size=(1, 1))
    loss, _ = SvObjective(loss_type='mse', targets_key='teacher_q')(objective_data, predictions)
    assert loss.item() > 0.0
