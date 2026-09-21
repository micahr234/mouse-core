from __future__ import annotations
import torch
from mouse_core.objectives import SvObjective

def test_sv_objective_default_targets_key() -> None:
    objective_data = {'info_q_star': torch.tensor([[[1.0, 2.0]]])}
    predictions = torch.tensor([[[1.0, 2.0]]])
    loss, metrics = SvObjective(loss_type="mse")(objective_data=objective_data, predictions=predictions)
    assert loss.item() == 0.0
    assert metrics['value'] == 0.0

def test_sv_objective_custom_targets_key() -> None:
    objective_data = {'teacher_q': torch.tensor([[[0.0, 1.0]]])}
    predictions = torch.tensor([[[0.5, 0.5]]])
    loss, _ = SvObjective(loss_type="mse", targets_key="teacher_q")(objective_data=objective_data, predictions=predictions)
    assert loss.item() > 0.0
