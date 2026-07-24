from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import SpObjective

def test_sp_objective_allows_negative_infinity_action_padding() -> None:
    objective_data = TensorDict({'info_q_star': torch.tensor([[[0.0, 1.0, -torch.inf]]])}, batch_size=(1, 1))
    predictions = TensorDict({'action': torch.tensor([[[0.0, 1.0, 100.0]]])}, batch_size=(1, 1))
    loss, metrics = SpObjective(loss_type='ce')(objective_data, predictions)
    assert loss.ndim == 0
    assert metrics['action'] >= 0.0

def test_sp_objective_skips_rows_with_no_finite_action_targets() -> None:
    objective_data = TensorDict({'info_q_star': torch.tensor([[[-torch.inf, -torch.inf], [0.0, 1.0]]])}, batch_size=(1, 2))
    predictions = TensorDict({'action': torch.tensor([[[100.0, 0.0], [0.0, 100.0]]])}, batch_size=(1, 2))
    loss, _ = SpObjective(loss_type='ce')(objective_data, predictions)
    assert loss.item() < 0.0001

def test_sp_objective_soft_losses_finite_with_padded_actions() -> None:
    """-inf padding sentinels must not blow up any soft loss or its gradient."""
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4)
    q[..., -1] = -torch.inf
    objective_data = TensorDict({'info_q_star': q}, batch_size=(2, 3))
    for loss_type in ('kl-fwd', 'kl-bwd', 'ce-soft-fwd', 'ce-soft-bwd', 'js'):
        logits = torch.randn(2, 3, 4, requires_grad=True)
        predictions = TensorDict({'action': logits}, batch_size=(2, 3))
        loss, _ = SpObjective(loss_type=loss_type, label_smoothing=0.1)(objective_data, predictions)
        loss.backward()
        assert torch.isfinite(loss), loss_type
        assert logits.grad is not None and torch.isfinite(logits.grad).all(), loss_type
        assert logits.grad[..., -1].abs().max() == 0.0, loss_type

def test_sp_objective_soft_losses_ignore_padded_student_logits() -> None:
    """A junk student logit at a padded slot must not affect the loss."""
    q = torch.tensor([[[1.0, 2.0, 3.0, -torch.inf]]])
    objective_data = TensorDict({'info_q_star': q}, batch_size=(1, 1))
    matching = TensorDict({'action': torch.tensor([[[1.0, 2.0, 3.0, 100.0]]])}, batch_size=(1, 1))
    for direction in ('kl-fwd', 'kl-bwd'):
        loss, _ = SpObjective(loss_type=direction)(objective_data, matching)
        assert loss.item() < 1e-06, direction

def test_sp_objective_custom_targets_key() -> None:
    objective_data = TensorDict({'action_value': torch.tensor([[[0.0, 2.0]]])}, batch_size=(1, 1))
    predictions = TensorDict({'action': torch.tensor([[[0.0, 1.0]]])}, batch_size=(1, 1))
    loss, _ = SpObjective(loss_type='ce', targets_key='action_value')(objective_data, predictions)
    assert loss.item() > 0.0
