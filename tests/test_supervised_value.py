from __future__ import annotations

from typing import Literal

import pytest
import torch

from mouse_core.objectives import SupervisedObjective


@pytest.mark.parametrize(("loss_type", "expected"), [("mse", 2.5), ("mae", 1.5)])
def test_supervised_value_objective_uses_only_finite_targets(loss_type, expected) -> None:
    predictions = torch.tensor([[[1.0, 3.0, float("nan")]]], requires_grad=True)
    loss, metrics = SupervisedObjective(loss_type=loss_type, oracle_temperature=None)(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[[0.0, 1.0, -torch.inf]]]),
    )
    assert loss["supervised"].item() == expected
    assert metrics["supervised"] == expected
    loss["supervised"].backward()
    assert predictions.grad is not None
    assert predictions.grad[0, 0, 2] == 0


@pytest.mark.parametrize("shape", [(2, 3, 2), (6, 2), (3, 4)])
def test_supervised_value_objective_rejects_misaligned_targets_with_same_element_count(shape) -> None:
    with pytest.raises(ValueError, match="same shape"):
        SupervisedObjective(loss_type="mse", oracle_temperature=None)(
            objective_data={},
            predictions=torch.zeros(3, 2, 2),
            targets=torch.zeros(shape),
        )


def test_supervised_value_default_three_dimensional_input_keeps_scalar_loss() -> None:
    predictions = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    loss, metrics = SupervisedObjective(loss_type="mse", oracle_temperature=None)(objective_data={}, predictions=predictions, targets=targets)
    assert loss["supervised"].ndim == 0
    assert isinstance(metrics["supervised"], torch.Tensor)
    loss["supervised"].backward()
    assert predictions.grad is not None


@pytest.mark.parametrize("loss_type", ["mse", "mae"])
@pytest.mark.parametrize("per_head_targets", [False, True])
def test_supervised_value_batched_heads_match_independent_calls(
    loss_type: Literal["mse", "mae"], per_head_targets: bool,
) -> None:
    torch.manual_seed(43)
    predictions = torch.randn(4, 3, 4, requires_grad=True)
    shared_q = torch.tensor([
        [0.0, 1.0, -torch.inf, 0.5],
        [2.0, -1.0, torch.nan, -torch.inf],
        [1.0, 0.0, -torch.inf, -2.0],
        [0.0, 0.0, 0.0, -torch.inf],
    ])
    q = shared_q[:, None, :].expand(-1, 3, -1).clone() if per_head_targets else shared_q
    if per_head_targets:
        q[:, 0, 0] += 0.4
        q[:, 1, 1] -= 0.8
        q[2, 2, 1] = torch.nan
    batched_objective = SupervisedObjective(loss_type=loss_type, oracle_temperature=None, batched_heads=True)
    single_objective = SupervisedObjective(loss_type=loss_type, oracle_temperature=None)
    losses, metrics = batched_objective(
        objective_data={}, predictions=predictions, targets=q,
    )
    assert losses["supervised"].shape == (3,)
    assert isinstance(metrics["supervised"], torch.Tensor)
    torch.testing.assert_close(metrics["supervised"], losses["supervised"].detach())
    losses["supervised"].sum().backward()
    batched_grad = predictions.grad
    assert batched_grad is not None
    for head in range(3):
        single = predictions.detach()[:, head].requires_grad_()
        single_q = q[:, head] if per_head_targets else q
        expected, _ = single_objective(
            objective_data={}, predictions=single, targets=single_q,
        )
        expected["supervised"].backward()
        single_grad = single.grad
        assert single_grad is not None
        torch.testing.assert_close(losses["supervised"][head], expected["supervised"])
        torch.testing.assert_close(batched_grad[:, head], single_grad)
    assert torch.isfinite(batched_grad).all()


def test_supervised_value_batched_heads_reject_misaligned_targets() -> None:
    with pytest.raises(ValueError, match="same shape"):
        SupervisedObjective(loss_type="mse", oracle_temperature=None, batched_heads=True)(
            objective_data={}, predictions=torch.zeros(2, 3, 4),
            targets=torch.zeros(2, 4, 3),
        )


def test_supervised_value_batched_heads_reject_head_without_finite_targets() -> None:
    targets = torch.zeros(2, 3, 4)
    targets[:, 1] = -torch.inf
    with pytest.raises(ValueError, match="one head"):
        SupervisedObjective(loss_type="mse", oracle_temperature=None, batched_heads=True)(
            objective_data={}, predictions=torch.zeros(2, 3, 4),
            targets=targets,
        )
