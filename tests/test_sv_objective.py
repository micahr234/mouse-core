from __future__ import annotations

import pytest
import torch

from mouse_core.objectives import SvObjective


@pytest.mark.parametrize(("loss_type", "expected"), [("mse", 2.5), ("mae", 1.5)])
def test_sv_objective_uses_only_finite_targets(loss_type, expected) -> None:
    predictions = torch.tensor([[[1.0, 3.0, float("nan")]]], requires_grad=True)
    loss, metrics = SvObjective(loss_type=loss_type)(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[[0.0, 1.0, -torch.inf]]]),
    )
    assert loss["value"].item() == expected
    assert metrics["value"] == expected
    loss["value"].backward()
    assert predictions.grad is not None
    assert predictions.grad[0, 0, 2] == 0


@pytest.mark.parametrize("shape", [(2, 3, 2), (6, 2), (3, 4)])
def test_sv_objective_rejects_misaligned_targets_with_same_element_count(shape) -> None:
    with pytest.raises(ValueError, match="same shape"):
        SvObjective()(
            objective_data={},
            predictions=torch.zeros(3, 2, 2),
            targets=torch.zeros(shape),
        )
