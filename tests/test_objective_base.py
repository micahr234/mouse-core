"""Shared objective contracts without imposing one concrete call signature."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import assert_type

import pytest
import torch

from mouse_core.objectives import (
    DqnObjective,
    GrpoObjective,
    Objective,
    PpoObjective,
    SupervisedObjective,
)


type _Result = tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]


class _ForwardObjective[**P](Objective[P]):
    def __init__(self, *, objective: Callable[P, _Result]) -> None:
        self.objective = objective

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> _Result:
        return self.objective(*args, **kwargs)


def test_objective_base_is_abstract() -> None:
    assert inspect.isabstract(Objective)
    assert "__call__" in Objective.__abstractmethods__


def test_objective_can_preserve_a_specific_keyword_signature() -> None:
    objective = _ForwardObjective(objective=SupervisedObjective(mask_key=None))
    result = objective(
        objective_data={},
        predictions=torch.tensor([[0.0, 1.0]], requires_grad=True),
        targets=torch.tensor([[0.0, 1.0]]),
    )
    assert_type(result, _Result)
    losses, metrics = result
    loss = losses["supervised"]
    assert loss.ndim == 0
    assert loss.requires_grad
    assert "supervised" in metrics


_CASES: list[tuple[Objective[...], dict[str, object], tuple[str, ...]]] = [
    (
        DqnObjective(
            discount=None, reward=None, value=None, gate=None,
            temperature=0.0, cross_group_backups="bootstrap",
        ),
        {
            "group_id": torch.zeros(2, dtype=torch.int64),
            "backup_value_predictions": torch.zeros(2, 2),
            "backup_action_predictions": torch.zeros(2, 2),
        },
        ("group_id", "backup_value_predictions", "backup_action_predictions"),
    ),
    (
        PpoObjective(
            discount=None, reward=None, value=None, cross_group_backups="bootstrap",
        ),
        {
            "group_id": torch.zeros(2, dtype=torch.int64),
            "value_predictions": torch.zeros(2),
        },
        ("group_id", "value_predictions"),
    ),
    (
        GrpoObjective(),
        {"group_id": torch.zeros(2, dtype=torch.int64)},
        ("group_id",),
    ),
    (SupervisedObjective(mask_key=None), {"targets": torch.zeros(2, 2)}, ("targets",)),
    (SupervisedObjective(loss_type="mse", target_temperature=None), {"targets": torch.zeros(2, 2)}, ("targets",)),
    (SupervisedObjective(loss_type="ce", target_temperature=None), {"actions": torch.zeros(2, dtype=torch.int64)}, ("actions",)),
]


@pytest.mark.parametrize(("objective", "extra_inputs", "required"), _CASES)
def test_objectives_still_require_their_specific_inputs(
    *,
    objective: Objective[...],
    extra_inputs: dict[str, object],
    required: tuple[str, ...],
) -> None:
    # The erased signature is intentional here: exercise invalid runtime calls.
    inputs: dict[str, object] = {
        "objective_data": {},
        "predictions": torch.zeros(2, 2),
        **extra_inputs,
    }
    for name in ("objective_data", "predictions", *required):
        missing = {key: value for key, value in inputs.items() if key != name}
        with pytest.raises(TypeError, match=name):
            objective(**missing)


@pytest.mark.parametrize(("objective", "extra_inputs", "required"), _CASES)
def test_objectives_still_reject_unused_prediction_tensors(
    *,
    objective: Objective[...],
    extra_inputs: dict[str, object],
    required: tuple[str, ...],
) -> None:
    inputs: dict[str, object] = {
        "objective_data": {},
        "predictions": torch.zeros(2, 2),
        **extra_inputs,
    }
    for name in ("backup_value_predictions", "backup_action_predictions", "value_predictions", "targets", "actions"):
        if name not in required:
            with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
                objective(**{**inputs, name: torch.zeros(2, 2)})
