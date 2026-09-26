from __future__ import annotations

import torch

from mouse_core.objectives import SvObjective


def test_sv_objective_mse_onto_targets() -> None:
    predictions = torch.tensor([[[1.0, 2.0]]])
    loss, metrics = SvObjective(loss_type="mse")(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[[1.0, 2.0]]]),
    )
    assert loss.item() == 0.0
    assert metrics["value"] == 0.0


def test_sv_objective_accepts_direct_q_targets() -> None:
    predictions = torch.tensor([[[0.5, 0.5]]])
    loss, _ = SvObjective(loss_type="mse")(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[[0.0, 1.0]]]),
    )
    assert loss.item() > 0.0


def test_sv_objective_requires_targets() -> None:
    predictions = torch.tensor([[[1.0, 2.0]]])
    try:
        SvObjective(loss_type="mse")(objective_data={}, predictions=predictions)
    except TypeError as exc:
        assert "targets" in str(exc)
    else:
        raise AssertionError("expected TypeError for missing targets")
