from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import SpObjective


def test_sp_objective_allows_negative_infinity_action_padding() -> None:
    objective_data = TensorDict(
        {"info_q_star": torch.tensor([[[0.0, 1.0, -torch.inf]]])},
        batch_size=(1, 1),
    )
    predictions = TensorDict(
        {"action": torch.tensor([[[0.0, 1.0, 100.0]]])},
        batch_size=(1, 1),
    )

    loss, metrics = SpObjective(loss_type="ce")(objective_data, predictions)

    assert loss.ndim == 0
    assert metrics["action"] >= 0.0


def test_sp_objective_skips_rows_with_no_finite_action_targets() -> None:
    objective_data = TensorDict(
        {"info_q_star": torch.tensor([[[-torch.inf, -torch.inf], [0.0, 1.0]]])},
        batch_size=(1, 2),
    )
    predictions = TensorDict(
        {"action": torch.tensor([[[100.0, 0.0], [0.0, 100.0]]])},
        batch_size=(1, 2),
    )

    loss, _ = SpObjective(loss_type="ce")(objective_data, predictions)

    assert loss.item() < 1.0e-4
