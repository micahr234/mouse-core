"""Tests for LayerwiseDiscreteActionValueHead and Model integration."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.models.backbone import Qwen3Backbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import LayerwiseDiscreteActionValueHead
from mouse_core.models.base import Model
from mouse_core.objectives import LayerwiseDqnObjective


def _tiny_batch() -> list[list[dict]]:
    return [
        [
            {"action": 0, "observation": 1, "reward": 0.0, "done": 0},
            {"action": 1, "observation": 2, "reward": 1.0, "done": 0},
            {"action": 0, "observation": 3, "reward": 0.5, "done": 0},
        ]
    ]


def test_layerwise_head_forward_shape() -> None:
    head = LayerwiseDiscreteActionValueHead(
        num_backbone_layers=2,
        in_features=8,
        out_features=4,
        hidden_dim=8,
        num_layers=1,
        scale=0.1,
    )
    h = torch.randn(1, 2, 3, 8)
    q = head.forward(h)
    q_target = head.target_forward(h)
    assert q.shape == (1, 3, 2, 4)
    assert q_target.shape == (1, 3, 2, 4)


def test_model_layerwise_forward_and_objective() -> None:
    backbone = Qwen3Backbone(hidden_dim=16, num_layers=2, num_heads=2)
    encoder = NumericEmbedder(
        hidden_dim=backbone.hidden_dim,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4, "std": 0.02},
            {"field": "observation", "type": "discrete", "vocab_size": 8, "std": 0.02},
            {"field": "reward", "type": "rff", "std": 0.02, "in_min": 0.01, "in_max": 100.0},
            {"field": "done", "type": "discrete", "vocab_size": 5, "std": 0.02},
        ],
    )
    head = LayerwiseDiscreteActionValueHead(
        num_backbone_layers=2,
        in_features=backbone.hidden_dim,
        out_features=4,
        hidden_dim=backbone.hidden_dim,
        num_layers=1,
        scale=0.1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=head)
    batch = _tiny_batch()
    predictions, objective_data, _ = model(batch)
    assert "action_value_layerwise" in predictions.keys()
    assert predictions["action_value_layerwise"].shape[-2:] == (2, 4)

    objective = LayerwiseDqnObjective(
        num_backbone_layers=2,
        gamma_step_start=0.0,
        gamma_step=0.99,
        tau=0.1,
    )
    loss, metrics = objective(objective_data, predictions)
    assert loss.ndim == 0
    assert metrics["action_value_layerwise"] >= 0.0

    action = model.get_action(predictions, temperature=0.0, num_actions=4)
    assert action.shape == (1,)

    model.polyak_update(action_value_layerwise_tau=objective.tau)


def test_layerwise_objective_q_metrics_use_curr_max_q() -> None:
    """q_values_mean and layer_q_mean report max online Q at the current state."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([[0, 1, 0]]),
            "reward": torch.tensor([[0.0, 1.0, 5.0]]),
            "done": torch.tensor([[0, 0, 0]]),
        },
        batch_size=(1, 3),
    )
    out = TensorDict(
        {
            "action_value_layerwise": torch.tensor(
                [[[[0.0, 2.0], [3.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]]
            ),
            "action_value_layerwise_target": torch.zeros(1, 3, 2, 2),
        },
        batch_size=(1, 3),
    )

    _, metrics = LayerwiseDqnObjective(
        num_backbone_layers=2,
        gamma_step_start=0.0,
        gamma_step=0.0,
    )(step_stream, out)

    # Deepest layer (L-1): max online Q at s_0 is 3, at s_1 is 0 → mean 1.5.
    assert abs(metrics["q_values_mean"] - 1.5) < 1e-5
    assert abs(metrics["layer_0_q_mean"] - 1.0) < 1e-5
    assert abs(metrics["layer_1_q_mean"] - 1.5) < 1e-5
