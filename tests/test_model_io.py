from __future__ import annotations

import json

import torch
from tensordict import TensorDict

from mouse_core.models import Model, load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead


def test_composed_model_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = StepEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {"name": "action", "embed": "discrete", "vocab_size": 4},
            {"name": "reward", "embed": "rff"},
            {"name": "done", "embed": "discrete", "vocab_size": 3},
        ],
    )
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionValueHead(
        in_features=hidden_dim,
        out_features=4,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=heads).eval()

    batch = TensorDict(
        {
            "action": torch.tensor([[0, 1, 2]]),
            "reward": torch.tensor([[0.0, 1.0, 2.0]]),
            "done": torch.tensor([[0, 0, 1]]),
        },
        batch_size=(1, 3),
    )

    expected, _ = model(batch)
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    actual, _ = loaded(batch)

    assert torch.allclose(actual["action_value"], expected["action_value"])
    assert loaded.hidden_dim == hidden_dim

    with (tmp_path / "config.json").open() as fh:
        config = json.load(fh)
    assert config["format"] == "mouse-core-model-v1"
    assert config["backbone"]["type"] == "identity"
