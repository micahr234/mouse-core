from __future__ import annotations

import json

import torch

from mouse_core.models import Model, load_model, save_model
from mouse_core.models.base import _write_model_card
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead


def test_composed_model_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = StepEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
            {"field": "done", "type": "discrete", "vocab_size": 3},
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

    batch = [
        [
            {"action": 0, "reward": 0.0, "done": 0},
            {"action": 1, "reward": 1.0, "done": 0},
            {"action": 2, "reward": 2.0, "done": 1},
        ]
    ]

    expected, _, _ = model(batch)
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    actual, _, _ = loaded(batch)

    assert torch.allclose(actual["action_value"], expected["action_value"])
    assert loaded.hidden_dim == hidden_dim

    with (tmp_path / "config.json").open() as fh:
        config = json.load(fh)
    assert config["format"] == "mouse-core-model-v1"
    assert config["backbone"]["type"] == "identity"


def test_model_card_includes_usage_and_architecture(tmp_path) -> None:
    model = Model(
        encoder=StepEmbedder(
            hidden_dim=8,
            modalities=[
                {"field": "action", "type": "discrete", "vocab_size": 4},
                {"field": "reward", "type": "rff"},
                {"field": "done", "type": "discrete", "vocab_size": 3},
            ],
        ),
        backbone=IdentityBackbone(hidden_dim=8),
        heads=DiscreteActionValueHead(
            in_features=8,
            out_features=4,
            hidden_dim=8,
            num_layers=1,
        ),
    )
    path = tmp_path / "README.md"

    _write_model_card(model, path, repo_id="user/mouse-example-model")

    text = path.read_text()
    assert "library_name: mouse-core" in text
    assert text.index("## Architecture") < text.index("### Encoder")
    assert text.index("### Encoder") < text.index("## Install MouseCore")
    assert text.index("## Install MouseCore") < text.index("## Load The Model")
    assert text.index("## Load The Model") < text.index("## Run Inference")
    assert "What This Contains" not in text
    assert "pip install mouse-core" in text
    assert 'load_model("user/mouse-example-model"' in text
    assert "list[list[dict]]" in text
    assert "| `action` | `discrete` | yes | `[B, S]` | `torch.long` | integer ids in `[0, 3]` |" in text
    assert '"action": 0,' in text
    assert '"reward": 0.0,' in text
    assert "out, step_stream, cache = model(batch)" not in text
    assert "predictions, objective_data, cache = model(batch)" in text
    assert "Backbone: `identity`" in text
    assert "Heads: `action_value`" in text
