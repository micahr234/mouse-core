from __future__ import annotations

import json

import torch

from mouse_core.models import Model, load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.base import Model as ModelClass
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionHead


def test_discrete_action_head_forward_shape() -> None:
    head = DiscreteActionHead(
        in_features=8,
        out_features=4,
        hidden_dim=8,
        num_layers=1,
    )
    out = head(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)


def test_infer_head_name_is_action() -> None:
    head = DiscreteActionHead(
        in_features=8,
        out_features=4,
        hidden_dim=8,
        num_layers=1,
    )
    assert ModelClass._infer_head_name(head) == "action"


def test_discrete_action_head_save_load_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = NumericEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
        ],
    )
    model = Model(
        encoder=encoder,
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=DiscreteActionHead(
            in_features=hidden_dim,
            out_features=4,
            hidden_dim=hidden_dim,
            num_layers=1,
        ),
    ).eval()

    batch = [[{"action": 0, "reward": 0.0}, {"action": 1, "reward": 1.0}]]
    expected, _, _ = model(batch)
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    actual, _, _ = loaded(batch)

    assert torch.allclose(actual["action"], expected["action"])
    assert loaded.action_head == "action"

    with (tmp_path / "config.json").open() as fh:
        config = json.load(fh)
    head_specs = config["heads"]["heads"]
    assert len(head_specs) == 1
    assert head_specs[0]["type"] == "discrete_action"
