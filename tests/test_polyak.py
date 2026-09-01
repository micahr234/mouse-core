from __future__ import annotations

import pytest
import torch

from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import (
    DiscreteActionHead,
    DiscreteActionValueHead,
    LayerwiseDiscreteActionValueHead,
)
from mouse_core.polyak import PolyakAverager
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder

_MODALITIES = [
    {"type": "discrete", "field": "action", "vocab_size": 4},
    {"type": "fourier", "field": "reward"},
    {"type": "discrete", "field": "episode_done", "vocab_size": 3},
]
_BATCH = [
    [
        {"action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0},
        {"action": 1, "reward": 1.0, "episode_done": 0, "task_done": 0},
        {"action": 2, "reward": 2.0, "episode_done": 1, "task_done": 0},
    ]
]


def _tiny_model() -> Model:
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionValueHead(
        in_features=hidden_dim,
        out_features=4,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    return Model(encoder=encoder, backbone=backbone, heads=heads)


def _token_batch(model: Model):
    return batch_to_token_batch(_tok(model.encoder), _BATCH)


def _perturb_encoder(model: Model) -> None:
    with torch.no_grad():
        for param in model.encoder.parameters():
            param.add_(1.0)


def _q_and_target(model: Model, averager: PolyakAverager):
    batch = _token_batch(model)
    predictions, _, _ = model(batch)
    averager.write_targets(batch, predictions)
    return predictions


def test_head_scope_target_uses_current_features() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="head", tau=1.0)
    averager.update()
    before = _q_and_target(model, averager)
    _perturb_encoder(model)
    after = _q_and_target(model, averager)
    assert not torch.allclose(after["action_value_target"], before["action_value_target"])
    assert torch.allclose(after["action_value"], after["action_value_target"])


def test_model_scope_recomputes_delayed_representation() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="model", tau=1.0)
    averager.update()
    for online_p, delayed_p in zip(
        model.encoder.parameters(), averager.delayed.encoder.parameters(), strict=True
    ):
        assert torch.equal(online_p, delayed_p)
    before = _q_and_target(model, averager)
    assert torch.allclose(before["action_value"], before["action_value_target"])
    _perturb_encoder(model)
    after = _q_and_target(model, averager)
    assert not torch.allclose(after["action_value"], before["action_value"])
    assert torch.allclose(after["action_value_target"], before["action_value_target"])
    assert not torch.allclose(after["action_value"], after["action_value_target"])


def test_polyak_tau_is_convex_combination() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    averager = PolyakAverager(model, scope="model", tau=0.5)
    online = next(model.encoder.parameters())
    delayed = next(averager.delayed.encoder.parameters())
    online.data.fill_(1.0)
    delayed.data.fill_(0.0)
    averager.update()
    assert torch.allclose(delayed, torch.full_like(delayed, 0.5))


def test_averager_without_dqn_head_raises() -> None:
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = DiscreteActionHead(
        in_features=hidden_dim,
        out_features=4,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=heads)
    with pytest.raises(ValueError, match="DQN-style"):
        PolyakAverager(model, scope="model")


def test_invalid_scope_raises() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="scope"):
        PolyakAverager(model, scope="encoder")  # type: ignore[arg-type]


def test_layerwise_model_scope_recomputes_delayed_layers() -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    heads = LayerwiseDiscreteActionValueHead(
        num_backbone_layers=1,
        in_features=hidden_dim,
        out_features=4,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=heads).eval()
    averager = PolyakAverager(model, scope="model", tau=1.0)
    averager.update()
    before = _q_and_target(model, averager)
    _perturb_encoder(model)
    after = _q_and_target(model, averager)
    assert not torch.allclose(
        after["action_value_layerwise"], before["action_value_layerwise"]
    )
    assert torch.allclose(
        after["action_value_layerwise_target"],
        before["action_value_layerwise_target"],
    )


def test_delayed_stays_eval() -> None:
    model = _tiny_model().train()
    averager = PolyakAverager(model, scope="head")
    assert model.training
    assert not averager.delayed.training
    model.train()
    averager.write_targets(_token_batch(model), model(_token_batch(model))[0])
    assert not averager.delayed.training
