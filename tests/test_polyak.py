from __future__ import annotations

import pytest
import torch

from mouse_core.models import AveragerInputs, Model
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
    predictions, averager_inputs = model(batch)
    return predictions, averager(averager_inputs)


def test_head_scope_target_uses_current_features() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="head", tau=1.0)
    averager.update()
    before_pred, before_delayed = _q_and_target(model, averager)
    _perturb_encoder(model)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert not torch.allclose(after_delayed["action_value"], before_delayed["action_value"])
    assert torch.allclose(after_pred["action_value"], after_delayed["action_value"])


def test_model_scope_recomputes_delayed_representation() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="model", tau=1.0)
    averager.update()
    for online_p, delayed_p in zip(
        model.encoder.parameters(), averager.model.encoder.parameters(), strict=True
    ):
        assert torch.equal(online_p, delayed_p)
    before_pred, before_delayed = _q_and_target(model, averager)
    assert torch.allclose(before_pred["action_value"], before_delayed["action_value"])
    _perturb_encoder(model)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert not torch.allclose(after_pred["action_value"], before_pred["action_value"])
    assert torch.allclose(after_delayed["action_value"], before_delayed["action_value"])
    assert not torch.allclose(after_pred["action_value"], after_delayed["action_value"])


def test_polyak_tau_is_convex_combination() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    averager = PolyakAverager(model, scope="model", tau=0.5)
    online = next(model.encoder.parameters())
    delayed = next(averager.model.encoder.parameters())
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
    before_pred, before_delayed = _q_and_target(model, averager)
    _perturb_encoder(model)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert not torch.allclose(
        after_pred["action_value_layerwise"], before_pred["action_value_layerwise"]
    )
    assert torch.allclose(
        after_delayed["action_value_layerwise"],
        before_delayed["action_value_layerwise"],
    )


def test_online_and_delayed_run_in_train() -> None:
    model = _tiny_model().train()
    averager = PolyakAverager(model, scope="head")
    assert model.training
    assert averager.model.training
    averager.model.eval()
    predictions, averager_inputs = model(_token_batch(model))
    averager(averager_inputs)
    assert model.training
    assert averager.model.training


def test_head_scope_does_not_rerun_encoder() -> None:
    model = _tiny_model().train()
    averager = PolyakAverager(model, scope="head")
    enc_calls = 0
    orig_enc = model.encoder.forward

    def _enc(*args, **kwargs):
        nonlocal enc_calls
        enc_calls += 1
        return orig_enc(*args, **kwargs)

    model.encoder.forward = _enc  # type: ignore[method-assign]
    batch = _token_batch(model)
    _, averager_inputs = model(batch)
    assert enc_calls == 1
    averager(averager_inputs)
    assert enc_calls == 1


def test_head_scope_does_not_copy_encoder_backbone() -> None:
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="head")
    assert not hasattr(averager.model, "encoder")
    assert not hasattr(averager.model, "backbone")
    delayed_names = {n for n, _ in averager.model.heads.named_parameters()}
    online_head_names = {n for n, _ in model.heads.named_parameters()}
    assert delayed_names == online_head_names
    _q_and_target(model, averager)


def test_model_scope_runs_delayed_encoder_backbone() -> None:
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="model")
    enc_calls = 0
    orig_enc = averager.model.encoder.forward

    def _enc(*args, **kwargs):
        nonlocal enc_calls
        enc_calls += 1
        return orig_enc(*args, **kwargs)

    averager.model.encoder.forward = _enc  # type: ignore[method-assign]
    _q_and_target(model, averager)
    assert enc_calls == 1


def test_averager_rejects_non_averager_inputs() -> None:
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="head")
    with pytest.raises(TypeError, match="AveragerInputs"):
        averager(_token_batch(model))  # type: ignore[arg-type]


def test_head_scope_uses_passed_inputs_not_later_forward() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, scope="head", tau=1.0)
    averager.update()
    batch = _token_batch(model)
    first_pred, first_inputs = model(batch)
    first_delayed = averager(first_inputs)
    _perturb_encoder(model)
    model(batch)
    replayed = averager(first_inputs)
    assert torch.allclose(replayed["action_value"], first_delayed["action_value"])
    assert torch.allclose(first_pred["action_value"], first_delayed["action_value"])


def test_forward_returns_averager_inputs() -> None:
    model = _tiny_model().eval()
    batch = _token_batch(model)
    predictions, averager_inputs = model(batch)
    assert isinstance(averager_inputs, AveragerInputs)
    assert averager_inputs.batch is batch
    assert averager_inputs.h is not None
    assert averager_inputs.cache is None
    assert predictions["action_value"].shape[0] == averager_inputs.h.shape[0]


def test_averager_inputs_h_is_detached() -> None:
    model = _tiny_model().train()
    predictions, averager_inputs = model(_token_batch(model))
    assert predictions["action_value"].requires_grad
    assert not averager_inputs.h.requires_grad
    assert averager_inputs.h.grad_fn is None
    live = predictions["action_value"].detach().clone().requires_grad_(True)
    wrapped = AveragerInputs(h=live, batch=averager_inputs.batch)
    assert not wrapped.h.requires_grad
    averager = PolyakAverager(model, scope="head")
    delayed = averager(averager_inputs)
    assert not delayed["action_value"].requires_grad
