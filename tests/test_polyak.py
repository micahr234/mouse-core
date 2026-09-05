from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

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


class _ScaleBackbone(IdentityBackbone):
    """Identity plus a learned scale, so backbone delay is observable."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(hidden_dim=hidden_dim)
        self.scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        embeds: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        out = embeds * self.scale
        if output_hidden_states:
            return out, (out,)
        return out


def _tiny_model(*, scaled_backbone: bool = False) -> Model:
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone: IdentityBackbone
    if scaled_backbone:
        backbone = _ScaleBackbone(hidden_dim=hidden_dim)
    else:
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


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)


def _q_and_target(model: Model, averager: PolyakAverager):
    batch = _token_batch(model)
    predictions, averager_inputs = model(batch)
    return predictions, averager(averager_inputs)


def _count_calls(module: nn.Module, name: str = "forward"):
    orig = getattr(module, name)
    calls = {"n": 0}

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    setattr(module, name, _wrapped)
    return calls


def test_head_delay_target_uses_current_features() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, tau_head=1.0)
    averager.update()
    before_pred, before_delayed = _q_and_target(model, averager)
    _perturb(model.encoder)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert not torch.allclose(after_delayed["action_value"], before_delayed["action_value"])
    assert torch.allclose(after_pred["action_value"], after_delayed["action_value"])


def test_encoder_delay_recomputes_delayed_representation() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(
        model, tau_encoder=1.0, tau_backbone=1.0, tau_head=1.0
    )
    averager.update()
    assert averager.encoder is not None
    for online_p, delayed_p in zip(
        model.encoder.parameters(), averager.encoder.parameters(), strict=True
    ):
        assert torch.equal(online_p, delayed_p)
    before_pred, before_delayed = _q_and_target(model, averager)
    assert torch.allclose(before_pred["action_value"], before_delayed["action_value"])
    _perturb(model.encoder)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert not torch.allclose(after_pred["action_value"], before_pred["action_value"])
    assert torch.allclose(after_delayed["action_value"], before_delayed["action_value"])
    assert not torch.allclose(after_pred["action_value"], after_delayed["action_value"])


def test_polyak_tau_is_convex_combination() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    averager = PolyakAverager(
        model, tau_encoder=0.5, tau_backbone=0.5, tau_head=0.5
    )
    assert averager.encoder is not None
    online = next(model.encoder.parameters())
    delayed = next(averager.encoder.parameters())
    online.data.fill_(1.0)
    delayed.data.fill_(0.0)
    averager.update()
    assert torch.allclose(delayed, torch.full_like(delayed, 0.5))


def test_polyak_taus_update_independently() -> None:
    torch.manual_seed(0)
    model = _tiny_model(scaled_backbone=True)
    averager = PolyakAverager(
        model, tau_encoder=1.0, tau_backbone=0.5, tau_head=0.0
    )
    assert averager.encoder is not None
    assert averager.backbone is not None
    assert averager.heads is None
    next(model.encoder.parameters()).data.fill_(1.0)
    next(averager.encoder.parameters()).data.fill_(0.0)
    model.backbone.scale.data.fill_(1.0)
    averager.backbone.scale.data.fill_(0.0)
    averager.update()
    assert torch.allclose(
        next(averager.encoder.parameters()),
        torch.ones_like(next(averager.encoder.parameters())),
    )
    assert torch.allclose(averager.backbone.scale, torch.full_like(averager.backbone.scale, 0.5))


def test_polyak_small_tau_accumulates_in_bf16() -> None:
    torch.manual_seed(0)
    model = _tiny_model().to(dtype=torch.bfloat16)
    online = next(model.encoder.parameters())
    online.data.fill_(0.9)
    tau = 0.0005
    averager = PolyakAverager(model, tau_encoder=tau, tau_backbone=tau, tau_head=tau)
    assert averager.encoder is not None
    delayed = next(averager.encoder.parameters())
    assert delayed.dtype == torch.bfloat16
    online.data.fill_(1.0)
    steps = 2000
    for _ in range(steps):
        averager.update()
    # Direct bf16 lerp would stay at ~0.898 forever; fp32 accumulation converges.
    expected = 1.0 - 0.1 * (1.0 - tau) ** steps
    assert torch.allclose(
        delayed.float(), torch.full_like(delayed.float(), expected), atol=1e-2
    )


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
        PolyakAverager(model, tau_encoder=0.01)


def test_layerwise_encoder_delay_recomputes_delayed_layers() -> None:
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
    averager = PolyakAverager(
        model, tau_encoder=1.0, tau_backbone=1.0, tau_head=1.0
    )
    averager.update()
    before_pred, before_delayed = _q_and_target(model, averager)
    _perturb(model.encoder)
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
    averager = PolyakAverager(model, tau_head=0.01)
    assert model.training
    assert averager.heads is not None
    assert averager.heads.training
    averager.heads.eval()
    predictions, averager_inputs = model(_token_batch(model))
    averager(averager_inputs)
    assert model.training
    assert averager.heads.training


def test_head_delay_does_not_rerun_encoder() -> None:
    model = _tiny_model().train()
    averager = PolyakAverager(model, tau_head=0.01)
    enc_calls = _count_calls(model.encoder)
    batch = _token_batch(model)
    _, averager_inputs = model(batch)
    assert enc_calls["n"] == 1
    averager(averager_inputs)
    assert enc_calls["n"] == 1


def test_head_delay_does_not_copy_encoder_backbone() -> None:
    model = _tiny_model().eval()
    averager = PolyakAverager(model, tau_head=0.01)
    assert averager.encoder is None
    assert averager.backbone is None
    assert averager.heads is not None
    delayed_names = {n for n, _ in averager.heads.named_parameters()}
    online_head_names = {n for n, _ in model.heads.named_parameters()}
    assert delayed_names == online_head_names
    _q_and_target(model, averager)


def test_zero_encoder_skips_encoder_when_backbone_delayed() -> None:
    model = _tiny_model(scaled_backbone=True).eval()
    averager = PolyakAverager(model, tau_backbone=0.01, tau_head=0.01)
    enc_calls = _count_calls(model.encoder)
    assert averager.encoder is None
    assert averager.backbone is not None
    bb_calls = _count_calls(averager.backbone)
    _q_and_target(model, averager)
    assert enc_calls["n"] == 1
    assert bb_calls["n"] == 1


def test_zero_backbone_runs_online_backbone_on_delayed_encoder() -> None:
    torch.manual_seed(0)
    model = _tiny_model(scaled_backbone=True).eval()
    averager = PolyakAverager(model, tau_encoder=1.0, tau_head=1.0)
    averager.update()
    assert averager.encoder is not None
    assert averager.backbone is None
    enc_calls = _count_calls(averager.encoder)
    bb_calls = _count_calls(model.backbone)
    before_pred, before_delayed = _q_and_target(model, averager)
    assert enc_calls["n"] == 1
    assert bb_calls["n"] == 2  # online forward + delayed-encoder path
    assert torch.allclose(before_pred["action_value"], before_delayed["action_value"])
    _perturb(model.backbone)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert torch.allclose(after_pred["action_value"], after_delayed["action_value"])
    assert not torch.allclose(after_delayed["action_value"], before_delayed["action_value"])


def test_zero_encoder_delayed_backbone_sees_online_embeds() -> None:
    torch.manual_seed(0)
    model = _tiny_model(scaled_backbone=True).eval()
    averager = PolyakAverager(model, tau_backbone=1.0, tau_head=1.0)
    averager.update()
    before_pred, before_delayed = _q_and_target(model, averager)
    assert torch.allclose(before_pred["action_value"], before_delayed["action_value"])
    _perturb(model.encoder)
    after_pred, after_delayed = _q_and_target(model, averager)
    assert torch.allclose(after_pred["action_value"], after_delayed["action_value"])
    assert not torch.allclose(after_delayed["action_value"], before_delayed["action_value"])


def test_all_zero_tau_returns_online_predictions() -> None:
    model = _tiny_model().train()
    averager = PolyakAverager(
        model, tau_encoder=0.0, tau_backbone=0.0, tau_head=0.0
    )
    assert averager.encoder is None
    assert averager.backbone is None
    assert averager.heads is None
    enc_calls = _count_calls(model.encoder)
    bb_calls = _count_calls(model.backbone)
    head_calls = _count_calls(model.heads["action_value"])
    predictions, delayed = _q_and_target(model, averager)
    assert enc_calls["n"] == 1
    assert bb_calls["n"] == 1
    assert head_calls["n"] == 1
    assert torch.equal(predictions["action_value"], delayed["action_value"])
    assert not delayed["action_value"].requires_grad


def test_encoder_delay_runs_delayed_encoder() -> None:
    model = _tiny_model().eval()
    averager = PolyakAverager(
        model, tau_encoder=0.01, tau_backbone=0.01, tau_head=0.01
    )
    assert averager.encoder is not None
    enc_calls = _count_calls(averager.encoder)
    _q_and_target(model, averager)
    assert enc_calls["n"] == 1


def test_averager_rejects_non_averager_inputs() -> None:
    model = _tiny_model().eval()
    averager = PolyakAverager(model, tau_head=0.01)
    with pytest.raises(TypeError, match="AveragerInputs"):
        averager(_token_batch(model))  # type: ignore[arg-type]


def test_head_delay_uses_passed_inputs_not_later_forward() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    averager = PolyakAverager(model, tau_head=1.0)
    averager.update()
    batch = _token_batch(model)
    first_pred, first_inputs = model(batch)
    first_delayed = averager(first_inputs)
    _perturb(model.encoder)
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
    assert averager_inputs.embeds is not None
    assert averager_inputs.prediction_indices is not None
    assert averager_inputs.predictions is not None
    assert averager_inputs.cache is None
    assert predictions["action_value"].shape[0] == averager_inputs.h.shape[0]
    assert averager_inputs.embeds.shape[-1] == averager_inputs.h.shape[-1]


def test_averager_inputs_are_detached() -> None:
    model = _tiny_model().train()
    predictions, averager_inputs = model(_token_batch(model))
    assert predictions["action_value"].requires_grad
    assert not averager_inputs.h.requires_grad
    assert averager_inputs.h.grad_fn is None
    assert averager_inputs.embeds is not None
    assert not averager_inputs.embeds.requires_grad
    assert averager_inputs.predictions is not None
    assert not averager_inputs.predictions["action_value"].requires_grad
    live = predictions["action_value"].detach().clone().requires_grad_(True)
    wrapped = AveragerInputs(h=live, batch=averager_inputs.batch)
    assert not wrapped.h.requires_grad
    averager = PolyakAverager(model, tau_head=0.01)
    delayed = averager(averager_inputs)
    assert not delayed["action_value"].requires_grad
