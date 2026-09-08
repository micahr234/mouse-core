from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

from mouse_core.models import Model, ModelOutput
from mouse_core.models.backbone import IdentityBackbone, LlamaBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import (
    DiscreteActionValueHead,
    LayerwiseDiscreteActionValueHead,
)
from mouse_core.polyak import Polyak, _PolyakState
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder

_MODALITIES = [
    {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
    {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1},
    {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
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


def _layerwise_model() -> Model:
    hidden_dim = 16
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone = LlamaBackbone(
        hidden_dim=hidden_dim, num_layers=2, num_heads=2, max_position_embeddings=64
    )
    head = LayerwiseDiscreteActionValueHead(
        num_backbone_layers=2,
        in_features=hidden_dim,
        out_features=4,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    return Model(encoder=encoder, backbone=backbone, heads=head)


def _token_batch(model: Model):
    assert model.encoder is not None
    return batch_to_token_batch(_tok(model.encoder), _BATCH)


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)


def _delayed_heads(delayed: Model, out: ModelOutput) -> ModelOutput:
    return delayed(
        last_hidden_state=out.last_hidden_state,
        head_output_indices=out.head_output_indices,
        hidden_states=out.hidden_states,
    )


def _count_calls(module: nn.Module, name: str = "forward"):
    orig = getattr(module, name)
    calls = {"n": 0}

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    setattr(module, name, _wrapped)
    return calls


def test_delayed_copy_is_heads_only_and_frozen() -> None:
    model = _tiny_model()
    delayed = model.delayed_copy()
    assert delayed.encoder is None and delayed.backbone is None
    assert delayed.reasoner is None and delayed.recurrence is None
    assert delayed.training
    assert all(not p.requires_grad for p in delayed.parameters())
    assert set(dict(delayed.heads.named_parameters())) == set(
        dict(model.heads.named_parameters())
    )
    with pytest.raises(ValueError, match="online model"):
        delayed.delayed_copy()


def test_delayed_heads_read_current_online_features() -> None:
    """Only the heads are delayed: encoder/backbone changes reach the target."""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy()
    batch = _token_batch(model)
    before = model(batch)
    with torch.no_grad():
        before_delayed = _delayed_heads(delayed, before)
    assert model.encoder is not None
    _perturb(model.encoder)
    after = model(batch)
    with torch.no_grad():
        after_delayed = _delayed_heads(delayed, after)
    assert not torch.allclose(
        after_delayed.predictions["action_value"],
        before_delayed.predictions["action_value"],
    )
    # Delayed heads still equal online heads (no update yet) on the same states.
    assert torch.allclose(
        after.predictions["action_value"], after_delayed.predictions["action_value"]
    )


def test_delayed_heads_lag_online_heads_until_update() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    batch = _token_batch(model)
    _perturb(model.heads)
    out = model(batch)
    with torch.no_grad():
        target = _delayed_heads(delayed, out)
    assert not torch.allclose(out.predictions["action_value"], target.predictions["action_value"])
    polyak.update(1.0)
    with torch.no_grad():
        target = _delayed_heads(delayed, out)
    assert torch.allclose(out.predictions["action_value"], target.predictions["action_value"])


def test_delayed_heads_do_not_rerun_encoder_or_backbone() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy()
    batch = _token_batch(model)
    out = model(batch)
    assert model.encoder is not None and model.backbone is not None
    enc_calls = _count_calls(model.encoder)
    bb_calls = _count_calls(model.backbone)
    with torch.no_grad():
        _delayed_heads(delayed, out)
    assert enc_calls["n"] == 0 and bb_calls["n"] == 0


def test_layerwise_delayed_heads_use_hidden_states() -> None:
    torch.manual_seed(0)
    model = _layerwise_model().eval()
    delayed = model.delayed_copy()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
        assert out.hidden_states is not None and len(out.hidden_states) == 2
        target = _delayed_heads(delayed, out)
    assert torch.allclose(
        out.predictions["action_value_layerwise"],
        target.predictions["action_value_layerwise"],
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="hidden_states"):
        delayed(
            last_hidden_state=out.last_hidden_state,
            head_output_indices=out.head_output_indices,
        )


def test_polyak_tau_is_convex_combination_and_can_change() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    online = next(model.heads.parameters())
    delayed_p = next(delayed.heads.parameters())
    online.data.fill_(1.0)
    delayed_p.data.fill_(0.0)
    polyak.update(0.5)
    assert torch.allclose(delayed_p, torch.full_like(delayed_p, 0.5))
    polyak.update(1.0)
    assert torch.allclose(delayed_p, torch.ones_like(delayed_p))


def test_zero_tau_freezes_delayed_heads() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    snapshot = [p.detach().clone() for p in delayed.heads.parameters()]
    _perturb(model.heads)
    polyak.update(0.0)
    for p, s in zip(delayed.heads.parameters(), snapshot, strict=True):
        assert torch.equal(p, s)


def test_polyak_rejects_tau_out_of_range() -> None:
    model = _tiny_model()
    polyak = Polyak(model, model.delayed_copy())
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        polyak.update(1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        polyak.update(-0.1)


def test_polyak_small_tau_accumulates_in_fp32() -> None:
    online = nn.Linear(8, 8, bias=False)
    delayed = nn.Linear(8, 8, bias=False)
    online.weight.data.fill_(1.0)
    delayed.weight.data.fill_(0.9)
    state = _PolyakState(online, delayed)
    tau = 0.0005
    steps = 2000
    for _ in range(steps):
        state.update(tau)
    expected = 1.0 - 0.1 * (1.0 - tau) ** steps
    assert torch.allclose(delayed.weight, torch.full_like(delayed.weight, expected), atol=1e-4)


def test_polyak_rejects_non_fp32_heads() -> None:
    online = nn.Linear(8, 8, bias=False)
    delayed = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        _PolyakState(online, delayed)


def test_polyak_rejects_wrong_models() -> None:
    model = _tiny_model()
    delayed = model.delayed_copy()
    with pytest.raises(TypeError):
        Polyak(model, nn.Linear(2, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="heads-only"):
        Polyak(model, model)
    with pytest.raises(ValueError, match="full model"):
        Polyak(delayed, delayed)


def test_polyak_rejects_mismatched_head_parameters() -> None:
    model = _tiny_model()
    other = Model(
        heads=DiscreteActionValueHead(in_features=8, out_features=4, hidden_dim=8, num_layers=2)
    )
    other.requires_grad_(False)
    with pytest.raises(ValueError, match="parameter names"):
        Polyak(model, other)


def test_forward_returns_model_output() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, ModelOutput)
    assert out.last_hidden_state.shape == (batch.L, model.hidden_dim)
    assert out.head_output_indices.shape == (batch.P,)
    assert len(out.passes) == 1
    assert out.passes[0].predictions is out.predictions
    assert out.cache is None


def test_last_hidden_state_stays_on_the_tape() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    out = model(_token_batch(model))
    assert out.last_hidden_state.requires_grad
    assert out.predictions["action_value"].requires_grad


def _assert_no_autograd_graph(fn: Callable[[], Any]) -> None:
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    assert saved["n"] == 0
    return result


def test_delayed_heads_under_no_grad_build_no_graph() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    delayed = model.delayed_copy()
    out = model(_token_batch(model))

    def run():
        with torch.no_grad():
            return _delayed_heads(delayed, out)

    delayed_out = _assert_no_autograd_graph(run)
    assert delayed_out.predictions["action_value"].grad_fn is None


def test_forward_rejects_stray_state_arguments() -> None:
    model = _tiny_model().eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
    with pytest.raises(ValueError, match="only accepted with last_hidden_state"):
        model(batch, head_output_indices=out.head_output_indices)
    with pytest.raises(ValueError, match="only accepted with last_hidden_state"):
        model(batch, hidden_states=())
    with pytest.raises(ValueError, match="requires head_output_indices"):
        model(last_hidden_state=out.last_hidden_state)
    with pytest.raises(ValueError, match="not both"):
        model(
            batch,
            last_hidden_state=out.last_hidden_state,
            head_output_indices=out.head_output_indices,
        )
