from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from mouse_core.models import LatentReasoner, Model, ModelOutput, Recurrence
from mouse_core.models.backbone import IdentityBackbone, LlamaBackbone
from mouse_core.models.embedding import NumericEmbedder, NumericEmbedderModalitySpec
from mouse_core.models.heads import (
    BaseHead,
    DiscreteActionValueHead,
    LayerwiseDiscreteActionValueHead,
)
from mouse_core.polyak import Polyak, _PolyakState
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder

_MODALITIES: list[dict[str, Any] | NumericEmbedderModalitySpec] = [
    {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1},
    {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
    {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
]
_BATCH = [
    [
        {"action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0},
        {"action": 1, "reward": 1.0, "episode_done": 0, "task_done": 0},
        {"action": 2, "reward": 2.0, "episode_done": 1, "task_done": 0},
    ]
]


def _head(hidden_dim: int) -> DiscreteActionValueHead:
    return DiscreteActionValueHead(
        in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1
    )


def _tiny_model() -> Model:
    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    return Model(encoder=encoder, backbone=backbone, heads=_head(hidden_dim), action_head="action_value", reasoner=None, recurrence=None)


def _llama_model(*, layerwise: bool = False) -> Model:
    hidden_dim = 16
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    backbone = LlamaBackbone(
        train_kernel="varlen", decode_kernel="flex", dtype=torch.float32,
        hidden_dim=hidden_dim, num_layers=2, num_heads=2, max_position_embeddings=64
    )
    head: BaseHead
    if layerwise:
        head = LayerwiseDiscreteActionValueHead(
            num_backbone_layers=2,
            in_features=hidden_dim,
            out_features=4,
            hidden_dim=hidden_dim,
            num_layers=1,
        )
    else:
        head = _head(hidden_dim)
    return Model(encoder=encoder, backbone=backbone, heads=head, action_head="action_value_layerwise" if layerwise else "action_value", reasoner=None, recurrence=None)


def _token_batch(model: Model):
    return batch_to_token_batch(_tok(model.encoder), _BATCH)


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)


def _q_close(a: ModelOutput, b: ModelOutput, key: str = "action_value") -> bool:
    return torch.allclose(a.predictions[key], b.predictions[key], atol=1e-5)


def _count_calls(module: nn.Module, name: str = "forward"):
    orig = getattr(module, name)
    calls = {"n": 0}

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    setattr(module, name, _wrapped)
    return calls


# ---- delayed_copy ---------------------------------------------------------


def test_delayed_copy_is_a_frozen_full_copy() -> None:
    model = _llama_model()
    delayed = model.delayed_copy()
    assert delayed.encoder is not model.encoder
    assert delayed.backbone is not model.backbone
    assert delayed.reasoner is None and delayed.recurrence is None
    assert delayed.action_head == model.action_head
    assert delayed.training
    assert all(not p.requires_grad for p in delayed.parameters())
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        assert p is not online[name]  # every trainable parameter is copied
        assert torch.equal(p, online[name])
    # Copying did not freeze the online model.
    assert all(p.requires_grad for p in model.parameters())


def test_delayed_copy_shares_frozen_parameters_by_reference() -> None:
    model = _llama_model()
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.encoder.requires_grad_(False)
    delayed = model.delayed_copy()
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith("heads."):
            assert p is not online[name]
        else:
            assert p is online[name]  # frozen encoder / backbone: referenced


def test_delayed_copy_rejects_a_model_with_nothing_trainable() -> None:
    model = _tiny_model()
    delayed = model.delayed_copy()
    with pytest.raises(ValueError, match="trainable online model"):
        delayed.delayed_copy()
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable online model"):
        model.delayed_copy()


def test_delayed_copy_carries_reasoner_and_recurrence() -> None:
    hidden_dim = 8
    recurrent = Model(
        encoder=NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES),
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=_head(hidden_dim),
        action_head="action_value",
        reasoner=None,
        recurrence=Recurrence(hidden_dim=hidden_dim, num_passes=2),
    )
    d = recurrent.delayed_copy()
    assert d.recurrence is not None and d.recurrence is not recurrent.recurrence
    assert all(not p.requires_grad for p in d.recurrence.parameters())

    reasoning = Model(
        encoder=NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES),
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=_head(hidden_dim),
        action_head="action_value",
        reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1),
        recurrence=None,
    )
    dr = reasoning.delayed_copy()
    assert dr.reasoner is not None and dr.reasoner is not reasoning.reasoner


# ---- delayed forward --------------------------------------------------------


def test_delayed_model_matches_online_before_update_and_builds_no_graph() -> None:
    torch.manual_seed(0)
    model = _llama_model().eval()
    delayed = model.delayed_copy().eval()
    batch = _token_batch(model)
    out = model(batch)
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.no_grad():
            delayed_out = delayed(batch)
    assert saved["n"] == 0
    assert delayed_out.predictions["action_value"].grad_fn is None
    assert _q_close(out, delayed_out)


def test_delayed_model_reruns_its_own_trunk() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy().eval()
    batch = _token_batch(model)
    enc_calls = _count_calls(delayed.encoder)
    bb_calls = _count_calls(delayed.backbone)
    with torch.no_grad():
        delayed(batch)
    assert enc_calls["n"] == 1 and bb_calls["n"] == 1


def test_delayed_model_ignores_online_changes_until_update() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy().eval()
    polyak = Polyak(model, delayed)
    batch = _token_batch(model)
    with torch.no_grad():
        before = delayed(batch)
    _perturb(model.encoder)
    _perturb(model.heads)
    with torch.no_grad():
        online = model(batch)
        still_delayed = delayed(batch)
    assert _q_close(before, still_delayed)
    assert not _q_close(online, still_delayed)
    polyak.update(tau_heads=1.0, tau_encoder=1.0, tau_backbone=1.0)
    with torch.no_grad():
        copied = delayed(batch)
    assert _q_close(online, copied)


def test_layerwise_delayed_model_matches_online_before_update() -> None:
    torch.manual_seed(0)
    model = _llama_model(layerwise=True).eval()
    delayed = model.delayed_copy().eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
        delayed_out = delayed(batch)
    assert out.hidden_states is not None and len(out.hidden_states) == 2
    assert _q_close(out, delayed_out, key="action_value_layerwise")


# ---- Polyak ---------------------------------------------------------------


def test_polyak_requires_a_tau_per_section() -> None:
    model = _tiny_model()
    polyak = Polyak(model, model.delayed_copy())
    with pytest.raises(TypeError):
        polyak.update(tau_heads=0.1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        polyak.update(tau_heads=0.1, tau_encoder=0.1)  # type: ignore[call-arg]
    polyak.update(tau_heads=0.1, tau_encoder=0.1, tau_backbone=0.1)


def test_polyak_tau_is_convex_combination_and_can_change() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    online = next(model.heads.parameters())
    delayed_p = next(delayed.heads.parameters())
    online.data.fill_(1.0)
    delayed_p.data.fill_(0.0)
    polyak.update(tau_heads=0.5, tau_encoder=0.0, tau_backbone=0.0)
    assert torch.allclose(delayed_p, torch.full_like(delayed_p, 0.5))
    polyak.update(tau_heads=1.0, tau_encoder=0.0, tau_backbone=0.0)
    assert torch.allclose(delayed_p, torch.ones_like(delayed_p))


def test_each_tau_interpolates_only_its_section() -> None:
    torch.manual_seed(0)
    model = _llama_model()
    delayed = model.delayed_copy()
    polyak = Polyak(model, delayed)
    snapshot = {n: p.detach().clone() for n, p in delayed.named_parameters()}
    _perturb(model)
    polyak.update(tau_heads=0.0, tau_encoder=0.5, tau_backbone=0.25)
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith("heads."):
            assert torch.equal(p, snapshot[name])
        elif name.startswith("encoder."):
            assert torch.allclose(p, 0.5 * snapshot[name] + 0.5 * online[name])
        else:
            assert name.startswith("backbone.")
            assert torch.allclose(p, 0.75 * snapshot[name] + 0.25 * online[name])


def test_tau_backbone_also_moves_recurrence_and_reasoner() -> None:
    hidden_dim = 8
    recurrent = Model(
        encoder=NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES),
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=_head(hidden_dim),
        action_head="action_value",
        reasoner=None,
        recurrence=Recurrence(hidden_dim=hidden_dim, num_passes=2),
    )
    d = recurrent.delayed_copy()
    assert recurrent.recurrence is not None and d.recurrence is not None
    _perturb(recurrent.recurrence)
    Polyak(recurrent, d).update(tau_heads=0.0, tau_encoder=0.0, tau_backbone=1.0)
    for a, b in zip(d.recurrence.parameters(), recurrent.recurrence.parameters(), strict=True):
        assert torch.equal(a, b)

    reasoning = Model(
        encoder=NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES),
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=_head(hidden_dim),
        action_head="action_value",
        reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1),
        recurrence=None,
    )
    dr = reasoning.delayed_copy()
    assert reasoning.reasoner is not None and dr.reasoner is not None
    _perturb(reasoning.reasoner)
    Polyak(reasoning, dr).update(tau_heads=0.0, tau_encoder=0.0, tau_backbone=1.0)
    for a, b in zip(dr.reasoner.parameters(), reasoning.reasoner.parameters(), strict=True):
        assert torch.equal(a, b)


def test_polyak_rejects_tau_out_of_range() -> None:
    model = _tiny_model()
    polyak = Polyak(model, model.delayed_copy())
    with pytest.raises(ValueError, match=r"tau_heads must be in \[0, 1\]"):
        polyak.update(tau_heads=1.5, tau_encoder=0.1, tau_backbone=0.1)
    with pytest.raises(ValueError, match=r"tau_encoder must be in \[0, 1\]"):
        polyak.update(tau_heads=0.1, tau_encoder=-0.1, tau_backbone=0.1)
    with pytest.raises(ValueError, match=r"tau_backbone must be in \[0, 1\]"):
        polyak.update(tau_heads=0.1, tau_encoder=0.1, tau_backbone=2.0)


def test_polyak_small_tau_accumulates_in_fp32() -> None:
    online = nn.Linear(8, 8, bias=False)
    delayed = nn.Linear(8, 8, bias=False)
    online.weight.data.fill_(1.0)
    delayed.weight.data.fill_(0.9)
    state = _PolyakState(online, delayed, section="heads")
    tau = 0.0005
    steps = 2000
    for _ in range(steps):
        state.update(tau)
    expected = 1.0 - 0.1 * (1.0 - tau) ** steps
    assert torch.allclose(delayed.weight, torch.full_like(delayed.weight, expected), atol=1e-4)


def test_polyak_rejects_non_fp32_interpolated_params() -> None:
    """A trainable bf16 copy would round a tiny tau away; Polyak refuses it."""
    online = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    delayed = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="fp32 parameters only"):
        _PolyakState(online, delayed, section="backbone")


def test_polyak_skips_shared_frozen_params_and_rejects_shared_trainable() -> None:
    online = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    online[0].requires_grad_(False)
    delayed = nn.Sequential(online[0], nn.Linear(8, 8, bias=False))
    state = _PolyakState(online, delayed, section="backbone")
    assert len(state) == 1  # the shared frozen layer is not interpolated
    online_trainable = cast(nn.Linear, online[1])
    delayed_trainable = cast(nn.Linear, delayed[1])
    online_trainable.weight.data.fill_(1.0)
    delayed_trainable.weight.data.fill_(0.0)
    state.update(0.5)
    assert torch.allclose(delayed_trainable.weight, torch.full((8, 8), 0.5))

    shared_trainable = nn.Sequential(online[0], online[1])  # trainable layer shared too
    with pytest.raises(ValueError, match="same tensor online and delayed"):
        _PolyakState(online, shared_trainable, section="backbone")
    with pytest.raises(ValueError, match="delayed_copy"):
        _PolyakState(online, online, section="backbone")


def test_polyak_rejects_wrong_models() -> None:
    model = _tiny_model()
    delayed = model.delayed_copy()
    with pytest.raises(TypeError):
        Polyak(model, nn.Linear(2, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delayed_copy"):
        Polyak(model, model)
    with pytest.raises(ValueError, match="trainable model"):
        Polyak(delayed, model.delayed_copy())
    other = _tiny_model()
    with pytest.raises(ValueError, match="delayed_copy"):
        Polyak(model, other)  # trainable sections that are not copies
    mismatched = Model(
        encoder=NumericEmbedder(hidden_dim=8, modalities=_MODALITIES),
        backbone=IdentityBackbone(hidden_dim=8),
        heads=DiscreteActionValueHead(in_features=8, out_features=4, hidden_dim=8, num_layers=2),
        action_head="action_value",
        reasoner=None,
        recurrence=None,
    ).requires_grad_(False)
    with pytest.raises(ValueError, match="parameter names"):
        Polyak(model, mismatched)


# ---- forward contract -------------------------------------------------------


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


def _assert_no_autograd_graph(fn: Callable[[], Any]) -> Any:
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    assert saved["n"] == 0
    return result


def test_delayed_forward_under_no_grad_builds_no_graph_in_train_mode() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    delayed = model.delayed_copy()
    batch = _token_batch(model)

    def run():
        with torch.no_grad():
            return delayed(batch)

    delayed_out = _assert_no_autograd_graph(run)
    assert delayed_out.predictions["action_value"].grad_fn is None


def test_forward_rejects_non_token_batch() -> None:
    model = _tiny_model().eval()
    with pytest.raises(TypeError, match="TokenBatch"):
        model(torch.zeros(3, 8))  # type: ignore[arg-type]
