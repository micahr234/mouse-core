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
    delayed = model.delayed_copy(heads=True)
    assert delayed.encoder is None and delayed.backbone is None
    assert delayed.reasoner is None and delayed.recurrence is None
    assert delayed.training
    assert all(not p.requires_grad for p in delayed.parameters())
    assert set(dict(delayed.heads.named_parameters())) == set(
        dict(model.heads.named_parameters())
    )
    with pytest.raises(ValueError, match="online model"):
        delayed.delayed_copy(heads=True)


def test_delayed_heads_read_current_online_features() -> None:
    """Only the heads are delayed: encoder/backbone changes reach the target."""
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy(heads=True)
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
    delayed = model.delayed_copy(heads=True)
    polyak = Polyak(model, delayed)
    batch = _token_batch(model)
    _perturb(model.heads)
    out = model(batch)
    with torch.no_grad():
        target = _delayed_heads(delayed, out)
    assert not torch.allclose(out.predictions["action_value"], target.predictions["action_value"])
    polyak.update(tau_heads=1.0)
    with torch.no_grad():
        target = _delayed_heads(delayed, out)
    assert torch.allclose(out.predictions["action_value"], target.predictions["action_value"])


def test_delayed_heads_do_not_rerun_encoder_or_backbone() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy(heads=True)
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
    delayed = model.delayed_copy(heads=True)
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
    delayed = model.delayed_copy(heads=True)
    polyak = Polyak(model, delayed)
    online = next(model.heads.parameters())
    delayed_p = next(delayed.heads.parameters())
    online.data.fill_(1.0)
    delayed_p.data.fill_(0.0)
    polyak.update(tau_heads=0.5)
    assert torch.allclose(delayed_p, torch.full_like(delayed_p, 0.5))
    polyak.update(tau_heads=1.0)
    assert torch.allclose(delayed_p, torch.ones_like(delayed_p))


def test_zero_tau_freezes_delayed_heads() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = model.delayed_copy(heads=True)
    polyak = Polyak(model, delayed)
    snapshot = [p.detach().clone() for p in delayed.heads.parameters()]
    _perturb(model.heads)
    polyak.update(tau_heads=0.0)
    for p, s in zip(delayed.heads.parameters(), snapshot, strict=True):
        assert torch.equal(p, s)


def test_polyak_rejects_tau_out_of_range() -> None:
    model = _tiny_model()
    polyak = Polyak(model, model.delayed_copy(heads=True))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        polyak.update(tau_heads=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        polyak.update(tau_heads=-0.1)


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


def test_polyak_bf16_delayed_accumulates_in_fp32_shadow() -> None:
    """A bf16 delayed copy would round a tiny tau away; the fp32 shadow must not."""
    online = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    delayed = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    online.weight.data.fill_(1.0)
    delayed.weight.data.fill_(0.9)
    start = float(delayed.weight.float()[0, 0])  # 0.9 rounded to bf16
    state = _PolyakState(online, delayed, section="backbone")
    tau = 0.0005
    steps = 2000
    for _ in range(steps):
        state.update(tau)
    expected = 1.0 - (1.0 - start) * (1.0 - tau) ** steps  # ≈ 0.963
    assert delayed.weight.dtype == torch.bfloat16
    # Final value is the fp32 shadow rounded once to bf16 (half-ULP ≈ 2e-3 here).
    assert torch.allclose(
        delayed.weight.float(), torch.full((8, 8), expected), atol=2.5e-3
    )
    # Sanity: a direct bf16 lerp with this tau does not move at all.
    naive = torch.full((8, 8), 0.9, dtype=torch.bfloat16)
    naive.lerp_(torch.ones_like(naive), tau)
    assert torch.equal(naive, torch.full((8, 8), 0.9, dtype=torch.bfloat16))


def test_polyak_rejects_wrong_models() -> None:
    model = _tiny_model()
    delayed = model.delayed_copy(heads=True)
    with pytest.raises(TypeError):
        Polyak(model, nn.Linear(2, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="delayed_copy"):
        Polyak(model, model)
    with pytest.raises(ValueError, match="full model"):
        Polyak(delayed, delayed)
    with pytest.raises(ValueError, match="at least one delayed section"):
        model.delayed_copy()


def test_shared_heads_are_not_interpolated_and_reject_tau() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy(backbone=True).eval()
    assert all(delayed.heads[n] is model.heads[n] for n in model.heads.keys())
    assert all(p.requires_grad for p in model.heads.parameters())
    polyak = Polyak(model, delayed)
    assert not polyak.delays_heads and polyak.delays_backbone
    polyak.update(tau_backbone=0.1)
    polyak.update(tau_backbone=0.1, tau_heads=1.0)
    with pytest.raises(ValueError, match="shares the online heads"):
        polyak.update(tau_backbone=0.1, tau_heads=0.5)
    # Shared heads follow the online heads immediately (identity backbone → equal).
    batch = _token_batch(model)
    _perturb(model.heads)
    with torch.no_grad():
        assert _full_forward_close(model(batch), delayed(batch))


# ---- full-model delay (delayed encoder / backbone) -------------------------


def _full_forward_close(a: ModelOutput, b: ModelOutput) -> bool:
    return torch.allclose(a.predictions["action_value"], b.predictions["action_value"], atol=1e-6)


def test_delayed_copy_trunk_sections_copied_or_shared() -> None:
    model = _tiny_model()
    both = model.delayed_copy(encoder=True, backbone=True, heads=True)
    assert both.encoder is not model.encoder and both.backbone is not model.backbone
    assert all(not p.requires_grad for p in both.parameters())
    assert both.training

    only_backbone = model.delayed_copy(backbone=True, heads=True)
    assert only_backbone.encoder is model.encoder  # shared by reference
    assert only_backbone.backbone is not model.backbone
    # Sharing must not freeze the online module.
    assert model.encoder is not None
    assert all(p.requires_grad for p in model.encoder.parameters())

    only_encoder = model.delayed_copy(encoder=True, heads=True)
    assert only_encoder.encoder is not model.encoder
    assert only_encoder.backbone is model.backbone


def test_full_delayed_model_runs_on_token_batch_and_matches_online_before_update() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy(encoder=True, backbone=True, heads=True).eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
        delayed_out = delayed(batch)
    assert _full_forward_close(out, delayed_out)
    assert delayed_out.predictions["action_value"].grad_fn is None
    with pytest.raises(ValueError, match="heads-only"):
        model.delayed_copy(heads=True)(batch)


def test_delayed_trunk_lags_online_trunk_until_update() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy(encoder=True, backbone=True, heads=True).eval()
    polyak = Polyak(model, delayed)
    assert polyak.delays_encoder and polyak.delays_backbone
    batch = _token_batch(model)
    with torch.no_grad():
        before = delayed(batch)
    assert model.encoder is not None
    _perturb(model.encoder)
    with torch.no_grad():
        online = model(batch)
        still_delayed = delayed(batch)
    # Delayed encoder has not moved: the target ignores the online change.
    assert _full_forward_close(before, still_delayed)
    assert not _full_forward_close(online, still_delayed)
    polyak.update(tau_heads=1.0, tau_encoder=1.0, tau_backbone=1.0)
    with torch.no_grad():
        copied = delayed(batch)
    assert _full_forward_close(online, copied)


def test_shared_encoder_follows_online_immediately() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = model.delayed_copy(backbone=True, heads=True).eval()
    polyak = Polyak(model, delayed)
    assert not polyak.delays_encoder and polyak.delays_backbone
    batch = _token_batch(model)
    assert model.encoder is not None
    _perturb(model.encoder)
    with torch.no_grad():
        online = model(batch)
        delayed_out = delayed(batch)
    # Encoder shared (tau = 1); backbone (identity) and heads unchanged → equal.
    assert _full_forward_close(online, delayed_out)


def test_update_requires_tau_for_delayed_trunk_and_rejects_it_for_shared() -> None:
    model = _tiny_model()
    full = Polyak(model, model.delayed_copy(encoder=True, backbone=True, heads=True))
    with pytest.raises(ValueError, match="pass tau_encoder="):
        full.update(tau_heads=0.1, tau_backbone=0.1)
    with pytest.raises(ValueError, match="pass tau_backbone="):
        full.update(tau_heads=0.1, tau_encoder=0.1)
    full.update(tau_heads=0.1, tau_encoder=0.1, tau_backbone=0.1)

    heads_only = Polyak(model, model.delayed_copy(heads=True))
    heads_only.update(tau_heads=0.1)
    heads_only.update(tau_heads=0.1, tau_encoder=1.0, tau_backbone=1.0)
    with pytest.raises(ValueError, match="shares the online encoder"):
        heads_only.update(tau_heads=0.1, tau_encoder=0.5)
    with pytest.raises(ValueError, match="shares the online backbone"):
        heads_only.update(tau_heads=0.1, tau_backbone=0.0)
    with pytest.raises(ValueError, match=r"tau_backbone must be in \[0, 1\]"):
        full.update(tau_heads=0.1, tau_encoder=0.1, tau_backbone=2.0)


def test_trunk_tau_interpolates_only_that_section() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = model.delayed_copy(encoder=True, backbone=True, heads=True)
    polyak = Polyak(model, delayed)
    assert model.encoder is not None and delayed.encoder is not None
    enc_snapshot = [p.detach().clone() for p in delayed.encoder.parameters()]
    head_snapshot = [p.detach().clone() for p in delayed.heads.parameters()]
    _perturb(model.encoder)
    _perturb(model.heads)
    polyak.update(tau_heads=0.0, tau_encoder=0.5, tau_backbone=0.0)
    for p, s in zip(delayed.heads.parameters(), head_snapshot, strict=True):
        assert torch.equal(p, s)
    for p, s, o in zip(
        delayed.encoder.parameters(), enc_snapshot, model.encoder.parameters(), strict=True
    ):
        assert torch.allclose(p, 0.5 * s + 0.5 * o)


def test_delayed_copy_backbone_flag_carries_recurrence_and_reasoner() -> None:
    from mouse_core.models import LatentReasoner
    from mouse_core.models.recurrence import Recurrence

    hidden_dim = 8
    encoder = NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES)
    head = DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1)
    recurrent = Model(
        encoder=encoder,
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=head,
        recurrence=Recurrence(hidden_dim=hidden_dim, num_passes=2),
    )
    d = recurrent.delayed_copy(backbone=True, heads=True)
    assert d.recurrence is not None and d.recurrence is not recurrent.recurrence
    assert d.encoder is recurrent.encoder
    shared = recurrent.delayed_copy(encoder=True, heads=True)
    assert shared.recurrence is recurrent.recurrence
    Polyak(recurrent, d).update(tau_heads=0.1, tau_backbone=0.1)

    reasoning = Model(
        encoder=NumericEmbedder(hidden_dim=hidden_dim, modalities=_MODALITIES),
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads=DiscreteActionValueHead(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1),
        reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1),
    )
    dr = reasoning.delayed_copy(encoder=True, backbone=True, heads=True)
    assert dr.reasoner is not None and dr.reasoner is not reasoning.reasoner
    Polyak(reasoning, dr).update(tau_heads=0.1, tau_encoder=0.1, tau_backbone=0.1)


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
    delayed = model.delayed_copy(heads=True)
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
