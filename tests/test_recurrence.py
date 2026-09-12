"""Recurrent-depth passes: ``Model(recurrence=Recurrence(...))``."""

from __future__ import annotations

import json

import pytest
import torch

from mouse_core.models import (
    LatentReasoner,
    Model,
    ModelOutput,
    Recurrence,
    load_model,
    save_model,
)
from mouse_core.models.backbone import LlamaBackbone, LoRAConfig
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionValueHead, LayerwiseDiscreteActionValueHead
from mouse_core.objectives import DqnObjective
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch, tok_from_encoder

_HIDDEN = 32
_ACTIONS = 4

_MODALITIES = [
    {"type": "discrete", "field": "action", "vocab_size": _ACTIONS, "std": 0.02, "positions": 1},
    {"type": "discrete", "field": "observation", "vocab_size": 16, "std": 0.02, "positions": 1},
    {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
    {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
]


def _tiny_model(*, num_passes: int | None = 3, layerwise: bool = False) -> Model:
    encoder = NumericEmbedder(hidden_dim=_HIDDEN, modalities=_MODALITIES)
    backbone = LlamaBackbone(
        train_kernel="varlen", decode_kernel="flex", dtype=torch.float32,
        hidden_dim=_HIDDEN,
        num_layers=2,
        num_heads=2,
        max_position_embeddings=128,
        lora=LoRAConfig(rank=4),
    )
    if layerwise:
        heads = LayerwiseDiscreteActionValueHead(
            num_backbone_layers=2,
            in_features=_HIDDEN,
            out_features=_ACTIONS,
            hidden_dim=_HIDDEN,
            num_layers=1,
        )
    else:
        heads = DiscreteActionValueHead(
            in_features=_HIDDEN,
            out_features=_ACTIONS,
            hidden_dim=_HIDDEN,
            num_layers=1,
        )
    recurrence = (
        Recurrence(hidden_dim=_HIDDEN, num_passes=num_passes)
        if num_passes is not None
        else None
    )
    return Model(encoder=encoder, backbone=backbone, heads=heads, action_head="action_value_layerwise" if layerwise else "action_value", reasoner=None, recurrence=recurrence)


def _rows(n: int, offset: int = 0) -> list[dict]:
    return [
        {
            "action": (i + offset) % _ACTIONS,
            "observation": (3 * i + offset) % 16,
            "reward": float(i),
            "episode_done": 0,
            "task_index": 0,
        }
        for i in range(n)
    ]


def _token_batch(model: Model, batch: list[list[dict]]):
    assert model.encoder is not None
    return batch_to_token_batch(
        tok_from_encoder(model.encoder, grouping_field="task_index"),
        batch,
        grouping_field="task_index",
    )


def _turn_on_recurrence(model: Model) -> None:
    """Zero-init makes every pass identical; give the adapter a real projection."""
    assert model.recurrence is not None
    with torch.no_grad():
        torch.nn.init.normal_(model.recurrence.proj.weight, std=0.05)


_BATCH = [_rows(4), _rows(3, offset=1)]


def test_recurrence_validates_arguments() -> None:
    with pytest.raises(ValueError, match="num_passes"):
        Recurrence(hidden_dim=_HIDDEN, num_passes=1)
    with pytest.raises(ValueError, match="cannot be combined"):
        Model(
            encoder=NumericEmbedder(hidden_dim=_HIDDEN, modalities=_MODALITIES),
            backbone=LlamaBackbone(train_kernel="varlen", decode_kernel="flex", dtype=torch.float32, hidden_dim=_HIDDEN, num_layers=1, num_heads=2),
            heads=DiscreteActionValueHead(
                in_features=_HIDDEN, out_features=_ACTIONS, hidden_dim=_HIDDEN, num_layers=1
            ),
            action_head="action_value",
            reasoner=LatentReasoner(hidden_dim=_HIDDEN, num_thoughts=2),
            recurrence=Recurrence(hidden_dim=_HIDDEN, num_passes=2),
        )
    with pytest.raises(ValueError, match="hidden_dim mismatch"):
        Model(
            encoder=NumericEmbedder(hidden_dim=_HIDDEN, modalities=_MODALITIES),
            backbone=LlamaBackbone(train_kernel="varlen", decode_kernel="flex", dtype=torch.float32, hidden_dim=_HIDDEN, num_layers=1, num_heads=2),
            heads=DiscreteActionValueHead(
                in_features=_HIDDEN, out_features=_ACTIONS, hidden_dim=_HIDDEN, num_layers=1
            ),
            action_head="action_value",
            reasoner=None,
            recurrence=Recurrence(hidden_dim=_HIDDEN * 2, num_passes=2),
        )


def test_forward_runs_num_passes_and_reports_final_pass() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).eval()
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, ModelOutput)
    assert len(out.passes) == 3
    assert out.predictions is out.passes[-1].predictions
    assert out.last_hidden_state is out.passes[-1].last_hidden_state
    for p in out.passes:
        assert p.last_hidden_state.shape == (batch.L, _HIDDEN)
        assert p.predictions["action_value"].shape == (batch.N, _ACTIONS)


def test_zero_init_adapter_makes_every_pass_equal_plain_forward() -> None:
    """At construction the recurrence is an exact no-op: every pass equals a
    single-pass model with the same weights."""
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).eval()
    plain = _tiny_model(num_passes=None).eval()
    plain.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("recurrence.")}
    )
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        out = model(batch)
        ref = plain(batch)
    for p in out.passes:
        assert torch.allclose(p.predictions["action_value"], ref.predictions["action_value"], atol=1e-6)
        assert torch.allclose(p.last_hidden_state, ref.last_hidden_state, atol=1e-6)


def test_adapter_bounds_recycled_state_scale() -> None:
    """A huge residual stream is normalized before re-entering the backbone."""
    torch.manual_seed(0)
    rec = Recurrence(hidden_dim=_HIDDEN, num_passes=2)
    torch.nn.init.normal_(rec.proj.weight, std=0.05)
    encodings = torch.randn(10, _HIDDEN) * 0.02
    huge = torch.randn(10, _HIDDEN) * 1e4
    huge[:, 0] = 1e6  # outlier dim
    nxt = rec(encodings, huge)
    rms = lambda x: x.pow(2).mean().sqrt().item()  # noqa: E731
    assert rms(nxt) < 1.0
    assert torch.isfinite(nxt).all()
    # Injection: the original encodings are part of every pass's input.
    assert torch.allclose(rec(encodings, torch.zeros_like(huge)), encodings)


def test_recurrent_passes_stay_bounded_on_backbone() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=4).eval()
    _turn_on_recurrence(model)
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        out = model(batch)
    rms = [p.last_hidden_state.pow(2).mean().sqrt().item() for p in out.passes]
    assert all(torch.isfinite(p.last_hidden_state).all() for p in out.passes)
    # Recycling goes through RMSNorm, so the scale does not compound pass over pass.
    later = rms[1:]
    assert max(later) < 2 * min(later)
    # Passes differ once the adapter is non-zero.
    assert not torch.allclose(
        out.passes[0].predictions["action_value"], out.passes[-1].predictions["action_value"]
    )


def test_gradients_flow_through_all_passes() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).train()
    _turn_on_recurrence(model)
    batch = _token_batch(model, _BATCH)
    out = model(batch)
    for p in out.passes:
        assert p.predictions["action_value"].requires_grad
        assert p.last_hidden_state.requires_grad
    # Final pass alone reaches the adapter, the backbone, and the encoder.
    out.passes[-1].predictions["action_value"].sum().backward()
    assert model.recurrence is not None
    assert model.recurrence.proj.weight.grad is not None
    assert float(model.recurrence.proj.weight.grad.abs().sum()) > 0.0
    q_proj = model.backbone.model.layers[0].self_attn.q_proj  # type: ignore[union-attr]
    assert q_proj.base.weight.grad is None  # frozen base
    bb_grad = q_proj.lora_B.weight.grad
    assert bb_grad is not None and float(bb_grad.abs().sum()) > 0.0
    assert model.encoder is not None
    enc_grad = next(model.encoder.parameters()).grad
    assert enc_grad is not None and float(enc_grad.abs().sum()) > 0.0


def test_zero_init_adapter_still_receives_gradient() -> None:
    """Zero projection is not a dead start: its gradient is non-zero."""
    torch.manual_seed(0)
    model = _tiny_model(num_passes=2).train()
    out = model(_token_batch(model, _BATCH))
    out.predictions["action_value"].sum().backward()
    assert model.recurrence is not None
    grad = model.recurrence.proj.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_delayed_model_parity_per_pass() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).eval()
    _turn_on_recurrence(model)
    delayed = model.delayed_copy().eval()
    assert delayed.recurrence is not None and delayed.recurrence is not model.recurrence
    batch = _token_batch(model, _BATCH)
    out = model(batch)
    with torch.no_grad():
        delayed_out = delayed(batch)
    assert len(delayed_out.passes) == 3
    for p, target in zip(out.passes, delayed_out.passes, strict=True):
        assert torch.allclose(p.predictions["action_value"], target.predictions["action_value"], atol=1e-5)
        assert not target.predictions["action_value"].requires_grad


def test_layerwise_recurrent_passes_carry_hidden_states() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=2, layerwise=True).eval()
    _turn_on_recurrence(model)
    delayed = model.delayed_copy().eval()
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        out = model(batch)
        delayed_out = delayed(batch)
    for p, target in zip(out.passes, delayed_out.passes, strict=True):
        assert p.hidden_states is not None and len(p.hidden_states) == 2
        assert torch.allclose(
            p.predictions["action_value_layerwise"],
            target.predictions["action_value_layerwise"],
            atol=1e-5,
        )


def test_mean_loss_over_passes_trains() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).train()
    _turn_on_recurrence(model)
    delayed = model.delayed_copy()
    batch = [[{**row, "task_done": 0} for row in seq] for seq in _BATCH]
    assert model.encoder is not None
    tok = tok_from_encoder(
        model.encoder,
        objective_fields=["action", "reward", "episode_done", "task_done"],
        grouping_field="task_index",
    )
    inputs, objective_data = batch_to_packed(tok, batch, grouping_field="task_index")
    objective = DqnObjective(gamma_step=1.0, grouping_field="task_index", gamma_episode_terminal=0.0, gamma_episode_truncated=0.0, gamma_task_terminal=0.0, gamma_task_truncated=0.0)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed(inputs)
    losses = []
    metrics: dict[str, float] = {}
    for p, target in zip(out.passes, delayed_out.passes, strict=True):
        step_loss, metrics = objective(objective_data, p.predictions, target.predictions)
        losses.append(step_loss)
    loss = torch.stack(losses).mean()
    assert loss.ndim == 0
    loss.backward()
    head_grad = next(model.heads.parameters()).grad
    assert head_grad is not None and float(head_grad.abs().sum()) > 0.0
    assert "q_values_mean" in metrics


def test_cached_decode_matches_full_forward_with_recurrence() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).eval()
    _turn_on_recurrence(model)
    steps = _rows(6)
    with torch.no_grad():
        full = model(_token_batch(model, [steps]))
        cache = None
        chunks = []
        for lo, hi in ((0, 3), (3, 4), (4, 6)):
            out = model(_token_batch(model, [steps[lo:hi]]), cache=cache, use_cache=True)
            cache = out.cache
            chunks.append(out.predictions["action_value"])
        assert cache is not None and len(cache.sessions) == 3
        incremental = torch.cat(chunks, dim=1)
    assert torch.allclose(incremental, full.predictions["action_value"].unsqueeze(0), atol=1e-5)


def test_cached_decode_with_wrong_session_count_raises() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=2).eval()
    plain = _tiny_model(num_passes=None).eval()
    with torch.no_grad():
        out = plain(_token_batch(plain, [_rows(2)]), use_cache=True)
        with pytest.raises(ValueError, match="sessions"):
            model(_token_batch(model, [_rows(2)]), cache=out.cache, use_cache=True)


def test_decode_cache_reset_rows_resets_every_pass() -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=2).eval()
    with torch.no_grad():
        out = model(_token_batch(model, [_rows(3), _rows(2)]), use_cache=True)
    cache = out.cache
    assert cache is not None
    cache.reset_rows([0])
    for session in cache.sessions:
        assert int(session.lengths[0]) == 0
        assert int(session.lengths[1]) > 0


def test_save_load_roundtrip_keeps_recurrence(tmp_path) -> None:
    torch.manual_seed(0)
    model = _tiny_model(num_passes=3).eval()
    _turn_on_recurrence(model)
    save_model(model, tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["recurrence"] == {"num_passes": 3}
    loaded = load_model(str(tmp_path), train_kernel="varlen", decode_kernel="flex", dtype=torch.float32).eval()
    assert loaded.recurrence is not None and loaded.recurrence.num_passes == 3
    batch = _token_batch(model, _BATCH)
    with torch.no_grad():
        a = model(batch)
        b = loaded(batch)
    assert len(b.passes) == 3
    assert torch.allclose(a.predictions["action_value"], b.predictions["action_value"], atol=1e-6)
