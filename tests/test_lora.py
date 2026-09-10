"""fp32 LoRA adapters on a frozen (bf16-capable) backbone."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from mouse_core import AdamW, Polyak
from mouse_core.models import (
    LatentReasoner,
    LoRAConfig,
    Model,
    Recurrence,
    load_model,
    save_model,
)
from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionValueHead
from mouse_core.models.lora import LoRALinear, apply_lora, lora_modules
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder

_HIDDEN = 16
_ACTIONS = 4
_MODALITIES = [
    {"type": "discrete", "field": "action", "vocab_size": _ACTIONS, "std": 0.02, "positions": 1},
    {"type": "fourier", "field": "reward", "std": 0.02, "positions": 1, "fourier_min": 0.01, "fourier_max": 10.0},
    {"type": "discrete", "field": "episode_done", "vocab_size": 3, "std": 0.02, "positions": 1},
]
_BATCH = [
    [
        {"action": 0, "reward": 0.0, "episode_done": 0},
        {"action": 1, "reward": 1.0, "episode_done": 0},
        {"action": 2, "reward": 2.0, "episode_done": 1},
    ],
    [{"action": 3, "reward": -1.0, "episode_done": 0}],
]


def _backbone(lora: LoRAConfig | None, *, cls=Qwen3Backbone, dtype: torch.dtype = torch.float32):
    return cls(train_kernel="varlen", decode_kernel="flex", dtype=dtype, hidden_dim=_HIDDEN, num_layers=2, num_heads=2, max_position_embeddings=64, lora=lora)


def _model(
    lora: LoRAConfig | None = LoRAConfig(rank=4, alpha=8.0),
    *,
    reasoner: bool = False,
    recurrence: bool = False,
    dtype: torch.dtype = torch.float32,
) -> Model:
    return Model(
        encoder=NumericEmbedder(hidden_dim=_HIDDEN, modalities=_MODALITIES),
        backbone=_backbone(lora, dtype=dtype),
        heads=DiscreteActionValueHead(
            in_features=_HIDDEN, out_features=_ACTIONS, hidden_dim=_HIDDEN, num_layers=1
        ),
        reasoner=LatentReasoner(hidden_dim=_HIDDEN, num_thoughts=1) if reasoner else None,
        recurrence=Recurrence(hidden_dim=_HIDDEN, num_passes=2) if recurrence else None,
    )


def _batch(model: Model):
    assert model.encoder is not None
    return batch_to_token_batch(tok_from_encoder(model.encoder), _BATCH)


# ---- LoRAConfig / apply_lora -------------------------------------------------


def test_lora_config_validates_and_normalizes() -> None:
    cfg = LoRAConfig(rank=4, alpha=8, targets=["q_proj"])
    assert cfg.rank == 4 and cfg.alpha == 8.0 and cfg.targets == ("q_proj",)
    with pytest.raises(ValueError, match="rank"):
        LoRAConfig(rank=0)
    with pytest.raises(ValueError, match="alpha"):
        LoRAConfig(alpha=0.0)
    with pytest.raises(ValueError, match="dropout"):
        LoRAConfig(dropout=1.0)
    with pytest.raises(ValueError, match="targets"):
        LoRAConfig(targets=())


def test_apply_lora_wraps_targets_and_freezes_base() -> None:
    torch.manual_seed(0)
    cfg = LoRAConfig(rank=2, targets=("q_proj", "down_proj"))
    backbone = _backbone(None)
    inner = backbone.model
    assert all(p.requires_grad for p in inner.parameters())
    wrapped = apply_lora(inner, cfg)
    assert all(not p.requires_grad for n, p in inner.named_parameters() if ".lora_" not in n)
    assert wrapped == 2 * 2  # two targets per layer, two layers
    for raw_layer in inner.layers:
        layer = cast(Any, raw_layer)
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.mlp.down_proj, LoRALinear)
        assert isinstance(layer.self_attn.k_proj, nn.Linear)
    trainable = {n for n, p in inner.named_parameters() if p.requires_grad}
    assert trainable
    assert all(".lora_A." in n or ".lora_B." in n for n in trainable)
    assert len(list(lora_modules(inner))) == wrapped


def test_apply_lora_rejects_missing_targets() -> None:
    with pytest.raises(ValueError, match="no nn.Linear named"):
        apply_lora(nn.Sequential(nn.Linear(4, 4)), LoRAConfig(targets=("nope",)))


def test_backbone_without_lora_is_fully_trainable() -> None:
    backbone = _backbone(None)
    assert backbone.lora is None
    assert all(p.requires_grad for p in backbone.parameters())
    assert all(p.dtype == torch.float32 for p in backbone.parameters())
    assert list(lora_modules(backbone)) == []


@pytest.mark.parametrize("cls", [Qwen3Backbone, LlamaBackbone])
def test_backbone_lora_kwarg_on_both_transformers(cls) -> None:
    backbone = _backbone(LoRAConfig(rank=2), cls=cls)
    assert backbone.lora == LoRAConfig(rank=2)
    assert len(list(lora_modules(backbone))) == 7 * 2


# ---- LoRALinear forward / gradients -----------------------------------------


def test_lora_linear_matches_base_until_b_moves() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 6)
    x = torch.randn(3, 8)
    expected = base(x)
    lora = LoRALinear(base, LoRAConfig(rank=2, alpha=4.0))
    assert torch.equal(lora(x), expected)
    assert lora.in_features == 8 and lora.out_features == 6
    with torch.no_grad():
        lora.lora_B.weight.fill_(0.1)
    out = lora(x)
    assert not torch.allclose(out, expected)
    delta = (x @ lora.lora_A.weight.T @ lora.lora_B.weight.T) * (4.0 / 2)
    assert torch.allclose(out, expected + delta, atol=1e-6)


def test_lora_linear_runs_fp32_adapters_over_bf16_base() -> None:
    torch.manual_seed(0)
    base = nn.Linear(8, 6).to(torch.bfloat16)
    lora = LoRALinear(base, LoRAConfig(rank=2))
    assert lora.lora_A.weight.dtype == torch.float32
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    out = lora(x)
    assert out.dtype == torch.bfloat16
    out.float().sum().backward()
    assert base.weight.grad is None
    assert lora.lora_A.weight.grad is not None and lora.lora_A.weight.grad.dtype == torch.float32
    assert lora.lora_B.weight.grad is not None and lora.lora_B.weight.grad.dtype == torch.float32


def test_only_lora_params_receive_gradients() -> None:
    torch.manual_seed(0)
    model = _model().train()
    out = model(_batch(model))
    out.predictions["action_value"].sum().backward()
    assert model.backbone is not None
    for name, p in model.backbone.named_parameters():
        if ".lora_" in name:
            assert p.grad is not None, name
        else:
            assert p.grad is None, name


# ---- Model.to precision policy -----------------------------------------------


def test_module_device_dtype_skips_lora_adapters() -> None:
    """Flex train/decode must not take dtype from an fp32 adapter that happens to be first."""
    from mouse_core.models.backbone.flex_decode import module_device_dtype

    class _Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, dtype=torch.float32))

    class _Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Register adapters first so next(parameters()) is fp32.
            self.q_proj = nn.Module()
            self.q_proj.lora_A = _Adapter()
            self.q_proj.lora_B = _Adapter()
            self.base = nn.Module()
            self.base.weight = nn.Parameter(torch.ones(3, dtype=torch.bfloat16))

    stack = _Stack()
    first = next(stack.parameters())
    assert first.dtype == torch.float32
    device, dtype = module_device_dtype(stack)
    assert dtype == torch.bfloat16
    assert device == first.device


def test_decode_session_dtype_skips_lora_adapters() -> None:
    torch.manual_seed(0)
    model = _model(dtype=torch.bfloat16)
    assert model.backbone is not None
    session = model.backbone.decode_session(batch_size=1)
    assert session.dtype == torch.bfloat16
    assert model.backbone.dtype == torch.bfloat16


def test_model_to_bf16_keeps_every_trainable_section_fp32() -> None:
    torch.manual_seed(0)
    model = _model(reasoner=True, dtype=torch.bfloat16)
    assert model.backbone is not None and model.encoder is not None and model.reasoner is not None
    assert model.backbone.dtype == torch.bfloat16
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.dtype == torch.float32, name
        else:
            assert p.dtype == torch.bfloat16, name
    assert {p.dtype for p in model.encoder.parameters()} == {torch.float32}
    assert {p.dtype for p in model.reasoner.parameters()} == {torch.float32}
    assert {p.dtype for p in model.heads.parameters()} == {torch.float32}


def test_bf16_backbone_forward_backward_on_cpu() -> None:
    """fp32 encoder → bf16 backbone → fp32 heads, through masked SDPA on CPU."""
    torch.manual_seed(0)
    model = _model(recurrence=True, dtype=torch.bfloat16).train()
    batch = _batch(model)
    out = model(batch)
    q = out.predictions["action_value"]
    assert q.dtype == torch.float32
    assert out.last_hidden_state.dtype == torch.bfloat16
    q.sum().backward()
    assert model.backbone is not None and model.recurrence is not None
    lora_grads = [p.grad for n, p in model.backbone.named_parameters() if ".lora_B." in n]
    assert lora_grads and all(g is not None and g.dtype == torch.float32 for g in lora_grads)
    assert model.recurrence.proj.weight.grad is not None
    # Cached decode runs through the same bf16 backbone.
    model.eval()
    with torch.no_grad():
        decoded = model(batch, use_cache=True).predictions["action_value"]
    assert decoded.dtype == torch.float32


def test_bf16_reasoner_forward_on_cpu() -> None:
    torch.manual_seed(0)
    model = _model(reasoner=True, dtype=torch.bfloat16).train()
    batch = _batch(model)
    out = model(batch, reasoning=[1, -1])
    out.predictions["action_value"].sum().backward()
    assert model.reasoner is not None
    assert model.reasoner.proj.weight.grad is not None


def test_adamw_trains_bf16_backbone_model_through_lora() -> None:
    torch.manual_seed(0)
    model = _model(dtype=torch.bfloat16).train()
    optimizer = AdamW(model.parameters(), lr=1e-3, fused=False)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == n_trainable
    assert model.backbone is not None
    lora_b = next(p for n, p in model.backbone.named_parameters() if ".lora_B." in n)
    base = next(p for n, p in model.backbone.named_parameters() if n.endswith(".base.weight"))
    base_before = base.detach().clone()
    model(_batch(model)).predictions["action_value"].sum().backward()
    optimizer.step()
    assert not torch.equal(lora_b, torch.zeros_like(lora_b))
    assert torch.equal(base, base_before)


# ---- full fp32 fine-tuning (no LoRA) -----------------------------------------


def test_full_fp32_path_trains_backbone_directly_with_adamw_and_polyak() -> None:
    """No LoRA: keep the model fp32, the base weights step in place, Polyak lerps them."""
    torch.manual_seed(0)
    model = _model(None).train().to(device=torch.device("cpu"))
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    assert model.backbone is not None and model.backbone.dtype == torch.float32
    delayed = model.delayed_copy()
    assert delayed.backbone is not None
    online = dict(model.backbone.named_parameters())
    for name, p in delayed.backbone.named_parameters():
        assert p is not online[name]  # trainable base: copied, not shared
    polyak = Polyak(model, delayed)
    optimizer = AdamW(model.parameters(), lr=1e-3, fused=False)
    assert sum(len(g["params"]) for g in optimizer.param_groups) == sum(1 for _ in model.parameters())

    q_proj = model.backbone.model.layers[0].self_attn.q_proj  # type: ignore[union-attr]
    assert isinstance(q_proj, nn.Linear)
    before = q_proj.weight.detach().clone()
    delayed_before = delayed.backbone.model.layers[0].self_attn.q_proj.weight.detach().clone()  # type: ignore[union-attr]
    model(_batch(model)).predictions["action_value"].sum().backward()
    assert q_proj.weight.grad is not None
    optimizer.step()
    assert not torch.equal(q_proj.weight, before)
    polyak.update(tau_heads=0.5, tau_encoder=0.5, tau_backbone=0.5)
    delayed_after = delayed.backbone.model.layers[0].self_attn.q_proj.weight  # type: ignore[union-attr]
    assert torch.allclose(delayed_after, 0.5 * delayed_before + 0.5 * q_proj.weight)


def test_full_fp32_model_cast_to_bf16_is_rejected_by_adamw_and_polyak() -> None:
    torch.manual_seed(0)
    model = _model(None, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="dtype=torch.float32"):
        AdamW(model.parameters(), lr=1e-3, fused=False)
    delayed = model.delayed_copy()
    with pytest.raises(TypeError, match="fp32 parameters only"):
        Polyak(model, delayed)


# ---- save / load -------------------------------------------------------------


def test_save_load_roundtrip_with_lora(tmp_path) -> None:
    torch.manual_seed(0)
    model = _model(LoRAConfig(rank=2, alpha=4.0, targets=("q_proj", "v_proj"))).eval()
    with torch.no_grad():
        for adapter in lora_modules(model.backbone):  # type: ignore[arg-type]
            adapter.lora_B.weight.normal_()
    batch = _batch(model)
    with torch.no_grad():
        expected = model(batch).predictions["action_value"]
    save_model(model, tmp_path)
    with (tmp_path / "config.json").open() as fh:
        config = json.load(fh)
    assert config["backbone"]["lora"] == {
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.0,
        "targets": ["q_proj", "v_proj"],
    }
    loaded = load_model(tmp_path, train_kernel="varlen", decode_kernel="flex", dtype=torch.float32).eval()
    assert loaded.backbone is not None
    assert loaded.backbone.lora == model.backbone.lora  # type: ignore[union-attr]
    assert set(loaded.state_dict()) == set(model.state_dict())
    with torch.no_grad():
        actual = loaded(_batch(loaded)).predictions["action_value"]
    assert torch.allclose(actual, expected)


def test_save_load_roundtrip_without_lora_has_no_lora_key(tmp_path) -> None:
    model = _model(None).eval()
    save_model(model, tmp_path)
    with (tmp_path / "config.json").open() as fh:
        config = json.load(fh)
    assert "lora" not in config["backbone"]
    assert load_model(tmp_path, train_kernel="varlen", decode_kernel="flex", dtype=torch.float32).backbone.lora is None  # type: ignore[union-attr]


# ---- delayed copy / Polyak ---------------------------------------------------


def test_delayed_backbone_shares_frozen_base_and_copies_adapters() -> None:
    torch.manual_seed(0)
    model = _model(dtype=torch.bfloat16)
    delayed = model.delayed_copy()
    assert model.backbone is not None and delayed.backbone is not None
    online = dict(model.backbone.named_parameters())
    for name, p in delayed.backbone.named_parameters():
        if ".lora_" in name:
            assert p is not online[name]
            assert p.dtype == torch.float32 and not p.requires_grad
        else:
            assert p is online[name]  # frozen bf16 base shared by reference
    # Sharing did not freeze the online adapters.
    assert all(p.requires_grad for n, p in online.items() if ".lora_" in n)


def test_polyak_interpolates_lora_adapters_in_fp32_without_shadows() -> None:
    torch.manual_seed(0)
    model = _model(dtype=torch.bfloat16).eval()
    delayed = model.delayed_copy().eval()
    polyak = Polyak(model, delayed)
    assert model.backbone is not None and delayed.backbone is not None
    online_b = [m.lora_B.weight for m in lora_modules(model.backbone)]
    delayed_b = [m.lora_B.weight for m in lora_modules(delayed.backbone)]
    with torch.no_grad():
        for w in online_b:
            w.fill_(1.0)
    tau = 0.0005
    for _ in range(200):
        polyak.update(tau_heads=0.0, tau_encoder=0.0, tau_backbone=tau)
    expected = 1.0 - (1.0 - tau) ** 200
    for w in delayed_b:
        assert torch.allclose(w, torch.full_like(w, expected), atol=1e-6)
    batch = _batch(model)
    with torch.no_grad():
        online_q = model(batch).predictions["action_value"]
        delayed_q = delayed(batch).predictions["action_value"]
    assert not torch.allclose(online_q, delayed_q)
    polyak.update(tau_heads=1.0, tau_encoder=1.0, tau_backbone=1.0)
    with torch.no_grad():
        copied_q = delayed(batch).predictions["action_value"]
    assert torch.allclose(online_q, copied_q)
