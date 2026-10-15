"""Shared projection readouts, independent training rates, and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import torch

from mouse_core import AdamW, model_polyak
from mouse_core.models import LatentReasoner, Model, ProjectionHead, load_model, save_model
from mouse_core.models.backbone import IdentityBackbone, TransformerBackbone
from mouse_core.models.heads import ClassificationHead, DiscretizedActionHead, RegressionHead
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer


def _model(*, architecture: Literal["llama", "qwen3"] | None = None, reasoning: bool = False) -> Model:
    backbone = (
        IdentityBackbone(hidden_dim=8, vocab_size=32)
        if architecture is None
        else TransformerBackbone(
            architecture=architecture, hidden_dim=8, num_layers=1, num_heads=2,
            vocab_size=32, train_kernel="reference",
            decode_kernel="flex", dtype=torch.float32,
        )
    )
    return Model(
        backbone=backbone,
        projection=ProjectionHead(in_features=8, hidden_dim=12, num_layers=2),
        heads={
            "q": RegressionHead(
                in_features=8, out_features=4, hidden_dim=8, num_layers=1,
                propagate_gradient=1.0,
            ),
            "policy": ClassificationHead(
                in_features=8, out_features=4, hidden_dim=8, num_layers=1,
                propagate_gradient=1.0,
            ),
        },
        action_source="q",
        reasoner=LatentReasoner(hidden_dim=8, num_thoughts=2) if reasoning else None,
    )


def _batch(start: int = 0, stop: int = 3):
    tokenizer = token_tokenizer("action", "episode_done")
    rows = [[{"action": i % 4, "episode_done": 0} for i in range(start, stop)]]
    return batch_to_token_batch(tokenizer, rows)


@pytest.mark.parametrize("reasoning", [False, True])
def test_projection_runs_once_for_all_heads_and_keeps_backbone_states(reasoning: bool) -> None:
    model = _model(architecture="qwen3", reasoning=reasoning)
    assert model.projection is not None
    calls: list[torch.Tensor] = []

    def record(_module, _args, out) -> None:
        assert isinstance(out, torch.Tensor)
        calls.append(out)

    handle = model.projection.register_forward_hook(record)
    try:
        out = model(_batch(), reasoning=np.array([1]) if reasoning else None)
    finally:
        handle.remove()
    assert len(calls) == 1
    pooled = model.pool(output=out)
    assert pooled.shape == calls[0].shape == (3, 8)
    torch.testing.assert_close(calls[0], model.projection(pooled))
    for name, head in model.heads.items():
        torch.testing.assert_close(out.predictions[name], head(calls[0]))
    torch.stack([pred.square().mean() for pred in out.predictions.values()]).sum().backward()
    for section in (model.backbone, model.projection, model.heads, model.reasoner):
        if section is not None:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in section.parameters())


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_projection_cached_decode_matches_packed_forward(architecture: Literal["llama", "qwen3"]) -> None:
    model = _model(architecture=architecture).eval()
    with torch.no_grad():
        full = model(_batch())
        first = model(_batch(0, 2), use_cache=True)
        assert first.cache is not None
        try:
            last = model(_batch(2, 3), use_cache=True, cache=first.cache)
            for name in model.heads:
                actual = torch.cat([first.predictions[name], last.predictions[name]], dim=1)
                torch.testing.assert_close(actual.squeeze(0), full.predictions[name], atol=1e-5, rtol=1e-5)
        finally:
            first.cache.close()


def test_delayed_projection_reuses_raw_online_states() -> None:
    model = _model()
    delayed = model.copy(heads=True, backbone=False, reasoner=False, projection=True)
    assert delayed.projection is not model.projection
    assert delayed.projection is not None and model.projection is not None
    assert all(not p.requires_grad for p in delayed.projection.parameters())
    assert all(p.requires_grad for p in model.projection.parameters())
    with torch.no_grad():
        for p in model.projection.parameters():
            p.add_(0.2)
        online = model(_batch())
        reused = delayed.head(h=model.pool(output=online))
        separate = delayed(_batch())
    for name in delayed.heads:
        torch.testing.assert_close(reused[name], separate.predictions[name])
        assert not torch.allclose(reused[name], online.predictions[name])


def test_projection_learning_rate_steps_only_its_parameters() -> None:
    model = _model()
    assert model.projection is not None
    optimizer = AdamW(
        params=[
            {"params": model.backbone.parameters(), "lr": 0.0},
            {"params": model.projection.parameters(), "lr": 0.02},
            {"params": model.heads.parameters(), "lr": 0.0},
        ],
        lr=0.0, fused=False,
    )
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    for name, p in model.named_parameters():
        expected = before[name] - (0.02 if name.startswith("projection.") else 0.0)
        torch.testing.assert_close(p, expected)


def test_projection_polyak_rate_is_independent_and_can_change() -> None:
    model = _model()
    delayed = model.copy(heads=True, backbone=True, reasoner=False, projection=True)
    update = model_polyak(online=model, delayed=delayed)
    before = {name: p.detach().clone() for name, p in delayed.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(2.0)
    for tau in (0.0, 0.25, 1.0):
        update(tau_heads={"q": 0.0, "policy": 0.0}, tau_backbone=0.0, tau_projection=tau)
        for name, p in delayed.named_parameters():
            if name.startswith("projection."):
                expected = before[name].lerp(dict(model.named_parameters())[name], tau)
            else:
                expected = before[name]
            torch.testing.assert_close(p, expected)
            before[name] = p.detach().clone()


@pytest.mark.parametrize("tau", [None, -0.1, 1.1, float("nan")])
def test_invalid_projection_tau_cannot_partially_update_model(tau: float | None) -> None:
    model = _model()
    delayed = model.copy(heads=True, backbone=True, reasoner=False, projection=True)
    update = model_polyak(online=model, delayed=delayed)
    before = {name: p.detach().clone() for name, p in delayed.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    with pytest.raises(ValueError, match="tau_projection"):
        update(tau_heads=1.0, tau_backbone=1.0, tau_projection=tau)
    for name, p in delayed.named_parameters():
        assert torch.equal(p, before[name])


def test_shared_projection_and_projection_only_copy() -> None:
    model = _model()
    shared = model.copy(heads=True, backbone=False, reasoner=False, projection=False)
    assert shared.projection is model.projection
    update = model_polyak(online=model, delayed=shared)
    update(tau_heads=0.1, tau_backbone=None, tau_projection=None)
    update(tau_heads=0.1, tau_backbone=None, tau_projection=1.0)
    with pytest.raises(ValueError, match="tau_projection must be None or 1"):
        update(tau_heads=0.1, tau_backbone=None, tau_projection=0.1)
    projection_only = model.copy(heads=False, backbone=False, reasoner=False, projection=True)
    assert projection_only.heads["q"] is model.heads["q"]
    model_polyak(online=model, delayed=projection_only)(
        tau_heads=None, tau_backbone=None, tau_projection=0.1,
    )


def test_projection_copy_and_pairing_require_explicit_matching_sections() -> None:
    model = _model()
    with pytest.raises(ValueError, match="requires projection=True or False"):
        model.copy(heads=True, backbone=True, reasoner=False)
    delayed = model.copy(heads=True, backbone=True, reasoner=False, projection=True)
    delayed.projection = None
    with pytest.raises(ValueError, match="same projection section"):
        model_polyak(online=model, delayed=delayed)
    with pytest.raises(ValueError, match="projection=True requires a projection"):
        delayed.copy(heads=True, backbone=True, reasoner=False, projection=True)


@pytest.mark.parametrize("scale", [0.0, 0.3, 1.0])
def test_prediction_gradient_scale_also_controls_projection(scale: float) -> None:
    torch.manual_seed(0)
    model = _model()
    model(_batch()).predictions["q"].square().sum().backward()
    reference = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    cast(RegressionHead, model.heads["q"]).propagate_gradient = scale
    model(_batch()).predictions["q"].square().sum().backward()
    for name, p in model.named_parameters():
        if name not in reference:
            continue
        factor = 1.0 if name.startswith("heads.") else scale
        actual = torch.zeros_like(p) if p.grad is None else p.grad
        torch.testing.assert_close(actual, reference[name] * factor)


@pytest.mark.parametrize("num_layers", [2, 3])
def test_projection_checkpoint_roundtrip(tmp_path: Path, num_layers: int) -> None:
    model = _model()
    model.projection = ProjectionHead(in_features=8, hidden_dim=16, num_layers=num_layers)
    expected = model(_batch())
    save_model(model=model, path=tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["projection"] == {"in_features": 8, "hidden_dim": 16, "num_layers": num_layers}
    loaded = load_model(
        repo_id_or_path=str(tmp_path), train_kernel="reference", decode_kernel="flex", dtype=torch.float32,
    )
    assert loaded.projection is not None
    assert loaded.projection.num_layers == num_layers
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
    for name, pred in loaded(_batch()).predictions.items():
        torch.testing.assert_close(pred, expected.predictions[name])


@pytest.mark.parametrize("kwargs", [
    {"in_features": 0, "hidden_dim": 8},
    {"in_features": 8, "hidden_dim": 0},
    {"in_features": 8, "hidden_dim": 8, "num_layers": 1},
])
def test_projection_rejects_invalid_dimensions(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ProjectionHead(**kwargs)


def test_projection_rejects_incompatible_backbone_width() -> None:
    model = _model()
    with pytest.raises(ValueError, match="hidden_dim mismatch between projection"):
        Model(
            backbone=model.backbone, heads=model._heads, action_source="q", reasoner=None,
            projection=ProjectionHead(in_features=16, hidden_dim=16),
        )


def test_projection_and_prediction_prenorms_run_on_head_tokens() -> None:
    model = _model(architecture="qwen3")
    assert model.projection is not None
    seen: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, args) -> None:
            seen[name] = args[0].detach().clone()
        return hook

    modules = {"projection": model.projection.layers}
    for name, head in model.heads.items():
        modules[name] = cast(RegressionHead | ClassificationHead, head).layers
    handles = [module.register_forward_pre_hook(capture(name)) for name, module in modules.items()]
    try:
        with torch.no_grad():
            model.projection.input_norm.weight.fill_(2.0)
            cast(RegressionHead, model.heads["q"]).norm.weight.fill_(3.0)
            cast(ClassificationHead, model.heads["policy"]).norm.weight.fill_(4.0)
            output = model(_batch())
    finally:
        for handle in handles:
            handle.remove()
    assert output.last_hidden_state.shape[0] > 3
    for name, gain in (("projection", 2.0), ("q", 3.0), ("policy", 4.0)):
        assert seen[name].shape == (3, 8)
        torch.testing.assert_close(
            seen[name].square().mean(-1).sqrt(), torch.full((3,), gain), atol=1e-3, rtol=1e-3,
        )


@pytest.mark.parametrize("head_cls", [ProjectionHead, RegressionHead, ClassificationHead, DiscretizedActionHead])
@pytest.mark.parametrize("use_norm", [True, False])
def test_head_norms_are_not_configurable(head_cls, use_norm: bool) -> None:
    kwargs = {"in_features": 8, "hidden_dim": 8, "num_layers": 2, "use_norm": use_norm}
    if head_cls is DiscretizedActionHead:
        kwargs.update(action_vectors=torch.tensor([[-1.0], [1.0]]), propagate_gradient=1.0)
    elif head_cls is not ProjectionHead:
        kwargs.update(out_features=4, propagate_gradient=1.0)
    with pytest.raises(TypeError, match="use_norm"):
        head_cls(**kwargs)


@pytest.mark.parametrize("head_cls", [RegressionHead, ClassificationHead, DiscretizedActionHead])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("scale", [0.0, 0.1])
def test_prediction_scale_changes_only_final_linear_initialization(head_cls, num_layers: int, scale: float) -> None:
    def build(scale: float) -> RegressionHead | ClassificationHead:
        torch.manual_seed(42)
        kwargs: dict[str, Any] = dict(in_features=8, hidden_dim=12, num_layers=num_layers, propagate_gradient=1.0, scale=scale)
        if head_cls is DiscretizedActionHead:
            return head_cls(action_vectors=torch.tensor([[-1.0], [1.0]]), **kwargs)
        return head_cls(out_features=2, **kwargs)

    reference, scaled = build(1.0), build(scale)
    reference_params = dict(reference.named_parameters())
    for name, param in scaled.named_parameters():
        factor = scale if name.startswith(f"layers.{num_layers - 1}.") else 1.0
        torch.testing.assert_close(param, reference_params[name] * factor, rtol=0, atol=0)
    # A runtime multiplier would scale the linear bias a second time.
    final = cast(torch.nn.Linear, scaled.layers[-1])
    assert final.bias is not None
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(2.0)
    torch.testing.assert_close(scaled(torch.randn(3, 8)), torch.full((3, 2), 2.0))


def test_projection_has_no_scale_option() -> None:
    with pytest.raises(TypeError, match="scale"):
        ProjectionHead(in_features=8, hidden_dim=8, scale=0.1)  # type: ignore[call-arg]
