"""Shared projection readouts, independent training rates, and checkpoints."""

from __future__ import annotations
from tests._model_helpers import fresh_model, ExampleModel as Model
from tests._model_helpers import copy_model, simple_model

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest
import torch

from mouse_core import AdamW
from mouse_core.models import LatentReasoner, load_model, save_model
from mouse_core.models.mlp import MLP
from mouse_core.models.backbone import IdentityBackbone, Backbone
from mouse_core.models.heads import DiscretizedActionHead
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer


def _model(*, architecture: Literal['llama', 'qwen3'] | None=None, reasoning: bool=False) -> Model:
    backbone = IdentityBackbone(hidden_dim=8) if architecture is None else Backbone(architecture=architecture, hidden_dim=8, num_layers=1, num_heads=2, train_kernel='reference', decode_kernel='flex', dtype=torch.float32)
    return simple_model(backbone=backbone, projection=MLP(in_features=8, out_features=8, hidden_dim=12, num_layers=2), heads={'q': MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0), 'policy': MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0)}, reasoner=LatentReasoner(hidden_dim=8, num_thoughts=2) if reasoning else None)


def _batch(start: int = 0, stop: int = 3):
    tokenizer = token_tokenizer("action", "episode_done")
    rows = [[{"action": i % 4, "episode_done": 0} for i in range(start, stop)]]
    return batch_to_token_batch(tokenizer, rows)


@pytest.mark.parametrize('reasoning', [False, True])
def test_projection_runs_once_for_all_heads_and_keeps_backbone_states(reasoning: bool) -> None:
    model = _model(architecture='qwen3', reasoning=reasoning)
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
    selected = out["last_hidden_state"][out["head_token_ids"]]
    assert selected.shape == calls[0].shape == (3, 8)
    torch.testing.assert_close(calls[0], model.projection(selected))
    for name, head in model.heads.items():
        torch.testing.assert_close(out["predictions"][name], head(calls[0]))
    torch.stack([pred.square().mean() for pred in out["predictions"].values()]).sum().backward()
    for section in (model.embedding, model.backbone, model.projection, model.heads, model.reasoner):
        if section is not None:
            assert any((p.grad is not None and p.grad.abs().sum() > 0 for p in section.parameters()))


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_projection_cached_decode_matches_packed_forward(architecture: Literal["llama", "qwen3"]) -> None:
    model = _model(architecture=architecture).eval()
    with torch.no_grad():
        full = model(_batch())
        first = model(_batch(0, 2), use_cache=True)
        assert first["cache"] is not None
        try:
            last = model(_batch(2, 3), use_cache=True, cache=first["cache"])
            for name in model.heads:
                actual = torch.cat([first["predictions"][name], last["predictions"][name]], dim=1)
                torch.testing.assert_close(actual.squeeze(0), full["predictions"][name], atol=1e-5, rtol=1e-5)
        finally:
            first["cache"].close()


def test_delayed_projection_reuses_raw_online_states() -> None:
    model = _model()
    delayed = copy_model(model, heads=True, backbone=False, reasoner=False, projection=True)
    assert delayed.projection is not model.projection
    assert delayed.projection is not None and model.projection is not None
    assert all((not p.requires_grad for p in delayed.projection.parameters()))
    assert all((p.requires_grad for p in model.projection.parameters()))
    with torch.no_grad():
        for p in model.projection.parameters():
            p.add_(0.2)
        online = model(_batch())
        reused = delayed.head(h=online["last_hidden_state"][online["head_token_ids"]])
        separate = delayed(_batch())
    for name in delayed.heads:
        torch.testing.assert_close(reused[name], separate["predictions"][name])
        assert not torch.allclose(reused[name], online["predictions"][name])


def test_projection_learning_rate_steps_only_its_parameters() -> None:
    model = _model()
    assert model.projection is not None
    optimizer = AdamW(params=[{'params': model.backbone.parameters(), 'lr': 0.0}, {'params': model.projection.parameters(), 'lr': 0.02}, {'params': model.heads.parameters(), 'lr': 0.0}], lr=0.0, fused=False)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    for name, p in model.named_parameters():
        expected = before[name] - (0.02 if name.startswith('projection.') else 0.0)
        torch.testing.assert_close(p, expected)


def test_shared_projection_and_projection_only_copy() -> None:
    model = _model()
    shared = copy_model(model, heads=True, backbone=False, reasoner=False, projection=False)
    assert shared.projection is model.projection
    projection_only = copy_model(model, heads=False, backbone=False, reasoner=False, projection=True)
    assert projection_only.heads['q'] is model.heads['q']
    assert projection_only.projection is not model.projection
    for source, target in zip(model.projection.parameters(), projection_only.projection.parameters(), strict=True):
        torch.testing.assert_close(target, source)
        assert not target.requires_grad


def test_projection_copy_validates_module_paths() -> None:
    model = _model()
    delayed = model.copy(modules=["projection", "heads"])
    assert delayed.projection is not model.projection
    with pytest.raises(ValueError, match="unknown module path"):
        model.copy(modules=["missing"])


@pytest.mark.parametrize("scale", [0.0, 0.3, 1.0])
def test_prediction_gradient_scale_also_controls_projection(scale: float) -> None:
    torch.manual_seed(0)
    model = _model()
    model(_batch())["predictions"]["q"].square().sum().backward()
    reference = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    cast(MLP, model.heads["q"]).propagate_gradient = scale
    model(_batch())["predictions"]["q"].square().sum().backward()
    for name, p in model.named_parameters():
        if name not in reference:
            continue
        factor = 1.0 if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')) else scale
        actual = torch.zeros_like(p) if p.grad is None else p.grad
        torch.testing.assert_close(actual, reference[name] * factor)


@pytest.mark.parametrize('num_layers', [2, 3])
def test_projection_checkpoint_roundtrip(tmp_path: Path, num_layers: int) -> None:
    model = _model()
    model.projection = MLP(in_features=8, out_features=8, hidden_dim=16, num_layers=num_layers)
    expected = model(_batch())
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=str(tmp_path))
    assert isinstance(loaded.projection, MLP)
    assert loaded.projection.num_layers == num_layers
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
    for name, pred in loaded(_batch())["predictions"].items():
        torch.testing.assert_close(pred, expected["predictions"][name])


@pytest.mark.parametrize("kwargs", [
    {"in_features": 0, "hidden_dim": 8},
    {"in_features": 8, "hidden_dim": 0},
    {"in_features": 8, "hidden_dim": 8, "num_layers": 0},
])
def test_projection_rejects_invalid_dimensions(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MLP(out_features=8, **kwargs)


def test_projection_rejects_incompatible_backbone_width() -> None:
    model = _model()
    model.projection = MLP(in_features=16, out_features=16, hidden_dim=16)
    with pytest.raises(RuntimeError):
        model(_batch())


def test_projection_and_prediction_prenorms_run_on_head_tokens() -> None:
    model = _model(architecture='qwen3')
    assert model.projection is not None
    seen: dict[str, torch.Tensor] = {}

    def capture(name: str):

        def hook(_module, args) -> None:
            seen[name] = args[0].detach().clone()
        return hook
    modules = {'projection': cast(MLP, model.projection).layers}
    for name, head in model.heads.items():
        modules[name] = cast(MLP, head).layers
    handles = [module.register_forward_pre_hook(capture(name)) for name, module in modules.items()]
    norms = {"projection": cast(MLP, model.projection).input_norm,
             **{name: cast(MLP, head).input_norm for name, head in model.heads.items()}}
    handles += [norm.register_forward_pre_hook(capture(name + "_input")) for name, norm in norms.items()]
    try:
        with torch.no_grad():
            for name, gain in (("projection", 2.0), ("q", 3.0), ("policy", 4.0)):
                norm = norms[name]
                assert isinstance(norm, torch.nn.RMSNorm)
                norm.weight.fill_(gain)
            output = model(_batch())
    finally:
        for handle in handles:
            handle.remove()
    assert output["last_hidden_state"].shape[0] > 3
    for name, gain in (('projection', 2.0), ('q', 3.0), ('policy', 4.0)):
        assert seen[name].shape == (3, 8)
        torch.testing.assert_close(seen[name], torch.nn.functional.rms_norm(seen[name + "_input"], (8,), torch.full((8,), gain), eps=1e-5))


@pytest.mark.parametrize("head_cls", [MLP, DiscretizedActionHead])
@pytest.mark.parametrize("use_norm", [True, False])
def test_head_norms_are_not_configurable(head_cls, use_norm: bool) -> None:
    kwargs = {"in_features": 8, "hidden_dim": 8, "num_layers": 2, "use_norm": use_norm}
    if head_cls is DiscretizedActionHead:
        kwargs.update(action_vectors=torch.tensor([[-1.0], [1.0]]), propagate_gradient=1.0)
    else:
        kwargs.update(out_features=4, propagate_gradient=1.0)
    with pytest.raises(TypeError, match="use_norm"):
        head_cls(**kwargs)


@pytest.mark.parametrize("head_cls", [MLP, DiscretizedActionHead])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("scale", [0.0, 0.1])
def test_prediction_scale_changes_only_final_linear_initialization(head_cls, num_layers: int, scale: float) -> None:
    def build(scale: float) -> MLP:
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


def test_feature_projection_omits_output_scale() -> None:
    mlp = MLP(in_features=8, out_features=8, hidden_dim=8)
    assert mlp.scale is None
    assert type(mlp.layers[-1]) is torch.nn.Linear
