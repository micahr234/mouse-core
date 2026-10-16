"""Execute the actual custom notebook agents with small, local modules."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from mouse_core import polyak_update
from mouse_core.data.token_batch import TokenBatch
from mouse_core.models import (
    Backbone, Embedding, DecodeCache, LatentReasoner, MLP, Model,
    prepare_decode, select_discrete_action,
)
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = [path for path in sorted(EXAMPLES.glob("*.ipynb"))
             if "class Agent(Model):" in path.read_text()]


def _classes(path):
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "class Agent(Model):" in "".join(cell["source"]))
    definitions: list[ast.stmt] = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    namespace = {**globals(), "head_keys": ("short", "medium", "long")}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _online(path, classes):
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2, train_kernel='reference', decode_kernel='flex', dtype=torch.float32)
    if "n_step" in path.stem:
        outputs = {"short_head": "short", "medium_head": "medium", "long_head": "long"}
    elif "ppo" in path.stem:
        outputs = {"policy_head": "action", "value_head": "value"}
    elif "grpo" in path.stem or "_sp" in path.stem:
        outputs = {"policy_head": "action"}
    else:
        outputs = {"value_head": "action_value"}
    heads = {name: MLP(in_features=8, out_features=1 if key == "value" else 3, hidden_dim=8)
             for name, key in outputs.items()}
    reasoner = {"reasoner": LatentReasoner(hidden_dim=8, num_thoughts=2)} if "reasoning_dqn" in path.stem else {}
    model = classes["Agent"](
        embedding=Embedding(hidden_dim=8, vocab_size=32), backbone=backbone,
        projection=MLP(in_features=8, out_features=8, hidden_dim=8), **heads, **reasoner,
    )
    return model, outputs


def _tokens(*actions):
    return batch_to_token_batch(token_tokenizer("action"), [[{"action": action} for action in actions]])


def _delayed(path, classes, online):
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "delayed_model = DelayedAgent(" in "".join(cell["source"]))
    definition = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "delayed_model"
                              for target in node.targets))
    namespace = {**classes, "model": online}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["delayed_model"]


@pytest.mark.parametrize(("path", "use_delayed"), [
    pytest.param(path, delayed, id=f"{path.stem}-{'delayed' if delayed else 'online'}")
    for path in NOTEBOOKS for delayed in (False, True)
    if not delayed or "class DelayedAgent(Model):" in path.read_text()
])
def test_notebook_forward_uses_tensor_modules_and_preserves_decode(path, use_delayed, monkeypatch):
    classes = _classes(path)
    model, outputs = _online(path, classes)
    expected_keys = set(outputs.values())
    if use_delayed:
        model = _delayed(path, classes, model)
        expected_keys = ({kind + "_" + key for kind in ("value", "action") for key in outputs.values()}
                         if "n_step" in path.stem else {"backup_value", "backup_action"})
    model.eval()
    assert not {id(p) for p in model.embedding.parameters()} & {id(p) for p in model.backbone.parameters()}
    calls = []
    prepared = []
    original_to_tensors = TokenBatch.to_tensors

    def prepare(batch, device=None):
        assert device == next(model.parameters()).device
        tensors = original_to_tensors(batch, device)
        prepared.append(tensors)
        return tensors

    def before_embedding(module, args, kwargs):
        assert args[0] is prepared[-1]["token_ids"]
        for name in ("numeric_mask", "numeric_values"):
            assert kwargs[name] is prepared[-1][name]
        assert all(tensor.device == next(model.parameters()).device
                   for tensor in (*args, *kwargs.values()) if isinstance(tensor, torch.Tensor))

    monkeypatch.setattr(TokenBatch, "to_tensors", prepare)
    embedding_handle = model.embedding.register_forward_pre_hook(before_embedding, with_kwargs=True)

    def observe(module, args, output):
        assert isinstance(output, torch.Tensor)
        calls.append(module)

    stages = (model.embedding, model.backbone, model.projection)
    handles = [module.register_forward_hook(observe) for module in stages]
    out = model(_tokens(0, 1))
    assert type(out) is dict
    assert len(prepared) == 1
    for handle in handles:
        handle.remove()
    assert calls == list(stages)
    assert set(out["predictions"]) == expected_keys
    if use_delayed:
        assert all(not tensor.requires_grad for tensor in out["predictions"].values())
    else:
        torch.stack([tensor.square().mean() for tensor in out["predictions"].values()]).sum().backward()
        for module in (*stages, *(getattr(model, name) for name in outputs)):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with torch.no_grad():
        first = model(_tokens(0), use_cache=True)
        try:
            last = model(_tokens(1), cache=first["cache"], use_cache=True)
            for key in expected_keys:
                actual = torch.cat([first["predictions"][key], last["predictions"][key]], dim=1)
                torch.testing.assert_close(actual.squeeze(0), out["predictions"][key], atol=2e-5, rtol=2e-5)
        finally:
            first["cache"].close()
    assert len(prepared) == 3  # one preparation per full/cached forward
    embedding_handle.remove()


def test_embedding_requires_prepared_inputs_on_the_model_device():
    model, _ = _online(NOTEBOOKS[0], _classes(NOTEBOOKS[0]))
    tokens = _tokens(0)
    tensors = tokens.to_tensors("cpu")
    kwargs = {name: tensors[name] for name in ("numeric_mask", "numeric_values")}
    with pytest.raises(TypeError, match="expects tensor"):
        model.embedding(tokens, **kwargs)
    model.to("meta")
    with pytest.raises(ValueError, match="same device"):
        model.embedding(tensors["token_ids"], **kwargs)


def test_embedding_and_backbone_are_independently_constructed():
    embedding = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=2)
    backbone = Backbone(architecture="qwen3", hidden_dim=8, num_layers=1,
                        num_heads=2, train_kernel="reference",
                        decode_kernel="flex", dtype=torch.float32)
    assert not hasattr(backbone, "embedding")
    assert getattr(backbone.model, "embed_tokens") is None
    assert embedding.fourier is not None
    assert not {id(p) for p in embedding.parameters()} & {id(p) for p in backbone.parameters()}


def test_backbone_requires_explicit_packed_metadata():
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2, train_kernel='reference', decode_kernel='flex', dtype=torch.float32)
    embed = torch.randn(5, 8)
    with pytest.raises(ValueError, match="require sequence_ids"):
        backbone(embed)
    with pytest.raises(ValueError, match="sequence_ids"):
        backbone(embed, sequence_ids=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="attention kwargs"):
        backbone(embed, sequence_ids=torch.zeros(5, dtype=torch.long), attention_mask=torch.ones(5, 5))
    features = backbone(embed, sequence_ids=torch.zeros(5, dtype=torch.long))
    assert isinstance(features, torch.Tensor) and features.shape == embed.shape


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if "class DelayedAgent(Model):" in p.read_text()], ids=lambda path: path.stem)
def test_delayed_agent_has_independent_module_polyak_rates(path, monkeypatch):
    classes = _classes(path)
    online, outputs = _online(path, classes)
    multi = "n_step" in path.stem
    delayed = _delayed(path, classes, online)
    pairs = []
    for name in outputs:
        value = getattr(delayed, "value_" + name if multi else "value_head")
        action = getattr(delayed, "action_" + name if multi else "action_head")
        pairs.append((name, value, action, [p.clone() for p in value.parameters()]))
    assert type(delayed) is not type(online)
    assert not {id(p) for p in online.parameters()} & {id(p) for p in delayed.parameters()}
    assert all(not p.requires_grad for p in delayed.parameters())
    assert not hasattr(online, "action_head")
    before_modules = {name: [p.clone() for p in getattr(delayed, name).parameters()]
                      for name in ("embedding", "backbone", "projection", "reasoner") if hasattr(delayed, name)}
    with torch.no_grad():
        for param in online.parameters():
            param.add_(2)
    rates = {"embedding_tau": 0.375, "backbone_tau": 0.5,
             "projection_tau": 0.125, "value_tau": 0.25, "action_tau": 0.75}
    delayed.polyak(online=online, **rates)
    for name, before in before_modules.items():
        rate = rates["backbone_tau" if name == "reasoner" else f"{name}_tau"]
        for initial, target in zip(before, getattr(delayed, name).parameters(), strict=True):
            torch.testing.assert_close(target, initial + 2 * rate)
    for name, value, action, before in pairs:
        for initial, v, a in zip(before, value.parameters(), action.parameters(), strict=True):
            torch.testing.assert_close(v, initial + 0.5)
            torch.testing.assert_close(a, initial + 1.5)
            assert not v.requires_grad and not a.requires_grad
    # Invalid rates must fail before changing any module.
    snapshot = [p.clone() for p in delayed.parameters()]
    for name in rates:
        for invalid in (-0.1, 1.1, float("nan")):
            with pytest.raises(ValueError, match="Polyak rates"):
                delayed.polyak(online=online, **{**rates, name: invalid})
            for initial, target in zip(snapshot, delayed.parameters(), strict=True):
                torch.testing.assert_close(target, initial, rtol=0, atol=0)
    # A zero rate preserves values; a small rate accumulates in fp32.
    zero_rates = dict.fromkeys(rates, 0.)
    delayed.polyak(online=online, **zero_rates)
    for initial, target in zip(snapshot, delayed.parameters(), strict=True):
        torch.testing.assert_close(target, initial, rtol=0, atol=0)
    initial_values = [[p.clone() for p in value.parameters()] for _, value, _, _ in pairs]
    for _ in range(20):
        delayed.polyak(online=online, **{**zero_rates, "value_tau": 0.0005})
    for (name, value, _, _), before in zip(pairs, initial_values, strict=True):
        for source, initial, target in zip(getattr(online, name).parameters(), before, value.parameters(), strict=True):
            assert target.dtype == torch.float32
            torch.testing.assert_close(target, initial.lerp(source, 1 - (1 - 0.0005) ** 20))
    # Replacing online heads must affect the next call, with no stale bindings.
    for name in outputs:
        replacement = deepcopy(getattr(online, name))
        with torch.no_grad():
            for param in replacement.parameters():
                param.add_(3)
        setattr(online, name, replacement)
    delayed.polyak(online=online, **{**rates, "value_tau": 1., "action_tau": 1.})
    for name, value, action, _ in pairs:
        for source, v, a in zip(getattr(online, name).parameters(), value.parameters(), action.parameters(), strict=True):
            torch.testing.assert_close(v, source)
            torch.testing.assert_close(a, source)
    assert all(p.requires_grad for p in online.parameters())
    prepared = []
    original_to_tensors = TokenBatch.to_tensors

    def prepare(batch, device=None):
        assert device == next(delayed.parameters()).device
        tensors = original_to_tensors(batch, device)
        prepared.append(tensors)
        return tensors

    def before_embedding(module, args, kwargs):
        assert args[0] is prepared[-1]["token_ids"]
        for name in ("numeric_mask", "numeric_values"):
            assert kwargs[name] is prepared[-1][name]

    monkeypatch.setattr(TokenBatch, "to_tensors", prepare)
    handle = delayed.embedding.register_forward_pre_hook(before_embedding, with_kwargs=True)
    with torch.no_grad():
        out = delayed(_tokens(0, 1))
        assert type(out) is dict
        predictions = out["predictions"]
    handle.remove()
    assert len(prepared) == 1
    keys = {kind + "_" + key for kind in ("value", "action") for key in outputs.values()} if multi else {"backup_value", "backup_action"}
    assert set(predictions) == keys
    assert all(tensor.shape == (2, 3) for tensor in predictions.values())
