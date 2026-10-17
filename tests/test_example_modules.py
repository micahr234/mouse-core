"""Exercise the actual notebook encoders, separate heads, and training loops."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from mouse_core import AdamW, polyak_update
from mouse_core.data import DataLoader, Tokenizer, to_device
from mouse_core.data.token_batch import TokenBatch
from mouse_core.models import (
    Backbone, Embedding, DecodeCache, LatentReasoner, MLP, Model,
    DiscretizedActionHead, prepare_decode, select_discrete_action,
    load_model, save_model, sample_reasoning_splits,
)
from mouse_core.objectives import (
    DqnObjective, SvObjective, SpObjective, PpoObjective, GrpoObjective,
    CosineObjective, best_action, boundary_discount, nstep_gate,
    value_gap_gate, general_gate,
)
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = [path for path in sorted(EXAMPLES.glob("*.ipynb"))
             if "class Encoder(Model):" in path.read_text()]


def _tokens(*actions):
    return batch_to_token_batch(token_tokenizer("action"), [[{"action": action} for action in actions]])


def _example(path):
    """Use the notebook's construction code with small local backend factories."""
    notebook = json.loads(path.read_text())
    namespace: dict[str, Any] = {**globals(), "device": torch.device("cpu")}
    # Preserve symbolic settings; no network, environment, or publication calls.
    for node in ast.parse("".join(notebook["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    namespace.update(
        PRETRAINED="local-test", MAX_ACTIONS=3, HIDDEN_DIM=8, BACKBONE_LAYERS=1,
        BACKBONE_HEADS=2, NUM_FREQUENCIES=2, NUM_THOUGHTS=2,
        vocabulary={str(i): i for i in range(32)},
        action_vectors=torch.tensor([[-2.], [0.], [2.]]),
    )
    construction = []

    def embedding_factory(**kwargs):
        construction.append(("embedding", kwargs))
        return Embedding(hidden_dim=8, vocab_size=32, num_frequencies=2)

    def backbone_factory(**kwargs):
        construction.append(("backbone", kwargs))
        return Backbone(architecture="qwen3", hidden_dim=8, num_layers=1, num_heads=2,
                        train_kernel="reference", decode_kernel="flex", dtype=torch.float32)

    namespace.update(Embedding=embedding_factory, Backbone=backbone_factory)
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "class Encoder(Model):" in "".join(cell["source"]))
    body = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            body.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "tokenizer" for t in node.targets):
                continue
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    assert [name for name, _ in construction] == ["embedding", "backbone"]
    if "modules" not in namespace:  # cosine example saves separate state dicts
        namespace["modules"] = nn.ModuleDict({
            name: namespace[name] for name in ("encoder", "q_head", "transition")
        })
    return notebook, namespace


def _heads(ns):
    if "q_heads" in ns:
        return list(ns["q_heads"].values())
    return [ns[name] for name in ("q_head", "policy_head", "value_head") if name in ns]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_encoders_and_heads_have_separate_gradients_and_cached_readouts(path):
    _, ns = _example(path)
    encoder = ns["encoder"]
    encoder.eval()
    assert set(dict(encoder.named_children())) == (
        {"embedding", "backbone", "reasoner"} if "reasoning" in path.stem
        else {"embedding", "backbone"}
    )
    assert not {id(p) for p in encoder.embedding.parameters()} & {id(p) for p in encoder.backbone.parameters()}
    out = encoder(_tokens(0, 1))
    assert "predictions" not in out
    torch.testing.assert_close(out["z"], out["last_hidden_state"][out["head_token_ids"]])
    heads = _heads(ns)
    torch.stack([head(out["z"]).square().mean() for head in heads]).sum().backward()
    for module in (encoder.embedding, encoder.backbone, *heads):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with torch.no_grad():
        first = encoder(_tokens(0), use_cache=True)
        try:
            last = encoder(_tokens(1), cache=first["cache"], use_cache=True)
            full_z = torch.cat([first["z"], last["z"]], dim=1).squeeze(0)
            torch.testing.assert_close(full_z, out["z"], atol=2e-5, rtol=2e-5)
            for head in heads:
                torch.testing.assert_close(head(full_z), head(out["z"]), atol=2e-5, rtol=2e-5)
        finally:
            first["cache"].close()
    if "delayed_encoder" in ns:
        target = ns["delayed_encoder"]
        assert not {id(p) for p in target.parameters()} & {id(p) for p in encoder.parameters()}
        assert all(not p.requires_grad for p in target.parameters())
        assert not target(_tokens(0, 1))["z"].requires_grad


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if "15_inference" not in p.stem], ids=lambda p: p.stem)
def test_notebook_training_uses_all_online_modules_and_updates_targets(path):
    notebook, ns = _example(path)
    tokens = _tokens(0, 1, 2)
    data = {
        "action": torch.tensor([0, 1, 2]), "reward": torch.tensor([0., 0.5, 1.]),
        "terminated": torch.tensor([False, False, True]),
        "truncated": torch.zeros(3, dtype=torch.bool),
        "repeat_episode_start": torch.tensor([True, False, False]),
        "repeat_episode_ended": torch.tensor([False, False, True]),
        "episode_ended": torch.tensor([False, False, True]),
        "old_log_prob": torch.full((3,), -1.1), "advantage": torch.tensor([0., 1., -1.]),
        "info_q_star": torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        "environment_step": torch.ones(3, dtype=torch.bool),
    }
    group_id = torch.zeros(3, dtype=torch.int64)

    class Loader:
        def __init__(self, **kwargs):
            pass
        def next_batch(self):
            return tokens, data, group_id
        def refresh(self):
            pass
        def close(self):
            pass
        def __len__(self):
            return 1

    ns.update(loader=Loader(), DataLoader=Loader, EpochDataLoader=Loader,
              stores=[[1]], store=[1], MAX_STEPS_PER_EPISODE=30,
              transform=None, tokenizer=None, ACTION_COMPONENTS=2)
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if "def run_train(" in "".join(c["source"]))
    exec(compile(cell, str(path), "exec"), ns)
    optimizer = ns["optimizer"]
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = [id(p) for p in ns["modules"].parameters() if p.requires_grad]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == set(expected)
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in ns["modules"].items()}
    delayed_before = ([p.detach().clone() for p in ns["delayed_encoder"].parameters()]
                      if "delayed_encoder" in ns else None)
    kwargs = {"epochs": 1} if "grpo" in path.stem else {"num_steps": 1}
    if path.stem.startswith(("19", "20")):
        kwargs["cycle"] = 0
    result = ns["run_train"](**kwargs)
    assert torch.isfinite(torch.as_tensor(result[0]))
    for name, module in ns["modules"].items():
        assert any(not torch.equal(old, new) for old, new in zip(before[name], module.parameters()))
    if delayed_before is not None:
        target = ns["delayed_encoder"]
        assert all(p.grad is None for p in target.parameters())
        assert any(not torch.equal(old, new) for old, new in zip(delayed_before, target.parameters()))
        # Check that both distinct backup heads track the online Q heads at their own rates.
        if "q_heads" in ns:
            triples = [(ns["q_heads"][key], ns["backup_value_heads"][key], ns["backup_action_heads"][key])
                       for key in ns["head_keys"]]
            initial = before["q_heads"]
        else:
            triples = [(ns["q_head"], ns["backup_value_head"], ns["backup_action_head"])]
            initial = before["q_head"]
        offset = 0
        for source, value, action in triples:
            for online, v, a in zip(source.parameters(), value.parameters(), action.parameters()):
                old = initial[offset]
                torch.testing.assert_close(v, old.lerp(online.detach(), ns["POLYAK_TAU_VALUE"]))
                torch.testing.assert_close(a, old.lerp(online.detach(), ns["POLYAK_TAU_ACTION"]))
                assert not v.requires_grad and not a.requires_grad
                offset += 1


def test_training_checkpoint_loads_in_inference_example(tmp_path):
    _, train = _example(EXAMPLES / "02_train_offline_dqn.ipynb")
    _, inference = _example(EXAMPLES / "15_inference.ipynb")
    train["modules"].eval()
    inference["modules"].eval()
    tokens = _tokens(0, 1)
    with torch.no_grad():
        expected = train["q_head"](train["encoder"](tokens)["z"])
    save_model(model=train["modules"], path=tmp_path)
    loaded = load_model(model=inference["modules"], repo_id_or_path=tmp_path)
    assert loaded is inference["modules"]
    with torch.no_grad():
        actual = inference["q_head"](inference["encoder"](tokens)["z"])
    torch.testing.assert_close(actual, expected)


def test_no_examples_keep_agent_projection_or_redundant_training_parameters():
    for path in EXAMPLES.glob("*.ipynb"):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    assert node.name not in {"Agent", "DelayedAgent"}
                if isinstance(node, ast.Name):
                    assert "projection" not in node.id.lower()
                if isinstance(node, ast.FunctionDef) and node.name == "run_train":
                    assert not {arg.arg for arg in node.args.kwonlyargs} & {
                        "model", "encoder", "q_head", "optimizer", "objective", "loader", "stores", "store",
                    }


def test_embedding_requires_prepared_inputs_on_the_model_device():
    _, namespace = _example(NOTEBOOKS[0])
    model = namespace["encoder"]
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
