"""Notebook 02's compute text preserves DQN labels and receives training gradients."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mouse_core.data import pack_token_batch
from tests.test_collection_events import _collect_outputs, _notebook_transform
from tests.test_example_modules import _example

EXAMPLES = Path(__file__).parents[1] / "examples"
NOTEBOOK = EXAMPLES / "02_train_offline_dqn.ipynb"


def test_offline_tokenizer_adds_fixed_compute_before_each_value_without_extra_readouts():
    source = [
        dict(
            provider="environment",
            step_index=i,
            payload=dict(
                observation=i,
                action_requested=True if i < 15 else None,
                _terminated=i == 15,
                reward=float(i),
            ),
        )
        for i in range(16)
    ]
    plain = _notebook_transform(NOTEBOOK.name, tokenizer_only=True)
    compute = _notebook_transform(NOTEBOOK.name, tokenizer_only=True, compute_tokens=3)
    baseline = plain.encode(source)
    sampled = compute.encode(source)
    for before, after in zip(baseline, sampled):
        delta = len(after.token_ids) - len(before.token_ids)
        assert after.objective_fields == before.objective_fields
        assert after.head_token_mask.sum() == before.head_token_mask.sum()
        if before.head_token_mask.any():
            count = delta // 2
            assert count == 3 and delta == 6
            np.testing.assert_array_equal(
                after.token_ids[-delta - 1 : -1], [32, 46] * count
            )
            assert after.token_ids[-1] == 10 and after.head_token_mask[-1]
        else:
            np.testing.assert_array_equal(after.token_ids, before.token_ids)
    assert [len(s.token_ids) for s in compute.encode(source)] == [
        len(s.token_ids) for s in sampled
    ]
    _, data = pack_token_batch(steps=sampled)
    assert data["head_step_ids"].tolist() == list(range(15))


def test_inference_callback_reconstructs_identical_text_without_sampling():
    notebook = json.loads((EXAMPLES / "15_inference.ipynb").read_text())
    node = next(
        node
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_text"
    )
    ns: dict[str, Any] = {"INFERENCE_COMPUTE_TOKENS": 3}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "<inference-callback>", "exec"
        ),
        ns,
    )
    formatter = ns["compute_text"]
    assert formatter({"action_requested": True}) == " ." * 3
    assert formatter({"action_requested": True}) == " ." * 3
    assert formatter({}) == ""


@pytest.mark.flex_training
def test_offline_dqn_update_backpropagates_into_compute_positions():
    notebook, ns = _example(NOTEBOOK, MAX_ACTIONS=4)
    log, _ = _collect_outputs(
        "01_collect_dataset.ipynb",
        [
            (0, 0.0, False, False, False, False),
            (1, 0.0, False, False, False, False),
            (2, 1.0, True, False, True, False),
        ],
    )
    tokenizer = _notebook_transform(
        NOTEBOOK.name, tokenizer_only=True, compute_tokens=None
    )
    inputs, data = pack_token_batch(
        steps=tokenizer.encode(log, context={"compute_tokens": 3})
    )
    # The test model has a 32-entry table; keep layout, features, and labels intact.
    inputs["token_ids"] %= 32
    ns["loader"] = SimpleNamespace(next_batch=lambda: (inputs, data))
    train = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def run_train(" in "".join(c["source"])
    )
    exec(compile(train, str(NOTEBOOK), "exec"), ns)
    embeddings = []

    def capture(module, args, output):
        output.retain_grad()
        embeddings.append(output)

    hook = ns["encoder"].embedding.register_forward_hook(capture)
    try:
        loss, _ = ns["run_train"](num_steps=1)
    finally:
        hook.remove()
    assert torch.isfinite(loss)
    last_compute = inputs["head_token_mask"].nonzero(as_tuple=True)[0] - 1
    assert embeddings[0].grad is not None
    assert embeddings[0].grad[last_compute].abs().sum() > 0
    assert all(p.grad is None for p in ns["delayed_encoder"].parameters())


def _compute_notebook_setup(filename, backend, **overrides):
    notebook = json.loads((EXAMPLES / filename).read_text())
    namespace: dict[str, Any] = {
        "tokenizer": SimpleNamespace(tokenizers={"text": backend})
    }
    settings = ast.parse("".join(notebook["cells"][1]["source"])).body
    for node in settings:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and "COMPUTE" in target.id:
                    namespace[target.id] = ast.literal_eval(node.value)
    namespace.update(overrides)
    body = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "compute_text",
                "training_context",
            }:
                body.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "compute_token_ids"
                for target in node.targets
            ):
                body.append(node)
            elif isinstance(node, ast.If) and "compute_token_ids" in ast.unparse(
                node.test
            ):
                body.append(node)
            elif (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and node.target.id == "compute_count"
            ):
                body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("filename", [NOTEBOOK.name, "15_inference.ipynb"])
def test_compute_setup_rejects_merging_or_multitoken_markers(filename):
    class MergingBackend:
        def encode(self, text, **kwargs):
            return [13] if text else []

    with pytest.raises(ValueError, match="exact token count"):
        _compute_notebook_setup(filename, MergingBackend())

    class MultitokenBackend:
        def encode(self, text, **kwargs):
            return [32, 46] if text else []

    with pytest.raises(ValueError, match="exactly one token"):
        _compute_notebook_setup(filename, MultitokenBackend())


def test_qwen_compute_text_has_exact_configured_token_counts():
    from transformers import AutoTokenizer

    from mouse_core.data import Tokenizer

    try:
        backend = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", local_files_only=True
        )
    except OSError:
        pytest.skip("Qwen tokenizer is not cached locally")
    namespace = _compute_notebook_setup(NOTEBOOK.name, backend)
    (token,) = namespace["compute_token_ids"]
    assert token == 659
    tok = Tokenizer(
        tokenizers={"text": backend},
        input_fields=[
            dict(type="text", provider="environment", format=namespace["compute_text"]),
            dict(
                type="token",
                provider="environment",
                input_field="action_requested",
                format="\n",
                head_output=True,
            ),
        ],
    )
    for count in (0, 1, 3, 10):
        piece = tok(
            dict(provider="environment", payload={"action_requested": True}),
            context={"compute_tokens": count},
        )
        assert piece["token_ids"].tolist() == [token] * count + backend.encode(
            "\n", add_special_tokens=False
        )
        assert piece["head_token_mask"].tolist() == [False] * count + [True]
    inference = _compute_notebook_setup("15_inference.ipynb", backend)
    text = inference["compute_text"]({"action_requested": True})
    assert (
        backend.encode(text, add_special_tokens=False)
        == [token] * inference["INFERENCE_COMPUTE_TOKENS"]
    )
    assert len(backend.encode("." * 10, add_special_tokens=False)) == 1


def test_notebook_compute_count_is_constant_across_batches_and_can_be_changed():
    class Backend:
        def encode(self, text, **kwargs):
            return [659] * (len(text) // 2)

    ns = _compute_notebook_setup(NOTEBOOK.name, Backend(), COMPUTE_TOKENS=3)
    context_for_batch = ns["training_context"]
    for index in (0, 1, 100):
        context = context_for_batch(index)
        assert context == {"compute_tokens": 3}
        assert (
            ns["compute_text"]({"action_requested": True}, context=context) == " ." * 3
        )
    ns["COMPUTE_TOKENS"] = 0
    assert (
        ns["compute_text"]({"action_requested": True}, context=context_for_batch(101))
        == ""
    )
    ns["COMPUTE_TOKENS"] = 8
    assert context_for_batch(102) == {"compute_tokens": 8}
    assert ns["compute_text"]({}, context=context_for_batch(102)) == ""
