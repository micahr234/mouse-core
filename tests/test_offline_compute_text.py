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


def test_offline_tokenizer_samples_compute_before_each_value_without_extra_readouts():
    source = [dict(provider="environment", step_index=i,
                   payload=dict(observation=i, action_requested=True if i < 15 else None,
                                _terminated=i == 15, reward=float(i))) for i in range(16)]
    plain = _notebook_transform(NOTEBOOK.name, tokenizer_only=True)
    compute = _notebook_transform(NOTEBOOK.name, tokenizer_only=True, compute_range=(1, 10))
    baseline = plain.encode(source)
    sampled = compute.encode(source)
    counts = []
    for before, after in zip(baseline, sampled):
        delta = len(after.token_ids) - len(before.token_ids)
        assert after.objective_fields == before.objective_fields
        assert after.head_token_mask.sum() == before.head_token_mask.sum()
        if before.head_token_mask.any():
            count = delta // 2
            counts.append(count)
            assert 1 <= count <= 10 and delta == 2 * count
            np.testing.assert_array_equal(after.token_ids[-delta-1:-1], [32, 46] * count)
            assert after.token_ids[-1] == 10 and after.head_token_mask[-1]
        else:
            np.testing.assert_array_equal(after.token_ids, before.token_ids)
    assert len(set(counts)) > 1
    assert [len(s.token_ids) for s in compute.encode(source)] != [len(s.token_ids) for s in sampled]
    _, data = pack_token_batch(steps=sampled)
    assert data["head_step_ids"].tolist() == list(range(15))


def test_inference_callback_reconstructs_identical_text_without_sampling():
    notebook = json.loads((EXAMPLES / "15_inference.ipynb").read_text())
    node = next(node for cell in notebook["cells"] if cell["cell_type"] == "code"
                for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.FunctionDef) and node.name == "compute_text")
    ns: dict[str, Any] = {"INFERENCE_COMPUTE_REPEATS": 3}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<inference-callback>", "exec"), ns)
    formatter = ns["compute_text"]
    assert formatter({"action_requested": True}) == " ." * 3
    assert formatter({"action_requested": True}) == " ." * 3
    assert formatter({}) == ""


@pytest.mark.flex_training
def test_offline_dqn_update_backpropagates_into_compute_positions():
    notebook, ns = _example(NOTEBOOK, MAX_ACTIONS=4)
    log, _ = _collect_outputs("01_collect_dataset.ipynb", [
        (0, 0., False, False, False, False),
        (1, 0., False, False, False, False),
        (2, 1., True, False, True, False),
    ])
    tokenizer = _notebook_transform(NOTEBOOK.name, tokenizer_only=True, compute_range=(2, 4))
    inputs, data = pack_token_batch(steps=tokenizer.encode(log))
    # The test model has a 32-entry table; keep layout, features, and labels intact.
    inputs["token_ids"] %= 32
    ns["loader"] = SimpleNamespace(next_batch=lambda: (inputs, data))
    train = next("".join(c["source"]) for c in notebook["cells"]
                 if c["cell_type"] == "code" and "def run_train(" in "".join(c["source"]))
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
