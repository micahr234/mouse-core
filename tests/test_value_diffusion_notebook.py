"""Exercise example 24's actual notebook definitions without Hub downloads."""

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mouse_core import AdamW
from mouse_core.data import make_token_batch, Tokenizer
from mouse_core.models import Backbone, Embedding
from mouse_core.objectives import boundary_discount


NOTEBOOK = Path(__file__).parents[1] / "examples/24_train_offline_value_diffusion.ipynb"


def definitions():
    notebook = json.loads(NOTEBOOK.read_text())
    nodes = [node for cell in notebook["cells"] if cell["cell_type"] == "code"
             for node in ast.parse("".join(cell["source"])).body
             if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def small_encoder(namespace):
    namespace["Embedding"] = lambda **_: Embedding(vocab_size=32, hidden_dim=16)
    namespace["Backbone"] = lambda **_: Backbone(
        architecture="llama", dtype=torch.float32, hidden_dim=16, num_layers=2, num_heads=2,
    )
    return namespace["Encoder"](
        pretrained="local", tokenizer_sources={"text": "local"}, dtype=torch.float32,
    )


def tokens():
    return make_token_batch(
        token_ids=[1, 2, 3, 4, 5, 6], numeric_values=np.zeros(6), numeric_mask=np.zeros(6, dtype=bool),
        sequence_ids=[17, 17, 17, 17, 9, 9], head_mask=[False, True] * 3,
        num_sequences=2, tokenizer_names=("text",),
    )


def small_head(ns):
    return ns["ValueFlow"](context_dim=16, hidden_dim=8, num_actions=2,
                       num_layers=2, scale=.1, return_num_frequencies=2, time_num_frequencies=2,
                       num_steps=3)


def test_context_is_causal_and_does_not_reveal_seed_beyond_coupled_input():
    ns = definitions()
    encoder = small_encoder(ns).eval()
    batch = tokens()
    changed = batch.clone()
    with torch.no_grad():
        out = encoder(batch)
        other = encoder(changed)
        assert out["z"].shape == (3, 16)
        torch.testing.assert_close(out["z"], other["z"])
        # Future tokens and another task cannot change the first readout.
        changed["token_ids"][2:] = 1
        future = encoder(changed)
        torch.testing.assert_close(out["z"][0], future["z"][0])
    interpolants = torch.randn(3, 2)
    head = small_head(ns)
    expected_context = head.context_projection(torch.nn.functional.rms_norm(
        out["z"], (16,), head.context_norm.weight, eps=1e-5))
    torch.testing.assert_close(head._context(out["z"]), expected_context)
    times = torch.tensor([0., .5])
    velocities = head(out["z"], interpolants, times=times)
    assert velocities.shape == (3, 2, 2)
    torch.testing.assert_close(velocities, head(other["z"], interpolants, times=times))
    assert not torch.equal(velocities, head(out["z"], interpolants + 1., times=times))
    assert not torch.equal(velocities, head(out["z"], interpolants, times=times + .25))
    velocities.square().mean().backward()
    for module in (head.mlp, head.return_features, head.time_features, head.context_norm, head.context_projection, head.input_projection):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    groups = encoder.parameter_groups(lr={"embedding": 1e-4, "backbone": 1e-4})
    assert {id(p) for group in groups for p in group["params"]} == {id(p) for p in encoder.parameters()}


def test_models_encode_identically_without_sampling():
    ns = definitions()
    encoder = small_encoder(ns).eval()
    delayed = deepcopy(encoder).requires_grad_(False).eval()
    batch = tokens()
    with torch.no_grad():
        before = torch.random.get_rng_state()
        online, target = encoder(batch), delayed(batch)
        assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(online["z"], target["z"])


def test_standalone_flow_steps_are_independent_of_transformer_depth():
    ns = definitions()
    encoder = small_encoder(ns).eval()
    head = small_head(ns).eval()
    with torch.no_grad():
        out = encoder(tokens())
        noise = torch.randn(3, 1)
        trajectory = head.trajectory(out["z"], initial_samples=noise[:, 0])
        samples, velocities, returns = trajectory.samples, trajectory.velocities, trajectory.returns
        assert samples.shape == velocities.shape == (3, 3, 2)
        torch.testing.assert_close(samples[:, 0], noise.expand(-1, 2))
        for step in range(3):
            expected = head(out["z"], samples[:, step:step + 1],
                                     times=trajectory.noise_levels[step:step + 1]).squeeze(1)
            torch.testing.assert_close(velocities[:, step], expected)
            following = samples[:, step + 1] if step < 2 else returns
            torch.testing.assert_close(following, samples[:, step] + expected / 3)



def test_frozenlake_pipeline_is_identical_to_example_two():
    from tests.test_collection_events import _collect_outputs, _notebook_transform
    log, _ = _collect_outputs("01_collect_dataset.ipynb", [
        (0, 0., False, False, False, False),
        (1, 0., True, False, False, False),
        (2, 0., False, False, False, False),
        (3, 1., True, False, True, False),
    ])
    base = _notebook_transform("02_train_offline_dqn.ipynb", compute_tokens=2)(log)
    actual = _notebook_transform(NOTEBOOK.name, compute_tokens=2)(log)
    assert len(base) == len(actual)
    for before, after in zip(base, actual):
        np.testing.assert_array_equal(before.token_ids, after.token_ids)
        np.testing.assert_array_equal(before.head_mask, after.head_mask)
        assert before.objective_fields == after.objective_fields


@pytest.mark.parametrize("backend", ["pointwise", pytest.param("flex", marks=pytest.mark.flex_training)])
def test_notebook_training_loop_updates_transformer_and_conditional_network(backend):
    ns = definitions()
    device = "cuda" if backend == "flex" else "cpu"
    encoder = small_encoder(ns)
    if backend == "pointwise":
        # Exercise the real notebook training plumbing on CPU. Full causal
        # attention backward is tested by the separate CUDA parameterization.
        class PointwiseBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
            def forward(self, x, **kwargs):
                layers = []
                for layer in self.model.layers:
                    x = layer(x).tanh()
                    layers.append(x)
                return x
        encoder.backbone = PointwiseBackbone()
    encoder.to(device)
    head = small_head(ns).to(device)
    delayed = deepcopy(encoder).requires_grad_(False).eval()
    backup_head = deepcopy(head).requires_grad_(False).eval()
    modules = nn.ModuleDict({"encoder": encoder, "flow": head})
    # Two complete groups: two decisions, then one decision, with terminal
    # result rows present in labels but no terminal readout in model inputs.
    from tensordict import TensorDict
    labels = TensorDict({
        "action": torch.tensor([torch.nan, 0., 1., torch.nan, 1.]),
        "reward": torch.tensor([0., 0.1, 1., 0., -1.]),
        "terminated": torch.tensor([False, False, True, False, True]),
        "truncated": torch.zeros(5, dtype=torch.bool),
        "sequence_ids": torch.tensor([17, 17, 17, 9, 9]),
        "step_ids": torch.tensor([0, 1, 3]),
    }, batch_size=[])
    ns.update(encoder=encoder, flow=head, delayed_encoder=delayed, delayed_flow=backup_head,
              modules=modules, device=torch.device(device), MAX_ACTIONS=2,
              loader=SimpleNamespace(next_batch=lambda: (tokens(), labels)),
              optimizer=AdamW(params=[{"params": modules.parameters(), "lr": 0.001}]),
              objective=ns["ValueDiffusionObjective"](
                  reward=None, discount=boundary_discount(
                      gamma_step=1., gamma_terminated=0., gamma_truncated=0.)),
              POLYAK_TAU_EMBEDDING=1., POLYAK_TAU_BACKBONE=1., POLYAK_TAU_DIFFUSION=1.)
    online_inputs, delayed_inputs, conditioned, layer_readouts = [], [], [], []
    def retain_readouts(module, args, out):
        out["z"].retain_grad()
        layer_readouts.append(out["z"])

    trajectories, correction_noise = [], []
    original_trajectory = backup_head.trajectory
    original_inputs = ns["objective"].inputs

    def record_trajectory(features, *, initial_samples):
        result = original_trajectory(features, initial_samples=initial_samples)
        trajectories.append((initial_samples.detach().clone(), result))
        return result

    def record_inputs(**kwargs):
        correction_noise.append(kwargs["noise"].detach().clone())
        return original_inputs(**kwargs)

    backup_head.trajectory = record_trajectory
    ns["objective"].inputs = record_inputs
    hooks = [
        encoder.register_forward_hook(retain_readouts),
        encoder.register_forward_pre_hook(lambda module, args: online_inputs.append(args[0])),
        delayed.register_forward_pre_hook(lambda module, args: delayed_inputs.append(args[0])),
        head.return_features.register_forward_pre_hook(
            lambda module, args: conditioned.append(args[0].detach().clone())),
    ]
    try:
        loss, metrics = ns["run_train"](num_steps=1)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(online_inputs) == len(delayed_inputs) == 1
    assert len(conditioned) == 1  # One external online evaluation at the transformed inputs.
    assert online_inputs[0] is delayed_inputs[0]
    torch.testing.assert_close(conditioned[-1], metrics["flow_inputs"].flatten())
    seeds, trajectory = trajectories[0]
    assert seeds.shape == correction_noise[0].shape == (3,)
    torch.testing.assert_close(seeds[0], seeds[1])  # s and s' belong to the same sequence.
    torch.testing.assert_close(trajectory.samples[:, 0], seeds[:, None].expand(-1, 2))
    next_action = trajectory.returns[1].argmax()
    times = trajectory.noise_levels
    torch.testing.assert_close(metrics["flow_inputs"][0],
                               times * .1 + trajectory.samples[1, :, next_action])
    torch.testing.assert_close(metrics["backup"][0], .1 + trajectory.velocities[1, :, next_action])
    for row, reward in ((1, 1.), (2, -1.)):
        torch.testing.assert_close(metrics["flow_inputs"][row],
                                   times * reward + (1 - times) * correction_noise[0][row])
        torch.testing.assert_close(metrics["backup"][row],
                                   (reward - correction_noise[0][row]).expand_as(times))
    assert torch.isfinite(loss)
    # The final context passes gradients back through the encoder.
    assert (layer_readouts[0].grad.abs().sum(-1) > 0).all()
    for component in (encoder.embedding, encoder.backbone, head.mlp, head.return_features, head.time_features, head.context_norm, head.context_projection, head.input_projection):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    for source, target in zip(encoder.parameters(), delayed.parameters()):
        torch.testing.assert_close(source, target)
    for source, target in zip(head.parameters(), backup_head.parameters()):
        torch.testing.assert_close(source, target)
