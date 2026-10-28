"""Direct latent matching and combined DQN example gradient contracts."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from mouse_core.objectives import CosineObjective


@pytest.mark.parametrize(("target", "expected"), [
    ([2., 0.], 0.), ([0., 1.], 1.), ([-1., 0.], 2.), ([0., 0.], 1.),
])
def test_cosine_geometry(target, expected):
    losses, metrics = CosineObjective(mask_key=None)(
        objective_data={}, predictions=torch.tensor([[1., 0.]]),
        targets=torch.tensor([target]),
    )
    torch.testing.assert_close(losses["cosine"], torch.tensor(expected))
    torch.testing.assert_close(metrics["cosine_similarity"], torch.tensor(1 - expected))


def test_only_prediction_receives_gradient_and_masked_rows_are_excluded():
    predictions = torch.tensor([[[1., 0.], [0., 0.]]], requires_grad=True)
    targets = torch.tensor([[[0., 1.], [float("nan"), 0.]]], requires_grad=True)
    losses, metrics = CosineObjective(mask_key="skip")(
        objective_data={"skip": torch.tensor([[0, 2]])},
        predictions=predictions, targets=targets,
    )
    losses["cosine"].backward()
    assert targets.grad is None
    assert predictions.grad is not None
    torch.testing.assert_close(predictions.grad, torch.tensor([[[0., -1.], [0., 0.]]]))
    assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_zero_vectors_have_finite_loss_and_backward(dtype):
    predictions = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    losses, _ = CosineObjective(mask_key=None)(
        objective_data={}, predictions=predictions, targets=torch.zeros_like(predictions),
    )
    losses["cosine"].backward()
    assert torch.isfinite(losses["cosine"])
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


@pytest.mark.parametrize("eps", [0., -1., float("nan"), float("inf")])
def test_invalid_norm_floor(eps):
    with pytest.raises(ValueError, match="eps"):
        CosineObjective(mask_key=None, eps=eps)


@pytest.mark.parametrize(("predictions", "targets", "message"), [
    (torch.ones(2, 3), torch.ones(1, 3), "identical shape"),
    (torch.tensor(1.), torch.tensor(1.), "identical shape"),
    (torch.empty(0, 3), torch.empty(0, 3), "nonempty"),
    (torch.empty(2, 0), torch.empty(2, 0), "nonempty"),
])
def test_rejects_invalid_shapes(predictions, targets, message):
    with pytest.raises(ValueError, match=message):
        CosineObjective(mask_key=None)(objective_data={}, predictions=predictions, targets=targets)


def test_rejects_integer_encodings():
    with pytest.raises(TypeError, match="floating-point"):
        CosineObjective(mask_key=None)(
            objective_data={}, predictions=torch.ones(2, 3, dtype=torch.long), targets=torch.ones(2, 3),
        )


@pytest.mark.parametrize(("mask", "message"), [
    (torch.ones(2, dtype=torch.bool), "no pairs"),
    (torch.zeros(2, 1), "leading dimensions"),
])
def test_rejects_invalid_masks(mask, message):
    with pytest.raises(ValueError, match=message):
        CosineObjective(mask_key="skip")(
            objective_data={"skip": mask}, predictions=torch.ones(2, 3), targets=torch.ones(2, 3),
        )


def test_offline_dqn_cosine_example(monkeypatch):
    import ast
    from mouse_core import AdamW, polyak_update
    from mouse_core.data import DataLoader, to_device
    from mouse_core.objectives import DqnObjective, boundary_discount
    from copy import deepcopy
    from torch import nn
    from mouse_core.models import Backbone, Embedding, MultiEmbedding, MLP, Model, DecodeCache, prepare_decode
    from tests.test_example_modules import _tokens

    path = Path(__file__).resolve().parents[1] / "examples/21_train_dqn_cosine.ipynb"
    notebook = json.loads(path.read_text())
    namespace: dict[str, Any] = dict(
        torch=torch, nn=nn, MultiEmbedding=MultiEmbedding, Model=Model, MLP=MLP, DecodeCache=DecodeCache,
        prepare_decode=prepare_decode, AdamW=AdamW, polyak_update=polyak_update,
        DataLoader=DataLoader, to_device=to_device, DqnObjective=DqnObjective,
        boundary_discount=boundary_discount, CosineObjective=CosineObjective, Path=Path,
    )
    definitions: list[ast.stmt] = [node for node in ast.parse(
        "".join(notebook["cells"][7]["source"])
    ).body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    # Keep construction inside Encoder, substituting small local factories
    # for pretrained downloads while checking all forwarded settings.
    construction = []

    def make_embedding(*, pretrained):
        construction.append(("embedding", pretrained))
        return Embedding(hidden_dim=8, vocab_size=32)

    def make_backbone(*, pretrained, train_kernel, decode_kernel, dtype):
        construction.append(("backbone", pretrained, train_kernel, decode_kernel, dtype))
        return Backbone(architecture="qwen3", hidden_dim=8, num_layers=1,
                        num_heads=2, train_kernel=train_kernel, decode_kernel=decode_kernel,
                        dtype=dtype)

    monkeypatch.setitem(namespace, "Embedding", make_embedding)
    monkeypatch.setitem(namespace, "Backbone", make_backbone)
    encoder = namespace["Encoder"](
        pretrained="local-test", tokenizer_sources={"text": "local-test"}, train_kernel="reference", decode_kernel="flex",
        dtype=torch.float32,
    )
    assert construction == [
        ("embedding", "local-test"),
        ("backbone", "local-test", "reference", "flex", torch.float32),
    ]
    q_head = namespace["QHead"](hidden_dim=8, num_actions=3, num_layers=1, scale=0.1)
    assert q_head.mlp.in_features == q_head.mlp.hidden_dim == 8
    assert q_head.mlp.out_features == 3
    assert q_head.mlp.num_layers == 1 and q_head.mlp.scale == 0.1
    assert q_head.mlp.propagate_gradient == 1.0
    transition = namespace["TransitionModel"](
        latent_dim=8, num_actions=3, num_blocks=2,
    )
    assert transition.action_embedding.embedding_dim == 8
    for block in transition.blocks:
        assert block.in_features == 8
        assert block.num_layers == 2 and block.hidden_dim == 32 and block.scale is None
    assert isinstance(transition.output_norm, nn.RMSNorm)
    delayed_encoder = deepcopy(encoder).requires_grad_(False).eval()
    backup_value_head = deepcopy(q_head).requires_grad_(False).eval()
    namespace.update(
        encoder=encoder, q_head=q_head, transition=transition,
        delayed_encoder=delayed_encoder, backup_value_head=backup_value_head,
    )
    settings = ast.parse("".join(notebook["cells"][1]["source"]))
    assignments: list[ast.stmt] = [node for node in settings.body if isinstance(node, ast.Assign)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    namespace["device"] = torch.device("cpu")
    exec(compile("".join(notebook["cells"][9]["source"]), str(path), "exec"), namespace)
    tokens = _tokens(0, 1, 2, 0, 1)
    data = {
        "action": torch.tensor([0, 1, 2, 0, 1]),
        "reward": torch.tensor([0., 1., 0., 0., 1.]),
        "terminated": torch.tensor([False, False, True, False, True]),
        "truncated": torch.zeros(5, dtype=torch.bool),
        "episode_reset": torch.tensor([True, False, False, True, False]),
    }
    group_id = torch.tensor([0, 0, 0, 1, 1])
    out, target = encoder(tokens), delayed_encoder(tokens)
    dqn_losses, _ = namespace["objective"](
        objective_data=data, group_id=group_id, predictions=q_head(out["z"]),
        backup_value_predictions=backup_value_head(target["z"]),
        backup_action_predictions=q_head(target["z"]).detach(),
    )
    losses, _ = namespace["cosine_objective"].latent_cosine_loss(
        transition=transition, predictions=out["z"], targets=target["z"],
        reset_key="episode_reset", objective_data=data, group_id=group_id,
    )
    for loss, trained, untrained in (
        (torch.stack(tuple(dqn_losses.values())).sum(), (encoder.embedding, encoder.backbone, q_head), (transition,)),
        (losses["cosine"], (encoder.embedding, encoder.backbone, transition), (q_head,)),
    ):
        namespace["optimizer"].zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        for module in trained:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        for module in untrained:
            assert all(p.grad is None for p in module.parameters())
    assert all(p.grad is None for module in (delayed_encoder, backup_value_head)
               for p in module.parameters())

    # Capture pairs to verify incoming actions and reset/boundary exclusion.
    captured = []
    handle = transition.register_forward_pre_hook(lambda module, args: captured.append(args))
    namespace["cosine_objective"].latent_cosine_loss(
        transition=transition, predictions=out["z"], targets=target["z"],
        reset_key="episode_reset", objective_data=data,
        group_id=torch.zeros_like(group_id),
    )
    handle.remove()
    torch.testing.assert_close(captured[0][0], out["z"][[0, 1, 3]])
    assert torch.equal(captured[0][1], torch.tensor([1, 2, 1]))
    z = out["z"][0:1].expand(2, -1)
    predicted = transition(z, torch.tensor([0, 1]))
    assert not torch.allclose(predicted[0], predicted[1])

    # The action table is separate: with detached z, only the transition
    # embedding receives gradients, not the agent's text-token embedding.
    namespace["optimizer"].zero_grad(set_to_none=True)
    action_prediction = transition(z.detach(), torch.tensor([0, 1]))
    action_prediction.square().sum().backward()
    embedding_grad = transition.action_embedding.weight.grad
    assert embedding_grad is not None
    assert embedding_grad[:2].abs().sum() > 0
    assert torch.count_nonzero(embedding_grad[2:]) == 0
    assert encoder.embedding.embeddings["text"].embed_tokens.weight.grad is None
    assert transition.action_embedding.weight is not encoder.embedding.embeddings["text"].embed_tokens.weight
    assert "action_embedding.weight" in transition.state_dict()

    # With zero residual updates, every block preserves the conditioned input before final RMSNorm.
    from copy import deepcopy
    residual = deepcopy(transition)
    with torch.no_grad():
        for parameter in residual.blocks.parameters():
            parameter.zero_()
    state = torch.randn(2, 8, requires_grad=True)
    identity = residual(state, torch.tensor([0, 1]))
    torch.testing.assert_close(identity, residual.output_norm(
        residual.input_norm(state) + residual.action_embedding(torch.tensor([0, 1]))))
    identity.sum().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0

    class Loader:
        def next_batch(self):
            return tokens, data, group_id

    before = [p.detach().clone() for p in transition.parameters()]
    namespace["loader"] = Loader()
    total, metrics = namespace["run_train"](num_steps=1)
    assert torch.isfinite(total) and "cosine" in metrics
    assert any(not torch.equal(old, new) for old, new in zip(before, transition.parameters()))
    # No valid adjacent pairs contributes a differentiable zero.
    empty, _ = namespace["cosine_objective"].latent_cosine_loss(
        transition=transition, predictions=encoder(tokens)["z"], targets=delayed_encoder(tokens)["z"],
        reset_key="episode_reset",
        objective_data=data, group_id=torch.arange(5),
    )
    assert empty["cosine"].item() == 0
    empty["cosine"].backward()

    # The encoder returns the raw readout, with no Q head or projection.
    assert set(dict(encoder.named_children())) == {"embedding", "backbone"}
    assert set(out) == {"z", "cache"}
    encoder.eval()
    with torch.no_grad():
        full = encoder(_tokens(0, 1))["z"]
        first = encoder(_tokens(0), use_cache=True)
        try:
            second = encoder(_tokens(1), cache=first["cache"], use_cache=True)
            torch.testing.assert_close(
                torch.cat((first["z"], second["z"]), dim=1).squeeze(0),
                full, atol=2e-5, rtol=2e-5,
            )
        finally:
            first["cache"].close()


def test_sequential_cosine_pairs_boundaries_masks_and_detaches_targets():
    z = torch.randn(8, 3, requires_grad=True)
    target = torch.randn(8, 3, requires_grad=True)
    actions = torch.arange(8)
    data = {
        "action": actions,
        "terminated": torch.tensor([False, True, False, False, False, False, False, False]),
        "truncated": torch.tensor([False, False, True, False, False, False, False, False]),
        "reset": torch.tensor([False, False, False, False, True, False, False, False]),
        "skip": torch.tensor([0, 0, 0, 0, 1, 0, 0, 0]),
    }
    captured = []

    def transition(state, action):
        captured.append((state, action))
        return state

    losses, _ = CosineObjective(mask_key="skip").latent_cosine_loss(
        objective_data=data, predictions=z, targets=target,
        group_id=torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]),
        transition=transition, reset_key="reset",
    )
    torch.testing.assert_close(captured[0][0], z[[0, 6]])
    assert torch.equal(captured[0][1], actions[[1, 7]])
    losses["cosine"].backward()
    assert target.grad is None
    assert z.grad is not None and z.grad[[0, 6]].abs().sum() > 0
    assert torch.count_nonzero(z.grad[1:6]) == 0


@pytest.mark.parametrize("n", [0, 1])
def test_sequential_cosine_without_pairs_does_not_call_transition(n):
    def transition(state, action):
        raise AssertionError("transition should not run without pairs")

    z = torch.randn(n, 3, requires_grad=True)
    losses, _ = CosineObjective(mask_key=None).latent_cosine_loss(
        objective_data={
            "action": torch.zeros(n, dtype=torch.long),
            "terminated": torch.zeros(n, dtype=torch.bool),
            "truncated": torch.zeros(n, dtype=torch.bool),
        },
        predictions=z, targets=z.detach(), group_id=torch.zeros(n, dtype=torch.long),
        transition=transition, reset_key=None,
    )
    assert losses["cosine"].item() == 0
    losses["cosine"].backward()
    assert z.grad is not None


def test_sequential_cosine_skips_unexecuted_placeholder_actions():
    z = torch.randn(3, 2, requires_grad=True)
    captured = []

    def transition(state, action):
        captured.append(action)
        return state

    losses, _ = CosineObjective(mask_key=None).latent_cosine_loss(
        objective_data={
            "action": torch.tensor([999, 999, 1]),
            "action_executed": torch.tensor([False, False, True]),
            "terminated": torch.zeros(3, dtype=torch.bool),
            "truncated": torch.zeros(3, dtype=torch.bool),
        },
        predictions=z, targets=z.detach(), group_id=torch.zeros(3, dtype=torch.int64),
        transition=transition, reset_key=None,
    )
    torch.testing.assert_close(captured[0], torch.tensor([1]))
    losses["cosine"].backward()
    assert z.grad is not None
    torch.testing.assert_close(z.grad[0], torch.zeros(2))


def test_example_zero_weight_omits_transition_and_auxiliary_loss():
    from tests.test_example_modules import _example, EXAMPLES, _tokens
    notebook, ns = _example(EXAMPLES / "21_train_dqn_cosine.ipynb", COSINE_LOSS_WEIGHT=0.0)
    assert ns["transition"] is None
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "def run_train(" in "".join(c["source"]))
    exec(compile(source, "cosine-example", "exec"), ns)
    assert ns["cosine_objective"] is None
    data = dict(action=torch.tensor([0, 1]), reward=torch.tensor([0., 1.]),
                terminated=torch.tensor([False, True]), truncated=torch.tensor([False, False]))
    from types import SimpleNamespace
    ns["loader"] = SimpleNamespace(next_batch=lambda: (_tokens(0, 1), data, torch.zeros(2, dtype=torch.long)))
    loss, metrics = ns["run_train"](num_steps=1)
    assert torch.isfinite(loss) and "cosine" not in metrics
    optimized = {id(p) for g in ns["optimizer"].param_groups for p in g["params"]}
    expected = {id(p) for module in (ns["encoder"], ns["q_head"]) for p in module.parameters()}
    assert optimized == expected
