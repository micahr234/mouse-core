from __future__ import annotations
from tests._model_helpers import ExampleModel as Model
from tests._model_helpers import copy_model, simple_model

from typing import Any

import pytest
import torch
import torch.nn as nn

from mouse_core.models import LatentReasoner
from mouse_core.models.backbone import IdentityBackbone, Backbone
from mouse_core.models.mlp import MLP
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action", "repeat_episode_done")
_BATCH = [
    [
        {"action": 0, "reward": 0.0, "repeat_episode_done": 0, "episode_done": 0},
        {"action": 1, "reward": 1.0, "repeat_episode_done": 0, "episode_done": 0},
        {"action": 2, "reward": 2.0, "repeat_episode_done": 1, "episode_done": 0},
    ]
]


def _head(hidden_dim: int) -> MLP:
    return MLP(
        in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1,
        propagate_gradient=1.0,
    )


def _tiny_model() -> Model:
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim)
    head = _head(hidden_dim)
    return simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None)


def _llama_model() -> Model:
    hidden_dim = 16
    backbone = Backbone(architecture='llama', train_kernel='reference', decode_kernel='flex', dtype=torch.float32, hidden_dim=hidden_dim, num_layers=2, num_heads=2, max_position_embeddings=64)
    head = _head(hidden_dim)
    return simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None)


def _token_batch(model: Model):
    return batch_to_token_batch(_TOK, _BATCH)


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)


def _q_close(a: dict[str, Any], b: dict[str, Any], key: str = "action_value") -> bool:
    return torch.allclose(a["predictions"][key], b["predictions"][key], atol=1e-5)


def _count_calls(module: nn.Module, name: str = "forward"):
    orig = getattr(module, name)
    calls = {"n": 0}

    def _wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    setattr(module, name, _wrapped)
    return calls


# ---- copy ---------------------------------------------------------


def test_copy_is_a_frozen_full_copy() -> None:
    model = _llama_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    assert delayed.backbone is not model.backbone
    assert delayed.reasoner is None
    assert delayed.training
    assert all((not p.requires_grad for p in delayed.parameters()))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        assert p is not online[name]
        assert torch.equal(p, online[name])
    assert all((p.requires_grad for p in model.parameters()))


def test_copy_shares_frozen_parameters_by_reference() -> None:
    model = _llama_model()
    model.embedding.requires_grad_(False)
    model.backbone.requires_grad_(False)
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')):
            assert p is not online[name]
        else:
            assert p is online[name]


def test_copy_rejects_a_model_with_nothing_trainable() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    with pytest.raises(ValueError, match='trainable online model'):
        copy_model(delayed, heads=True, backbone=True, reasoner=False)
    model.requires_grad_(False)
    with pytest.raises(ValueError, match='trainable online model'):
        copy_model(model, heads=True, backbone=True, reasoner=False)


def _two_head_model(hidden_dim: int=8) -> Model:
    q_head = _head(hidden_dim)
    return simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim), heads={'action_value': q_head, 'behavior': MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)}, reasoner=None)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_tensor_reuse_preserves_separate_delayed_forward_and_online_gradients(monkeypatch, device) -> None:
    from mouse_core.data import reuse_token_tensors
    if device == 'cuda' and (not torch.cuda.is_available()):
        pytest.skip('CUDA required')
    torch.manual_seed(0)
    model = _llama_model().to(device=torch.device(device)).eval()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    _perturb(model.heads)
    batch = _token_batch(model)
    conversions = []
    from_numpy = torch.from_numpy

    def record(array):
        conversions.append(array)
        return from_numpy(array)
    monkeypatch.setattr(torch, 'from_numpy', record)

    def forward_pair():
        online_q = model(batch)["predictions"]['action_value']
        with torch.no_grad():
            delayed_q = delayed(batch)["predictions"]['action_value']
        (online_q - delayed_q).square().mean().backward()
        return (online_q, delayed_q)
    expected = forward_pair()
    expected_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    uncached_count = len(conversions)
    model.zero_grad(set_to_none=True)
    conversions.clear()
    with reuse_token_tensors():
        actual = forward_pair()
    assert len(conversions) < uncached_count
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    for name, param in model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(param.grad, expected_grads[name], rtol=0, atol=0)
    assert delayed.backbone is not model.backbone
    assert all((p.grad is None for p in delayed.parameters()))


def test_copy_heads_true_copies_every_head() -> None:
    model = _two_head_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    assert tuple(delayed.heads) == ('action_value', 'behavior')
    assert delayed.heads['action_value'] is not model.heads['action_value']
    assert delayed.heads['behavior'] is not model.heads['behavior']


def test_copy_carries_only_the_named_heads() -> None:
    model = _two_head_model()
    delayed = copy_model(model, heads=(model.heads['action_value'],), backbone=True, reasoner=False)
    assert tuple(delayed.heads) == ('action_value',)
    assert not any((name.startswith('heads.behavior') for name, _ in delayed.named_parameters()))
    behavior_only = copy_model(model, heads=(model.heads['behavior'],), backbone=True, reasoner=False)
    assert tuple(behavior_only.heads) == ('behavior',)


def test_copy_validates_module_paths() -> None:
    model = _two_head_model()
    for paths in ("heads.action_value", ["heads.action_value", "heads.action_value"], []):
        with pytest.raises(ValueError, match="distinct module paths"):
            model.copy(modules=paths)
    with pytest.raises(ValueError, match="unknown module path"):
        model.copy(modules=["missing"])
    with pytest.raises(ValueError, match="overlap"):
        model.copy(modules=["heads", "heads.action_value"])


def test_copy_carries_reasoner() -> None:
    hidden_dim = 8
    reasoning = simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim), heads=({'action_value': (head := _head(hidden_dim))}), reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1))
    dr = copy_model(reasoning, heads=True, backbone=True, reasoner=True)
    assert dr.reasoner is not None and dr.reasoner is not reasoning.reasoner
    shared = copy_model(reasoning, heads=True, backbone=False, reasoner=False)
    assert shared.backbone is reasoning.backbone
    assert shared.reasoner is reasoning.reasoner


# ---- delayed forward --------------------------------------------------------


def test_delayed_model_matches_online_before_update_and_builds_no_graph() -> None:
    torch.manual_seed(0)
    model = _llama_model().eval()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    batch = _token_batch(model)
    out = model(batch)
    saved = {'n': 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved['n'] += 1
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.no_grad():
            delayed_out = delayed(batch)
    assert saved['n'] == 0
    assert delayed_out["predictions"]['action_value'].grad_fn is None
    assert _q_close(out, delayed_out)


def test_delayed_model_reruns_its_own_trunk() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    batch = _token_batch(model)
    emb_calls = _count_calls(delayed.embedding)
    bb_calls = _count_calls(delayed.backbone)
    with torch.no_grad():
        delayed(batch)
    assert emb_calls['n'] == 1 and bb_calls['n'] == 1


def test_delayed_copy_ignores_online_changes() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    batch = _token_batch(model)
    with torch.no_grad():
        before = delayed(batch)
    _perturb(model.embedding)
    _perturb(model.backbone)
    _perturb(model.heads)
    with torch.no_grad():
        online = model(batch)
        still_delayed = delayed(batch)
    assert _q_close(before, still_delayed)
    assert not _q_close(online, still_delayed)


def test_copy_shares_uncopied_backbone() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=False, reasoner=False)
    assert delayed.backbone is model.backbone
    assert all((p.requires_grad for p in model.backbone.parameters()))
    assert all((not p.requires_grad for p in delayed.heads.parameters()))


def test_copy_shares_uncopied_heads() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=False, backbone=True, reasoner=False)
    assert delayed.heads['action_value'] is model.heads['action_value']
    assert all((p.requires_grad for p in model.heads.parameters()))
    assert all((not p.requires_grad for p in delayed.backbone.parameters()))


def test_selected_tokens_and_head_match_delayed_heads_on_shared_backbone() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    delayed = copy_model(model, heads=True, backbone=False, reasoner=False)
    out = model(_token_batch(model))
    saved = {'n': 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved['n'] += 1
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.no_grad():
            preds = delayed.head(h=out["last_hidden_state"][out["head_token_ids"]])
    assert saved['n'] == 0
    assert preds['action_value'].grad_fn is None
    assert torch.allclose(preds['action_value'], out["predictions"]['action_value'])


# ---- parameter updates ------------------------------------------------------


@pytest.mark.parametrize("tau", [0., 0.25, 1.])
def test_polyak_matches_names_independent_of_registration_order(tau):
    from mouse_core import polyak_update
    target, source = nn.Module(), nn.Module()
    target.register_parameter("first", nn.Parameter(torch.tensor([1., 2.])))
    target.register_parameter("second", nn.Parameter(torch.tensor([3., 4.])))
    source.register_parameter("second", nn.Parameter(torch.tensor([7., 8.])))
    source.register_parameter("first", nn.Parameter(torch.tensor([5., 6.])))
    target.register_buffer("counter", torch.tensor(1))
    source.register_buffer("counter", torch.tensor(9))
    before = {name: p.detach().clone() for name, p in target.named_parameters()}
    polyak_update(target=target, source=source, tau=tau)
    for name, p in target.named_parameters():
        torch.testing.assert_close(p, before[name] + 4 * tau)
    assert target.counter == 1


@pytest.mark.parametrize("mismatch", ["names", "broadcast_shape", "count", "dtype", "integer"])
def test_polyak_rejects_mismatch_before_updating_any_parameter(mismatch):
    from mouse_core import polyak_update
    target, source = nn.Module(), nn.Module()
    target.register_parameter("first", nn.Parameter(torch.zeros(2)))
    source.register_parameter("first", nn.Parameter(torch.ones(2)))
    dtype = torch.int64 if mismatch == "integer" else torch.float32
    target.register_parameter("second", nn.Parameter(torch.zeros(2, 3, dtype=dtype), requires_grad=False))
    if mismatch != "count":
        source.register_parameter(
            "other" if mismatch == "names" else "second",
            nn.Parameter(torch.ones(1 if mismatch == "broadcast_shape" else 2, 3,
                                    dtype=torch.float64 if mismatch == "dtype" else dtype), requires_grad=False),
        )
    before = {name: p.detach().clone() for name, p in target.named_parameters()}
    with pytest.raises(ValueError, match="Polyak"):
        polyak_update(target=target, source=source, tau=0.5)
    for name, p in target.named_parameters():
        torch.testing.assert_close(p, before[name])
