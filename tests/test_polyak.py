from __future__ import annotations
from tests._model_helpers import ExampleModel as Model
from tests._model_helpers import copy_model, node_rates, simple_model

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from mouse_core.models import LatentReasoner, ModelOutput
from mouse_core.models.backbone import IdentityBackbone, Backbone
from mouse_core.models.mlp import MLP
from mouse_core.polyak import _PolyakState
from tests._model_helpers import model_polyak
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action", "episode_done")
_BATCH = [
    [
        {"action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0},
        {"action": 1, "reward": 1.0, "episode_done": 0, "task_done": 0},
        {"action": 2, "reward": 2.0, "episode_done": 1, "task_done": 0},
    ]
]


def _head(hidden_dim: int) -> MLP:
    return MLP(
        in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1,
        propagate_gradient=1.0,
    )


def _tiny_model() -> Model:
    hidden_dim = 8
    backbone = IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32)
    head = _head(hidden_dim)
    return simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None)


def _llama_model() -> Model:
    hidden_dim = 16
    backbone = Backbone(architecture='llama', train_kernel='reference', decode_kernel='flex', dtype=torch.float32, hidden_dim=hidden_dim, num_layers=2, num_heads=2, max_position_embeddings=64, vocab_size=32)
    head = _head(hidden_dim)
    return simple_model(backbone=backbone, heads={'action_value': head}, reasoner=None)


def _token_batch(model: Model):
    return batch_to_token_batch(_TOK, _BATCH)


def _perturb(module: nn.Module) -> None:
    with torch.no_grad():
        for param in module.parameters():
            param.add_(1.0)


def _q_close(a: ModelOutput, b: ModelOutput, key: str = "action_value") -> bool:
    return torch.allclose(a.predictions[key], b.predictions[key], atol=1e-5)


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
    assert delayed.encoder.backbones[0] is not model.encoder.backbones[0]
    assert delayed.encoder.reasoner is None
    assert delayed.training
    assert all((not p.requires_grad for p in delayed.parameters()))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        assert p is not online[name]
        assert torch.equal(p, online[name])
    assert all((p.requires_grad for p in model.parameters()))


def test_copy_shares_frozen_parameters_by_reference() -> None:
    model = _llama_model()
    for p in model.encoder.backbones[0].parameters():
        p.requires_grad_(False)
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
    return simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32), heads={'action_value': q_head, 'behavior': MLP(in_features=hidden_dim, out_features=4, hidden_dim=hidden_dim, num_layers=1, propagate_gradient=1.0)}, reasoner=None)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bound_polyak_matches_per_tensor_updates_with_changing_head_rates(monkeypatch, device) -> None:
    if device == 'cuda' and (not torch.cuda.is_available()):
        pytest.skip('CUDA required')
    model = _two_head_model().to(device=torch.device(device))
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    update = model_polyak(online=model, delayed=delayed)
    sources = dict(model.named_parameters())
    targets = dict(delayed.named_parameters())
    reference = {name: p.detach().clone() for name, p in targets.items()}

    def no_traversal(*args, **kwargs):
        raise AssertionError('a bound update must not traverse the model again')
    monkeypatch.setattr(nn.Module, 'named_parameters', no_traversal)
    for step in range(12):
        rates = {'action_value': [0.0, 0.0005, 1.0][step % 3], 'behavior': 0.2}
        with torch.no_grad():
            for name, source in sources.items():
                source.add_(0.1)
                rate = rates[name.split('.')[1]] if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')) else 0.05
                reference[name].lerp_(source, rate)
        update(tau=node_rates(delayed, heads=rates, backbone=0.05))
        for name, target in targets.items():
            torch.testing.assert_close(target, reference[name], rtol=0, atol=0)
            assert target is not sources[name]
            assert not target.requires_grad


@pytest.mark.parametrize('rates,backbone_tau', [({'action_value': 0.1}, 0.2), ({'action_value': 0.1, 'behavior': -1.0}, 0.2), ({'action_value': 0.1, 'behavior': 0.2}, float('nan'))])
def test_bound_polyak_validates_all_rates_before_writing(rates, backbone_tau) -> None:
    model = _two_head_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    update = model_polyak(online=model, delayed=delayed)
    before = {name: p.detach().clone() for name, p in delayed.named_parameters()}
    _perturb(model)
    with pytest.raises(ValueError):
        update(tau=node_rates(delayed, heads=rates, backbone=backbone_tau))
    for name, param in delayed.named_parameters():
        torch.testing.assert_close(param, before[name], rtol=0, atol=0)


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
        online_q = model(batch).predictions['action_value']
        with torch.no_grad():
            delayed_q = delayed(batch).predictions['action_value']
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
    assert delayed.encoder.backbones[0] is not model.encoder.backbones[0]
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




def test_polyak_skips_online_heads_the_delayed_model_does_not_carry() -> None:
    model = _two_head_model()
    delayed = copy_model(model, heads=(model.heads['action_value'],), backbone=True, reasoner=False)
    _perturb(model.heads)
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.0, backbone=0.0))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')):
            assert torch.equal(p, online[name])
    assert not any((name.startswith('heads.behavior') for name, _ in delayed.named_parameters()))


def test_polyak_rejects_delayed_heads_missing_online() -> None:
    model = _two_head_model()
    other = _tiny_model()
    with pytest.raises(ValueError, match='unknown module path'):
        model_polyak(online=other, delayed=copy_model(model, heads=(model.heads['action_value'], model.heads['behavior']), backbone=True, reasoner=False))


def test_copy_carries_reasoner() -> None:
    hidden_dim = 8
    reasoning = simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32), heads=({'action_value': (head := _head(hidden_dim))}), reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1))
    dr = copy_model(reasoning, heads=True, backbone=True, reasoner=True)
    assert dr.encoder.reasoner is not None and dr.encoder.reasoner is not reasoning.encoder.reasoner
    shared = copy_model(reasoning, heads=True, backbone=False, reasoner=False)
    assert shared.encoder.backbones[0] is reasoning.encoder.backbones[0]
    assert shared.encoder.reasoner is reasoning.encoder.reasoner


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
    assert delayed_out.predictions['action_value'].grad_fn is None
    assert _q_close(out, delayed_out)


def test_delayed_model_reruns_its_own_trunk() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    batch = _token_batch(model)
    emb_calls = _count_calls(delayed.encoder.backbones[0], name='embed')
    bb_calls = _count_calls(delayed.encoder.backbones[0])
    with torch.no_grad():
        delayed(batch)
    assert emb_calls['n'] == 1 and bb_calls['n'] == 1


def test_delayed_model_ignores_online_changes_until_update() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False).eval()
    batch = _token_batch(model)
    with torch.no_grad():
        before = delayed(batch)
    _perturb(model.encoder.backbones[0])
    _perturb(model.heads)
    with torch.no_grad():
        online = model(batch)
        still_delayed = delayed(batch)
    assert _q_close(before, still_delayed)
    assert not _q_close(online, still_delayed)
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.0, backbone=1.0))
    with torch.no_grad():
        copied = delayed(batch)
    assert _q_close(online, copied)


def test_copy_shares_uncopied_backbone() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=False, reasoner=False)
    assert delayed.encoder.backbones[0] is model.encoder.backbones[0]
    assert all((p.requires_grad for p in model.encoder.backbones[0].parameters()))
    assert all((not p.requires_grad for p in delayed.heads.parameters()))


def test_copy_shares_uncopied_heads() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=False, backbone=True, reasoner=False)
    assert delayed.heads['action_value'] is model.heads['action_value']
    assert all((p.requires_grad for p in model.heads.parameters()))
    assert all((not p.requires_grad for p in delayed.encoder.backbones[0].parameters()))


def test_pool_and_head_match_delayed_heads_on_shared_backbone() -> None:
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
            preds = delayed.head(h=out.pool())
    assert saved['n'] == 0
    assert preds['action_value'].grad_fn is None
    assert torch.allclose(preds['action_value'], out.predictions['action_value'])


# ---- model_polyak ---------------------------------------------------------------


def test_all_zero_tau_does_not_write_delayed_params() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    versions = [param._version for param in delayed.parameters()]
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.0, backbone=0.0))
    assert [param._version for param in delayed.parameters()] == versions


def test_polyak_requires_a_tau_per_copied_section() -> None:
    model = _tiny_model()
    both = copy_model(model, heads=True, backbone=True, reasoner=False)
    with pytest.raises(ValueError, match='backbone.* is required'):
        model_polyak(online=model, delayed=both)(tau=node_rates(both, heads=0.1, backbone=None))
    with pytest.raises(ValueError, match='.* is required'):
        model_polyak(online=model, delayed=both)(tau=node_rates(both, heads=None, backbone=0.1))
    model_polyak(online=model, delayed=both)(tau=node_rates(both, heads=0.1, backbone=0.1))
    model_polyak(online=model, delayed=copy_model(model, heads=True, backbone=False, reasoner=False))(tau=node_rates(copy_model(model, heads=True, backbone=False, reasoner=False), heads=0.1, backbone=None))
    model_polyak(online=model, delayed=copy_model(model, heads=False, backbone=True, reasoner=False))(tau=node_rates(copy_model(model, heads=False, backbone=True, reasoner=False), heads=None, backbone=0.1))


def test_polyak_shared_backbone_omits_tau_backbone() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=False, reasoner=False)
    _perturb(model.heads)
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.0, backbone=None))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')):
            assert torch.equal(p, online[name])
    _perturb(model.heads)
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.0, backbone=1.0))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')):
            assert torch.equal(p, online[name])
    with pytest.raises(ValueError, match='backbone.* must be None or 1'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.1, backbone=0.1))
    with pytest.raises(ValueError, match='backbone.* must be None or 1'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.1, backbone=0.0))


def test_polyak_copied_section_rejects_tau_none() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    with pytest.raises(ValueError, match='backbone.* is required'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.1, backbone=None))
    with pytest.raises(ValueError, match='.* is required'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=None, backbone=0.1))


def test_polyak_shared_heads_omits_tau_heads() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=False, backbone=True, reasoner=False)
    _perturb(model.encoder.backbones[0])
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=None, backbone=1.0))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith('encoder.backbones.0.'):
            assert torch.equal(p, online[name])
    _perturb(model.encoder.backbones[0])
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.0, backbone=1.0))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith('encoder.backbones.0.'):
            assert torch.equal(p, online[name])
    with pytest.raises(ValueError, match='.* must be None or 1'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.1, backbone=0.1))
    with pytest.raises(ValueError, match='.* must be None or 1'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.0, backbone=0.1))


def test_polyak_tau_is_convex_combination_and_can_change() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    online = next(model.heads.parameters())
    delayed_p = next(delayed.heads.parameters())
    online.data.fill_(1.0)
    delayed_p.data.fill_(0.0)
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.5, backbone=0.0))
    assert torch.allclose(delayed_p, torch.full_like(delayed_p, 0.5))
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.0, backbone=0.0))
    assert torch.allclose(delayed_p, torch.ones_like(delayed_p))


def test_each_tau_interpolates_only_its_section() -> None:
    torch.manual_seed(0)
    model = _llama_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    snapshot = {n: p.detach().clone() for n, p in delayed.named_parameters()}
    _perturb(model)
    model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.0, backbone=0.25))
    online = dict(model.named_parameters())
    for name, p in delayed.named_parameters():
        if name.startswith(('heads.action_value.', 'heads.behavior.', 'heads.q.', 'heads.policy.')):
            assert torch.equal(p, snapshot[name])
        else:
            assert name.startswith('encoder.backbones.0.')
            assert torch.allclose(p, 0.75 * snapshot[name] + 0.25 * online[name])


def test_tau_backbone_also_moves_reasoner() -> None:
    hidden_dim = 8
    reasoning = simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim, vocab_size=32), heads=({'action_value': (head := _head(hidden_dim))}), reasoner=LatentReasoner(hidden_dim=hidden_dim, num_thoughts=1))
    dr = copy_model(reasoning, heads=True, backbone=True, reasoner=True)
    assert reasoning.encoder.reasoner is not None and dr.encoder.reasoner is not None
    _perturb(reasoning.encoder.reasoner)
    model_polyak(online=reasoning, delayed=dr)(tau=node_rates(dr, heads=0.0, backbone=1.0, reasoner=1.0))
    for a, b in zip(dr.encoder.reasoner.parameters(), reasoning.encoder.reasoner.parameters(), strict=True):
        assert torch.equal(a, b)


def test_polyak_rejects_tau_out_of_range() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    with pytest.raises(ValueError, match='.* must be in \\[0, 1\\]'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=1.5, backbone=0.1))
    with pytest.raises(ValueError, match='backbone.* must be in \\[0, 1\\]'):
        model_polyak(online=model, delayed=delayed)(tau=node_rates(delayed, heads=0.1, backbone=2.0))


def test_polyak_small_tau_accumulates_in_fp32() -> None:
    online = nn.Linear(8, 8, bias=False)
    delayed = nn.Linear(8, 8, bias=False)
    online.weight.data.fill_(1.0)
    delayed.weight.data.fill_(0.9)
    state = _PolyakState(online, delayed, section="heads")
    tau = 0.0005
    steps = 2000
    for _ in range(steps):
        state.update(tau)
    expected = 1.0 - 0.1 * (1.0 - tau) ** steps
    assert torch.allclose(delayed.weight, torch.full_like(delayed.weight, expected), atol=1e-4)


def test_polyak_rejects_non_fp32_interpolated_params() -> None:
    """A trainable bf16 copy would round a tiny tau away; model_polyak refuses it."""
    online = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    delayed = nn.Linear(8, 8, bias=False).to(dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="fp32 parameters only"):
        _PolyakState(online, delayed, section="backbone")


def test_polyak_skips_shared_frozen_params_and_rejects_shared_trainable() -> None:
    online = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
    online[0].requires_grad_(False)
    delayed = nn.Sequential(online[0], nn.Linear(8, 8, bias=False))
    state = _PolyakState(online, delayed, section="backbone")
    assert len(state) == 1  # the shared frozen layer is not interpolated
    online_trainable = cast(nn.Linear, online[1])
    delayed_trainable = cast(nn.Linear, delayed[1])
    online_trainable.weight.data.fill_(1.0)
    delayed_trainable.weight.data.fill_(0.0)
    state.update(0.5)
    assert torch.allclose(delayed_trainable.weight, torch.full((8, 8), 0.5))

    shared_trainable = nn.Sequential(online[0], online[1])  # trainable layer shared too
    with pytest.raises(ValueError, match="same tensor online and delayed"):
        _PolyakState(online, shared_trainable, section="backbone")
    with pytest.raises(ValueError, match="copy"):
        _PolyakState(online, online, section="backbone")


def test_polyak_rejects_wrong_models() -> None:
    model = _tiny_model()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    with pytest.raises(TypeError):
        model_polyak(online=model, delayed=nn.Linear(2, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='copy'):
        model_polyak(online=model, delayed=model)
    with pytest.raises(ValueError, match='trainable model'):
        model_polyak(online=delayed, delayed=copy_model(model, heads=True, backbone=True, reasoner=False))
    other = _tiny_model()
    with pytest.raises(ValueError, match='copy'):
        model_polyak(online=model, delayed=other)
    mismatched = simple_model(backbone=IdentityBackbone(hidden_dim=8, vocab_size=32), heads=({'action_value': (head := MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=2, propagate_gradient=1.0))}), reasoner=None).requires_grad_(False)
    with pytest.raises(ValueError, match='parameter names'):
        model_polyak(online=model, delayed=mismatched)


# ---- forward contract -------------------------------------------------------


def test_forward_returns_model_output() -> None:
    torch.manual_seed(0)
    model = _tiny_model().eval()
    batch = _token_batch(model)
    with torch.no_grad():
        out = model(batch)
    assert isinstance(out, ModelOutput)
    assert out.last_hidden_state.shape == (batch.L, model.encoder.hidden_dim)
    assert out.head_output_indices.shape == (batch.P,)
    assert out.cache is None


def test_last_hidden_state_stays_on_the_tape() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    out = model(_token_batch(model))
    assert out.last_hidden_state.requires_grad
    assert out.predictions["action_value"].requires_grad


def _assert_no_autograd_graph(fn: Callable[[], Any]) -> Any:
    saved = {"n": 0}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved["n"] += 1
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    assert saved["n"] == 0
    return result


def test_delayed_forward_under_no_grad_builds_no_graph_in_train_mode() -> None:
    torch.manual_seed(0)
    model = _tiny_model().train()
    delayed = copy_model(model, heads=True, backbone=True, reasoner=False)
    batch = _token_batch(model)

    def run():
        with torch.no_grad():
            return delayed(batch)
    delayed_out = _assert_no_autograd_graph(run)
    assert delayed_out.predictions['action_value'].grad_fn is None


def test_forward_rejects_non_token_batch() -> None:
    model = _tiny_model().eval()
    with pytest.raises(TypeError, match="TokenBatch"):
        model(torch.zeros(3, 8))  # type: ignore[arg-type]
