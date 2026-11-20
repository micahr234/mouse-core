"""Optimizer dtype controls arithmetic as well as state storage."""
import copy

import pytest
import torch
from torch import nn

from mouse_core import MouseOptimizer
from mouse_core.precision_optim import PrecisionOptimizer, _rmsprop_update_
from mouse_core.rounding import stochastic_round_to, polyak_parameter_, scalar_tensor


def options(name):
    settings = dict(weight_decay=.13)
    if name in ('adam', 'adamw'):
        settings.update(betas=(.7, .9), eps=.003)
    elif name == 'sgd':
        settings.update(momentum=.8)
    else:
        settings.update(alpha=.8, eps=.003, momentum=.6, centered=True)
    return settings


@pytest.mark.parametrize('name', ['rmsprop', 'adam', 'adamw', 'sgd'])
@pytest.mark.parametrize('rounding', [False, True])
def test_bf16_arithmetic_matches_native_bf16(name, rounding):
    torch.manual_seed(71)
    p = nn.Parameter(torch.randn(5, 3, dtype=torch.bfloat16))
    q = nn.Parameter(p.detach().clone())
    opt = PrecisionOptimizer([p], name=name, options=options(name), dtype=torch.bfloat16,
                             stochastic_rounding=rounding)
    opt.param_groups[0]['lr'] = .001
    native = getattr(torch.optim, {'rmsprop':'RMSprop','adam':'Adam','adamw':'AdamW','sgd':'SGD'}[name])(
        [q], lr=.001, **options(name), foreach=False)
    for _ in range(4):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
        native.step()
        opt.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key, value in native.state[q].items():
            torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0)
            assert value.dtype == (torch.float32 if key == 'step' else torch.bfloat16)


@pytest.mark.parametrize('name', ['rmsprop', 'adam', 'adamw', 'sgd'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize('rounding', [False, True])
def test_dtype_and_rounding_roundtrip(name, dtype, rounding):
    p = nn.Parameter(torch.linspace(.1, .9, 100))
    opt = PrecisionOptimizer([p], name=name, options=options(name), dtype=dtype,
                             stochastic_rounding=rounding)
    p.grad = torch.full_like(p, .17)
    opt.step()
    q = nn.Parameter(p.detach().clone())
    restored = PrecisionOptimizer([q], name=name, options=options(name), dtype=dtype,
                                  stochastic_rounding=rounding)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for key, value in opt.state[p].items():
        assert value.dtype == (torch.float32 if key == 'step' else dtype)
        torch.testing.assert_close(restored.state[q][key], value, rtol=0, atol=0)
    q.grad = p.grad.clone()
    rng = torch.random.get_rng_state()
    opt.step()
    torch.random.set_rng_state(rng)
    restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_rounding_flag_does_not_change_bf16_moment_arithmetic():
    means = []
    for rounding in (False, True):
        p = nn.Parameter(torch.ones(100_000, dtype=torch.bfloat16))
        opt = PrecisionOptimizer([p], name='rmsprop', options=dict(alpha=.9999),
                                 dtype=torch.bfloat16, stochastic_rounding=rounding)
        opt.state[p] = dict(step=torch.tensor(0.), square_avg=torch.ones_like(p))
        p.grad = torch.zeros_like(p)
        opt.step()
        means.append(opt.state[p]['square_avg'].float().mean().item())
    assert means[0] == 1.
    assert means[1] == 1.


def test_bf16_dtype_changes_arithmetic_for_fp32_model_weights():
    p = nn.Parameter(torch.full((100,), 1.001))
    opt = PrecisionOptimizer([p], name='sgd', options={}, dtype=torch.bfloat16,
                             stochastic_rounding=False)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert p.dtype == torch.float32
    torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)


def test_one_rounding_flag_controls_model_and_target_writeback():
    online = nn.Linear(100_000, 1, bias=False, dtype=torch.bfloat16)
    online.weight.data.fill_(1.)
    target = copy.deepcopy(online).requires_grad_(False)
    opt = MouseOptimizer(groups={'x':dict(online=online, target=target, lr=.001, polyak_tau=.1)},
                         optimizer='sgd', dtype=torch.float32, stochastic_rounding=True)
    online.weight.grad = -torch.ones_like(online.weight)
    opt.step(0)
    assert online.weight.float().mean().item() == pytest.approx(1.001, abs=3e-5)
    assert target.weight.float().mean().item() == pytest.approx(1.0001, abs=2e-5)
    for bad in (torch.float16, torch.int32, 'bfloat16'):
        with pytest.raises(ValueError, match='dtype'):
            MouseOptimizer(groups={'x':dict(online=online, lr=.001)}, dtype=bad)
    with pytest.raises(TypeError, match='polyak_stochastic_rounding'):
        MouseOptimizer(groups={'x':dict(online=online, lr=.001)}, polyak_stochastic_rounding=True)


def test_checkpoint_rejects_different_dtype_or_rounding():
    model = nn.Linear(2, 1)
    make = lambda **kw: MouseOptimizer(groups={'x':dict(online=model, lr=.001)}, optimizer='rmsprop', **kw)
    state = make(dtype=torch.bfloat16, stochastic_rounding=True).state_dict()
    for dtype, rounding in [(torch.float32, True), (torch.bfloat16, False)]:
        with pytest.raises(ValueError, match='precision'):
            make(dtype=dtype, stochastic_rounding=rounding).load_state_dict(state)


@pytest.mark.parametrize('rounding', [False, True])
def test_bf16_update_transform_receives_optimizer_dtype(rounding):
    p = nn.Parameter(torch.ones(32))
    opt = PrecisionOptimizer([p], name='sgd', options=dict(momentum=.8),
                             dtype=torch.bfloat16, stochastic_rounding=rounding)
    def transform(parameter, update):
        assert parameter.dtype == torch.float32
        assert update.dtype == torch.bfloat16
        update.add_(.001)
    opt.update_transform = transform
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)
    assert opt.state[p]['momentum_buffer'].dtype == torch.bfloat16


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
def test_polyak_uses_optimizer_arithmetic_dtype(dtype):
    online = nn.Linear(100_000, 1, bias=False, dtype=torch.bfloat16)
    online.weight.data.fill_(1.0078125)
    target = copy.deepcopy(online).requires_grad_(False)
    target.weight.data.fill_(1.)
    opt = MouseOptimizer(groups={'x':dict(online=online, target=target, lr=0., polyak_tau=.1)},
                         optimizer='sgd', dtype=dtype, stochastic_rounding=True)
    opt.step(0)
    expected = 1.00078125 if dtype != torch.bfloat16 else 1.
    assert target.weight.float().mean().item() == pytest.approx(expected, abs=3e-5)


@pytest.mark.parametrize('rounding', [False, True])
def test_same_dtype_model_write_does_not_consume_randomness(rounding):
    p = nn.Parameter(torch.ones(16, dtype=torch.bfloat16))
    opt = PrecisionOptimizer([p], name='sgd', options={}, dtype=torch.bfloat16,
                             stochastic_rounding=rounding)
    p.grad = torch.ones_like(p)
    rng = torch.random.get_rng_state()
    opt.step()
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('name', ['rmsprop', 'adam', 'adamw', 'sgd'])
@pytest.mark.parametrize('model_dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('rounding', [False, True])
def test_fp64_math_matches_native_before_model_write(name, model_dtype, rounding):
    torch.manual_seed(513)
    p = nn.Parameter(torch.randn(7, 3, dtype=model_dtype))
    q = nn.Parameter(p.detach().double())
    opt = PrecisionOptimizer([p], name=name, options=options(name), dtype=torch.float64,
                             stochastic_rounding=rounding)
    opt.param_groups[0]['lr'] = .0001234567890123
    native = getattr(torch.optim, {'rmsprop':'RMSprop','adam':'Adam','adamw':'AdamW','sgd':'SGD'}[name])(
        [q], lr=.0001234567890123, **options(name), foreach=False)
    for _ in range(4):
        q.data.copy_(p)
        p.grad = torch.randn_like(p)
        q.grad = p.grad.double()
        native.step()
        rng = torch.random.get_rng_state()
        expected = stochastic_round_to(q, model_dtype) if rounding else q.to(model_dtype)
        torch.random.set_rng_state(rng)
        opt.step()
        torch.testing.assert_close(p, expected, rtol=0, atol=0)
        for key, value in native.state[q].items():
            torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0)
            assert value.dtype == (torch.float32 if key == 'step' else torch.float64)


def test_fp64_tiny_model_and_target_updates_survive_fp32_storage():
    torch.manual_seed(723)
    online = nn.Linear(100_000, 1, bias=False)
    online.weight.data.fill_(1.)
    target = copy.deepcopy(online).requires_grad_(False)
    opt = MouseOptimizer(groups={'x':dict(online=online, target=target, lr=2**-27, polyak_tau=.25)},
                         optimizer='sgd', dtype=torch.float64, stochastic_rounding=True)
    online.weight.grad = -torch.ones_like(online.weight)
    opt.step(0)
    assert online.weight.double().mean().item() == pytest.approx(1 + 2**-27, abs=3e-10)
    assert target.weight.double().mean().item() == pytest.approx(1 + 2**-29, abs=2e-10)
    assert (online.weight > 1).any() and (target.weight > 1).any()


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_fp64_compiled_rmsprop_preserves_small_updates_and_scalar_precision(model_dtype):
    update = torch.compile(_rmsprop_update_, fullgraph=True, dynamic=True)
    p = torch.ones(100_000, dtype=model_dtype)
    g = -torch.ones_like(p)
    square = torch.ones_like(p, dtype=torch.float64)
    lr = torch.finfo(model_dtype).eps / 16 + 2**-54
    assert scalar_tensor(lr, torch.float64).item() == lr
    update(p, g, square, None, None, lr=scalar_tensor(lr, torch.float64), alpha=.95,
           eps=1e-8, weight_decay=0., momentum=0., stochastic_rounding=True)
    assert p.double().mean().item() == pytest.approx(1 + lr / (1 + 1e-8), abs=torch.finfo(model_dtype).eps * .003)
    assert square.dtype == torch.float64
    torch.testing.assert_close(square, torch.ones_like(square), rtol=0, atol=0)
