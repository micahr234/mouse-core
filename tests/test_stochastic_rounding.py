"""BF16 rounding statistics, native optimizer parity, and target updates."""
import copy

import pytest
import torch
from torch import nn

from mouse_core import MouseOptimizer
from mouse_core.polyak import polyak_update
from mouse_core.rounding import stochastic_round_bfloat16, stochastic_round_to


@pytest.mark.parametrize('value', [1.001, -1.001, .02001, -.02001, 1e-40, -1e-40])
def test_rounding_is_unbiased_and_uses_adjacent_values(value):
    torch.manual_seed(123)
    x = torch.full((200_000,), value, dtype=torch.float32)
    rounded = stochastic_round_bfloat16(x).float()
    neighbors = rounded.unique().sort().values
    assert len(neighbors) == 2
    a, b = neighbors.tolist()
    assert a < x[0] < b
    # Six standard deviations, measured relative to one representable interval.
    probability = (float(x[0]) - a) / (b - a)
    observed = (rounded == b).float().mean().item()
    assert abs(observed - probability) < 6 * (probability * (1-probability) / len(x)) ** .5


def test_rounding_exact_noncontiguous_special_values_and_seed():
    x = torch.tensor([[0., -0., 1., -1.], [float('inf'), -float('inf'), float('nan'), 2.]]).t()
    result = stochastic_round_bfloat16(x)
    torch.testing.assert_close(result.float(), x, rtol=0, atol=0, equal_nan=True)
    assert torch.signbit(result[1, 0])
    limit = torch.finfo(torch.bfloat16).max
    result = stochastic_round_bfloat16(torch.tensor([torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]))
    torch.testing.assert_close(result.float(), torch.tensor([limit, -limit]), rtol=0, atol=0)
    def sample():
        return stochastic_round_bfloat16(torch.full((1000,), 1.001), generator=torch.Generator().manual_seed(7))
    assert torch.equal(sample(), sample())


def options(name):
    common = {'name': name, 'weight_decay': .13}
    if name in ('adam', 'adamw'):
        common.update(betas=(.7, .9), eps=.003, fused=False)
    elif name == 'sgd':
        common.update(momentum=.8, fused=False)
    else:
        common.update(alpha=.8, eps=.003, momentum=.6, centered=True)
    return common


@pytest.mark.parametrize('name', ['adam', 'adamw', 'sgd', 'rmsprop'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_optimizer_matches_native_fp32_math_then_one_round(name, dtype):
    model = nn.Linear(5, 3, dtype=dtype)
    ref = copy.deepcopy(model).float()
    settings = options(name)
    opt = MouseOptimizer(groups={'m': {'online': model, 'lr': .0001}}, optimizer=settings,
                         stochastic_rounding=True)
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
           'sgd': torch.optim.SGD, 'rmsprop': torch.optim.RMSprop}[name]
    native = cls(ref.parameters(), lr=.0001, **{k:v for k,v in settings.items() if k != 'name'}, foreach=False)
    for step in range(4):
        for p, q in zip(model.parameters(), ref.parameters()):
            q.data.copy_(p.float())
            p.grad = torch.randn_like(p)
            q.grad = p.grad.float().clone()
        native.step()
        rng = torch.random.get_rng_state()
        expected = [stochastic_round_bfloat16(p) if dtype == torch.bfloat16 else p.detach().clone()
                    for p in ref.parameters()]
        torch.random.set_rng_state(rng)
        opt.step(step)
        for p, value in zip(model.parameters(), expected):
            torch.testing.assert_close(p, value, rtol=0, atol=0)
        for p in model.parameters():
            assert all(not isinstance(v, torch.Tensor) or v.dtype == torch.float32
                       for v in opt.inner.state[p].values())
        assert not opt.inner._kernel.state and not opt.inner._kernel.param_groups
    # Reload preserves full moment precision, even though weights are BF16.
    other = copy.deepcopy(model)
    restored = MouseOptimizer(groups={'m': {'online': other, 'lr': .0001}}, optimizer=settings,
                              stochastic_rounding=True)
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p, q in zip(model.parameters(), other.parameters()):
        for key, value in opt.inner.state[p].items():
            torch.testing.assert_close(restored.inner.state[q][key], value, rtol=0, atol=0)
        p.grad = torch.full_like(p, .25)
        q.grad = p.grad.clone()
    rng = torch.random.get_rng_state()
    opt.step(4)
    torch.random.set_rng_state(rng)
    restored.step(4)
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_tiny_optimizer_and_polyak_updates_survive():
    model = nn.Linear(100_000, 1, bias=False, dtype=torch.bfloat16)
    target = copy.deepcopy(model).requires_grad_(False)
    model.weight.data.fill_(1)
    target.weight.data.fill_(1)
    opt = MouseOptimizer(groups={'m': {'online': model, 'lr': .001,
        'target': target, 'polyak_tau': .1}}, optimizer='sgd',
        stochastic_rounding=True)
    torch.manual_seed(15)
    model.weight.grad = -torch.ones_like(model.weight)
    opt.step(0)
    assert model.weight.float().mean().item() == pytest.approx(1.001, abs=3e-5)
    assert target.weight.float().mean().item() == pytest.approx(1.0001, abs=2e-5)
    assert bool((model.weight != 1).any()) and bool((target.weight != 1).any())


@pytest.mark.parametrize('standalone', [False, True])
def test_polyak_fp32_arithmetic_exact_endpoints_and_frozen_sharing(standalone):
    source = nn.Linear(10000, 1, bias=False, dtype=torch.bfloat16)
    target = copy.deepcopy(source).requires_grad_(False)
    source.weight.data.fill_(1.0078125)
    target.weight.data.fill_(1.)
    source.register_parameter('frozen', nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False))
    target.register_parameter('frozen', source.frozen)
    for tau in (0., .001, .2, 1.):
        before = target.weight.detach().float().clone()
        rng = torch.random.get_rng_state()
        intended = before.lerp(source.weight.float(), tau)
        expected = stochastic_round_bfloat16(intended) if 0 < tau < 1 else intended.bfloat16()
        torch.random.set_rng_state(rng)
        if standalone:
            polyak_update(source=source, target=target, tau=tau, stochastic_rounding=True)
        else:
            opt = MouseOptimizer(groups={'m': {'online': source, 'lr': 0., 'target': target, 'polyak_tau': tau}},
                                 optimizer='sgd', stochastic_rounding=True)
            opt.step(0)
        torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
        if tau in (0., 1.):
            assert torch.equal(torch.random.get_rng_state(), rng)
        assert target.frozen is source.frozen
        torch.testing.assert_close(target.frozen, torch.ones_like(target.frozen))


def test_tensor_polyak_rates_keep_fp32_precision():
    source = nn.Linear(30000, 3, bias=False, dtype=torch.bfloat16)
    target = copy.deepcopy(source).requires_grad_(False)
    source.weight.data.fill_(1.0078125)
    target.weight.data.fill_(1.)
    rate = torch.tensor([[0.], [.12345], [1.]], dtype=torch.float32)
    rng = torch.random.get_rng_state()
    expected = stochastic_round_bfloat16(target.weight.float().lerp(source.weight.float(), rate))
    torch.random.set_rng_state(rng)
    opt = MouseOptimizer(groups={'m': {'online': source, 'lr': 0., 'target': target,
        'polyak_tau': {'weight': rate}}}, optimizer='sgd',
        stochastic_rounding=True)
    opt.step(0)
    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)


def test_invalid_options_fail_before_updates():
    model = nn.Linear(2, 1, dtype=torch.bfloat16)
    target = copy.deepcopy(model).requires_grad_(False)
    groups = {'m': {'online': model, 'lr': .1, 'target': target, 'polyak_tau': .1}}
    with pytest.raises(ValueError, match='stochastic_rounding'):
        MouseOptimizer(groups=groups)
    with pytest.raises(TypeError, match='polyak_stochastic_rounding'):
        MouseOptimizer(groups=groups, polyak_stochastic_rounding=True)
    with pytest.raises(ValueError, match='fused'):
        MouseOptimizer(groups=groups, optimizer='adamw_fused', stochastic_rounding=True)
    with pytest.raises(TypeError, match='booleans'):
        MouseOptimizer(groups=groups, stochastic_rounding=1)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_fp64_rounding_statistics_and_special_values(dtype, sign):
    torch.manual_seed(333)
    quantum = torch.finfo(dtype).eps
    x = torch.full((100_000,), sign * (1 + quantum * .125), dtype=torch.float64)
    rounded = stochastic_round_to(x, dtype).double()
    a, b = rounded.unique().sort().values.tolist()
    probability = (x[0].item() - a) / (b - a)
    observed = (rounded == b).double().mean().item()
    assert abs(observed - probability) < 6 * (probability * (1-probability) / x.numel()) ** .5
    special = torch.tensor([[0., -0., 1., float('inf')],
                            [-float('inf'), float('nan'), 1e300, -1e300]], dtype=torch.float64).t()
    expected = special.clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max)
    expected = torch.where(torch.isfinite(special), expected, special).to(dtype)
    result = stochastic_round_to(special, dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.signbit(result[1, 0])


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fp64_rounding_has_no_intermediate_fp32_cast(dtype, monkeypatch):
    midpoint = 1. + torch.finfo(dtype).eps / 2
    x = torch.tensor([midpoint + 2**-45], dtype=torch.float64)
    monkeypatch.setattr(torch, 'rand_like', lambda value: torch.full_like(value, .5))
    result = stochastic_round_to(x, dtype)
    assert result.item() == 1. + torch.finfo(dtype).eps
