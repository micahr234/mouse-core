"""Standalone optimizer noise, precision ordering and no-noise fast paths."""
import copy
import math
import pytest
import torch
from torch import nn
from mouse_core import MouseOptimizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sgd_noise_is_inside_optimizer_before_target_averaging(dtype, monkeypatch):
    source = nn.Linear(1, 1, bias=False)
    source.weight.data.fill_(1.)
    target = copy.deepcopy(source).requires_grad_(False)
    optimizer = MouseOptimizer(groups={"weights": {"online": source, "lr": .1,
        "target": target, "polyak_tau": .5}}, optimizer="sgd", dtype=dtype,
        langevin_noise_beta=lambda step: 2. if step >= 10 else 0.)
    source.weight.grad = torch.ones_like(source.weight)
    draws = []
    def noise(value):
        draws.append(value.dtype)
        return torch.ones_like(value)
    monkeypatch.setattr(torch, "randn_like", noise)
    optimizer.step(10, online_transition_count=20)
    expected = .9 - 2 * math.sqrt(.2 / 20)
    assert source.weight.item() == pytest.approx(expected)
    assert target.weight.item() == pytest.approx((1 + expected) / 2)
    assert draws == [dtype]


@pytest.mark.parametrize("count", [None, True, 0, -2, 1.5])
def test_noise_validates_transition_count_before_update(count):
    source = nn.Linear(1, 1, bias=False)
    original = source.weight.detach().clone()
    optimizer = MouseOptimizer(groups={"weights": {"online": source, "lr": .1}},
                               optimizer="rmsprop", langevin_noise_beta=1.)
    source.weight.grad = torch.ones_like(source.weight)
    with pytest.raises(ValueError, match="positive online_transition_count"):
        optimizer.step(0, online_transition_count=count)
    torch.testing.assert_close(source.weight, original)
    assert not optimizer.inner.state


def test_disabled_noise_preserves_rng_and_precision_rmsprop_fast_path(monkeypatch):
    import mouse_core.precision_optim as precision
    source = nn.Linear(1, 1, bias=False)
    source.weight.grad = torch.ones_like(source.weight)
    optimizer = MouseOptimizer(groups={"weights": {"online": source, "lr": .1}},
                               optimizer="rmsprop", dtype=torch.float64, langevin_noise_beta=0.)
    called = []
    kernel = precision._rmsprop_update_
    def record(*args, **kwargs):
        called.append(True)
        return kernel(*args, **kwargs)
    monkeypatch.setattr(precision, "_rmsprop_kernel", lambda device: record)
    rng = torch.random.get_rng_state()
    optimizer.step(0)
    assert called == [True]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert optimizer.inner.update_transform is None


@pytest.mark.parametrize("beta", [-1., float("nan"), float("inf"), True])
def test_invalid_scheduled_beta_is_rejected_before_update(beta):
    source = nn.Linear(1, 1, bias=False)
    original = source.weight.detach().clone()
    optimizer = MouseOptimizer(groups={"weights": {"online": source, "lr": .1}},
        optimizer="sgd", langevin_noise_beta=lambda step: beta)
    source.weight.grad = torch.ones_like(source.weight)
    with pytest.raises(ValueError, match="langevin_noise_beta"):
        optimizer.step(0, online_transition_count=10)
    torch.testing.assert_close(source.weight, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("name", ["sgd", "rmsprop", "adamw"])
def test_noise_enters_moments_in_arithmetic_dtype(name, dtype, monkeypatch):
    source = nn.Linear(3, 1, bias=False)
    source.weight.data.fill_(.5)
    options = {"sgd": {"momentum": .8},
               "rmsprop": {"alpha": .95, "momentum": .8, "eps": 1e-8},
               "adamw": {"betas": (.9, .95), "eps": 1e-8, "weight_decay": 0.}}[name]
    optimizer = MouseOptimizer(groups={"weights": {"online": source, "lr": .01}},
        optimizer={"name": name, **options}, dtype=dtype, langevin_noise_beta=.2)
    parameter = nn.Parameter(source.weight.detach().to(dtype, copy=True))
    reference = {"sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop,
                 "adamw": torch.optim.AdamW}[name]([parameter], lr=.01, foreach=False, **options)
    seen = []
    def noise(gradient):
        seen.append(gradient.dtype)
        return torch.ones_like(gradient)
    monkeypatch.setattr(torch, "randn_like", noise)
    for step in range(2):
        source.weight.grad = torch.full_like(source.weight, .1)
        parameter.grad = source.weight.grad.to(dtype, copy=True)
        parameter.grad.add_(torch.ones_like(parameter), alpha=.2 * math.sqrt(2 / .01 / 10))
        reference.step()
        optimizer.step(step, online_transition_count=10)
        torch.testing.assert_close(source.weight, parameter.float(), rtol=1e-5, atol=1e-6)
        actual = optimizer.inner
        if not isinstance(actual, torch.optim.Optimizer):
            actual = actual._inner
        for key, value in reference.state[parameter].items():
            torch.testing.assert_close(actual.state[source.weight][key], value)
        torch.testing.assert_close(source.weight.grad, torch.full_like(source.weight, .1))
        parameter.data.copy_(source.weight.to(dtype))
    assert seen == [dtype, dtype]


def test_bf16_weights_get_noise_in_fp64_before_final_rounding(monkeypatch):
    import mouse_core.precision_optim as precision
    source = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    source.weight.data.fill_(1.)
    source.weight.grad = torch.zeros_like(source.weight)
    optimizer = MouseOptimizer(groups={"weights": {"online": source, "lr": .001}},
        optimizer="sgd", dtype=torch.float64, stochastic_rounding=True, langevin_noise_beta=.01)
    monkeypatch.setattr(torch, "randn_like", torch.ones_like)
    writes = []
    copy_update = precision.copy_update_
    def capture(parameter, update, **kwargs):
        writes.append(update.detach().clone())
        return copy_update(parameter, update, **kwargs)
    monkeypatch.setattr(precision, "copy_update_", capture)
    optimizer.step(0, online_transition_count=10)
    assert writes[0].dtype == torch.float64
    assert writes[0].item() == pytest.approx(1 - .01 * math.sqrt(2 * .001 / 10), abs=1e-14)
    assert optimizer.inner.gradient_transform is None
    assert optimizer.inner.update_transform is None
