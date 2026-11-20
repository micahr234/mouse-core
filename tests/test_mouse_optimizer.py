"""Coupled optimizer/target updates, schedules and failure ordering."""
from collections.abc import Callable
from typing import Any, cast
import copy

import pytest
import torch
from torch import nn

from mouse_core import MouseOptimizer, Piecewise


def setup(*, tau: float | Callable[[int], float]=.2, lr: float | Callable[[int], float]=.1):
    online = nn.ModuleDict({"norm": nn.RMSNorm(2), "head": nn.Linear(2, 1)})
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = MouseOptimizer(groups={
        "norm": {"online": online["norm"], "lr": lr, "target": target["norm"], "polyak_tau": tau},
        "head": {"online": online["head"], "lr": lr, "target": target["head"], "polyak_tau": .7},
    }, optimizer={"name": "adamw", "fused": False})
    return online, target, optimizer


def test_target_follows_post_adam_weights_with_separate_taus():
    online, target, optimizer = setup()
    initial = copy.deepcopy(target.state_dict())
    online["head"](online["norm"](torch.tensor([[1., 3.]]))).sum().backward()
    optimizer.step(0)
    for name, p in online.named_parameters():
        rate = .2 if name.startswith("norm") else .7
        torch.testing.assert_close(target.state_dict()[name], initial[name].lerp(p, rate))
    assert not torch.equal(online["norm"].get_parameter("weight"), initial["norm.weight"])
    assert all(p.grad is None for p in target.parameters())


def test_schedules_resume_at_explicit_batch_step_with_optimizer_moments():
    tau = Piecewise(knots=[(0, .1), (2, .9)])
    lr = Piecewise(knots=[(0, .01), (3, .04)])
    online, target, optimizer = setup(tau=tau, lr=lr)
    for p in online.parameters(): p.grad = torch.ones_like(p)
    optimizer.step(0)
    restored, delayed, second = setup(tau=tau, lr=lr)
    restored.load_state_dict(online.state_dict())
    delayed.load_state_dict(target.state_dict())
    second.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for model, opt in [(online, optimizer), (restored, second)]:
        for p in model.parameters(): p.grad = torch.full_like(p, .3)
        opt.step(3)
    for a, b in zip(online.parameters(), restored.parameters()): torch.testing.assert_close(a, b)
    for a, b in zip(target.parameters(), delayed.parameters()): torch.testing.assert_close(a, b)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.04)


def test_invalid_later_group_does_not_update_online_or_any_target():
    online, target, optimizer = setup()
    optimizer.groups["head"]["polyak_tau"] = lambda step: float("nan")
    before = copy.deepcopy(online.state_dict()), copy.deepcopy(target.state_dict())
    for p in online.parameters(): p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="Polyak rate"):
        optimizer.step(0)
    for model, state in zip((online, target), before):
        for name, p in model.state_dict().items(): torch.testing.assert_close(p, state[name])
    assert not optimizer.state_dict()["inner"]["state"]


def test_inner_failure_does_not_update_target(monkeypatch):
    online, target, optimizer = setup()
    with torch.no_grad(): online["norm"].get_parameter("weight").add_(1)
    initial = target["norm"].get_parameter("weight").clone()
    def fail(): raise RuntimeError("step failed")
    monkeypatch.setattr(optimizer.inner, "step", fail)
    with pytest.raises(RuntimeError, match="step failed"): optimizer.step(0)
    torch.testing.assert_close(target["norm"].get_parameter("weight"), initial)


def test_no_target_and_frozen_base_with_trainable_adapter_and_norm():
    online = nn.ModuleDict({"base": nn.Linear(2, 2).requires_grad_(False),
                           "adapter": nn.Linear(2, 2), "norm": nn.RMSNorm(2)})
    target = copy.deepcopy(online).requires_grad_(False)
    target["base"] = online["base"]
    frozen = copy.deepcopy(online["base"].state_dict())
    optimizer = MouseOptimizer(groups={
        "base": {"online": online["base"], "lr": 0., "target": target["base"], "polyak_tau": .3},
        "adapter": {"online": online["adapter"], "lr": .01, "target": target["adapter"], "polyak_tau": .3},
        "norm": {"online": online["norm"], "lr": .02},
    })
    for p in online.parameters():
        if p.requires_grad: p.grad = torch.ones_like(p)
    optimizer.step(0)
    for key, value in frozen.items(): torch.testing.assert_close(online["base"].state_dict()[key], value)
    assert not torch.equal(online["norm"].get_parameter("weight"), target["norm"].get_parameter("weight"))


def test_packed_head_tensor_taus_and_buffers():
    model = nn.ModuleDict({"head": nn.Linear(2, 2)})
    model["head"].register_buffer("buffer", torch.ones(1))
    target = copy.deepcopy(model).requires_grad_(False)
    opt = MouseOptimizer(groups={"head": {
        "online": model["head"], "lr": 0., "target": target["head"], "polyak_tau": {
            "weight": torch.tensor([[0.], [1.]]), "bias": torch.tensor([.2, .8]),
        }}})
    initial = copy.deepcopy(target.state_dict())
    with torch.no_grad():
        for p in model.parameters(): p.add_(1)
        model["head"].get_buffer("buffer").add_(1)
    opt.step(0)
    torch.testing.assert_close(target["head"].get_parameter("weight"), initial["head.weight"] + torch.tensor([[0.], [1.]]))
    torch.testing.assert_close(target["head"].get_parameter("bias"), initial["head.bias"] + torch.tensor([.2, .8]))
    torch.testing.assert_close(target["head"].get_buffer("buffer"), initial["head.buffer"])


@pytest.mark.parametrize("bad", ["missing", "overlap", "unfrozen_target", "no_tau", "no_target"])
def test_invalid_group_registration(bad):
    model = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 2)})
    if bad == "overlap": model["b"] = model["a"]
    groups = {"a": {"online": model["a"], "lr": .01}, "b": {"online": model["b"], "lr": .01}}
    if bad == "missing": del groups["b"]["online"]
    if bad == "unfrozen_target": groups["a"].update(target=copy.deepcopy(model["a"]), polyak_tau=.1)
    if bad == "no_tau": groups["a"].update(target=copy.deepcopy(model["a"]).requires_grad_(False))
    if bad == "no_target": groups["a"]["polyak_tau"] = .1
    with pytest.raises(ValueError): MouseOptimizer(groups=groups)


def test_weight_decay_is_global_while_learning_rates_are_per_group():
    model = nn.ModuleDict({"a": nn.Linear(1, 1, bias=False), "b": nn.Linear(1, 1, bias=False)})
    with torch.no_grad():
        for p in model.parameters(): p.fill_(1.)
    opt = MouseOptimizer(groups={"a": {"online": model["a"], "lr": .1},
                                 "b": {"online": model["b"], "lr": .2}},
                         optimizer={"name": "adamw", "weight_decay": .2})
    for p in model.parameters(): p.grad = torch.zeros_like(p)
    opt.step(0)
    assert [group["name"] for group in opt.param_groups] == ["a", "b"]
    assert model["a"].get_parameter("weight").item() == pytest.approx(.98)
    assert model["b"].get_parameter("weight").item() == pytest.approx(.96)


def test_effective_lr_overflow_fails_before_any_weight_update():
    model = nn.ModuleDict({"head": nn.Linear(1, 1)})
    class ScaledOptimizer(MouseOptimizer):
        def _build_optimizer(self, params, optimizer):
            params[0]["lr"] *= 2
            return super()._build_optimizer(params, optimizer)
    opt = ScaledOptimizer(groups={"head": {"online": model["head"], "lr": lambda step: .1 if step == 0 else 1e308}},
                          optimizer="sgd")
    before = copy.deepcopy(model.state_dict())
    for p in model.parameters(): p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="effective lr"):
        opt.step(1)
    for name, p in model.state_dict().items(): torch.testing.assert_close(p, before[name])


@pytest.mark.parametrize("name", ["adamw", "adam", "sgd", "rmsprop"])
def test_single_selected_optimizer_matches_torch_for_all_explicit_groups(name):
    model = nn.ModuleDict({"a": nn.Linear(2, 2), "b": nn.Linear(2, 1)})
    reference = copy.deepcopy(model)
    target = copy.deepcopy(model).requires_grad_(False)
    settings = {"name": name, "weight_decay": .2}
    if name in ("adamw", "adam"):
        settings.update(betas=(.5, .8), eps=.03, fused=False)
    elif name == "sgd":
        settings.update(momentum=.7)
    else:
        settings.update(alpha=.8, eps=.02, momentum=.3, centered=True)
    optimizer = MouseOptimizer(groups={
        "input": {"online": model["a"], "target": target["a"], "lr": .01, "polyak_tau": .2},
        "output": {"online": model["b"], "target": target["b"], "lr": .03, "polyak_tau": .7},
    }, optimizer=settings)
    cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
           "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}[name]
    expected = cls([{"params": reference["a"].parameters(), "lr": .01},
                    {"params": reference["b"].parameters(), "lr": .03}],
                   **{key: value for key, value in settings.items() if key != "name"})
    for step, grad in enumerate((.3, -.5, .2)):
        before = copy.deepcopy(target.state_dict())
        for p, q in zip(model.parameters(), reference.parameters()):
            p.grad = torch.full_like(p, grad)
            q.grad = torch.full_like(q, grad)
        optimizer.step(step)
        expected.step()
        for key, actual in model.state_dict().items():
            torch.testing.assert_close(actual, reference.state_dict()[key], rtol=0, atol=0)
            tau = .2 if key.startswith('a.') else .7
            torch.testing.assert_close(target.state_dict()[key], before[key].lerp(actual, tau))
    assert len(optimizer.param_groups) == 2
    assert optimizer.groups["input"]["online"] is model["a"]


@pytest.mark.parametrize("key,value", [("optimizer", "rmsprop"), ("weight_decay", .1),
                                       ("momentum", .9), ("betas", (.8, .9)), ("eps", 1e-6)])
def test_groups_reject_optimizer_overrides(key, value):
    with pytest.raises(ValueError, match="supported settings"):
        MouseOptimizer(groups={"weights": {"online": nn.Linear(2, 1), "lr": .1, key: value}})


@pytest.mark.parametrize("key", ["online", "target"])
def test_online_and_target_are_not_global_arguments(key):
    with pytest.raises(TypeError, match="unexpected keyword"):
        MouseOptimizer(groups={"weights": {"online": nn.Linear(2, 1), "lr": .1}}, **{key: nn.Linear(2, 1)})


@pytest.mark.parametrize("settings", [
    {"name": "unknown"}, {"name": "sgd", "betas": (.9, .95)},
    {"name": "adamw", "eps": float("nan")}, {"name": "adamw", "betas": (1., .95)},
    {"name": "rmsprop", "alpha": 1.}, {"name": "rmsprop", "centered": 1},
    {"name": "adamw", "weight_decay": -1.},
])
def test_global_optimizer_options_are_validated(settings):
    with pytest.raises(ValueError):
        MouseOptimizer(groups={"weights": {"online": nn.Linear(2, 1), "lr": .1}}, optimizer=settings)


@pytest.mark.parametrize("weight_update_interval_steps", [1, 3])
def test_accumulation_uses_batch_index_for_schedules_and_keeps_evaluation_cadence(weight_update_interval_steps):
    online = nn.Linear(1, 1, bias=False)
    with torch.no_grad(): online.weight.fill_(1.)
    target = copy.deepcopy(online).requires_grad_(False)
    schedule_calls = []
    def lr(step):
        schedule_calls.append(step)
        return .1 if step < 3 else .2
    optimizer = MouseOptimizer(groups={"head": {
        "online": online, "target": target, "lr": lr,
        "polyak_tau": lambda step: 0. if step < 3 else 1.,
    }}, optimizer="sgd")
    schedule_calls.clear()
    evaluations = []
    total = 7
    count = weight_update_interval_steps
    for step in range(total):
        if step % 2 == 0:
            evaluations.append(step)
        if step % weight_update_interval_steps == 0:
            optimizer.zero_grad()
            count = min(weight_update_interval_steps, total - step)
        (online.weight.sum() / count).backward()
        if (step + 1) % weight_update_interval_steps == 0 or step == total - 1:
            optimizer.step(step)
        if step < 3:
            torch.testing.assert_close(target.weight, torch.ones_like(target.weight))
    expected_steps = [s for s in range(total)
                      if (s + 1) % weight_update_interval_steps == 0 or s == total - 1]
    assert evaluations == [0, 2, 4, 6]
    assert schedule_calls == expected_steps
    expected_weight = 1. - sum(.1 if step < 3 else .2 for step in expected_steps)
    assert online.weight.item() == pytest.approx(expected_weight)
    torch.testing.assert_close(target.weight, online.weight)


def test_training_step_is_required_and_not_inferred_from_optimizer_state():
    _, _, optimizer = setup()
    with pytest.raises(TypeError):
        cast(Any, optimizer).step()
    assert "next_step" not in optimizer.state_dict()
    with pytest.raises(ValueError, match="fields"):
        optimizer.load_state_dict({**optimizer.state_dict(), "next_step": 2})


@pytest.mark.parametrize("step", [None, -1, True, 1.5])
def test_training_step_requires_a_nonnegative_integer(step):
    _, _, optimizer = setup()
    with pytest.raises(ValueError, match="training step"):
        optimizer.step(step)
