"""Optimizer construction owns parameter selection and skips inactive groups."""
import copy
import pytest
import torch
from torch import nn
from mouse_core import MouseOptimizer
from mouse_core.models.lora import LoRALinear, LoRAConfig


def test_frozen_and_empty_groups_need_no_rates_or_optimizer_state():
    frozen = nn.Linear(2, 2).requires_grad_(False)
    head = nn.Linear(2, 1)
    optimizer = MouseOptimizer(groups={
        "base": {"online": frozen},
        "empty": {"online": nn.Identity()},
        "head": {"online": head, "lr": .1},
    }, optimizer="rmsprop")
    inputs = torch.ones(1, 2, requires_grad=True)
    head(frozen(inputs)).sum().backward()
    optimizer.step(0)
    assert inputs.grad.abs().sum() > 0
    assert set(optimizer.groups) == {"head"}
    assert [g["name"] for g in optimizer.param_groups] == ["head"]
    assert {id(p) for p in optimizer.inner.state} == {id(p) for p in head.parameters()}
    assert all(p.grad is None for p in frozen.parameters())
    assert [entry[0] for entry in optimizer.state_dict()["groups"]] == ["head"]


def test_inactive_schedules_are_never_evaluated():
    def forbidden(step):
        raise AssertionError("Frozen parameters must have no optimizer work")
    frozen = nn.Linear(2, 2).requires_grad_(False)
    head = nn.Linear(2, 1)
    optimizer = MouseOptimizer(groups={
        "frozen": {"online": frozen, "lr": forbidden, "target": frozen, "polyak_tau": forbidden},
        "head": {"online": head, "lr": .1},
    }, optimizer="sgd")
    optimizer.step(0)


@pytest.mark.parametrize("train_base", [False, True])
def test_base_and_lora_selection_pairs_targets_and_keeps_independent_rates(train_base):
    source = LoRALinear(nn.Linear(2, 2), LoRAConfig(rank=1))
    source.base.requires_grad_(train_base)
    target = copy.deepcopy(source).requires_grad_(False)
    original = {name: p.detach().clone() for name, p in source.named_parameters()}
    optimizer = MouseOptimizer(groups={
        "base": {"online": source, "parameters": "base", "lr": .1,
                 "target": target, "polyak_tau": .2},
        "lora": {"online": source, "parameters": "lora", "lr": .3,
                 "target": target, "polyak_tau": .4},
    }, optimizer="sgd")
    for p in source.parameters():
        if p.requires_grad: p.grad = torch.ones_like(p)
    optimizer.step(0)
    assert set(optimizer.groups) == ({"base", "lora"} if train_base else {"lora"})
    registered = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(registered) == len(set(registered))
    for name, p in source.named_parameters():
        is_base = name.startswith("base.")
        rate = .1 if is_base else .3
        tau = .2 if is_base else .4
        torch.testing.assert_close(p, original[name] - (rate if p.requires_grad else 0.))
        torch.testing.assert_close(target.get_parameter(name), original[name] - (rate * tau if p.requires_grad else 0.))


def test_absent_lora_is_omitted_without_caller_filter():
    source = nn.Linear(2, 2)
    optimizer = MouseOptimizer(groups={
        "base": {"online": source, "parameters": "base", "lr": .1},
        "lora": {"online": source, "parameters": "lora"},
    }, optimizer="sgd")
    assert set(optimizer.groups) == {"base"}


def test_unfreezing_ignored_group_requires_reconstruction_before_any_update():
    frozen, head = nn.Linear(1, 1).requires_grad_(False), nn.Linear(1, 1)
    optimizer = MouseOptimizer(groups={"frozen": {"online": frozen}, "head": {"online": head, "lr": .1}})
    frozen.requires_grad_(True)
    before = head.weight.detach().clone()
    head.weight.grad = torch.ones_like(head.weight)
    with pytest.raises(ValueError, match="Trainability changed"):
        optimizer.step(0)
    torch.testing.assert_close(head.weight, before)


@pytest.mark.parametrize("settings", [
    {"parameters": "typo", "lr": .1}, {}, {"lr": None},
])
def test_trainable_groups_still_require_valid_settings(settings):
    with pytest.raises(ValueError):
        MouseOptimizer(groups={"head": {"online": nn.Linear(2, 1), **settings}})


def test_all_inactive_groups_raise_clear_error():
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        MouseOptimizer(groups={"frozen": {"online": nn.Linear(2, 2).requires_grad_(False)}})


def test_overlapping_selection_is_rejected():
    source = LoRALinear(nn.Linear(2, 2), LoRAConfig(rank=1))
    with pytest.raises(ValueError, match="overlap"):
        MouseOptimizer(groups={"all": {"online": source, "lr": .1},
                               "lora": {"online": source, "parameters": "lora", "lr": .2}})
