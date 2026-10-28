"""AdamW over fp32 trainable parameters (no GPU required)."""

from __future__ import annotations

import copy
import pytest
import torch

from mouse_core.optim import AdamW


def test_adamw_accumulates_sub_bf16_ulp_updates() -> None:
    """A 1e-5 step is below the bf16 ULP around 0.02; fp32 parameters still move."""
    w = torch.nn.Parameter(torch.tensor(0.02))
    start = w.detach().clone()
    opt = AdamW(params=[w], lr=1e-5, fused=False)
    for _ in range(20):
        w.grad = torch.tensor(1.0)
        opt.step()
        opt.zero_grad()
    assert not torch.equal(w, start)


def test_adamw_rejects_trainable_non_fp32_params() -> None:
    low = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="fp32 parameters only"):
        AdamW(params=[low], lr=1e-2, fused=False)
    half = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    with pytest.raises(TypeError, match="torch.float16"):
        AdamW(params=[torch.nn.Parameter(torch.ones(4)), half], lr=1e-2, fused=False)


def test_adamw_accepts_frozen_non_fp32_params() -> None:
    """Frozen bf16 backbone base weights pass through ``model.parameters()`` untouched."""
    frozen = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(4))
    opt = AdamW(params=[frozen, trainable], lr=1e-2, fused=False)
    assert sum(len(group["params"]) for group in opt.param_groups) == 1
    trainable.grad = torch.ones_like(trainable)
    opt.step()
    opt.zero_grad()
    assert torch.equal(frozen.float(), torch.ones(4))
    assert not torch.equal(trainable, torch.ones(4))


def test_adamw_state_dict_resumes_identical_updates() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW(params=[p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    opt2 = AdamW(params=[q], lr=0.1, fused=False)
    opt2.load_state_dict(state)
    assert opt2.param_groups[0]["lr"] == 1e-2
    for scale in (0.3, -2.0, 0.0):
        p.grad = torch.full_like(p, scale)
        q.grad = p.grad.clone()
        opt.step()
        opt2.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)


@pytest.mark.parametrize("set_to_none", [None, True, False])
def test_zero_grad_preserves_requested_storage_semantics(set_to_none) -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW(params=[p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    held = p.grad
    if set_to_none is None:
        opt.zero_grad()
    else:
        opt.zero_grad(set_to_none=set_to_none)
    if set_to_none is False:
        assert p.grad is held
        assert torch.equal(p.grad, torch.zeros_like(p))
    else:
        assert p.grad is None


def test_adamw_param_group_learning_rates_step_independently() -> None:
    slow = torch.nn.Parameter(torch.zeros(1))
    fast = torch.nn.Parameter(torch.zeros(1))
    opt = AdamW(
        params=[
            {"params": [slow], "lr": 1e-3},
            {"params": [fast], "lr": 1e-1},
        ],
        lr=1e-3,
        weight_decay=0.0,
        fused=False,
    )
    assert [group["lr"] for group in opt.param_groups] == [1e-3, 1e-1]
    slow.grad = torch.ones_like(slow)
    fast.grad = torch.ones_like(fast)
    opt.step()
    assert fast.abs().item() > slow.abs().item() > 0.0


def test_adamw_param_group_rejects_a_mix_and_unknown_keys() -> None:
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(TypeError, match="not a mix"):
        AdamW(params=[p, {"params": [p], "lr": 1e-3}], lr=1e-3, fused=False)
    with pytest.raises(TypeError, match="missing 'params'"):
        AdamW(params=[{"lr": 1e-3}], lr=1e-3, fused=False)
    with pytest.raises(TypeError, match="unsupported keys"):
        AdamW(params=[{"params": [p], "momentum": 0.9}], lr=1e-3, fused=False)
