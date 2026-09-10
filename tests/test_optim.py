"""AdamW over fp32 trainable parameters (no GPU required)."""

from __future__ import annotations

import pytest
import torch

from mouse_core.optim import AdamW


def _opt_dtypes(opt: AdamW) -> set[torch.dtype]:
    return {p.dtype for group in opt.param_groups for p in group["params"]}


def test_adamw_steps_fp32_params_in_place() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.step()
    assert p.dtype == torch.float32
    assert _opt_dtypes(opt) == {torch.float32}
    assert not torch.equal(p, torch.ones(4))


def test_adamw_accumulates_sub_bf16_ulp_updates() -> None:
    """A 1e-5 step is below the bf16 ULP around 0.02; fp32 parameters still move."""
    w = torch.nn.Parameter(torch.tensor(0.02))
    start = w.detach().clone()
    opt = AdamW([w], lr=1e-5, fused=False)
    for _ in range(20):
        w.grad = torch.tensor(1.0)
        opt.step()
        opt.zero_grad()
    assert not torch.equal(w, start)


def test_adamw_rejects_trainable_non_fp32_params() -> None:
    low = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    with pytest.raises(TypeError, match="fp32 parameters only"):
        AdamW([low], lr=1e-2, fused=False)
    half = torch.nn.Parameter(torch.ones(4, dtype=torch.float16))
    with pytest.raises(TypeError, match="torch.float16"):
        AdamW([torch.nn.Parameter(torch.ones(4)), half], lr=1e-2, fused=False)


def test_adamw_accepts_frozen_non_fp32_params() -> None:
    """Frozen bf16 backbone base weights pass through ``model.parameters()`` untouched."""
    frozen = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([frozen, trainable], lr=1e-2, fused=False)
    assert sum(len(group["params"]) for group in opt.param_groups) == 1
    trainable.grad = torch.ones_like(trainable)
    opt.step()
    opt.zero_grad()
    assert torch.equal(frozen.float(), torch.ones(4))
    assert not torch.equal(trainable, torch.ones(4))


def test_adamw_state_dict_roundtrip() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.step()
    state = opt.state_dict()
    q = torch.nn.Parameter(torch.ones(4))
    opt2 = AdamW([q], lr=1e-2, fused=False)
    opt2.load_state_dict(state)
    assert opt2.param_groups[0]["lr"] == 1e-2
    assert len(opt2._inner.state) == 1


def test_zero_grad_set_to_none() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.zero_grad(set_to_none=True)
    assert p.grad is None


def test_zero_grad_zeros_in_place() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    held = p.grad
    opt.zero_grad(set_to_none=False)
    assert p.grad is held
    assert torch.equal(p.grad, torch.zeros_like(p))


def test_zero_grad_default_drops_grad() -> None:
    p = torch.nn.Parameter(torch.ones(4))
    opt = AdamW([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.zero_grad()
    assert p.grad is None
