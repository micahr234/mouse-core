"""AdamW vs AdamWFp32 (no GPU required)."""

from __future__ import annotations

import torch

from mouse_core.optim import AdamW, AdamWFp32


def _opt_dtypes(opt: AdamW | AdamWFp32) -> set[torch.dtype]:
    return {p.dtype for group in opt.param_groups for p in group["params"]}


def test_adamw_steps_compute_dtype() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = AdamW([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.step()
    assert p.dtype == torch.bfloat16
    assert _opt_dtypes(opt) == {torch.bfloat16}
    assert not torch.equal(p.float(), torch.ones(4))


def test_adamw_does_not_accumulate_sub_ulp_updates() -> None:
    w = torch.nn.Parameter(torch.tensor(0.02, dtype=torch.bfloat16))
    start = w.detach().float().clone()
    opt = AdamW([w], lr=1e-5, fused=False)
    for _ in range(20):
        w.grad = torch.tensor(1.0, dtype=torch.bfloat16)
        opt.step()
        opt.zero_grad()
    assert torch.equal(w.float(), start)


def test_adamw_fp32_opt_state_is_fp32() -> None:
    low = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    head = torch.nn.Parameter(torch.ones(4))
    opt = AdamWFp32([low, head], lr=1e-2, fused=False)
    assert _opt_dtypes(opt) == {torch.float32}


def test_adamw_fp32_keeps_compute_dtype_and_moves() -> None:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    opt = AdamWFp32([p], lr=1e-2, fused=False)
    p.grad = torch.ones_like(p)
    opt.step()
    assert p.dtype == torch.bfloat16
    assert not torch.equal(p.float(), torch.ones(4))


def test_adamw_fp32_accumulates_sub_ulp_updates() -> None:
    """A 1e-5 step is below bf16 ULP around 0.02; masters still move the weight."""
    w = torch.nn.Parameter(torch.tensor(0.02, dtype=torch.bfloat16))
    start = w.detach().float().clone()
    opt = AdamWFp32([w], lr=1e-5, fused=False)
    for _ in range(20):
        w.grad = torch.tensor(1.0, dtype=torch.bfloat16)
        opt.step()
        opt.zero_grad()
    assert not torch.equal(w.float(), start)


def test_adamw_fp32_steps_fp32_params_in_place() -> None:
    low = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    head = torch.nn.Parameter(torch.ones(4))
    opt = AdamWFp32([low, head], lr=1e-2, fused=False)
    low.grad = torch.ones_like(low)
    head.grad = torch.ones_like(head)
    opt.step()
    assert low.dtype == torch.bfloat16
    assert head.dtype == torch.float32
    assert not torch.equal(low.float(), torch.ones(4))
    assert not torch.equal(head, torch.ones(4))


def test_skips_frozen_params() -> None:
    trainable = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    frozen = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    frozen.requires_grad_(False)
    for cls in (AdamW, AdamWFp32):
        opt = cls([trainable, frozen], lr=1e-2, fused=False)
        assert sum(len(group["params"]) for group in opt.param_groups) == 1
        before = frozen.detach().float().clone()
        trainable.grad = torch.ones_like(trainable)
        opt.step()
        opt.zero_grad()
        assert torch.equal(frozen.float(), before)
