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


def test_adamw_fp32_state_dict_roundtrip_resumes_sub_ulp_progress() -> None:
    """Resume must restore the fp32 masters, not just the bf16 compute weights."""
    def _run(opt_factory, w, steps):
        opt = opt_factory(w)
        for _ in range(steps):
            w.grad = torch.tensor(1.0, dtype=torch.bfloat16)
            opt.step()
            opt.zero_grad()
        return opt

    factory = lambda w: AdamWFp32([w], lr=1e-5, fused=False)  # noqa: E731
    w_ref = torch.nn.Parameter(torch.tensor(0.02, dtype=torch.bfloat16))
    _run(factory, w_ref, 40)

    w = torch.nn.Parameter(torch.tensor(0.02, dtype=torch.bfloat16))
    opt = _run(factory, w, 20)
    state = opt.state_dict()
    assert len(state["masters"]) == 1
    assert state["masters"][0].dtype == torch.float32

    w2 = torch.nn.Parameter(w.detach().clone())
    opt2 = AdamWFp32([w2], lr=1e-5, fused=False)
    opt2.load_state_dict(state)
    for _ in range(20):
        w2.grad = torch.tensor(1.0, dtype=torch.bfloat16)
        opt2.step()
        opt2.zero_grad()
    assert torch.equal(w2.float(), w_ref.float())
    master = opt2.state_dict()["masters"][0]
    assert torch.allclose(master, _run(factory, torch.nn.Parameter(torch.tensor(0.02, dtype=torch.bfloat16)), 40).state_dict()["masters"][0])


def test_adamw_fp32_load_state_dict_rejects_mismatch() -> None:
    import pytest

    a = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    b = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    state = AdamWFp32([a], lr=1e-2, fused=False).state_dict()
    with pytest.raises(ValueError, match="master shape"):
        AdamWFp32([b], lr=1e-2, fused=False).load_state_dict(state)
    with pytest.raises(ValueError, match="master tensors"):
        AdamWFp32([a, b], lr=1e-2, fused=False).load_state_dict(state)


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
