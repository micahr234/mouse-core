"""Independent numerical checks for AdamP and its fused CUDA implementation."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch

from mouse_core.adamp import AdamP


def _reference(p, g, m, v, step, *, lr, betas, eps, weight_decay, delta, wd_ratio, nesterov):
    """Float64 NumPy oracle for the published AdamP equations and gate order."""
    b1, b2 = betas
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    update = (b1 * m + (1 - b1) * g if nesterov else m) / (np.sqrt(v / (1 - b2 ** step)) + eps)
    decay = 1.0
    if p.ndim > 1:
        for rows in [p.shape[0], 1]:
            w, grad = p.reshape(rows, -1), g.reshape(rows, -1)
            wn, gn = np.linalg.norm(w, axis=1), np.linalg.norm(grad, axis=1)
            cosine = np.abs(np.sum(w * grad, axis=1)) / (np.maximum(wn, eps) * np.maximum(gn, eps))
            if np.max(cosine) < delta / math.sqrt(w.shape[1]):
                unit = w / (wn[:, None] + eps)
                u = update.reshape(rows, -1)
                update = (u - unit * np.sum(unit * u, axis=1, keepdims=True)).reshape(p.shape)
                decay = wd_ratio
                break
    return p * (1 - lr * weight_decay * decay) - lr / (1 - b1 ** step) * update, m, v


CASES = [
    ([[1., 0.], [0., 1.]], [[0., 2.], [3., 0.]]),  # channel projection
    ([[1., 0.], [1., 0.]], [[2., 1.], [-2., 1.]]),  # layer-only projection
    ([[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]),  # no projection
    ([1., 2., 3.], [3., -2., 1.]),  # vectors never project
    (1., -2.),
    ([[0., 0.], [0., 0.]], [[1., -1.], [2., 3.]]),
]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("nesterov", [False, True])
@pytest.mark.parametrize("initial,gradient", CASES)
def test_adamp_matches_numpy(initial, gradient, nesterov, fused):
    if fused and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if fused else "cpu"
    p = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32, device=device))
    kwargs = dict(lr=0.03, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.2,
                  delta=0.1, wd_ratio=0.1, nesterov=nesterov)
    opt = AdamP(params=[p], fused=fused, **kwargs)
    ref = np.array(initial, dtype=np.float64)
    m, v = np.zeros_like(ref), np.zeros_like(ref)
    for step in range(1, 8):
        g = np.array(gradient, dtype=np.float64) * (1 if step % 2 else -0.7)
        p.grad = torch.tensor(g, dtype=torch.float32, device=device)
        opt.step()
        ref, m, v = _reference(ref, g, m, v, step, **kwargs)
        np.testing.assert_allclose(p.detach().cpu().numpy(), ref, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("fused", [False, True])
def test_adamp_missing_grads_groups_and_resume(fused):
    if fused and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if fused else "cpu"
    ps = [torch.nn.Parameter(torch.ones((3, 5), device=device)) for _ in range(3)]
    def build(params):
        return AdamP(params=[{"params": params[:2], "lr": 0.01}, {"params": params[2:], "lr": 0.0}],
                     lr=0.1, fused=fused)
    opt = build(ps)
    for i, p in enumerate(ps):
        p.grad = torch.full_like(p, i + 1.) if i != 1 else None
    opt.step()
    assert ps[1] not in opt._inner.state
    assert torch.equal(ps[2], torch.ones_like(ps[2]))
    saved = copy.deepcopy(opt.state_dict())
    qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    restored = build(qs)
    restored.load_state_dict(saved)
    for i, (p, q) in enumerate(zip(ps, qs)):
        p.grad = torch.full_like(p, i + 0.3)
        q.grad = p.grad.clone()
    opt.step()
    restored.step()
    for p, q in zip(ps, qs):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert [opt._inner.state[p]["step"] for p in ps] == [2, 1, 2]
    opt.zero_grad()
    assert all(p.grad is None for p in ps)


def test_adamp_validation_and_frozen_parameters():
    frozen = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)
    p = torch.nn.Parameter(torch.ones(2))
    opt = AdamP(params=[frozen, p], lr=0.01)
    assert len(opt.param_groups[0]["params"]) == 1
    with pytest.raises(TypeError, match="fp32"):
        AdamP(params=[torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))], lr=0.01)
    with pytest.raises(ValueError, match="lr"):
        AdamP(params=[p], lr=float("nan"))
    with pytest.raises(ValueError, match="AdamP checkpoint"):
        opt.load_state_dict({"inner": {}})
    strict = AdamP(params=[p], lr=0.01, fused=True)
    p.grad = torch.ones_like(p)
    with pytest.raises(ValueError, match="contiguous CUDA"):
        strict.step()
    assert not strict._inner.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(7, 35), (32, 1024), (1024, 32), (32, 3072), (3072, 32), (2, 3, 5)])
def test_fused_batched_shapes_and_new_gradient_storage(shape):
    torch.manual_seed(42)
    ps = [torch.nn.Parameter(torch.randn(shape, device="cuda")) for _ in range(3)]
    qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    kwargs = dict(lr=0.002, betas=(0.9, 0.999), weight_decay=0.03, nesterov=True)
    fused = AdamP(params=ps, fused=True, **kwargs)
    eager = AdamP(params=qs, fused=False, **kwargs)
    for step in range(5):
        for i, (p, q) in enumerate(zip(ps, qs)):
            if i == 0 and step == 2:
                p.grad = q.grad = None
            else:
                p.grad = torch.randn_like(p)
                q.grad = p.grad.clone()
        fused.step()
        eager.step()
        for p, q in zip(ps, qs):
            torch.testing.assert_close(p, q, rtol=3e-5, atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_projection_modes():
    ps = [torch.nn.Parameter(torch.tensor(p, device="cuda")) for p, _ in CASES[:3]]
    opt = AdamP(params=ps, lr=0.01, fused=True)
    for p, (_, g) in zip(ps, CASES[:3]):
        p.grad = torch.tensor(g, device="cuda")
    opt.step()
    (_, summary), = opt._inner._workspace.values()
    assert summary[:, 0].tolist() == [1, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_falls_back_for_noncontiguous_tensors():
    p = torch.nn.Parameter(torch.randn(3, 5, device="cuda").T)
    q = torch.nn.Parameter(p.detach().clone())
    auto = AdamP(params=[p], lr=0.01)
    eager = AdamP(params=[q], lr=0.01, fused=False)
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    auto.step()
    eager.step()
    torch.testing.assert_close(p, q)
    assert not auto._inner._workspace


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("epsilon", [1e-3, 1e-8])
def test_fused_small_norms_and_cpu_checkpoint(epsilon):
    torch.manual_seed(14)
    p = torch.nn.Parameter(torch.randn(5, 7, device="cuda") * 1e-8)
    q = torch.nn.Parameter(p.detach().cpu())
    kwargs = dict(lr=1e-4, eps=epsilon, weight_decay=0.1)
    fused, eager = AdamP(params=[p], fused=True, **kwargs), AdamP(params=[q], fused=False, **kwargs)
    for _ in range(3):
        p.grad = torch.randn_like(p) * 1e-8
        q.grad = p.grad.cpu()
        fused.step()
        eager.step()
        torch.testing.assert_close(p.cpu(), q, atol=1e-8, rtol=2e-5)
    # Move state through torch.save/load(map_location='cpu'), as real checkpoints do.
    import io
    buffer = io.BytesIO()
    torch.save(fused.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location="cpu", weights_only=False)
    r = torch.nn.Parameter(p.detach().cpu())
    restored = AdamP(params=[r], fused=False, **kwargs)
    restored.load_state_dict(state)
    p.grad = torch.randn_like(p) * 0.1
    r.grad = p.grad.cpu()
    fused.step()
    restored.step()
    torch.testing.assert_close(p.cpu(), r, rtol=2e-5, atol=1e-8)
