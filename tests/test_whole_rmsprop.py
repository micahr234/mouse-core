"""Whole-step compilation preserves heterogeneous RMSprop updates."""
import copy

import pytest
import torch

from mouse_core.precision_optim import PrecisionOptimizer, _rmsprop_step_, _rmsprop_update_
from mouse_core.rounding import scalar_tensor


def test_whole_step_reuses_one_graph_for_shapes_and_learning_rates():
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    parameters = [torch.nn.Parameter(torch.ones(n, 3)) for n in range(4, 24)]
    states = [torch.zeros_like(p) for p in parameters]
    compiled = torch.compile(_rmsprop_step_, backend=backend, fullgraph=True,
                             dynamic=False, isolate_recompiles=True)
    with torch.no_grad():
        for rate in (.01, .001, 0.):
            updates = [(p.detach(), torch.ones_like(p), s, None, None,
                        dict(lr=scalar_tensor(rate), alpha=.95, eps=1e-8,
                             weight_decay=0., momentum=0.))
                       for p, s in zip(parameters, states, strict=True)]
            compiled(updates, stochastic_rounding=False)
    assert len(graphs) == 1
    for p in parameters:
        assert p.is_leaf and p.requires_grad and bool((p < 1).all())
        p.square().sum().backward()
        torch.testing.assert_close(p.grad, 2 * p.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reference_device', ['cpu', 'cuda'])
def test_whole_step_gpu_matches_reference_and_resumes(dtype, reference_device):
    torch.manual_seed(42)
    originals = [torch.randn(7, 11, dtype=torch.bfloat16).t(),
                 torch.randn(13, dtype=torch.float32), torch.randn(3, 5, dtype=torch.float64)]
    ps = [torch.nn.Parameter(p.cuda()) for p in originals]
    qs = [torch.nn.Parameter(p.to(reference_device, copy=True)) for p in originals]
    frozen = torch.nn.Parameter(torch.ones(5, device='cuda'), requires_grad=False)
    frozen.grad = torch.ones_like(frozen)
    options = [dict(alpha=.93, eps=.003, weight_decay=.12, centered=True, momentum=.7),
               dict(alpha=.95, eps=1e-8, weight_decay=0., centered=False, momentum=0.),
               dict(alpha=.91, eps=.001, weight_decay=.03, centered=False, momentum=.4)]
    opt = PrecisionOptimizer([dict(params=[p], compile=True, **o) for p, o in zip(ps, options, strict=True)] +
                             [dict(params=[frozen])], name='rmsprop', options={}, dtype=dtype,
                             stochastic_rounding=False)
    ref = PrecisionOptimizer([dict(params=[p], compile=True, **o) for p, o in zip(qs, options, strict=True)],
                             name='rmsprop', options={}, dtype=dtype, stochastic_rounding=False)
    if reference_device == 'cuda':
        single = torch.compile(_rmsprop_update_, fullgraph=True, dynamic=True,
                               isolate_recompiles=True, options={"emulate_precision_casts": True})
        def per_tensor(updates):
            for p, g, s, ga, buf, options in updates:
                single(p, g, s, ga, buf, **options, stochastic_rounding=False)
        ref._apply_rmsprop_step = per_tensor
    for index, rate in enumerate((.01, .001, 0., .002)):
        for i, (p, q) in enumerate(zip(ps, qs, strict=True)):
            opt.param_groups[i]['lr'] = ref.param_groups[i]['lr'] = rate / (i+1)
            q.data.copy_(p.detach().to(reference_device))
            grad = torch.randn_like(q)
            p.grad = None if index == 1 and i == 1 else grad.cuda()
            q.grad = None if p.grad is None else grad
        opt.step()
        ref.step()
        for p, q in zip(ps, qs, strict=True):
            torch.testing.assert_close(p.cpu(), q.cpu(), rtol=.008 if p.dtype == torch.bfloat16 else 2e-5,
                                       atol=2e-7)
            for key, value in ref.state[q].items():
                # CPU and CUDA reciprocal/sqrt arithmetic differ slightly in
                # FP32 momentum; the previous per-tensor CUDA path does too.
                # Require exact state agreement with that compiled CUDA path.
                rtol, atol = ((0., 0.) if reference_device == 'cuda' else
                              (2e-6, 1e-6) if dtype == torch.float32 else (1e-12, 1e-12))
                torch.testing.assert_close(opt.state[p][key].cpu(), value.cpu(), rtol=rtol, atol=atol)
        if index == 2:
            restored = PrecisionOptimizer(opt.param_groups, name='rmsprop', options={}, dtype=dtype,
                                          stochastic_rounding=False)
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
            opt = restored
    assert opt._compiled_rmsprop_step is not None
    assert frozen not in opt.state
    torch.testing.assert_close(frozen, torch.ones_like(frozen))
    for p in ps:
        p.grad = None
        p.square().sum().backward()
        torch.testing.assert_close(p.grad, 2 * p.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('transformed_gradient', [False, True])
def test_whole_step_gpu_stochastic_rounding_retains_small_updates(dtype, transformed_gradient):
    torch.manual_seed(99)
    ps = [torch.nn.Parameter(torch.ones(100_000, device='cuda', dtype=model_dtype))
          for model_dtype in (torch.bfloat16, torch.float32)]
    groups = [dict(params=[p], compile=True, lr=torch.finfo(p.dtype).eps / 16) for p in ps]
    opt = PrecisionOptimizer(groups, name='rmsprop', options={}, dtype=dtype, stochastic_rounding=True)
    if transformed_gradient:
        opt.gradient_transform = lambda parameter, gradient: gradient.mul_(1.)
    for p in ps:
        p.grad = -torch.ones_like(p)
    opt.step()
    for p in ps:
        expected = 1 + torch.finfo(p.dtype).eps / 16 / (.01**.5 + 1e-8)
        tolerance = torch.finfo(p.dtype).eps * .015
        # FP32 arithmetic rounds the FP32 result normally; FP64 arithmetic
        # stochastically narrows to either model dtype.
        if dtype == p.dtype:
            expected = torch.tensor(expected, dtype=dtype).item()
        assert p.double().mean().item() == pytest.approx(expected, abs=tolerance)
        assert torch.isfinite(p).all()
