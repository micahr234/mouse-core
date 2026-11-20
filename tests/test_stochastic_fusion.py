"""Fused stochastic updates preserve optimizer and rounding behavior."""
import copy
import math
from typing import Any

import pytest
import torch

from mouse_core.precision_optim import PrecisionOptimizer, _rmsprop_update_
from mouse_core.rounding import _polyak_bfloat16_, stochastic_round_bfloat16, scalar_tensor


@pytest.mark.parametrize('centered', [False, True])
@pytest.mark.parametrize('momentum', [0., .7])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_rmsprop_fast_path_matches_native_with_schedules_and_strides(centered, momentum, dtype):
    p = torch.nn.Parameter(torch.linspace(-1, 1, 24).reshape(4, 6).t().to(dtype))
    q = torch.nn.Parameter(p.float().clone())
    options: dict[str, Any] = dict(alpha=.93, eps=.003, weight_decay=.12, momentum=momentum, centered=centered)
    opt = PrecisionOptimizer([p], name='rmsprop', options=options)
    ref = torch.optim.RMSprop([q], **options, foreach=False)
    for step, rate in enumerate((.01, 0., .0001)):
        opt.param_groups[0]['lr'] = ref.param_groups[0]['lr'] = rate
        q.data.copy_(p.float())
        p.grad = torch.randn_like(p)
        q.grad = p.grad.float().clone()
        ref.step()
        rng = torch.random.get_rng_state()
        expected = stochastic_round_bfloat16(q) if dtype == torch.bfloat16 else q.detach().clone()
        torch.random.set_rng_state(rng)
        opt.step()
        torch.testing.assert_close(p, expected, rtol=0, atol=0)
        for key, value in ref.state[q].items():
            torch.testing.assert_close(opt.state[p][key], value, rtol=0, atol=0)
        assert not opt._kernel.state and not opt._kernel.param_groups


def test_rmsprop_fast_path_skips_native_optimizer_and_missing_gradients(monkeypatch):
    p = torch.nn.Parameter(torch.ones(10, dtype=torch.bfloat16))
    unused = torch.nn.Parameter(torch.ones(10, dtype=torch.bfloat16))
    opt = PrecisionOptimizer([p, unused], name='rmsprop', options={})
    p.grad = -torch.ones_like(p)
    def unexpected():
        pytest.fail('ordinary RMSprop should not construct temporary parameters or call the native optimizer')
    monkeypatch.setattr(opt._kernel, 'step', unexpected)
    opt.step()
    assert set(opt.state) == {p}
    torch.testing.assert_close(unused, torch.ones_like(unused))


def test_switching_transform_preserves_moments_and_resume():
    p = torch.nn.Parameter(torch.ones(100, dtype=torch.bfloat16))
    opt = PrecisionOptimizer([p], name='rmsprop', options=dict(momentum=.8, centered=True))
    p.grad = -torch.ones_like(p)
    opt.step()
    def transform(parameter, update):
        update.add_(.001)
    opt.update_transform = transform
    opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    restored = PrecisionOptimizer([q], name='rmsprop', options=dict(momentum=.8, centered=True))
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    opt.update_transform = None
    q.grad = p.grad.clone()
    rng = torch.random.get_rng_state()
    opt.step()
    torch.random.set_rng_state(rng)
    restored.step()
    torch.testing.assert_close(p, q, rtol=0, atol=0)
    for key, value in opt.state[p].items():
        torch.testing.assert_close(value, restored.state[q][key], rtol=0, atol=0)


def test_tensor_learning_rates_do_not_specialize_graph_on_values():
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(_rmsprop_update_, backend=backend, fullgraph=True, dynamic=True)
    p = torch.ones(100)
    g = torch.ones_like(p)
    s = torch.zeros_like(p)
    for lr in (.001, .002, .003):
        compiled(p, g, s, None, None, lr=scalar_tensor(lr), alpha=.95,
                 eps=1e-8, weight_decay=0., momentum=0.)
    assert len(graphs) == 1


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('centered,momentum', [(False, 0.), (True, .7)])
def test_compiled_update_and_polyak_numerics(device, centered, momentum):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    # Exercise the actual compiler, not a mock or an eager graph backend.
    update = torch.compile(_rmsprop_update_, fullgraph=True, dynamic=True)
    target_update = torch.compile(_polyak_bfloat16_, fullgraph=True, dynamic=True)
    n = 100_000
    p = torch.ones(n, dtype=torch.bfloat16, device=device)
    g = -torch.ones_like(p)
    s = torch.zeros_like(p, dtype=torch.float32)
    ga = torch.zeros_like(s) if centered else None
    buf = torch.zeros_like(s) if momentum else None
    target = p.clone()
    update(p, g, s, ga, buf, lr=scalar_tensor(.0001), alpha=.95,
           eps=1e-8, weight_decay=0., momentum=momentum)
    expected_square = .05
    expected_mean = 1 + .0001 / (math.sqrt(.05 - (.05 ** 2 if centered else 0)) + 1e-8)
    quantum = .0078125
    prob = (expected_mean - 1) / quantum
    tolerance = 7 * quantum * math.sqrt(prob * (1-prob) / n)
    assert p.float().mean().item() == pytest.approx(expected_mean, abs=tolerance)
    torch.testing.assert_close(s, torch.full_like(s, expected_square))
    assert set(p.unique().tolist()).issubset({1., 1.0078125})
    torch.manual_seed(456)
    target_update(target, p, scalar_tensor(.1))
    expected_target = torch.ones_like(s).lerp(p.float(), .1).mean().item()
    assert target.float().mean().item() == pytest.approx(expected_target, abs=2e-5)
    assert bool((target != 1).any())
    # Re-seeding reproduces compiled stochastic writes.
    repeated = torch.ones_like(target)
    torch.manual_seed(456)
    target_update(repeated, p, scalar_tensor(.1))
    torch.testing.assert_close(repeated, target, rtol=0, atol=0)


def test_optimizer_dynamic_shapes_reuse_graph_and_preserve_live_parameters(monkeypatch):
    import mouse_core.precision_optim as module
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(_rmsprop_update_, backend=backend, fullgraph=True,
                             dynamic=True, isolate_recompiles=True)
    parameters = [torch.nn.Parameter(torch.ones(n, 3)) for n in range(4, 24)]
    pointers = [p.data_ptr() for p in parameters]
    def update(parameter, *args, **kwargs):
        assert type(parameter) is torch.Tensor and not parameter.requires_grad
        assert parameter.data_ptr() in pointers
        return compiled(parameter, *args, **kwargs)
    monkeypatch.setattr(module, '_rmsprop_kernel', lambda device: update)
    optimizer = PrecisionOptimizer(parameters, name='rmsprop', options={}, stochastic_rounding=False)
    for p in parameters:
        p.grad = torch.ones_like(p)
    for _ in range(3):
        optimizer.step()
    assert len(graphs) == 1
    assert set(optimizer.state) == set(parameters)
    for p, pointer in zip(parameters, pointers, strict=True):
        assert p.requires_grad and p.is_leaf and p.data_ptr() == pointer
        assert bool((p < 1).all())
        p.grad = None
        p.square().sum().backward()
        torch.testing.assert_close(p.grad, 2 * p.detach())
