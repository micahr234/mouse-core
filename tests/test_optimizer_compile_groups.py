"""Compile is an execution choice for each optimizer group."""
import copy

import pytest
import torch
from mouse_core import MouseOptimizer
from mouse_core.precision_optim import PrecisionOptimizer


@pytest.mark.parametrize('name', ['rmsprop', 'adam', 'adamw', 'sgd', 'adamw_fused', 'sgd_fused'])
def test_mouse_optimizer_routes_compile_for_every_algorithm(name):
    module = torch.nn.Linear(3, 2)
    opt = MouseOptimizer(groups={'head':dict(online=module, lr=.01, compile=True)}, optimizer=name)
    assert isinstance(opt.inner, PrecisionOptimizer)
    assert opt.inner.param_groups[0]['compile'] is True
    module(torch.ones(1, 3)).sum().backward()
    opt.step(0)


@pytest.mark.parametrize('invalid', [0, 1, 'true', None])
def test_compile_requires_boolean(invalid):
    with pytest.raises(TypeError, match='compile'):
        MouseOptimizer(groups={'head':dict(online=torch.nn.Linear(3, 2), lr=.01, compile=invalid)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('name', ['rmsprop', 'adam', 'adamw', 'sgd'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_mixed_compile_groups_match_eager_updates(name, dtype):
    torch.manual_seed(12)
    ps = [torch.nn.Parameter(torch.randn(4, 7, device='cuda')) for _ in range(2)]
    qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
    options = (dict(momentum=.7, centered=True, alpha=.95, eps=.001, weight_decay=.02) if name == 'rmsprop' else
               dict(momentum=.7, weight_decay=.02) if name == 'sgd' else
               dict(betas=(.8, .95), eps=.001, weight_decay=.02))
    opt = PrecisionOptimizer([dict(params=[ps[0]], compile=True, lr=.01),
                              dict(params=[ps[1]], compile=False, lr=.02)],
                             name=name, options=options, dtype=dtype, stochastic_rounding=False)
    ref = PrecisionOptimizer([dict(params=[q], compile=False, lr=.01*(i+1)) for i,q in enumerate(qs)],
                             name=name, options=options, dtype=dtype, stochastic_rounding=False)
    for step, rate in enumerate((.01, .001, 0., .02)):
        for i, (p, q) in enumerate(zip(ps,qs,strict=True)):
            q.data.copy_(p)
            grad=torch.randn_like(p)
            p.grad=q.grad=grad
            opt.param_groups[i]['lr']=ref.param_groups[i]['lr']=rate*(i+1)
        opt.step(); ref.step()
        for p,q in zip(ps,qs,strict=True):
            torch.testing.assert_close(p,q,rtol=.02 if dtype==torch.bfloat16 else 2e-5,atol=.004 if dtype==torch.bfloat16 else 2e-6)
            assert opt.state[p].keys()==ref.state[q].keys()
            for key,value in ref.state[q].items():
                actual = opt.state[p][key]
                assert actual.dtype == value.dtype
                # Inductor decomposes BF16 compound operations (e.g. lerp)
                # into extra rounded intermediates. Cancellation in momentum
                # makes relative error misleading: bound accumulated error by
                # BF16 spacing at the state's scale and the number of updates.
                tolerance = (torch.finfo(dtype).eps * (step + 1) * float(value.abs().max())
                             if dtype == torch.bfloat16 and key != 'step' else 2e-6)
                torch.testing.assert_close(actual, value, rtol=2e-5, atol=tolerance)
    assert opt._compiled_rmsprop_step is not None or opt._compiled_native_step is not None
    assert ref._compiled_rmsprop_step is None and ref._compiled_native_step is None
    # Execution choices belong to this run, not to the saved optimizer state.
    opt.load_state_dict(copy.deepcopy(ref.state_dict()))
    assert [g['compile'] for g in opt.param_groups]==[True,False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_disabled_groups_never_call_compiler(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('disabled optimizer groups must never compile')
    monkeypatch.setattr(torch,'compile',unexpected)
    for name in ('rmsprop','adam','adamw','sgd'):
        p=torch.nn.Parameter(torch.ones(4,device='cuda'))
        opt=PrecisionOptimizer([dict(params=[p],compile=False)],name=name,options={})
        p.grad=torch.ones_like(p)
        opt.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('name', ['rmsprop', 'adam', 'adamw', 'sgd'])
def test_compiled_transforms_preserve_raw_gradient_and_update_order(name):
    p=torch.nn.Parameter(torch.ones(32,device='cuda'))
    q=torch.nn.Parameter(p.detach().clone())
    opt=PrecisionOptimizer([dict(params=[p],compile=True)],name=name,options={},dtype=torch.float64,
                           stochastic_rounding=False)
    ref=PrecisionOptimizer([dict(params=[q],compile=False)],name=name,options={},dtype=torch.float64,
                           stochastic_rounding=False)
    for optimizer in (opt,ref):
        optimizer.gradient_transform=lambda parameter,gradient: gradient.add_(.2)
        optimizer.update_transform=lambda parameter,update: update.copy_(parameter.double().lerp(update,.125))
    for _ in range(3):
        p.grad=torch.ones_like(p)
        q.grad=torch.ones_like(q)
        opt.step();ref.step()
        torch.testing.assert_close(p,q,rtol=0,atol=1e-7)
        torch.testing.assert_close(p.grad,torch.ones_like(p))
        for key,value in ref.state[q].items():
            torch.testing.assert_close(opt.state[p][key],value,rtol=1e-10,atol=1e-10)
    assert opt._compiled_native_step is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('name', ['adam', 'adamw', 'sgd'])
@pytest.mark.parametrize('model_dtype', [torch.bfloat16, torch.float32])
def test_compiled_native_fp64_writeback_is_stochastic(name, model_dtype):
    torch.manual_seed(42)
    p=torch.nn.Parameter(torch.ones(100_000,device='cuda',dtype=model_dtype))
    lr=torch.finfo(model_dtype).eps / 16
    opt=PrecisionOptimizer([dict(params=[p],compile=True,lr=lr)],name=name,options={},
                           dtype=torch.float64,stochastic_rounding=True)
    p.grad=-torch.ones_like(p)
    opt.step()
    expected=1 + lr / (1 + (1e-8 if name != 'sgd' else 0))
    assert p.double().mean().item()==pytest.approx(expected,abs=torch.finfo(model_dtype).eps*.005)
    assert bool((p>1).any()) and bool((p==1).any())
