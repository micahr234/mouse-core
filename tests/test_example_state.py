"""Notebook run state is explicit and cannot leak between function calls."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from tensordict import TensorDict
from mouse_core import MouseOptimizer
from mouse_core.objectives import SupervisedObjective

EXAMPLES = Path(__file__).parents[1] / 'examples'


def source(path):
    return '\n'.join(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                     if c['cell_type'] == 'code')


@pytest.mark.parametrize('path', sorted(EXAMPLES.glob('*.ipynb')), ids=lambda p: p.stem)
def test_examples_do_not_declare_global_state(path):
    tree = ast.parse(source(path))
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'run_train':
            required = [arg.arg for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                        if default is None]
            assert 'training_step' in required
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'run_train':
            assert 'training_step' in {kw.arg for kw in node.keywords}


def training_function():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
        def forward(self, inputs):
            return {'z': self.linear(inputs)}
    modules = nn.ModuleDict({'encoder': Encoder(), 'q_head': nn.Linear(2, 2)})
    schedules, batches = [], []
    def rate(step):
        schedules.append(step)
        return .01
    optimizer = MouseOptimizer(groups={'model': {'online': modules, 'lr': rate}}, optimizer='sgd')
    schedules.clear()
    def next_batch():
        batches.append(True)
        return torch.ones(2, 2), TensorDict({'sequence_ids': torch.zeros(2, dtype=torch.long),
            'step_ids': torch.arange(2), 'q_star': torch.ones(2, 2)}, batch_size=[])
    namespace = dict(torch=torch, modules=modules, encoder=modules['encoder'], q_head=modules['q_head'],
        loader=SimpleNamespace(next_batch=next_batch), optimizer=optimizer, device=torch.device('cpu'),
        objective=SupervisedObjective(loss_type='mse', target_temperature=None), training_step=999)
    tree = ast.parse(source(EXAMPLES / '05_train_offline_sv.ipynb'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_train')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<notebook>', 'exec'), namespace)
    return namespace, schedules, batches


def test_training_returns_step_and_resumes_schedule_without_touching_global():
    namespace, schedules, batches = training_function()
    run_train = namespace['run_train']
    loss, metrics, step = run_train(training_step=11, num_steps=2)
    assert torch.isfinite(loss) and step == 13
    _, _, step = run_train(training_step=step, num_steps=3)
    assert step == 16 and schedules == [11, 12, 13, 14, 15]
    assert len(batches) == 5 and namespace['training_step'] == 999
    # A new caller's explicit clock wins over any prior invocation.
    _, _, step = run_train(training_step=2, num_steps=1)
    assert step == 3 and schedules[-1] == 2
    assert namespace['training_step'] == 999


@pytest.mark.parametrize('step', [-1, True, 1.5, None])
def test_invalid_training_step_fails_before_consuming_data(step):
    namespace, schedules, batches = training_function()
    with pytest.raises(ValueError, match='nonnegative integer'):
        namespace['run_train'](training_step=step, num_steps=1)
    assert not batches and not schedules


def test_ppo_closes_local_environment_on_collection_error():
    closed = []
    env = SimpleNamespace(names=[], envs=[], close=lambda: closed.append(True))
    def fail():
        raise RuntimeError('collection failed')
    namespace = dict(torch=torch, REPEAT_COUNT=1, MAX_STEPS_PER_EPISODE=3,
        make_rollout_env=lambda **kw: env, modules=SimpleNamespace(eval=fail))
    tree = ast.parse(source(EXAMPLES / '07_train_online_ppo.ipynb'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_rollout')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<notebook>', 'exec'), namespace)
    with pytest.raises(RuntimeError, match='collection failed'):
        namespace['run_rollout'](env_steps=0, num_episodes=1, max_cache=4, stores=[], contexts=[])
    assert closed == [True]
    assert not {'env', 'logs', 'trackers'} & namespace.keys()
