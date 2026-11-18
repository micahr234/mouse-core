"""Prevent notebook optimization and target-update conventions from drifting."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from mouse_core import AdamW, polyak_update
from mouse_core.models import Model

EXAMPLES = Path(__file__).parents[1] / 'examples'


def source(path):
    return '\n'.join(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                     if c['cell_type'] == 'code')


TRAINERS = [p for p in sorted(EXAMPLES.glob('*.ipynb')) if 'def run_train(' in source(p)]
DQN = [p for p in TRAINERS if 'polyak_update(' in source(p)]


def calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


@pytest.mark.parametrize('path', TRAINERS, ids=lambda p: p.stem)
def test_common_optimizer_and_loader_controls(path):
    tree = ast.parse(source(path))
    assignments = {t.id: n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    opt = assignments['optimizer'].value
    assert isinstance(opt, ast.Call) and isinstance(opt.func, ast.Name) and opt.func.id == 'AdamW'
    options = {k.arg: k.value for k in opt.keywords}
    assert ast.literal_eval(options['betas']) == (.9, .95)
    assert ast.literal_eval(options['eps']) == 1e-8
    assert ast.literal_eval(options['weight_decay']) == 0.
    rates = {n.id for n in ast.walk(options['params']) if isinstance(n, ast.Name) and n.id.startswith('LR_')}
    assert {'LR_EMBEDDING', 'LR_BACKBONE', 'LR_HEADS'} <= rates
    for loader in calls(tree, 'DataLoader') + calls(tree, 'EpochDataLoader'):
        assert isinstance(loader.func, ast.Name)
        settings = {k.arg: k.value for k in loader.keywords}
        assert ast.unparse(settings['num_workers']) == 'NUM_WORKERS'
        assert ast.unparse(settings['transform']) == 'train_transform'
        if loader.func.id == 'DataLoader':
            assert ast.unparse(settings['prefetch']) == 'PREFETCH'
    assert 'TARGET_TAU' not in assignments and 'LEARNING_RATE' not in assignments


@pytest.mark.parametrize('path', DQN, ids=lambda p: p.stem)
@pytest.mark.parametrize('rates', [(0., .25, 1., .6), (.7, 0., .1, 1.)])
def test_each_notebook_updates_components_with_independent_rates(path, rates):
    """Execute the actual notebook update calls with visibly different rates."""
    tree = ast.parse(source(path))
    update_calls = calls(tree, 'polyak_update')
    assert update_calls
    encoder = SimpleNamespace(embedding=nn.Linear(2, 2), backbone=nn.Linear(2, 2), reasoner=nn.Linear(2, 2))
    online = SimpleNamespace(embedding=nn.Linear(2, 2), backbone=nn.Linear(2, 2), value_head=nn.Linear(2, 2),
        reasoner=SimpleNamespace(norm=nn.Linear(2, 2), stop_head=nn.Linear(2, 2), token_head=nn.Linear(2, 2)))
    q_head = nn.Linear(2, 2)
    ns = dict(encoder=encoder, delayed_encoder=deepcopy(encoder), online=online, target=deepcopy(online),
              q_head=q_head, backup_value_head=deepcopy(q_head), q_heads={'n2': q_head},
              backup_value_heads={'n2': deepcopy(q_head)}, key='n2', polyak_update=polyak_update)
    ns.update(zip(('POLYAK_TAU_EMBEDDING', 'POLYAK_TAU_BACKBONE', 'POLYAK_TAU_VALUE',
                   'POLYAK_TAU_REASONING_HEAD'), rates, strict=True))
    seen = set()
    for call in update_calls:
        kw = {k.arg: k.value for k in call.keywords}
        component = ast.unparse(kw['source'])
        expected = (rates[0] if component.endswith('.embedding') else
                    rates[3] if component.endswith(('.token_head', '.stop_head')) else
                    rates[2] if component.endswith('.value_head') or component.startswith('q_head') else rates[1])
        evaluate = lambda expr: eval(compile(ast.Expression(expr), str(path), 'eval'), ns)
        src, dst = evaluate(kw['source']), evaluate(kw['target'])
        # A whole-model update is deliberately unsupported: it hides component rates.
        assert isinstance(src, nn.Linear) and isinstance(dst, nn.Linear)
        assert id(dst) not in seen, 'A delayed component must update once per step'
        seen.add(id(dst))
        with torch.no_grad():
            for param in src.parameters():
                param.fill_(4.)
            for param in dst.parameters():
                param.fill_(-2.)
                param.requires_grad_(False)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(call)], type_ignores=[])), str(path), 'exec'), ns)
        for param in dst.parameters():
            torch.testing.assert_close(param, torch.full_like(param, -2. + 6. * expected))
            assert param.grad is None
    assert len(seen) >= 3


class ReasoningModules(Model):
    def __init__(self, kind):
        super().__init__()
        self.embedding = nn.Linear(2, 2)
        self.backbone = nn.Linear(2, 2)
        self.value_head = nn.Linear(2, 2)
        self.reasoner = nn.Module()
        if kind == '23':
            self.reasoner.token_head = nn.Linear(2, 2)
        else:
            self.reasoner.norm = nn.Linear(2, 2)
            self.reasoner.stop_head = nn.Linear(2, 2)

    def forward(self, x):
        return x


@pytest.mark.parametrize('kind', ['23', '25'])
def test_reasoning_optimizers_cover_parameters_once_at_component_rates(kind):
    path = next(EXAMPLES.glob(kind + '_*.ipynb'))
    tree = ast.parse(source(path))
    optimizer = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'optimizer' for t in n.targets))
    model = ReasoningModules(kind)
    ns: dict[str, Any] = dict(online=model, AdamW=AdamW, LR_EMBEDDING=.001, LR_BACKBONE=.002, LR_HEADS=.003)
    exec(compile(ast.Module(body=[optimizer], type_ignores=[]), str(path), 'exec'), ns)
    actual = {}
    for group in ns['optimizer'].param_groups:
        for param in group['params']:
            assert id(param) not in actual
            actual[id(param)] = group['lr']
    for name, param in model.named_parameters():
        expected = .001 if name.startswith('embedding.') else .002 if name.startswith(('backbone.', 'reasoner.norm.')) else .003
        assert actual.pop(id(param)) == expected
    assert not actual
