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

from mouse_core import MouseOptimizer
from mouse_core.models import Model
from mouse_core.models.lora import LoRALinear, LoRAConfig

EXAMPLES = Path(__file__).parents[1] / 'examples'


def source(path):
    return '\n'.join(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                     if c['cell_type'] == 'code')


TRAINERS = [p for p in sorted(EXAMPLES.glob('*.ipynb')) if 'def run_train(' in source(p)]
DQN = [p for p in TRAINERS if 'polyak_tau' in source(p)]


def calls(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


@pytest.mark.parametrize('path', TRAINERS, ids=lambda p: p.stem)
def test_common_optimizer_and_loader_controls(path):
    tree = ast.parse(source(path))
    assignments = {t.id: n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}
    opt = assignments['optimizer'].value
    assert isinstance(opt, ast.Call) and isinstance(opt.func, ast.Name) and opt.func.id == 'MouseOptimizer'
    options = {k.arg: k.value for k in opt.keywords}
    settings = ast.literal_eval(options['optimizer'])
    assert settings == {'name': 'adamw', 'betas': (.9, .95), 'eps': 1e-8, 'weight_decay': 0.}
    assert 'online' not in options and 'target' not in options
    rates = {n.id for n in ast.walk(options['groups']) if isinstance(n, ast.Name) and n.id.startswith('LR_')}
    assert {'LR_EMBEDDING', 'LR_BACKBONE_LORA', 'LR_HEADS'} <= rates
    assert 'LR_BACKBONE' not in assignments and 'POLYAK_TAU_BACKBONE' not in assignments
    assert isinstance(options['groups'], ast.Dict)
    for settings in (n for n in ast.walk(options['groups']) if isinstance(n, ast.Dict)):
        values = {ast.literal_eval(k): v for k, v in zip(settings.keys, settings.values) if k is not None}
        if 'parameters' in values:
            assert ast.literal_eval(values['parameters']) == 'lora'
    for backbone in calls(tree, 'Backbone'):
        lora = next(k.value for k in backbone.keywords if k.arg == 'lora')
        assert isinstance(lora, ast.Call) and ast.unparse(lora.func) == 'LoRAConfig'
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
    """Execute the notebook optimizer with independent component rates."""
    tree = ast.parse(source(path))
    optimizer = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'optimizer' for t in n.targets))
    encoder = nn.ModuleDict(dict(embedding=nn.Linear(2, 2), backbone=nn.Linear(2, 2), reasoner=nn.Linear(2, 2)))
    # Only example 10 has the older latent reasoner.
    if not path.name.startswith('10_'):
        del encoder['reasoner']
    online = ReasoningModules('23' if path.name.startswith('23_') else '25')
    encoder["backbone"] = LoRALinear(encoder["backbone"], LoRAConfig(rank=1))
    q_head = nn.Linear(2, 2)
    q_heads = nn.ModuleDict({'n2': q_head})
    modules = nn.ModuleDict({'encoder': encoder, 'q_heads' if path.name.startswith('13_') else 'q_head':
                             q_heads if path.name.startswith('13_') else q_head})
    ns: dict[str, Any] = dict(encoder=encoder, delayed_encoder=deepcopy(encoder).requires_grad_(False),
              online=online, target=deepcopy(online).requires_grad_(False), modules=modules,
              q_head=q_head, backup_value_head=deepcopy(q_head).requires_grad_(False), q_heads=q_heads,
              backup_value_heads=deepcopy(q_heads).requires_grad_(False), transition=None,
              MouseOptimizer=MouseOptimizer, nn=nn,
              LR_EMBEDDING=0., LR_REASONING=0., LR_REASONING_INPUT=0., LR_BACKBONE_LORA=0., LR_HEADS=0., POLYAK_TAU_BACKBONE_LORA=.37)
    ns.update(zip(('POLYAK_TAU_EMBEDDING', 'POLYAK_TAU_REASONING', 'POLYAK_TAU_VALUE',
                   'POLYAK_TAU_REASONING_HEAD'), rates, strict=True))
    ns.update(POLYAK_TAU_REASONING=rates[1], POLYAK_TAU_REASONING_INPUT=rates[1])
    exec(compile(ast.Module(body=[optimizer], type_ignores=[]), str(path), 'exec'), ns)
    opt = ns['optimizer']
    assert any(name.endswith("backbone_lora") for name in opt.groups)
    assert not any(name.endswith("backbone") for name in opt.groups)
    seen = set()
    for name, settings in opt.groups.items():
        src = settings['online']
        dst = settings.get('target')
        if dst is None:
            continue
        assert id(dst) not in seen
        seen.add(id(dst))
        with torch.no_grad():
            for param in src.parameters(): param.fill_(4.)
            for param in dst.parameters(): param.fill_(-2.)
    opt.step(0)
    for name, settings in opt.groups.items():
        expected = (.37 if name.endswith('backbone_lora') else rates[0] if name in ('embedding', 'encoder.embedding') else
                    rates[3] if name.endswith('token_head') else
                    rates[2] if name.endswith('value_head') or name.startswith('q_head') else rates[1])
        dst = settings.get('target')
        if dst is not None:
            for param in dst.parameters():
                torch.testing.assert_close(param, torch.full_like(param, -2. + 6. * expected))
                assert param.grad is None
    assert len(seen) >= 3


class ReasoningModules(Model):
    def __init__(self, kind):
        super().__init__()
        self.embedding = nn.Linear(2, 2)
        self.backbone = LoRALinear(nn.Linear(2, 2), LoRAConfig(rank=1))
        self.value_head = nn.Linear(2, 2)
        self.reasoner = nn.Module()
        if kind == '23':
            self.reasoner.token_head = nn.Linear(2, 2)
        else:
            self.reasoner.norm = nn.Linear(2, 2)
            self.embedding = nn.Module()
            self.embedding.embeddings = nn.ModuleDict({'text': nn.Linear(2, 2)})
            self.reasoner.input_projection = nn.Linear(2, 2)

    def forward(self, x):
        return x


@pytest.mark.parametrize('kind', ['23', '25'])
def test_reasoning_optimizers_cover_parameters_once_at_component_rates(kind):
    path = next(EXAMPLES.glob(kind + '_*.ipynb'))
    tree = ast.parse(source(path))
    optimizer = next(n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'optimizer' for t in n.targets))
    model = ReasoningModules(kind)
    ns: dict[str, Any] = dict(online=model, target=deepcopy(model).requires_grad_(False), MouseOptimizer=MouseOptimizer, nn=nn,
        LR_BACKBONE_LORA=.004, POLYAK_TAU_BACKBONE_LORA=.5,
        LR_EMBEDDING=.001, LR_REASONING_INPUT=.002, LR_HEADS=.003, POLYAK_TAU_EMBEDDING=.1,
        POLYAK_TAU_REASONING_INPUT=.2, POLYAK_TAU_VALUE=.3, POLYAK_TAU_REASONING_HEAD=.4)
    exec(compile(ast.Module(body=[optimizer], type_ignores=[]), str(path), 'exec'), ns)
    actual = {}
    for group in ns['optimizer'].param_groups:
        for param in group['params']:
            assert id(param) not in actual
            actual[id(param)] = group['lr']
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert id(param) not in actual
            continue
        expected = .001 if name.startswith('embedding.') else .004 if name.startswith('backbone.') else .002 if name.startswith(('reasoner.norm.', 'reasoner.input_projection.')) else .003
        assert actual.pop(id(param)) == expected
    assert not actual
