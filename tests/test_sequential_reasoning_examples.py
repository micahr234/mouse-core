"""Execute each sequential reasoning notebook's real CPU training path."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mouse_core.models import Embedding, MultiEmbedding
from mouse_core.objectives import DqnObjective, boundary_discount
from tests._reasoning_replay_fixture import configure_notebook_optimizer, reasoning_loader
from tests.test_residual_reasoning import CausalBackbone

EXAMPLES = Path(__file__).parents[1] / 'examples'


def example_module(number):
    path = next(EXAMPLES.glob(f'{number}_*.ipynb'))
    nodes, constants = [], {}
    for cell in json.loads(path.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for name in node.targets:
                    if isinstance(name, ast.Name) and name.id.isupper():
                        constants[name.id] = value
    namespace = constants
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize('number,residual', [('25', True), ('26', False), ('26', True), ('27', False), ('27', True)])
@pytest.mark.parametrize('min_steps,max_steps', [(1, 1), (1, 3), (3, 4)])
def test_notebook_training_uses_its_selected_recurrence_and_target_depth(number, residual, min_steps, max_steps, monkeypatch):
    torch.manual_seed(37)
    example = example_module(number)
    adaptive = number != '25'
    head = nn.Linear(8, 5 if adaptive else 4)
    if adaptive:
        with torch.no_grad():
            head.weight[-1].zero_()
            head.bias[-1] = 12.
    kwargs = dict(embedding=MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8)}),
                  backbone=CausalBackbone(), value_head=head, max_steps=max_steps)
    if adaptive:
        online = example.ValueReasoningDqn(**kwargs, min_steps=min_steps, residual_connections=residual)
    else:
        online = example.ResidualDqn(**kwargs)
    target = deepcopy(online).requires_grad_(False).eval()
    with torch.no_grad():
        target.backbone.linear.bias.add_(.35)
        target.value_head.bias.add_(.5)
        if adaptive and min_steps > 1:
            target.value_head.bias[:-1].add_(100.)
    assert online.reasoner.residual_connections is residual
    assert online.reasoner.adaptive_depth is adaptive
    assert online.reasoner.min_steps == (min_steps if adaptive else max_steps)
    with reasoning_loader(example) as reference:
        tokens, data = reference.next_batch()
    trace = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        action_values=online.value_head, backend='uncached', max_active_requests=2)
    assert trace.lengths.tolist() == [max_steps] * len(trace.lengths)
    with torch.no_grad():
        online_states = online.reasoner(trace, embedding=online.embedding, backbone=online.backbone)
        delayed_states = target.reasoner(trace, embedding=target.embedding, backbone=target.backbone)
        online_q = online.value_head(online_states)
        delayed_q = target.value_head(delayed_states)
    expected_backup_values = delayed_q[:, 0 if adaptive else -1].clone()
    if adaptive and min_steps > 1:
        expected_backup_values[:, :-1] = -torch.inf
    if adaptive and max_steps == 1:
        expected_backup_values = expected_backup_values[:, :-1]
    captured = {}
    objective = DqnObjective(reward=None, value=None, gate=None, temperature=0.,
        discount=boundary_discount(gamma_step=.65, gamma_terminated=0., gamma_truncated=0.))

    def checked_objective(**kwargs):
        torch.testing.assert_close(kwargs['backup_value_predictions'], expected_backup_values)
        torch.testing.assert_close(kwargs['backup_action_predictions'], expected_backup_values)
        result = objective(**kwargs)
        captured.update(result[1])
        return result

    def forbidden_generation(*args, **kwargs):
        raise AssertionError('The delayed model must replay the exact online history')

    monkeypatch.setattr(target.reasoner, 'generate', forbidden_generation)
    original_generation = online.reasoner.generate
    def cpu_generation(*args, **kwargs):
        assert kwargs.pop('backend') == 'continuous'
        return original_generation(*args, backend='uncached', **kwargs)
    monkeypatch.setattr(online.reasoner, 'generate', cpu_generation)
    original_weights = {name: param.detach().clone() for name, param in online.named_parameters()}
    head_states = []

    def record_states(_module, _args, output):
        if output.requires_grad:
            output.retain_grad()
            head_states.append(output)

    handle = online.reasoner.register_forward_hook(record_states)
    with reasoning_loader(example) as loader:
        example.run_train.__globals__.update(
            online=online, target=target, LR_EMBEDDING=.001, LR_BACKBONE=.001, LR_HEADS=.001,
            objective=checked_objective, loader=loader, device=torch.device('cpu'),
            MAX_ACTIVE_REQUESTS=2,
            POLYAK_TAU_EMBEDDING=.1, POLYAK_TAU_BACKBONE=.2, POLYAK_TAU_VALUE=.3,
            MAX_GRAD_NORM=1., REASONING_STEP_COST=.2, GAMMA_THINK=.4)
        configure_notebook_optimizer(example, next(EXAMPLES.glob(f'{number}_*.ipynb')))
        try:
            loss, metrics = example.run_train(num_steps=1)
        finally:
            handle.remove()
    weighted_errors, transition_weights = [], []
    for row, (length, step) in enumerate(zip(trace.lengths.tolist(), data['step_ids'].tolist(), strict=True)):
        action = int(data['action'][step + 1])
        for depth in range(length) if adaptive else [length - 1]:
            weight = float(captured['backup_weight'][row])
            if adaptive and depth < length - 1:
                following = delayed_q[row, depth + 1]
                if depth + 2 < min_steps:
                    following = following[-1:]
                elif depth + 1 == max_steps - 1:
                    following = following[:-1]
                target_value = -.2 + .4 * float(following.max())
                prediction = float(online_q[row, depth, -1])
            else:
                target_value = float(captured['backup'][row])
                if not adaptive:
                    target_value -= .2 * (length - 1)
                prediction = float(online_q[row, depth, action])
            weighted_errors.append(weight * (prediction - target_value) ** 2)
            transition_weights.append(weight)
    assert float(loss) == pytest.approx(sum(weighted_errors) / sum(transition_weights), rel=2e-5)
    assert metrics['reasoning_cost'] == pytest.approx(.2 * (max_steps - 1))
    assert len(head_states) == 1
    gradient = head_states[0].grad
    assert gradient is not None
    if not adaptive:
        assert gradient[:, :-1].count_nonzero() == 0
    assert gradient[:, -1].abs().sum() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    for component in ('embedding.', 'backbone.', 'reasoner.norm.', 'value_head.'):
        assert any(not torch.equal(original_weights[name], param)
                   for name, param in online.named_parameters() if name.startswith(component))
