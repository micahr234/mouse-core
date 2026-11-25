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

from mouse_core.models import Embedding, MultiEmbedding, ReasoningActionHeads
from mouse_core.models.lora import apply_lora, LoRAConfig
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


@pytest.mark.parametrize('reasoning_loss_mode', ['transition_bellman', 'final_action'])
@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('min_depth,max_depth', [(1, 1), (1, 3), (3, 4)])
@pytest.mark.parametrize('wrap', ['none', 'decision', 'sequence'])
def test_notebook_training_uses_its_selected_recurrence_and_target_depth(reasoning_loss_mode, residual, min_depth, max_depth, wrap, monkeypatch):
    torch.manual_seed(37)
    example = example_module("25")
    adaptive = reasoning_loss_mode == "transition_bellman"
    if not adaptive:
        min_depth = max_depth
    head = (ReasoningActionHeads(action_head=nn.Linear(8, 4),
                                 reasoning_control_head=nn.Linear(8, 2)) if adaptive else nn.Linear(8, 4))
    if adaptive:
        with torch.no_grad():
            head.reasoning_control_head.weight.zero_()
            head.reasoning_control_head.bias.copy_(torch.tensor([0., 12.]))
    kwargs = dict(embedding=MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8), 'reasoning': Embedding(vocab_size=1, hidden_dim=8)}),
                  backbone=CausalBackbone(), value_head=head, max_depth=max_depth)
    online = example.ReasoningDqn(input_mlp_output_scale=100., **kwargs, min_depth=min_depth,
        latent_residual_enabled=residual, latent_feedback_scope=wrap, reasoning_loss_mode=reasoning_loss_mode)
    apply_lora(model=online.backbone, config=LoRAConfig(rank=2))
    target = deepcopy(online).requires_grad_(False).eval()
    with torch.no_grad():
        target.backbone.linear.base.bias.add_(.35)
        if adaptive:
            target.value_head.action_head.bias.add_(.5)
            target.value_head.reasoning_control_head.bias.add_(.5)
        else:
            target.value_head.bias.add_(.5)
        if adaptive and min_depth > 1:
            target.value_head.reasoning_control_head.bias[0].add_(100.)
    assert online.reasoner.latent_residual_enabled is residual
    assert online.reasoner.adaptive_depth is adaptive
    assert online.reasoner.min_depth == (min_depth if adaptive else max_depth)
    with reasoning_loader(example) as reference:
        tokens, data = reference.next_batch()
    trace = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        action_values=online.value_head, backend='uncached', max_active_requests=2)
    assert trace.lengths.tolist() == [max_depth] * len(trace.lengths)
    with torch.no_grad():
        online_states = online.reasoner(trace, embedding=online.embedding, backbone=online.backbone)
        delayed_states = target.reasoner(trace, embedding=target.embedding, backbone=target.backbone)
        online_q = online.value_head(online_states)
        delayed_q = target.value_head(delayed_states)
    expected_backup_values = (delayed_q[:, 0].masked(
        torch.ones(len(trace.lengths), dtype=torch.long), min_depth=min_depth, max_depth=max_depth
    ).control_values.amax(-1, keepdim=True).expand(-1, 4) if adaptive else delayed_q[:, -1])
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
        if max_depth == 1 and wrap != "sequence":
            raise AssertionError('One-step notebook training must skip generation')
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
            online=online, target=target, LR_EMBEDDING=.001, LR_BACKBONE_LORA=.001, LR_REASONING_INPUT=.001, LR_HEADS=.001,
            objective=checked_objective, loader=loader, device=torch.device('cpu'),
            MAX_ACTIVE_REQUESTS=2,
            POLYAK_TAU_EMBEDDING=.1, POLYAK_TAU_BACKBONE_LORA=.2, POLYAK_TAU_REASONING_INPUT=.2, POLYAK_TAU_VALUE=.3,
            MAX_GRAD_NORM=1., THINK_REWARD=-.2, GAMMA_THINK=.4)
        configure_notebook_optimizer(example, EXAMPLES / "25_train_reasoning_dqn.ipynb")
        try:
            loss, metrics, next_step = example.run_train(training_step=0, num_steps=1)
        finally:
            handle.remove()
    weighted_errors, transition_weights = [], []
    for row, (length, step) in enumerate(zip(trace.lengths.tolist(), data['step_ids'].tolist(), strict=True)):
        action = int(data['action'][step + 1])
        if adaptive:
            for depth in range(length):
                weight = float(captured['backup_weight'][row])
                if depth < length - 1:
                    legal = [1] if depth + 2 < min_depth else ([0] if depth + 2 == max_depth else [0, 1])
                    target_value = -.2 + .4 * float(delayed_q.control_values[row, depth + 1, legal].max())
                    prediction = online_q.control_values[row, depth, 1]
                else:
                    target_value = float(delayed_q.action_values[row, depth].max())
                    prediction = online_q.control_values[row, depth, 0]
                weighted_errors.append(weight * (float(prediction) - target_value) ** 2)
                transition_weights.append(weight)
        prediction = online_q.action_values[row, length - 1, action] if adaptive else online_q[row, length - 1, action]
        target_value = float(captured['backup'][row]) + (0. if adaptive else -.2 * (length - 1))
        weight = float(captured['backup_weight'][row])
        weighted_errors.append(weight * (float(prediction) - target_value) ** 2)
        transition_weights.append(weight)
    assert float(loss) == pytest.approx(sum(weighted_errors) / sum(transition_weights), rel=2e-5)
    assert metrics['think_reward'] == pytest.approx(-.2 * (max_depth - 1))
    assert len(head_states) == 1
    gradient = head_states[0].grad
    assert gradient is not None
    if not adaptive:
        assert gradient[:, :-1].count_nonzero() == 0
    assert gradient[:, -1].abs().sum() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    for name, param in online.named_parameters():
        if not param.requires_grad:
            assert param.grad is None
            torch.testing.assert_close(param, original_weights[name], rtol=0, atol=0)
    for component in ('embedding.', 'backbone.', 'reasoner.norm.', 'reasoner.input_projection.', 'value_head.'):
        changed = any(not torch.equal(original_weights[name], param)
                      for name, param in online.named_parameters() if name.startswith(component))
        feedback_is_zero = wrap == "none" or (wrap == "decision" and max_depth == 1)
        expect_change = not (feedback_is_zero and component in ('reasoner.norm.', 'reasoner.input_projection.'))
        assert changed is expect_change, component


@pytest.mark.parametrize('reasoning_loss_mode,min_depth,max_depth,message', [
    ('final_action', 1, 3, 'equal minimum and maximum'),
    ('final_action', 3, 1, 'equal minimum and maximum'),
    ('unsupported', 1, 3, 'reasoning_loss_mode must be'),
])
def test_notebook_rejects_invalid_loss_and_adaptive_last_token_budget(reasoning_loss_mode, min_depth, max_depth, message):
    example = example_module('25')
    with pytest.raises(ValueError, match=message):
        example.ReasoningDqn(input_mlp_output_scale=100.,
            embedding=MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8), 'reasoning': Embedding(vocab_size=1, hidden_dim=8)}),
            backbone=CausalBackbone(), value_head=nn.Linear(8, 4),
            min_depth=min_depth, max_depth=max_depth,
            latent_residual_enabled=False, latent_feedback_scope="none", reasoning_loss_mode=reasoning_loss_mode,
        )
