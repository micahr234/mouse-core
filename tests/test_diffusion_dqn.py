"""Analytic flow targets, replay alignment, and the actual example head/loop."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from tensordict import TensorDict

from mouse_core import AdamW, polyak_update
from mouse_core.objectives import DiffusionDqnObjective, boundary_discount

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
NOTEBOOK = EXAMPLES / '24_train_offline_diffusion_dqn.ipynb'


def notebook_sources():
    return [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']]


def head_class():
    source = next(s for s in notebook_sources() if 'class DiffusionHead(' in s)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DiffusionHead')
    ns: dict[str, Any] = dict(torch=torch, nn=nn)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
    return ns['DiffusionHead']


def prepare(*, g=.6, steps=4, terminal_readout=False):
    objective = DiffusionDqnObjective(discount=boundary_discount(
        gamma_step=g, gamma_terminated=0., gamma_truncated=0.))
    data = dict(action=torch.tensor([float('nan'), 1., 0.]),
                reward=torch.tensor([0., 2., 3.]),
                terminated=torch.tensor([False, False, True]),
                truncated=torch.zeros(3, dtype=torch.bool))
    p = 3 if terminal_readout else 2
    times = torch.arange(steps).float() / steps
    samples = torch.arange(p * steps * 2).float().reshape(p, steps, 2).requires_grad_()
    velocities = (samples.detach() + 10).requires_grad_()
    scores = torch.tensor([[10., 0.], [0., 10.], [float('nan'), float('nan')]])[:p]
    if terminal_readout:
        with torch.no_grad():
            samples[-1] = float('nan')
            velocities[-1] = float('nan')
    noise = torch.tensor([.5, -.25, 1.])[:p].requires_grad_()
    result = objective.prepare(objective_data=data, sequence_ids=torch.zeros(3, dtype=torch.long),
        step_ids=torch.arange(p), backup_samples=samples, backup_value_predictions=velocities,
        backup_action_predictions=scores, times=times, noise=noise)
    return objective, result, (samples, velocities, noise, times)


@pytest.mark.parametrize('steps', [1, 4, 10])
@pytest.mark.parametrize('g', [0., .6, 1.])
@pytest.mark.parametrize('terminal_readout', [False, True])
def test_exact_targets_at_every_diffusion_step(g, steps, terminal_readout):
    obj, (x, target, actions, weight), (samples, velocities, noise, times) = prepare(
        g=g, steps=steps, terminal_readout=terminal_readout)
    correction = (1 - g*g)**.5 * noise.detach()[0]
    torch.testing.assert_close(x[0, :, 1], times*2 + g*samples.detach()[1, :, 1] + (1-times)*correction)
    torch.testing.assert_close(target[0], 2 + g*velocities.detach()[1, :, 1] - correction)
    # Other coordinates retain the delayed CURRENT state path as input context.
    torch.testing.assert_close(x[0, :, 0], samples.detach()[0, :, 0])
    torch.testing.assert_close(x[1, :, 1], samples.detach()[1, :, 1])
    # Terminal has no next readout, yet has standard reward-to-noise flow loss.
    torch.testing.assert_close(x[1, :, 0], times*3 + (1-times)*noise.detach()[1])
    torch.testing.assert_close(target[1], torch.full((steps,), 3.25))
    assert actions[:2].tolist() == [1, 0]
    assert weight[:2].tolist() == [1., 1.]
    assert not x.requires_grad and not target.requires_grad
    predictions = torch.zeros_like(x, requires_grad=True)
    losses, metrics = obj(predictions=predictions, targets=target, actions=actions, weight=weight)
    torch.testing.assert_close(metrics['per_diffusion_step'], target[:2].square().mean(0))
    losses['diffusion_dqn'].backward()
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[0, :, 0]) == 0
    assert torch.count_nonzero(predictions.grad[1, :, 1]) == 0
    assert torch.count_nonzero(predictions.grad[:2]) == 2*steps
    assert all(x.grad is None for x in (samples, velocities, noise))
    if terminal_readout:
        assert torch.count_nonzero(predictions.grad[2]) == 0


def test_inner_reset_carries_reward_discount_but_stops_at_next_decision():
    # Decision 0 -> inner terminal 1 -> automatic reset 2 -> outer terminal 3.
    # Readouts exist only at decisions 0 and 2, exactly as in example 2.
    obj = DiffusionDqnObjective(discount=lambda **_: torch.tensor([1., .5, .8, 0.]))
    data = dict(action=torch.tensor([float('nan'), 1., float('nan'), 0.]),
                reward=torch.tensor([0., 2., 3., 999.]))
    times = torch.tensor([0., .5])
    samples = torch.tensor([[[0., 0.], [0., 0.]], [[4., 40.], [6., 60.]]])
    velocities = torch.ones_like(samples)*7
    scores = torch.tensor([[0., 1.], [10., 0.]])
    x, target, actions, weight = obj.prepare(objective_data=data,
        sequence_ids=torch.zeros(4, dtype=torch.long), step_ids=torch.tensor([0, 2]),
        backup_samples=samples, backup_value_predictions=velocities,
        backup_action_predictions=scores, times=times, noise=torch.ones(2))
    # R=2+.5*3, G=.5*.8. Reward 999 is NOT in decision 0's target.
    c = (1-.4**2)**.5
    torch.testing.assert_close(x[0, :, 1], times*3.5 + .4*samples[1, :, 0] + (1-times)*c)
    torch.testing.assert_close(target[0], torch.full((2,), 3.5+.4*7-c))
    torch.testing.assert_close(target[1], torch.full((2,), 998.))
    assert actions.tolist() == [1, 0] and weight.tolist() == [1., 1.]


def test_packed_groups_and_duplicate_readouts():
    obj = DiffusionDqnObjective(discount=lambda **_: torch.tensor([1., 0., 1., 0.]))
    kwargs: dict[str, Any] = dict(objective_data=dict(action=torch.tensor([float('nan'), 0., float('nan'), 1.]),
                    reward=torch.tensor([0., 2., 0., 7.])),
        sequence_ids=torch.tensor([0, 0, 1, 1]), step_ids=torch.tensor([0, 0, 2]),
        backup_samples=torch.zeros(3, 2, 2), backup_value_predictions=torch.zeros(3, 2, 2),
        backup_action_predictions=torch.zeros(3, 2), times=torch.tensor([0., .5]), noise=torch.zeros(3))
    _, target, actions, weight = obj.prepare(**kwargs)
    torch.testing.assert_close(target, torch.tensor([[2., 2.], [2., 2.], [7., 7.]]))
    assert actions.tolist() == [0, 0, 1] and weight.tolist() == [1., 1., 1.]
    # A missing target cannot silently become a terminal or cross into group 1.
    obj.discount = lambda **_: torch.tensor([1., .9, 1., 0.])
    with pytest.raises(ValueError, match='cross-group'):
        obj.prepare(**kwargs)


def test_empty_readouts_have_differentiable_zero_loss():
    obj = DiffusionDqnObjective(discount=None)
    x, targets, actions, weight = obj.prepare(
        objective_data=dict(action=torch.full((2,), float('nan')), reward=torch.zeros(2)),
        sequence_ids=torch.zeros(2, dtype=torch.long), step_ids=torch.empty(0, dtype=torch.long),
        backup_samples=torch.empty(0, 3, 2), backup_value_predictions=torch.empty(0, 3, 2),
        backup_action_predictions=torch.empty(0, 2), times=torch.tensor([0., .3, .6]), noise=torch.empty(0))
    x.requires_grad_()
    losses, _ = obj(predictions=x, targets=targets, actions=actions, weight=weight)
    assert losses['diffusion_dqn'].item() == 0
    losses['diffusion_dqn'].backward()
    assert x.grad is not None


def test_example_preserves_encoder_tokenizer_and_data_pipeline():
    base = [''.join(c['source']) for c in json.loads((EXAMPLES/'02_train_offline_dqn.ipynb').read_text())['cells']]
    sources = notebook_sources()
    assert sources[3] == base[3]  # Dataset loading.
    assert sources[5] == base[5]  # Whole tokenizer, augmentation and loader cell.
    def encoder(source):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Encoder'))
    assert encoder(sources[7]) == encoder(base[7])
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
            assert cell['outputs'] == []


def test_head_conditions_on_final_features_and_integrates_its_velocities():
    torch.manual_seed(12)
    head = head_class()(hidden_dim=8, num_actions=3, width=12, num_steps=4, scale=.1)
    z = torch.randn(2, 8, requires_grad=True)
    noise = torch.tensor([[-.4, .8, .1], [.2, -.6, .9]])
    samples, velocities, endpoint = head.trajectory(z, initial_noise=noise)
    torch.testing.assert_close(samples[:, 0], noise)
    torch.testing.assert_close(samples[:, 1:], samples[:, :-1]+velocities[:, :-1]/4)
    torch.testing.assert_close(endpoint, samples[:, -1]+velocities[:, -1]/4)
    predictions = head(z, samples, times=head.times)
    torch.testing.assert_close(predictions, velocities)
    predictions.square().mean().backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert head.q_values(z.detach(), num_samples=5).shape == (2, 3)
    # Changing one action's noisy value changes other actions' velocities.
    changed = samples.clone()
    changed[:, :, 0] += 10
    assert not torch.allclose(head(z, changed, times=head.times)[:, :, 1:], predictions[:, :, 1:])


@pytest.mark.parametrize('heads', [1, 3])
def test_q_values_accepts_cached_encoder_feature_shape(heads):
    head = head_class()(hidden_dim=8, num_actions=3, width=12, num_steps=2, scale=.1)
    features = torch.randn(2, heads, 8)
    torch.manual_seed(13)
    values = head.q_values(features, num_samples=4)
    torch.manual_seed(13)
    flat_values = head.q_values(features.reshape(-1, 8), num_samples=4)
    assert values.shape == (2, heads, 3)
    torch.testing.assert_close(values.reshape(-1, 3), flat_values)


def test_head_sampling_draws_independent_noise_for_each_action():
    torch.manual_seed(7)
    head = head_class()(hidden_dim=8, num_actions=3, width=12, num_steps=2, scale=.1)
    initial_draws = []
    trajectory = head.trajectory

    def capture(z, *, initial_noise):
        initial_draws.append(initial_noise.clone())
        return trajectory(z, initial_noise=initial_noise)

    head.trajectory = capture
    head.q_values(torch.randn(2, 8), num_samples=4)
    assert initial_draws[0].shape == (8, 3)
    assert not torch.equal(initial_draws[0][:, 0], initial_draws[0][:, 1])
    assert not torch.equal(initial_draws[0][:, 1], initial_draws[0][:, 2])


def test_taken_action_loss_trains_cross_action_inputs_without_other_output_losses():
    torch.manual_seed(19)
    head = head_class()(hidden_dim=8, num_actions=3, width=12, num_steps=3, scale=.1)
    features = torch.randn(1, 8, requires_grad=True)
    samples = torch.randn(1, 3, 3, requires_grad=True)
    predictions = head(features, samples, times=head.times)
    predictions.retain_grad()
    objective = DiffusionDqnObjective(discount=None)
    kwargs = dict(targets=torch.ones(1, 3), actions=torch.tensor([1]), weight=torch.ones(1))
    losses, _ = objective(predictions=predictions, **kwargs)
    # Even invalid/unbounded unused outputs must not affect the selected loss.
    altered = predictions.detach().clone()
    altered[:, :, 0] = float('nan')
    altered[:, :, 2] = 1e9
    altered_losses, _ = objective(predictions=altered, **kwargs)
    torch.testing.assert_close(losses['diffusion_dqn'], altered_losses['diffusion_dqn'])
    losses['diffusion_dqn'].backward()
    assert predictions.grad is not None and samples.grad is not None
    assert torch.count_nonzero(predictions.grad[:, :, [0, 2]]) == 0
    assert torch.count_nonzero(predictions.grad[:, :, 1]) == 3
    assert torch.count_nonzero(samples.grad[:, :, [0, 2]]) == 6
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_actual_notebook_training_loop_on_cpu():
    torch.manual_seed(4)
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(4, 8)
            self.backbone = nn.Linear(8, 8)
        def forward(self, inputs):
            return {'z': self.backbone(self.embedding(inputs))}
        def parameter_groups(self, *, lr):
            return [dict(params=getattr(self, name).parameters(), lr=rate) for name, rate in lr.items()]
    encoder = TinyEncoder()
    head = head_class()(hidden_dim=8, num_actions=2, width=12, num_steps=4, scale=.1)
    modules = nn.ModuleDict(dict(encoder=encoder, q_head=head))
    target_encoder = deepcopy(encoder).requires_grad_(False)
    target_head = deepcopy(head).requires_grad_(False)
    data = TensorDict(dict(action=torch.tensor([float('nan'), 1., float('nan'), 0.]),
        reward=torch.tensor([0., 1., 0., 2.]), terminated=torch.tensor([False, False, False, True]),
        truncated=torch.zeros(4, dtype=torch.bool), sequence_ids=torch.zeros(4, dtype=torch.long),
        step_ids=torch.tensor([0, 2])), batch_size=[])
    inputs = torch.randn(2, 4)
    ns: dict[str, Any] = dict(torch=torch, AdamW=AdamW, polyak_update=polyak_update,
        DiffusionDqnObjective=DiffusionDqnObjective, boundary_discount=boundary_discount,
        device=torch.device('cpu'), encoder=encoder, q_head=head, modules=modules,
        delayed_encoder=target_encoder, backup_value_head=target_head,
        loader=SimpleNamespace(next_batch=lambda: (inputs, data)), RETURN_SAMPLES=3,
        LR_EMBEDDING=1e-3, LR_BACKBONE=1e-3, LR_HEADS=1e-3,
        POLYAK_TAU_EMBEDDING=.1, POLYAK_TAU_BACKBONE=.1, POLYAK_TAU_VALUE=.01)
    source = next(s for s in notebook_sources() if 'def run_train(' in s)
    exec(compile(source, str(NOTEBOOK), 'exec'), ns)
    before = {k: p.detach().clone() for k, p in modules.named_parameters()}
    loss, metrics = ns['run_train'](num_steps=1)
    assert torch.isfinite(loss) and metrics['per_diffusion_step'].shape == (4,)
    for name, p in modules.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
        assert not torch.equal(p, before[name]), name
    for source, target, prefix, tau in [(encoder.embedding, target_encoder.embedding, 'encoder.embedding', .1),
            (encoder.backbone, target_encoder.backbone, 'encoder.backbone', .1), (head, target_head, 'q_head', .01)]:
        for name, p in target.named_parameters():
            assert p.grad is None
            torch.testing.assert_close(p, before[f'{prefix}.{name}'].lerp(dict(source.named_parameters())[name], tau))


def test_real_notebook_encoder_feeds_diffusion_head():
    from tests.test_example_modules import _example, _tokens

    _, ns = _example(NOTEBOOK, DIFFUSION_STEPS=3, DIFFUSION_HIDDEN_DIM=12)
    with torch.no_grad():
        features = ns['encoder'].eval()(_tokens(0, 1))['z']
        head = ns['q_head'].eval()
        values = head.q_values(features, num_samples=2)
    assert values.shape == (2, 3) and torch.isfinite(values).all()
