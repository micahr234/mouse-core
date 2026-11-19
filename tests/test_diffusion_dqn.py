"""Analytic flow targets, replay alignment, and the actual example head/loop."""
from typing import Any
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

from mouse_core import MouseOptimizer
from mouse_core.models.mlp import SwiGLU
from mouse_core.objectives import DiffusionDqnObjective, boundary_discount

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
NOTEBOOK = EXAMPLES / '24_train_offline_diffusion_dqn.ipynb'


def notebook_sources():
    return [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']]


def head_class():
    source = next(s for s in notebook_sources() if 'class DiffusionHead(' in s)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'DiffusionHead')
    ns: dict[str, Any] = dict(torch=torch, nn=nn, SwiGLU=SwiGLU)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), ns)
    return ns['DiffusionHead']


def make_head(**settings):
    settings = dict(latent_frequencies=4, latent_min=0., latent_max=20.,
                    time_min=0., time_max=1., noise_mean=0., noise_std=1.) | settings
    return head_class()(**settings)


def prepare(*, g=1., steps=4, terminal_readout=False, noise_mean=0., noise_std=1.,
            noise_values=(.5, -.25, 1.)):
    objective = DiffusionDqnObjective(discount=boundary_discount(
        gamma_step=g, gamma_terminated=0., gamma_truncated=0.),
        noise_mean=noise_mean, noise_std=noise_std)
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
    noise = torch.tensor(noise_values)[:p].requires_grad_()
    result = objective.prepare(objective_data=data, sequence_ids=torch.zeros(3, dtype=torch.long),
        step_ids=torch.arange(p), backup_samples=samples, backup_value_predictions=velocities,
        backup_action_predictions=scores, times=times, noise=noise)
    return objective, result, (samples, velocities, noise, times)


@pytest.mark.parametrize('steps', [1, 4, 10])
@pytest.mark.parametrize('g', [0., .6, 1.])
@pytest.mark.parametrize('terminal_readout', [False, True])
@pytest.mark.parametrize('noise_mean,noise_std', [(0., 1.), (2.5, 2.9), (-1.2, 1.8), (2.5, 0.)])
def test_exact_targets_at_every_diffusion_step(g, steps, terminal_readout, noise_mean, noise_std):
    obj, (x, target, actions, weight), (samples, velocities, noise, times) = prepare(
        g=g, steps=steps, terminal_readout=terminal_readout, noise_mean=noise_mean, noise_std=noise_std)
    correction = (1 - g) * noise_mean + (1 - g*g)**.5 * noise_std * noise.detach()[0]
    torch.testing.assert_close(x[0, :, 1], times*2 + g*samples.detach()[1, :, 1] + (1-times)*correction)
    torch.testing.assert_close(target[0], 2 + g*velocities.detach()[1, :, 1] - correction)
    # Other coordinates retain the delayed CURRENT state path as input context.
    torch.testing.assert_close(x[0, :, 0], samples.detach()[0, :, 0])
    torch.testing.assert_close(x[1, :, 1], samples.detach()[1, :, 1])
    # Terminal has no next readout; its flow still starts at the configured Gaussian distribution.
    terminal_noise = noise_mean + noise_std * noise.detach()[1]
    torch.testing.assert_close(x[1, :, 0], times*3 + (1-times)*terminal_noise)
    torch.testing.assert_close(target[1], (3 - terminal_noise).expand(steps))
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


@pytest.mark.parametrize('noise_mean,noise_std', [(0., 1.), (2.5, 2.9)])
@pytest.mark.parametrize('first_discount,reset_discount', [(1., 1.), (.5, .8), (1., 0.), (0., 1.)])
def test_inner_reset_carries_reward_discount_but_stops_at_next_decision(
    noise_mean, noise_std, first_discount, reset_discount,
):
    # Decision 0 -> inner terminal 1 -> automatic reset 2 -> outer terminal 3.
    # Readouts exist only at decisions 0 and 2, exactly as in example 2.
    obj = DiffusionDqnObjective(discount=lambda **_: torch.tensor([1., first_discount, reset_discount, 0.]),
                                noise_mean=noise_mean, noise_std=noise_std)
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
    # Carry automatic-row rewards with the preceding discount.
    # Reward 999 belongs to the next decision and is excluded from this backup.
    reward = 2 + first_discount * 3
    discount = first_discount * reset_discount
    correction = (1 - discount) * noise_mean + (1 - discount**2)**.5 * noise_std
    torch.testing.assert_close(x[0, :, 1], times*reward + discount*samples[1, :, 0] + (1-times)*correction)
    torch.testing.assert_close(target[0], torch.full((2,), reward+discount*7-correction))
    torch.testing.assert_close(target[1], torch.full((2,), 999. - noise_mean - noise_std))
    assert actions.tolist() == [1, 0] and weight.tolist() == [1., 1.]


@pytest.mark.parametrize('discount', [-1., 1.1, float('nan'), float('inf')])
def test_gaussian_correction_rejects_invalid_discounts(discount):
    with pytest.raises(ValueError, match='discount'):
        prepare(g=discount)


@pytest.mark.parametrize('row', [0, 1, 2, 3])
@pytest.mark.parametrize('invalid_discount', [-.1, 1.1, float('nan')])
def test_every_raw_discount_is_valid_including_automatic_and_unused_rows(row, invalid_discount):
    discounts = torch.tensor([1., 0., 1., 0.])
    discounts[row] = invalid_discount
    obj = DiffusionDqnObjective(discount=lambda **_: discounts)
    # The automatic reset at row 2 cannot hide an invalid raw discount,
    # even when row 1's zero makes the effective backup discount zero.
    with pytest.raises(ValueError, match='discount'):
        obj.prepare(
            objective_data=dict(action=torch.tensor([float('nan'), 1., float('nan'), 0.]),
                                reward=torch.tensor([0., 2., 3., 4.])),
            sequence_ids=torch.zeros(4, dtype=torch.long), step_ids=torch.tensor([0, 2]),
            backup_samples=torch.zeros(2, 2, 2), backup_value_predictions=torch.zeros(2, 2, 2),
            backup_action_predictions=torch.zeros(2, 2), times=torch.tensor([0., .5]), noise=torch.zeros(2),
        )


@pytest.mark.parametrize('noise', [float('nan'), float('inf'), float('-inf')])
def test_gaussian_correction_rejects_nonfinite_draws(noise):
    with pytest.raises(ValueError, match='noise'):
        prepare(noise_values=(noise, .5, .5))


def test_terminal_correction_transforms_standard_gaussian_draws():
    _, (inputs, targets, _, _), (_, _, _, times) = prepare(
        g=0., noise_mean=-3., noise_std=7., noise_values=(-2., 2., .5),
    )
    torch.testing.assert_close(inputs[0, :, 1], 2*times - 17*(1-times))
    torch.testing.assert_close(inputs[1, :, 0], 3*times + 11*(1-times))
    torch.testing.assert_close(targets, torch.tensor([[19.]*4, [-8.]*4]))


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
    obj.discount = lambda **_: torch.tensor([1., 1., 1., 0.])
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
    def section_code(cells, title):
        index = next(i for i, source in enumerate(cells) if source.startswith(title + '\n'))
        return cells[index + 1]
    for title in ('## Load Data', '## Tokenizer', '## Replay'):
        assert section_code(sources, title) == section_code(base, title)
    def encoder(source):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Encoder'))
    assert encoder(section_code(sources, '## Build Encoder and Heads')) == encoder(section_code(base, '## Build Encoder and Heads'))
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), str(NOTEBOOK), 'exec')
            assert cell['outputs'] == []


@pytest.mark.parametrize('num_blocks', [0, 1, 2, 4])
def test_head_conditions_on_final_features_and_integrates_its_velocities(num_blocks):
    torch.manual_seed(12)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=4,
                     time_frequencies=4, num_blocks=num_blocks)
    assert len(head.blocks) == num_blocks
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
    assert head.q_values(z.detach()).shape == (2, 3)
    # Changing one action's noisy value changes other actions' velocities.
    changed = samples.clone()
    changed[:, :, 0] += 10
    assert not torch.allclose(head(z, changed, times=head.times)[:, :, 1:], predictions[:, :, 1:])


def test_head_uses_all_conditioning_inputs_with_gradients():
    torch.manual_seed(23)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3, time_frequencies=4)
    features = torch.randn(2, 8, requires_grad=True)
    samples = torch.randn(2, 3, 3, requires_grad=True)
    times = torch.tensor([0., .3, .8], requires_grad=True)
    velocities = head(features, samples, times=times)
    changed_features = features.detach().clone()
    changed_features[:, 0] += 2
    changed_samples = samples.detach().clone()
    changed_samples[:, :, 0] += 2
    assert not torch.allclose(head(changed_features, samples, times=times), velocities)
    assert not torch.allclose(head(features, changed_samples, times=times), velocities)
    assert not torch.allclose(head(features, samples, times=times + .05), velocities)
    velocities.square().mean().backward()
    for value in (features, samples, times):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('width,frequencies,time_min,time_max', [
    (1, 1, -1., 1.), (7, 4, 2., 6.), (12, 16, 0., 1.), (128, 64, -.5, .5),
])
def test_conditions_share_backbone_width_with_configurable_time_basis_and_range(
    width, frequencies, time_min, time_max,
):
    torch.manual_seed(29)
    head = make_head(hidden_dim=width, num_actions=3, num_steps=10,
                        time_frequencies=frequencies, time_min=time_min, time_max=time_max,
                        context_embedding_scale=.25, latent_embedding_scale=2., time_embedding_scale=3.)
    assert isinstance(head.time_embedding, nn.Linear)
    assert (head.time_embedding.in_features, head.time_embedding.out_features) == (2 * frequencies, width)
    assert head.time_embedding.bias is None
    assert head.width == width
    assert not hasattr(head, 'context')
    assert head.noisy_return.out_features == head.output.in_features == width
    assert head.noisy_return.weight.data_ptr() != head.time_embedding.weight.data_ptr()
    torch.testing.assert_close(head.time_bands, torch.arange(1, frequencies + 1) * torch.pi / frequencies)
    assert (head.time_min, head.time_max) == (time_min, time_max)
    assert type(head.time_min) is float and type(head.time_max) is float
    # Representation bounds do not alter the physical Euler-time grid.
    torch.testing.assert_close(head.times, torch.arange(10).float() / 10)
    # Constant features have nonzero RMS and zero centered standard deviation.
    features = torch.full((2, width), 3.)
    samples = torch.randn(2, 1, 3).expand(-1, 12, -1)
    times = torch.linspace(time_min, time_max, 12)
    projected_inputs, projected_outputs = {}, {}

    def capture(name):
        def hook(_module, inputs):
            projected_inputs[name] = inputs[0].detach().clone()
        return hook

    def capture_output(name):
        def hook(_module, _inputs, output):
            projected_outputs[name] = output.detach().clone()
        return hook

    hooks = [head.time_embedding.register_forward_pre_hook(capture('time')),
             head.blocks[0].register_forward_pre_hook(capture('combined')),
             head.norm.register_forward_hook(capture_output('context')),
             head.noisy_return.register_forward_hook(capture_output('returns')),
             head.time_embedding.register_forward_hook(capture_output('time'))]
    try:
        velocities = head(features, samples, times=times)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(projected_outputs['context'], head.norm(features))
    assert projected_outputs['context'].shape == (2, width)
    assert projected_outputs['returns'].shape == (2, 12, width)
    assert projected_outputs['time'].shape == (12, width)
    expected_sum = ((.25 / 3**.5) * projected_outputs['context'][:, None]
                    + (2. / 3**.5) * projected_outputs['returns']
                    + (3. / 3**.5) * projected_outputs['time'][None])
    torch.testing.assert_close(projected_inputs['combined'], expected_sum, rtol=0, atol=0)
    phases = ((times - time_min) / (time_max - time_min))[:, None] * head.time_bands
    assert torch.all((phases >= 0) & (phases <= torch.pi))
    torch.testing.assert_close(phases[0], torch.zeros(frequencies))
    torch.testing.assert_close(phases[-1].max(), torch.tensor(torch.pi))
    time_features = projected_inputs['time']
    assert time_features.shape == (12, 2 * frequencies)
    torch.testing.assert_close(time_features, torch.cat((phases.sin(), phases.cos()), dim=-1))
    torch.testing.assert_close(time_features.square().mean(-1), torch.full((12,), .5))
    assert not torch.allclose(time_features[0], time_features[-1])
    time_variation = projected_outputs['time'].amax(0) - projected_outputs['time'].amin(0)
    assert torch.all(time_variation > 1e-5)
    assert not torch.allclose(velocities[:, 0], velocities[:, -1])
    individual = torch.cat([
        head(features, samples[:, index:index + 1], times=times[index:index + 1])
        for index in range(len(times))
    ], dim=1)
    torch.testing.assert_close(individual, velocities)


@pytest.mark.parametrize('num_blocks', [0, 2])
@pytest.mark.parametrize('branch', ['context', 'latent', 'time'])
def test_zero_embedding_scale_disables_only_that_input_branch(branch, num_blocks):
    settings = dict(hidden_dim=8, num_actions=3, num_steps=3,
                    time_frequencies=4, num_blocks=num_blocks)
    torch.manual_seed(29)
    reference = make_head(**settings)
    torch.manual_seed(29)
    head = make_head(**settings, **{f'{branch}_embedding_scale': 0.})
    assert dict(head.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    for name, parameter in head.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)
    inputs = {
        'context': torch.randn(2, 8, requires_grad=True),
        'latent': torch.randn(2, 3, 3, requires_grad=True),
        'time': torch.tensor([0., .3, .8], requires_grad=True),
    }
    velocities = head(inputs['context'], inputs['latent'], times=inputs['time'])
    changed = {name: tensor.detach().clone() for name, tensor in inputs.items()}
    changed[branch] = changed[branch] + torch.randn_like(changed[branch])
    torch.testing.assert_close(
        velocities, head(changed['context'], changed['latent'], times=changed['time']),
        rtol=0, atol=0,
    )
    velocities.square().sum().backward()
    for name, tensor in inputs.items():
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert bool(tensor.grad.abs().sum() > 0) == (name != branch)
    for name, projection in [('latent', head.noisy_return), ('time', head.time_embedding)]:
        assert projection.weight.grad is not None
        assert bool(projection.weight.grad.abs().sum() > 0) == (name != branch)


@pytest.mark.parametrize('num_blocks', [0, 2])
def test_zero_embedding_scales_leave_only_the_velocity_bias(num_blocks):
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3,
                     time_frequencies=4, num_blocks=num_blocks,
                     context_embedding_scale=0., latent_embedding_scale=0., time_embedding_scale=0.)
    actual = head(torch.randn(2, 8), torch.randn(2, 3, 3), times=head.times)
    torch.testing.assert_close(actual, head.output.bias.expand_as(actual), rtol=0, atol=0)


@pytest.mark.parametrize('setting', ['context_embedding_scale', 'latent_embedding_scale', 'time_embedding_scale'])
@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), True, '1', None])
def test_embedding_scales_require_finite_nonnegative_numbers(setting, value):
    with pytest.raises(ValueError, match=setting):
        make_head(hidden_dim=8, num_actions=3, num_steps=3,
                  time_frequencies=4, **{setting: value})


@pytest.mark.parametrize('branch', ['time', 'latent'])
@pytest.mark.parametrize('frequencies', [0, -1, 1.5, True, '4', None])
def test_head_rejects_invalid_fourier_frequency_counts(branch, frequencies):
    settings = dict(time_frequencies=4) | {f'{branch}_frequencies': frequencies}
    with pytest.raises(ValueError, match=f'{branch}_frequencies'):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  **settings)


def test_velocity_readout_keeps_default_linear_initialization(monkeypatch):
    initial = {}
    reset_parameters = nn.Linear.reset_parameters

    def record_initialization(layer):
        reset_parameters(layer)
        initial[layer] = (layer.weight.detach().clone(),
                          None if layer.bias is None else layer.bias.detach().clone())

    monkeypatch.setattr(nn.Linear, 'reset_parameters', record_initialization)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3,
                     time_frequencies=4)
    weight, bias = initial[head.output]
    torch.testing.assert_close(head.output.weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(head.output.bias, bias, rtol=0, atol=0)


@pytest.mark.parametrize('scale', [0., .1, 1., 2.])
def test_velocity_initialization_scale_affects_only_final_linear_parameters(scale):
    settings = dict(hidden_dim=8, num_actions=3, num_steps=3,
                    time_frequencies=4)
    torch.manual_seed(43)
    reference = make_head(**settings)
    rng_state = torch.get_rng_state().clone()
    torch.manual_seed(43)
    head = make_head(**settings, velocity_output_initialization_scale=scale)
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name, parameter in head.state_dict().items():
        expected = reference.state_dict()[name]
        if name in ('output.weight', 'output.bias'):
            expected = scale * expected
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    features = torch.randn(2, 8)
    samples = torch.randn(2, 3, 3)
    torch.testing.assert_close(head(features, samples, times=head.times),
                               scale * reference(features, samples, times=reference.times))
    if scale == 0:
        initial_noise = torch.randn(2, 3)
        _, _, endpoint = head.trajectory(features, initial_noise=initial_noise)
        torch.testing.assert_close(endpoint, initial_noise, rtol=0, atol=0)


@pytest.mark.parametrize('scale', [0., .1, 2.])
def test_loading_velocity_weights_does_not_reapply_initialization_scale(scale):
    settings = dict(hidden_dim=8, num_actions=3, num_steps=3,
                    time_frequencies=4)
    reference = make_head(**settings)
    head = make_head(**settings, velocity_output_initialization_scale=scale)
    head.load_state_dict(reference.state_dict())
    features = torch.randn(2, 8)
    samples = torch.randn(2, 3, 3)
    expected = reference(features, samples, times=reference.times)
    actual = head(features, samples, times=head.times)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual.square().mean(), tuple(head.parameters()))
    expected_grads = torch.autograd.grad(expected.square().mean(), tuple(reference.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_grads[-1].abs().sum() > 0


@pytest.mark.parametrize('scale', [-.1, float('inf'), float('-inf'), float('nan'), True, '1', None])
def test_velocity_initialization_scale_requires_finite_nonnegative_number(scale):
    with pytest.raises(ValueError, match='velocity_output_initialization_scale'):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  time_frequencies=4, velocity_output_initialization_scale=scale)


@pytest.mark.parametrize('name,value', [
    ('noise_mean', float('inf')), ('noise_mean', float('-inf')), ('noise_mean', float('nan')),
    ('noise_mean', True), ('noise_mean', '1'), ('noise_mean', None),
    ('noise_std', float('-inf')), ('noise_std', float('inf')),
    ('noise_std', float('nan')), ('noise_std', True), ('noise_std', '1'), ('noise_std', None),
])
def test_gaussian_requires_finite_settings(name, value):
    with pytest.raises(ValueError, match=name):
        DiffusionDqnObjective(discount=None, **{name: value})
    with pytest.raises(ValueError, match=name):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  time_frequencies=4, **{name: value})


@pytest.mark.parametrize('noise_mean,noise_std', [(1., -.1), (-1., -2.)])
def test_gaussian_rejects_negative_std(noise_mean, noise_std):
    with pytest.raises(ValueError, match='noise_std'):
        DiffusionDqnObjective(discount=None, noise_mean=noise_mean, noise_std=noise_std)
    with pytest.raises(ValueError, match='noise_std'):
        make_head(hidden_dim=8, num_actions=3, num_steps=2, time_frequencies=4,
                  noise_mean=noise_mean, noise_std=noise_std)


@pytest.mark.parametrize('obsolete', ['noise_min', 'noise_max'])
def test_gaussian_api_rejects_obsolete_uniform_parameters(obsolete):
    kwargs: dict[str, Any] = {obsolete: 1.}
    with pytest.raises(TypeError, match=obsolete):
        DiffusionDqnObjective(discount=None, **kwargs)
    with pytest.raises(TypeError, match=obsolete):
        make_head(hidden_dim=8, num_actions=3, num_steps=2, time_frequencies=4, **kwargs)


@pytest.mark.parametrize('noise_mean,noise_std', [(0., 1.), (2.5, 2.9), (-1., 2.), (2.5, 0.)])
def test_gaussian_changes_start_samples_without_adding_parameters(noise_mean, noise_std):
    settings = dict(hidden_dim=8, num_actions=3, num_steps=3,
                    time_frequencies=4, velocity_output_initialization_scale=0.)
    torch.manual_seed(43)
    reference = make_head(**settings)
    torch.manual_seed(43)
    head = make_head(**settings, noise_mean=noise_mean, noise_std=noise_std)
    assert type(head.noise_mean) is float and type(head.noise_std) is float
    assert dict(head.named_parameters()).keys() == dict(reference.named_parameters()).keys()
    for name, parameter in head.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)
    features = torch.ones(2, 8)
    for count in (1, 3):
        raw_noise = torch.randn(2 * count, 3, generator=torch.Generator().manual_seed(17))
        initial = noise_mean + noise_std * raw_noise
        actual = head.q_values(features, action_samples=count, generator=torch.Generator().manual_seed(17))
        torch.testing.assert_close(actual, initial.reshape(2, count, 3).mean(1), rtol=0, atol=0)
        # Explicit starting samples must not receive the affine transform again.
        _, _, endpoint = head.trajectory(features.repeat_interleave(count, dim=0), initial_noise=initial)
        torch.testing.assert_close(endpoint, initial, rtol=0, atol=0)


def test_gaussian_settings_are_independent_of_fourier_encoding_bounds():
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3, time_frequencies=4,
                     noise_mean=-2., noise_std=1., latent_min=0., latent_max=20.,
                     velocity_output_initialization_scale=0.)
    expected = -2. + torch.randn(16, 3, generator=torch.Generator().manual_seed(17))
    actual = head.q_values(torch.ones(16, 8), generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual < head.latent_min).any()


@pytest.mark.parametrize('num_blocks', [0, 1, 2, 4])
def test_feedforward_blocks_preserve_skip_path_with_zero_residual_branches(num_blocks):
    torch.manual_seed(31)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3,
                     time_frequencies=4, num_blocks=num_blocks)
    assert len(head.blocks) == num_blocks
    with torch.no_grad():
        for block in head.blocks:
            assert isinstance(block[0], nn.RMSNorm) and block[0].eps == 1e-6
            assert isinstance(block[2], SwiGLU)
            assert isinstance(block[-1], nn.Linear)
            assert (block[1].in_features, block[1].out_features) == (8, 48)
            assert (block[-1].in_features, block[-1].out_features) == (24, 8)
            assert block[1].bias is None and block[-1].bias is None
            block[-1].weight.zero_()
    branches = {}

    def capture(name):
        def hook(_module, _inputs, output):
            branches[name] = output.detach().clone()
        return hook

    hooks = [head.norm.register_forward_hook(capture('context')),
             head.noisy_return.register_forward_hook(capture('returns')),
             head.time_embedding.register_forward_hook(capture('time')),
             head.output.register_forward_pre_hook(
                 lambda _module, inputs: branches.update(before_output=inputs[0].detach().clone()))]
    try:
        velocities = head(torch.randn(2, 8), torch.randn(2, 3, 3), times=head.times)
    finally:
        for hook in hooks:
            hook.remove()
    # Zero residual branches preserve the embedding sum until the final linear layer.
    combined = ((1 / 3**.5) * branches['context'][:, None]
                + (1 / 3**.5) * branches['returns']
                + (1 / 3**.5) * branches['time'][None])
    torch.testing.assert_close(branches['before_output'], combined, rtol=0, atol=0)
    torch.testing.assert_close(velocities, head.output(combined))


@pytest.mark.parametrize('num_blocks', [-1, True, 1.5, '2', None])
def test_block_count_requires_nonnegative_integer(num_blocks):
    with pytest.raises(ValueError, match='num_blocks'):
        make_head(hidden_dim=8, num_actions=3, num_steps=3,
                  time_frequencies=4, num_blocks=num_blocks)


@pytest.mark.parametrize('num_blocks', [0, 2])
def test_zero_std_is_deterministic_for_nonzero_velocity(num_blocks):
    torch.manual_seed(23)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=4,
                     time_frequencies=4, num_blocks=num_blocks,
                     noise_mean=2.5, noise_std=0.)
    features = torch.randn(2, 8)
    initial = torch.full((2, 3), 2.5)
    samples, velocities, endpoint = head.trajectory(features, initial_noise=initial)
    assert velocities.abs().sum() > 0
    torch.testing.assert_close(samples[:, 0], initial)
    for seed, count in ((3, 1), (19, 3)):
        values = head.q_values(features, generator=torch.Generator().manual_seed(seed), action_samples=count)
        torch.testing.assert_close(values, endpoint)


@pytest.mark.parametrize('frequencies,latent_min,latent_max', [(1, -2., 2.), (4, 0., 20.), (7, 5., 8.)])
def test_latent_fourier_features_use_shared_bounds_and_their_full_period(frequencies, latent_min, latent_max):
    torch.manual_seed(37)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2,
                     time_frequencies=4, latent_frequencies=frequencies,
                     latent_min=latent_min, latent_max=latent_max)
    assert not hasattr(head, 'return_norm')
    assert head.noisy_return.in_features == 2 * frequencies * 3
    assert (head.latent_min, head.latent_max) == (latent_min, latent_max)
    assert type(head.latent_min) is float and type(head.latent_max) is float
    assert not head.return_bands.requires_grad
    assert not {'latent_min', 'latent_max'} & dict(head.named_parameters()).keys()
    torch.testing.assert_close(head.return_bands, torch.pi * torch.arange(1, frequencies + 1) / frequencies)
    features = torch.randn(1, 8).expand(4, -1)
    interval = latent_max - latent_min
    samples = (latent_min + interval * torch.tensor([0., .25, .5, 1.]))[:, None, None] * torch.ones(1, 2, 3)
    encoded = []
    hook = head.noisy_return.register_forward_pre_hook(
        lambda _module, inputs: encoded.append(inputs[0].detach().clone()))
    try:
        values = head(features, samples, times=head.times)
        # Repeat just the middle action by the common fundamental period.
        shifted = samples.clone()
        shifted[..., 1] += 2 * frequencies * interval
        shifted_values = head(features, shifted, times=head.times)
    finally:
        hook.remove()
    raw = encoded[0].reshape(4, 2, 3, 2 * frequencies)
    phases = ((samples - latent_min) / interval)[..., None] * (
        torch.pi * torch.arange(1, frequencies + 1) / frequencies)
    torch.testing.assert_close(raw, torch.cat((phases.sin(), phases.cos()), dim=-1))
    torch.testing.assert_close(raw.square().mean(-1), torch.full((4, 2, 3), .5))
    torch.testing.assert_close(raw[:, :, 0], raw[:, :, 1])
    torch.testing.assert_close(raw[:, :, 0], raw[:, :, 2])
    for index in range(3):
        assert not torch.allclose(encoded[0][index], encoded[0][index + 1])
        assert not torch.allclose(values[index], values[index + 1])
    # The basis is injective over the configured interval, but eventually repeats.
    torch.testing.assert_close(encoded[0], encoded[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values, shifted_values, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('setting', ['time_min', 'time_max', 'latent_min', 'latent_max'])
@pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan'), True, '0', None, []])
def test_fourier_bounds_require_finite_numbers(setting, value):
    with pytest.raises(ValueError, match=setting):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  time_frequencies=4, **{setting: value})


@pytest.mark.parametrize('branch', ['time', 'latent'])
@pytest.mark.parametrize('lower,upper', [(0., 0.), (2., 1.), (-1., -2.), (-1e308, 1e308)])
def test_fourier_intervals_require_positive_finite_width(branch, lower, upper):
    with pytest.raises(ValueError, match=f'{branch}_max - {branch}_min'):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  time_frequencies=4, **{f'{branch}_min': lower, f'{branch}_max': upper})


@pytest.mark.parametrize('setting', ['time_min', 'time_max', 'latent_min', 'latent_max',
                                   'time_frequencies', 'latent_frequencies'])
def test_fourier_bounds_and_counts_are_required_constructor_arguments(setting):
    settings = dict(hidden_dim=8, num_actions=3, num_steps=2,
                    time_frequencies=4, time_min=0., time_max=1.,
                    latent_frequencies=4, latent_min=0., latent_max=20.)
    del settings[setting]
    with pytest.raises(TypeError, match=setting):
        head_class()(**settings)


@pytest.mark.parametrize('setting', ['time_scale', 'return_scale', 'return_scales',
                                   'return_frequencies', 'return_min', 'return_max', 'return_embedding_scale'])
def test_obsolete_fourier_arguments_are_rejected(setting):
    with pytest.raises(TypeError, match=setting):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  time_frequencies=4, **{setting: .1})


def test_out_of_range_returns_and_times_are_not_clamped_and_keep_gradients():
    torch.manual_seed(38)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2, time_frequencies=4)
    features = torch.randn(2, 8)
    samples = torch.tensor([[[-5., 25., 35.], [-10., 40., 60.]]]).expand(2, -1, -1).clone().requires_grad_()
    times = torch.tensor([-.5, 1.5], requires_grad=True)
    values = head(features, samples, times=times)
    clamped = head(features, samples.clamp(0., 20.), times=times.clamp(0., 1.))
    assert not torch.allclose(values, clamped)
    values.square().sum().backward()
    for tensor in (samples, times):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert torch.all(tensor.grad != 0)


def test_zero_return_values_have_finite_nonzero_gradients():
    torch.manual_seed(39)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2,
                     time_frequencies=4)
    samples = torch.zeros(2, 2, 3, requires_grad=True)
    values = head(torch.randn(2, 8), samples, times=head.times)
    values.square().mean().backward()
    assert samples.grad is not None and torch.isfinite(samples.grad).all()
    assert torch.all(samples.grad != 0)


@pytest.mark.parametrize('latent_frequencies,latent_min,latent_max', [(4, -10., 10.), (64, 0., 20.)])
def test_all_three_conditioning_branches_start_with_expected_unit_rms(latent_frequencies, latent_min, latent_max):
    # Context is normalized; return and time projections have expected unit RMS.
    torch.manual_seed(43)
    branch_mean_squares: dict[str, list[torch.Tensor]] = {name: [] for name in ('context', 'returns', 'time')}
    features = torch.randn(32, 128)
    samples = torch.randn(32, 17, 3)
    times = torch.linspace(0, 1, 17)

    def capture(name):
        def hook(_module, _inputs, output):
            branch_mean_squares[name].append(output.detach().square().mean())
        return hook

    for _ in range(16):
        head = make_head(hidden_dim=128, num_actions=3, num_steps=10,
                         time_frequencies=64,
                         latent_frequencies=latent_frequencies, latent_min=latent_min, latent_max=latent_max)
        assert head.norm.eps == 1e-5 and not head.norm.elementwise_affine
        assert head.norm.normalized_shape == (128,)
        assert head.norm.weight is None
        hooks = [head.norm.register_forward_hook(capture('context')),
                 head.noisy_return.register_forward_hook(capture('returns')),
                 head.time_embedding.register_forward_hook(capture('time'))]
        try:
            with torch.no_grad():
                head(features, samples, times=times)
        finally:
            for hook in hooks:
                hook.remove()
    for mean_squares in branch_mean_squares.values():
        rms = torch.stack(mean_squares).mean().sqrt()
        torch.testing.assert_close(rms, torch.tensor(1.), atol=.08, rtol=0)


def test_context_normalization_enters_backbone_width_sum_without_projection():
    torch.manual_seed(47)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3,
                     time_frequencies=4)
    assert not head.norm.elementwise_affine and head.norm.weight is None
    assert list(head.norm.parameters()) == []
    assert not hasattr(head, 'context')
    assert all(block[0].elementwise_affine and block[0].weight is not None for block in head.blocks)
    features = torch.randn(2, 8)
    samples = torch.randn(2, 3, 3)
    combined_inputs, branch_outputs = {}, {}

    def capture(name):
        def hook(_module, _inputs, output):
            branch_outputs[name] = output.detach().clone()
        return hook

    hooks = [head.blocks[0].register_forward_pre_hook(
                 lambda _module, inputs: combined_inputs.update(hidden=inputs[0].detach().clone())),
             head.norm.register_forward_hook(capture('context')),
             head.noisy_return.register_forward_hook(capture('returns')),
             head.time_embedding.register_forward_hook(capture('time'))]
    try:
        head(features, samples, times=head.times)
        torch.testing.assert_close(branch_outputs['context'], head.norm(features))
        expected = ((1 / 3**.5) * head.norm(features)[:, None]
                    + (1 / 3**.5) * branch_outputs['returns']
                    + (1 / 3**.5) * branch_outputs['time'][None])
        torch.testing.assert_close(combined_inputs['hidden'], expected)
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize('hidden_dim', [7, 128, 1024])
@pytest.mark.parametrize('context_scale', [.5, 3**.5])
def test_context_only_flow_matches_direct_linear_outputs_and_gradients(hidden_dim, context_scale):
    torch.manual_seed(51)
    head = make_head(hidden_dim=hidden_dim, num_actions=4, num_steps=1,
                     time_frequencies=4, num_blocks=0,
                     noise_mean=0., noise_std=0., context_embedding_scale=context_scale,
                     latent_embedding_scale=0., time_embedding_scale=0.)
    features = torch.randn(3, hidden_dim, requires_grad=True)
    samples = torch.randn(3, 1, 4)
    before_output = []
    hook = head.output.register_forward_pre_hook(
        lambda _module, inputs: before_output.append(inputs[0].detach().clone()))
    try:
        actual = head(features, samples, times=head.times)[:, 0]
    finally:
        hook.remove()
    # At sqrt(3), compare directly against the ordinary normalized linear head.
    hidden = head.norm(features)
    if context_scale != 3**.5:
        hidden = (context_scale / 3**.5) * hidden
    torch.testing.assert_close(before_output[0][:, 0], hidden, rtol=0, atol=0)
    expected = nn.functional.linear(hidden,
                                    head.output.weight, head.output.bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(head.q_values(features), expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    parameters = (features, head.output.weight, head.output.bias)
    actual_gradients = torch.autograd.grad(actual, parameters, upstream, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters, upstream)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        assert actual_gradient.abs().sum() > 0


def test_head_time_embedding_and_swiglu_computation_stays_float32_inside_autocast():
    torch.manual_seed(41)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3, time_frequencies=4)
    features = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    samples = torch.randn(2, 3, 3, dtype=torch.bfloat16, requires_grad=True)
    parameters = (*head.parameters(), features, samples)
    reference = head(features, samples, times=head.times)
    reference_grads = torch.autograd.grad(reference.square().mean(), parameters)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = head(features, samples, times=head.times)
    grads = torch.autograd.grad(output.square().mean(), parameters)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for actual, expected in zip(grads, reference_grads, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('heads', [1, 3])
@pytest.mark.parametrize('action_samples', [1, 3])
def test_q_values_accepts_cached_encoder_feature_shape(heads, action_samples):
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2,
                     time_frequencies=4, action_samples=action_samples)
    features = torch.randn(2, heads, 8)
    torch.manual_seed(13)
    values = head.q_values(features)
    torch.manual_seed(13)
    flat_values = head.q_values(features.reshape(-1, 8))
    assert values.shape == (2, heads, 3)
    torch.testing.assert_close(values.reshape(-1, 3), flat_values)


def test_q_values_uses_one_full_trajectory_with_independent_action_noise():
    torch.manual_seed(7)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2, time_frequencies=4)
    calls = []
    trajectory = head.trajectory

    def capture(z, *, initial_noise):
        result = trajectory(z, initial_noise=initial_noise)
        calls.append((initial_noise.clone(), result))
        return result

    head.trajectory = capture
    features = torch.randn(2, 8)
    values = head.q_values(features)
    assert len(calls) == 1
    noise, (samples, velocities, endpoint) = calls[0]
    assert noise.shape == (2, 3)
    assert not torch.equal(noise[:, 0], noise[:, 1])
    assert not torch.equal(noise[:, 1], noise[:, 2])
    assert samples.shape == velocities.shape == (2, head.num_steps, 3)
    torch.testing.assert_close(values, endpoint, rtol=0, atol=0)
    head.q_values(features)
    assert len(calls) == 2 and not torch.equal(calls[1][0], noise)


@pytest.mark.parametrize('action_samples', [0, -1, True, 1.5, '3', None])
def test_action_samples_requires_positive_integer(action_samples):
    with pytest.raises(ValueError, match='action_samples must be a positive integer'):
        make_head(hidden_dim=8, num_actions=3, num_steps=2,
                  time_frequencies=4, action_samples=action_samples)


def test_action_selection_averages_endpoints_before_argmax():
    head = make_head(hidden_dim=8, num_actions=2, num_steps=2,
                     time_frequencies=4, action_samples=3)
    features = torch.randn(2, 8)
    # Mean chooses 1 even though voting and the single largest sample choose 0.
    endpoints = torch.tensor([[9., 8.], [1., 0.], [0., 8.],
                              [8., 9.], [0., 1.], [8., 0.]])
    calls = []

    def trajectory(z, *, initial_noise):
        calls.append((z, initial_noise))
        return None, None, endpoints

    head.trajectory = trajectory
    scores = head.q_values(features)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], features.repeat_interleave(3, dim=0))
    assert calls[0][1].shape == (6, 2)
    torch.testing.assert_close(scores, endpoints.reshape(2, 3, 2).mean(1))
    assert scores.argmax(-1).tolist() == [1, 0]
    assert endpoints.reshape(2, 3, 2).argmax(-1).mode(-1).values.tolist() == [0, 1]
    assert endpoints.reshape(2, 3, 2).amax(1).argmax(-1).tolist() == [0, 1]


def test_action_samples_integrate_independent_noise_before_averaging():
    torch.manual_seed(42)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=4,
                     time_frequencies=4, action_samples=3)
    features = torch.randn(2, 8)
    noise = torch.randn(6, 3, generator=torch.Generator().manual_seed(11))
    _, _, endpoints = head.trajectory(features.repeat_interleave(3, dim=0), initial_noise=noise)
    global_rng = torch.get_rng_state().clone()
    scores = head.q_values(features, generator=torch.Generator().manual_seed(11))
    torch.testing.assert_close(scores, endpoints.reshape(2, 3, 3).mean(1), rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), global_rng)
    _, _, wrong = head.trajectory(features, initial_noise=noise.reshape(2, 3, 3).mean(1))
    assert not torch.allclose(scores, wrong)
    assert head.q_values(features[:0]).shape == (0, 3)


@pytest.mark.parametrize('action_samples', [0, -1, True, 1.5, '3'])
def test_q_values_rejects_invalid_sample_override(action_samples):
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2,
                     time_frequencies=4)
    with pytest.raises(ValueError, match='action_samples must be a positive integer'):
        head.q_values(torch.zeros(2, 8), action_samples=action_samples)


def test_online_and_evaluation_sample_overrides_leave_training_count_unchanged():
    head = make_head(hidden_dim=8, num_actions=3, num_steps=2,
                     time_frequencies=4, action_samples=3)
    features = torch.randn(2, 8)
    trajectory = head.trajectory
    calls = []

    def capture(z, *, initial_noise):
        result = trajectory(z, initial_noise=initial_noise)
        calls.append((z, result[-1]))
        return result

    head.trajectory = capture
    for override, expected in [(2, 2), (5, 5), (1, 1), (None, 3)]:
        scores = head.q_values(features, action_samples=override)
        z, endpoints = calls[-1]
        torch.testing.assert_close(z, features.repeat_interleave(expected, dim=0))
        torch.testing.assert_close(scores, endpoints.reshape(2, expected, 3).mean(1))
        assert head.action_samples == 3


def test_taken_action_loss_trains_cross_action_inputs_without_other_output_losses():
    torch.manual_seed(19)
    head = make_head(hidden_dim=8, num_actions=3, num_steps=3, time_frequencies=4)
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


def notebook_training_namespace(*, action_samples=1, noise_mean=0., noise_std=1., num_blocks=2,
                                head_tau=.01):
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
    head = make_head(hidden_dim=8, num_actions=2, num_steps=4,
                     time_frequencies=4, action_samples=action_samples, num_blocks=num_blocks,
                     noise_mean=noise_mean, noise_std=noise_std)
    modules = nn.ModuleDict(dict(encoder=encoder, q_head=head))
    target_encoder = deepcopy(encoder).requires_grad_(False)
    target_head = deepcopy(head).requires_grad_(False)
    data = TensorDict(dict(action=torch.tensor([float('nan'), 1., float('nan'), 0.]),
        reward=torch.tensor([0., 1., 0., 2.]), terminated=torch.tensor([False, False, False, True]),
        truncated=torch.zeros(4, dtype=torch.bool), sequence_ids=torch.zeros(4, dtype=torch.long),
        step_ids=torch.tensor([0, 2])), batch_size=[])
    inputs = torch.randn(2, 4)
    ns: dict[str, Any] = dict(torch=torch, MouseOptimizer=MouseOptimizer,
        DiffusionDqnObjective=DiffusionDqnObjective, boundary_discount=boundary_discount,
        device=torch.device('cpu'), encoder=encoder, q_head=head, modules=modules,
        delayed_encoder=target_encoder, backup_value_head=target_head,
        loader=SimpleNamespace(next_batch=lambda: (inputs, data)),
        LR_EMBEDDING=1e-3, LR_BACKBONE=1e-3, LR_HEADS=1e-3,
        DIFFUSION_NOISE_MEAN=noise_mean, DIFFUSION_NOISE_STD=noise_std,
        POLYAK_TAU_EMBEDDING=.1, POLYAK_TAU_BACKBONE=.1, POLYAK_TAU_VALUE=head_tau)
    source = next(s for s in notebook_sources() if 'def run_train(' in s)
    exec(compile(source, str(NOTEBOOK), 'exec'), ns)
    return ns


@pytest.mark.parametrize('controlled_target', [False, True], ids=['real-head', 'fixed-selector-action'])
@pytest.mark.parametrize('action_samples', [1, 3])
@pytest.mark.parametrize('noise_mean,noise_std', [(0., 1.), (2.5, 2.9)])
def test_actual_notebook_training_loop_uses_configurable_selector_and_single_backup(
    controlled_target, action_samples, noise_mean, noise_std,
):
    ns = notebook_training_namespace(action_samples=action_samples, noise_mean=noise_mean, noise_std=noise_std)
    encoder, target_encoder = ns['encoder'], ns['delayed_encoder']
    head, target_head, modules = ns['q_head'], ns['backup_value_head'], ns['modules']
    trajectory_calls = []
    trajectory = target_head.trajectory

    def record_trajectory(features, *, initial_noise):
        samples, velocities, endpoint = trajectory(features, initial_noise=initial_noise)
        if controlled_target:
            if not trajectory_calls:
                # Only the final selector endpoint chooses the action.
                endpoint = torch.tensor([[9., 1.], [0., 10.]]).repeat_interleave(action_samples, dim=0)
            else:
                # Backup maxima change with tau and disagree at the endpoint.
                samples = samples.clone()
                velocities = velocities.clone()
                samples[1] = torch.tensor([[8., 1.], [2., 9.], [10., 3.], [4., 11.]])
                velocities[1] = torch.tensor([[1., 20.], [30., 4.], [5., 60.], [70., 8.]])
                endpoint = torch.tensor([[1., 9.], [10., 0.]])
        trajectory_calls.append((features.detach().clone(), initial_noise.clone(), samples, velocities, endpoint))
        return samples, velocities, endpoint

    target_head.trajectory = record_trajectory
    prepared = {}
    prepare = ns['objective'].prepare

    def record_prepare(**kwargs):
        result = prepare(**kwargs)
        prepared.update(inputs=kwargs, result=result)
        return result

    ns['objective'].prepare = record_prepare
    expected_rng = torch.Generator().set_state(torch.get_rng_state())
    expected_selector_noise = noise_mean + noise_std * torch.randn(2 * action_samples, 2, generator=expected_rng)
    expected_backup_noise = noise_mean + noise_std * torch.randn(2, 2, generator=expected_rng)
    expected_correction_noise = torch.randn(2, generator=expected_rng)
    before = {k: p.detach().clone() for k, p in modules.named_parameters()}
    loss, metrics = ns['run_train'](num_steps=1)
    assert torch.isfinite(loss) and metrics['per_diffusion_step'].shape == (4,)
    # K batched selector trajectories and one fresh backup; no tau-level reselection.
    assert len(trajectory_calls) == 2
    selector, backup = trajectory_calls
    assert selector[0].shape == (2 * action_samples, 8)
    assert backup[0].shape == (2, 8)
    torch.testing.assert_close(selector[0], backup[0].repeat_interleave(action_samples, dim=0))
    assert selector[1].shape == (2 * action_samples, 2)
    assert backup[1].shape == (2, 2)
    torch.testing.assert_close(selector[1], expected_selector_noise, rtol=0, atol=0)
    torch.testing.assert_close(backup[1], expected_backup_noise, rtol=0, atol=0)
    assert not torch.equal(selector[1][:2], backup[1])
    torch.testing.assert_close(prepared['inputs']['noise'], expected_correction_noise, rtol=0, atol=0)
    correction_draws = noise_mean + noise_std * prepared['inputs']['noise']
    # A third independent draw supplies the correction, not either action's Z0.
    for action in range(2):
        assert not torch.equal(correction_draws, backup[1][:, action])
        assert not torch.equal(correction_draws, selector[1][:2, action])
    selector_values = selector[4].reshape(2, action_samples, 2).mean(1)
    torch.testing.assert_close(prepared['inputs']['backup_action_predictions'], selector_values)
    torch.testing.assert_close(prepared['inputs']['backup_samples'], backup[2])
    torch.testing.assert_close(prepared['inputs']['backup_value_predictions'], backup[3])
    selected = selector_values[1].argmax()
    flow_inputs, targets, _, _ = prepared['result']
    torch.testing.assert_close(flow_inputs[0, :, 1], head.times + backup[2][1, :, selected])
    torch.testing.assert_close(targets[0], 1 + backup[3][1, :, selected])
    terminal_noise = noise_mean + noise_std * expected_correction_noise[1]
    torch.testing.assert_close(flow_inputs[1, :, 0], 2*head.times + (1-head.times)*terminal_noise)
    torch.testing.assert_close(targets[1], (2-terminal_noise).expand(4))
    if controlled_target:
        assert selected.item() == 1 and backup[4][1].argmax().item() == 0
        assert backup[2][1].argmax(-1).tolist() == [0, 1, 0, 1]
    for name, p in modules.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
        assert not torch.equal(p, before[name]), name
    for source, target, prefix, tau in [(encoder.embedding, target_encoder.embedding, 'encoder.embedding', .1),
            (encoder.backbone, target_encoder.backbone, 'encoder.backbone', .1)]:
        for name, p in target.named_parameters():
            assert p.grad is None
            torch.testing.assert_close(p, before[f'{prefix}.{name}'].lerp(dict(source.named_parameters())[name], tau))
    for name, p in target_head.named_parameters():
        assert p.grad is None
        # Projections, residual blocks and the final linear layer share one rate.
        torch.testing.assert_close(p, before[f'q_head.{name}'].lerp(dict(head.named_parameters())[name], .01),
                                   rtol=0, atol=0)


@pytest.mark.parametrize('num_blocks', [0, 2])
@pytest.mark.parametrize('head_tau', [0., 1.])
def test_velocity_target_can_freeze_or_copy_the_entire_head(num_blocks, head_tau):
    ns = notebook_training_namespace(num_blocks=num_blocks, head_tau=head_tau)
    source, target = ns['q_head'], ns['backup_value_head']
    before = {name: value.detach().clone() for name, value in target.named_parameters()}
    buffers_before = {name: value.detach().clone() for name, value in target.named_buffers()}
    ns['run_train'](num_steps=1)
    for name, parameter in target.named_parameters():
        online = dict(source.named_parameters())[name]
        assert not torch.equal(online, before[name]), name
        expected = online if head_tau == 1. else before[name]
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        assert parameter.grad is None
    for name, buffer in target.named_buffers():
        torch.testing.assert_close(buffer, buffers_before[name], rtol=0, atol=0)


@pytest.mark.parametrize('head_tau', [-1., 1.01, float('inf'), float('nan'), True, '1', None])
def test_invalid_velocity_target_rate_fails_when_building_notebook_optimizer(head_tau):
    with pytest.raises(ValueError):
        notebook_training_namespace(head_tau=head_tau)


@pytest.mark.parametrize('num_blocks', [0, 1, 2, 4])
def test_real_notebook_encoder_feeds_diffusion_head(num_blocks):
    from tests.test_example_modules import _example, _tokens

    _, ns = _example(NOTEBOOK, DIFFUSION_STEPS=3,
                     DIFFUSION_NUM_BLOCKS=num_blocks,
                     DIFFUSION_TIME_MIN=-.5, DIFFUSION_TIME_MAX=1.5,
                     DIFFUSION_LATENT_MIN=-10., DIFFUSION_LATENT_MAX=30.,
                     DIFFUSION_CONTEXT_EMBEDDING_SCALE=.5,
                     DIFFUSION_LATENT_EMBEDDING_SCALE=2., DIFFUSION_TIME_EMBEDDING_SCALE=.25,
                     DIFFUSION_ACTION_SAMPLES=3,
                     DIFFUSION_NOISE_MEAN=2.5, DIFFUSION_NOISE_STD=2.9,
                     VELOCITY_OUTPUT_INITIALIZATION_SCALE=.1,
                     SwiGLU=SwiGLU)
    with torch.no_grad():
        features = ns['encoder'].eval()(_tokens(0, 1))['z']
        head = ns['q_head'].eval()
        assert head.width == ns['encoder'].backbone.hidden_dim == features.shape[-1]
        assert head.output.in_features == head.noisy_return.out_features == head.time_embedding.out_features == head.width
        assert not hasattr(head, 'output_norm')
        assert head.action_samples == 3
        assert len(head.blocks) == num_blocks
        assert (head.time_min, head.time_max) == (-.5, 1.5)
        assert (head.latent_min, head.latent_max) == (-10., 30.)
        assert (head.context_embedding_scale, head.latent_embedding_scale, head.time_embedding_scale) == (.5, 2., .25)
        assert head.noise_mean == 2.5 and head.noise_std == 2.9
        values = head.q_values(features)
    assert values.shape == (2, 3) and torch.isfinite(values).all()
