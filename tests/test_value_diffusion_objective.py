"""Value Forcing input/target coupling, DQN alignment, and velocity MSE."""

from types import SimpleNamespace

from tests.test_value_diffusion_notebook import definitions

import pytest
import torch

from mouse_core.objectives import Objective, ValueDiffusionObjective, affine_reward, boundary_discount


_notebook = definitions()
ValueFlow = _notebook["ValueFlow"]


class AnalyticFlow(ValueFlow):
    def __init__(self, weight, *, noise_levels=None):
        super().__init__(context_dim=1, hidden_dim=2, num_actions=2, num_layers=1,
                         return_num_frequencies=1, time_num_frequencies=1, scale=.1,
                         num_steps=3, noise_levels=noise_levels)
        self.weight = weight

    def _context(self, features):
        return features

    def _predict(self, context, samples, times):
        context = context.reshape(len(context), *((1,) * samples.ndim))
        time = times.reshape(1, len(times), *((1,) * (samples.ndim - 1)))
        return (context + samples[..., None] + time) * self.weight


def inputs():
    return dict(
        backup_samples=torch.randn(4, 3, 2, requires_grad=True),
        backup_value_predictions=torch.randn(4, 3, 2, requires_grad=True),
        backup_action_predictions=torch.tensor([[0., 1.]] * 4, requires_grad=True),
        noise=torch.randn(4, requires_grad=True),
        objective_data={
            'action': torch.tensor([torch.nan, 0., 1., torch.nan, 1., 0.]),
            'reward': torch.tensor([0., 0.2, 1., 0., -0.3, -2.]),
            'terminated': torch.tensor([False, False, True, False, False, False]),
            'truncated': torch.tensor([False, False, False, False, False, True]),
        },
        sequence_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        step_ids=torch.tensor([0, 1, 3, 4]),
    )


def objective(*, discount=None, reward=None):
    return SimpleNamespace(
        loss=ValueDiffusionObjective(
            discount=discount if discount is not None else boundary_discount(
                gamma_step=0.6, gamma_terminated=0., gamma_truncated=0.), reward=reward),
    )


def prepare(fn, *, backup_samples, backup_value_predictions, backup_action_predictions,
            noise, objective_data, sequence_ids, step_ids, noise_levels=None):
    times = torch.tensor([0., .5, 1.]) if noise_levels is None else noise_levels
    arguments = dict(objective_data=objective_data, sequence_ids=sequence_ids, step_ids=step_ids,
                     backup_action_predictions=backup_action_predictions, noise=noise)
    samples = fn.loss.inputs(backup_samples=backup_samples, noise_levels=times, **arguments)
    with torch.no_grad():
        _, metrics = fn.loss(predictions=torch.zeros_like(samples),
                             backup_value_predictions=backup_value_predictions, **arguments)
    incoming = (step_ids + 1).clamp_max(len(objective_data['action']) - 1)
    return SimpleNamespace(samples=samples, velocities=metrics['backup'],
                           actions=objective_data['action'][incoming].nan_to_num(0).long(),
                           weights=metrics['backup_weight'], effective_discount=metrics['effective_discount'],
                           noise_levels=times, arguments=arguments,
                           backup_value_predictions=backup_value_predictions)


def velocity_loss(fn, *, predictions, targets):
    losses, metrics = fn.loss(predictions=predictions,
                             backup_value_predictions=targets.backup_value_predictions, **targets.arguments)
    metrics['flow_inputs'] = targets.samples
    return losses, metrics


def test_prepare_matches_paper_inputs_and_velocity_targets_at_every_noise_level():
    batch, fn = inputs(), objective()
    targets = prepare(fn, **batch)
    assert isinstance(fn.loss, Objective)
    noise_levels = torch.tensor([0., 0.5, 1.])
    for row, next_row, incoming in ((0, 1, 1), (2, 3, 4)):
        r = batch['objective_data']['reward'][incoming]
        eps = batch['noise'][row]
        z = batch['backup_samples'][next_row, :, 1]
        velocity = batch['backup_value_predictions'][next_row, :, 1]
        torch.testing.assert_close(targets.samples[row], noise_levels * r + .6 * z + (1 - noise_levels) * .8 * eps)
        torch.testing.assert_close(targets.velocities[row], r + .6 * velocity - .8 * eps)
    # Terminal AND truncated transitions need no following readout.
    for row, incoming in ((1, 2), (3, 5)):
        r, eps = batch['objective_data']['reward'][incoming], batch['noise'][row]
        torch.testing.assert_close(targets.samples[row], noise_levels * r + (1 - noise_levels) * eps)
        torch.testing.assert_close(targets.velocities[row], (r - eps).expand(3))
    torch.testing.assert_close(targets.effective_discount, torch.tensor([.6, 0., .6, 0.]))
    assert targets.weights.eq(1).all()


def test_same_correction_draw_changes_both_input_and_velocity_with_correct_signs():
    fn, batch = objective(), inputs()
    first = prepare(fn, **batch)
    second = prepare(fn, **(batch | {'noise': batch['noise'] + 2.}))
    scale = (1 - first.effective_discount.square()).sqrt()[:, None]
    noise_level = torch.tensor([0., .5, 1.])[None, :]
    torch.testing.assert_close(second.samples - first.samples, (2 * (1 - noise_level) * scale).expand_as(first.samples))
    torch.testing.assert_close(second.velocities - first.velocities, (-2 * scale).expand_as(first.velocities))
    # At noise_level=1 the input is clean but the derivative still contains epsilon.
    torch.testing.assert_close(second.samples[:, -1], first.samples[:, -1])
    assert not torch.equal(second.velocities[:, -1], first.velocities[:, -1])


def test_mse_on_paired_velocities_supports_one_sample_and_stops_target_gradients():
    fn, batch = objective(), inputs()
    rng = torch.random.get_rng_state()
    targets = prepare(fn, **batch)
    assert torch.equal(rng, torch.random.get_rng_state())
    predictions = (targets.velocities + 2.).detach().requires_grad_()
    losses, metrics = velocity_loss(fn, predictions=predictions, targets=targets)
    torch.testing.assert_close(losses['value_diffusion'], torch.tensor(4.))
    losses['value_diffusion'].backward()
    assert predictions.grad is not None and predictions.grad.abs().sum() > 0
    for key in ('backup_samples', 'backup_value_predictions', 'backup_action_predictions', 'noise'):
        assert batch[key].grad is None
    assert all(not value.requires_grad for value in metrics.values())
    zero, _ = velocity_loss(fn, predictions=targets.velocities.clone().requires_grad_(), targets=targets)
    assert zero['value_diffusion'].item() == 0


def test_loss_averages_distinct_errors_from_every_flow_step():
    fn = objective()
    targets = prepare(fn, **inputs())
    predictions = (targets.velocities + torch.tensor([1., 2., 4.])[None, :]).requires_grad_()
    losses, metrics = velocity_loss(fn, predictions=predictions, targets=targets)
    torch.testing.assert_close(metrics['per_flow_step'], torch.tensor([1., 4., 16.]))
    torch.testing.assert_close(losses['value_diffusion'], torch.tensor(7.))
    losses['value_diffusion'].backward()
    assert (predictions.grad.abs().sum(dim=0) > 0).all()


def test_flow_is_separate_from_loss_and_stops_target_gradients():
    batch = inputs()
    fn = objective(discount=boundary_discount(
        gamma_step=.6, gamma_terminated=0., gamma_truncated=0.))
    features = torch.randn(4, 1, requires_grad=True)
    delayed_features = torch.randn(4, 1, requires_grad=True)
    online_weight = torch.tensor([.3, .7], requires_grad=True)
    delayed_weight = torch.tensor([.2, .5], requires_grad=True)
    flow = AnalyticFlow(online_weight)
    backup = AnalyticFlow(delayed_weight)
    draws = torch.randn(4, 5, generator=torch.Generator().manual_seed(31))
    # Integrate independently to check the external module's trajectory.
    current = draws[:, :1].expand(-1, 2)
    samples, velocities = [], []
    for layer in range(3):
        velocity = (delayed_features + current + layer / 3) * delayed_weight
        samples.append(current)
        velocities.append(velocity)
        current = current + velocity / 3
    trajectory = backup.trajectory(delayed_features, initial_samples=draws[:, 0])
    torch.testing.assert_close(trajectory.samples, torch.stack(samples, 1))
    torch.testing.assert_close(trajectory.velocities, torch.stack(velocities, 1))
    torch.testing.assert_close(trajectory.returns, current)
    # The objective only consumes already-generated flow data and velocities.
    before = torch.random.get_rng_state()
    targets = prepare(
        fn, backup_samples=trajectory.samples, backup_value_predictions=trajectory.velocities,
        backup_action_predictions=trajectory.returns, noise=draws[:, 1],
        objective_data=batch['objective_data'], sequence_ids=batch['sequence_ids'],
        step_ids=batch['step_ids'], noise_levels=trajectory.noise_levels,
    )
    losses, metrics = velocity_loss(fn, predictions=flow(features, targets.samples, times=targets.noise_levels), targets=targets)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(metrics['flow_inputs'], targets.samples)
    torch.testing.assert_close(metrics['backup'], targets.velocities)
    losses['value_diffusion'].backward()
    assert (features.grad.abs().sum(-1) > 0).all()
    assert online_weight.grad.abs().sum() > 0
    for tensor in (delayed_features, delayed_weight):
        assert tensor.grad is None


def test_all_action_prediction_loss_selects_incoming_action_only():
    fn = objective()
    targets = prepare(fn, **inputs())
    prediction = torch.zeros(4, 3, 2, requires_grad=True)
    losses, _ = velocity_loss(fn, predictions=prediction, targets=targets)
    torch.testing.assert_close(losses['value_diffusion'], targets.velocities.square().mean())
    losses['value_diffusion'].backward()
    for row, action in enumerate(targets.actions):
        assert prediction.grad[row, :, 1 - action].count_nonzero() == 0


def test_automatic_steps_use_accumulated_reward_and_product_discount_for_both_parts():
    fn = objective()
    batch = inputs()
    for key in ('backup_samples', 'backup_value_predictions', 'backup_action_predictions', 'noise'):
        batch[key] = batch[key][:2].detach()
    batch['backup_samples'].fill_(4.)
    batch['backup_value_predictions'].fill_(2.)
    batch['objective_data'] = dict(
        action=torch.tensor([torch.nan, 0., torch.nan, 1.]),
        reward=torch.tensor([0., 2., 3., 5.]),
        terminated=torch.tensor([False, False, False, True]), truncated=torch.zeros(4, dtype=torch.bool),
    )
    batch['sequence_ids'] = torch.zeros(4, dtype=torch.long)
    batch['step_ids'] = torch.tensor([0, 2])
    targets = prepare(fn, **batch)
    reward, discount = 2. + .6 * 3., .6**2
    correction = (1 - discount**2)**.5 * batch['noise'][0]
    noise_levels = torch.tensor([0., .5, 1.])
    torch.testing.assert_close(targets.samples[0], noise_levels * reward + discount * 4. + (1 - noise_levels) * correction)
    torch.testing.assert_close(targets.velocities[0], (reward + discount * 2. - correction).expand(3))
    torch.testing.assert_close(targets.effective_discount, torch.tensor([.36, 0.]))


def test_reward_normalization_applies_before_noise_level_scaling_and_velocity_differentiation():
    batch = inputs()
    before = prepare(objective(), **batch)
    after = prepare(objective(reward=affine_reward(scale=2., shift=0.)), **batch)
    reward = batch['objective_data']['reward'][batch['step_ids'] + 1]
    delta_input = reward[:, None] * torch.tensor([0., .5, 1.])[None, :]
    torch.testing.assert_close(after.samples - before.samples, delta_input.expand_as(before.samples))
    torch.testing.assert_close(after.velocities - before.velocities, reward[:, None].expand_as(before.velocities))


def test_repeated_readouts_use_last_next_prediction_but_keep_own_correction_draws():
    batch = inputs()
    batch['step_ids'] = torch.tensor([0, 1, 1, 3, 4])
    for key in ('backup_samples', 'backup_value_predictions', 'backup_action_predictions', 'noise'):
        batch[key] = batch[key].detach()[[0, 1, 1, 2, 3]].clone()
    batch['noise'].zero_()
    batch['noise'][2].fill_(2.)
    batch['backup_samples'][1].fill_(500.)
    batch['backup_samples'][2].fill_(4.)
    batch['backup_value_predictions'][2].fill_(3.)
    targets = prepare(objective(), **batch)
    torch.testing.assert_close(targets.samples[0], (.2 * torch.tensor([0., .5, 1.]) + 2.4))
    torch.testing.assert_close(targets.velocities[0], torch.full((3,), 2.))
    torch.testing.assert_close(targets.velocities[2] - targets.velocities[1], torch.full((3,), -2.))


def test_inactive_rows_ignore_nan_predictions_without_crossing_groups():
    batch = inputs()
    batch['sequence_ids'] = torch.arange(6)
    fn = objective()
    targets = prepare(fn, **batch)
    prediction = torch.full_like(targets.velocities, torch.nan, requires_grad=True)
    loss, _ = velocity_loss(fn, predictions=prediction, targets=targets)
    assert loss['value_diffusion'].item() == 0
    loss['value_diffusion'].backward()
    assert prediction.grad.count_nonzero() == 0


@pytest.mark.parametrize('noise_levels', [[], [[.5]], [float('nan')], [-.1], [1.1]])
def test_invalid_schedule(noise_levels):
    with pytest.raises(ValueError, match='noise_levels'):
        AnalyticFlow(torch.tensor([.2, .5]), noise_levels=noise_levels)


@pytest.mark.parametrize('discount', [-.1, 1.1, float('nan')])
def test_invalid_discount(discount):
    fn = objective(discount=lambda terminated, truncated, **_: torch.where(
        terminated | truncated, 0., torch.full((6,), discount)))
    with pytest.raises(ValueError, match='discounts'):
        prepare(fn, **inputs())


def test_fractional_action_is_rejected():
    batch = inputs()
    batch['objective_data']['action'][1] = .5
    with pytest.raises(ValueError, match='integers'):
        prepare(objective(), **batch)
