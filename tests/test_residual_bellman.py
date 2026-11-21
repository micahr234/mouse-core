"""Numerical targets and gradients for environment/think Q-learning."""
import pytest
import torch
from mouse_core.objectives import value_reasoning_loss


def test_internal_backups_cap_recorded_action_padding_and_detached_targets():
    q = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                      [[0., 1., 2.], [4., 5., 6.], [100., 100., 100.]],
                      [[100., 100., 100.]] * 3], requires_grad=True)
    target = torch.tensor([[[0., 0., 0.], [2., 3., 9.], [4., 7., 100.]],
                           [[0., 0., 0.], [2., 8., 12.], [100., 100., 100.]],
                           [[100., 100., 100.]] * 3], requires_grad=True)
    environment = torch.tensor([11., -2., 999.], requires_grad=True)
    loss, details = value_reasoning_loss(min_steps=1, 
        q_values=q, target_q_values=target, lengths=torch.tensor([3, 2, 1]),
        environment_actions=torch.tensor([1, 0, 1]), environment_targets=environment,
        weights=torch.tensor([1., 2., 0.]), reasoning_reward=-.5, gamma_think=1.,
    )
    # A target may still choose think at an online stopping depth below the cap.
    torch.testing.assert_close(details['td_target'][:2], torch.tensor([[8.5, 6.5, 11.], [11.5, -2., -2.]]))
    torch.testing.assert_close(loss, torch.tensor(((3-8.5)**2 + (6-6.5)**2 + (8-11)**2
                                                  + 2*((2-11.5)**2 + (4+2)**2))/7))
    loss.backward()
    expected = torch.zeros_like(q)
    expected[0, 0, 2] = 2*(3-8.5)/7
    expected[0, 1, 2] = 2*(6-6.5)/7
    expected[0, 2, 1] = 2*(8-11)/7
    expected[1, 0, 2] = 4*(2-11.5)/7
    expected[1, 1, 0] = 4*(4+2)/7
    torch.testing.assert_close(q.grad, expected)
    assert target.grad is None and environment.grad is None


def test_first_state_is_free_and_affine_units_apply_to_internal_targets():
    q = torch.tensor([[[1., 3., 5.], [2., 4., 6.]]], requires_grad=True)
    target = torch.tensor([[[0., 0., 0.], [7., 9., 100.]]])
    args = dict(q_values=q, target_q_values=target, environment_actions=torch.tensor([0]),
                environment_targets=torch.tensor([10.]), weights=torch.ones(1),
                reasoning_reward=-2., gamma_think=1., value_scale=3., value_shift=-1.)
    loss, details = value_reasoning_loss(min_steps=1, **args, lengths=torch.tensor([2]))
    torch.testing.assert_close(details['td_target'], torch.tensor([[24., 10.]]))
    torch.testing.assert_close(loss, torch.tensor(((14-24)**2+(5-10)**2)/2))
    loss, _ = value_reasoning_loss(min_steps=1, **args, lengths=torch.tensor([1]))
    torch.testing.assert_close(loss, torch.tensor((2-10)**2, dtype=torch.float32))


def test_zero_weight_and_empty_batches_are_finite():
    for n in (0, 2):
        q = torch.zeros(n, 3, 5, requires_grad=True)
        loss, _ = value_reasoning_loss(min_steps=1, q_values=q, target_q_values=q.detach(),
            lengths=torch.ones(n, dtype=torch.long), environment_actions=torch.zeros(n, dtype=torch.long),
            environment_targets=torch.full((n,), float('nan')), weights=torch.zeros(n), reasoning_reward=-.1, gamma_think=1.)
        assert loss.item() == 0
        loss.backward()
        assert torch.isfinite(q.grad).all() and q.grad.count_nonzero() == 0


@pytest.mark.parametrize("gamma_think", [0., .4, 1.])
def test_think_discount_only_scales_the_internal_bootstrap(gamma_think):
    q = torch.zeros(1, 2, 3, requires_grad=True)
    delayed = torch.tensor([[[1., 2., 3.], [5., 10., 100.]]], requires_grad=True)
    loss, details = value_reasoning_loss(min_steps=1, 
        q_values=q, target_q_values=delayed, lengths=torch.tensor([2]),
        environment_actions=torch.tensor([1]), environment_targets=torch.tensor([7.]),
        weights=torch.ones(1), reasoning_reward=-2., gamma_think=gamma_think,
    )
    expected = torch.tensor([[gamma_think * 10 - 2, 7.]])
    torch.testing.assert_close(details['td_target'], expected)
    torch.testing.assert_close(loss, expected.square().mean())
    loss.backward()
    assert delayed.grad is None


@pytest.mark.parametrize("gamma_think", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_think_discount_is_rejected(gamma_think):
    with pytest.raises(ValueError, match="gamma_think"):
        value_reasoning_loss(min_steps=1, 
            q_values=torch.zeros(1, 1, 2), target_q_values=torch.zeros(1, 1, 2),
            lengths=torch.ones(1, dtype=torch.long), environment_actions=torch.zeros(1, dtype=torch.long),
            environment_targets=torch.zeros(1), weights=torch.ones(1),
            reasoning_reward=-0., gamma_think=gamma_think,
        )


@pytest.mark.parametrize('cost', [-.7, 0., .7])
def test_final_reasoning_loss_uses_selected_values_and_current_cost_only(cost):
    from mouse_core.objectives import final_reasoning_loss
    q = torch.tensor([2., 3., float('nan')], requires_grad=True)
    target = torch.tensor([8., 9., float('nan')], requires_grad=True)
    weights = torch.tensor([1., 2., 0.], requires_grad=True)
    lengths = torch.tensor([1, 4, 3])
    loss, details = final_reasoning_loss(q_values=q, environment_targets=target,
        lengths=lengths, weights=weights, reasoning_reward=-cost, value_scale=2., value_shift=-1.)
    expected_predictions = torch.tensor([3., 5.])
    expected_targets = torch.tensor([8., 9. - 3 * cost])
    expected = ((expected_predictions - expected_targets).square() * weights[:2].detach()).sum() / 3
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(details['td_target'][:2], expected_targets)
    loss.backward()
    torch.testing.assert_close(q.grad[:2], 4 * weights[:2].detach() * (expected_predictions - expected_targets) / 3)
    assert q.grad[2] == 0
    assert target.grad is None and weights.grad is None


@pytest.mark.parametrize('count', [0, 2])
def test_final_reasoning_loss_empty_or_masked_batches_are_finite(count):
    from mouse_core.objectives import final_reasoning_loss
    values = torch.full((count,), float('nan'), requires_grad=True)
    loss, _ = final_reasoning_loss(q_values=values, environment_targets=values.detach(),
        lengths=torch.ones(count, dtype=torch.long), weights=torch.zeros(count), reasoning_reward=-.3)
    assert loss == 0
    loss.backward()
    assert values.grad.count_nonzero() == 0


@pytest.mark.parametrize("reward", [-.5, 0., .5])
@pytest.mark.parametrize("gamma", [0., .4, 1.])
def test_signed_think_reward_only_affects_continuations(reward, gamma):
    q = torch.zeros(2, 2, 3, requires_grad=True)
    target = torch.tensor([[[0., 0., 0.], [2., 4., 100.]]] * 2, requires_grad=True)
    environment = torch.tensor([7., 9.], requires_grad=True)
    loss, details = value_reasoning_loss(
        q_values=q, target_q_values=target, lengths=torch.tensor([2, 1]),
        environment_actions=torch.tensor([0, 1]), environment_targets=environment,
        weights=torch.ones(2), reasoning_reward=reward, gamma_think=gamma, min_steps=1)
    expected = torch.tensor([[reward + gamma * 4, 7.], [9., 9.]])
    torch.testing.assert_close(details["td_target"], expected)
    torch.testing.assert_close(loss, (expected[0].square().sum() + 81) / 3)
    loss.backward()
    assert target.grad is None and environment.grad is None
    assert q.grad[1, 1].count_nonzero() == 0


@pytest.mark.parametrize("reward", [True, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("final", [False, True])
def test_reasoning_reward_requires_finite_number(reward, final):
    from mouse_core.objectives import final_reasoning_loss
    common = dict(lengths=torch.ones(1, dtype=torch.long),
                  environment_targets=torch.zeros(1), weights=torch.ones(1),
                  reasoning_reward=reward)
    with pytest.raises(ValueError, match="reasoning_reward"):
        if final:
            final_reasoning_loss(q_values=torch.zeros(1), **common)
        else:
            value_reasoning_loss(q_values=torch.zeros(1, 1, 2),
                target_q_values=torch.zeros(1, 1, 2), environment_actions=torch.zeros(1, dtype=torch.long),
                gamma_think=1., min_steps=1, **common)
