"""Numerical targets and gradients for environment/think Q-learning."""
import pytest
import torch
from mouse_core.models import ReasoningActionValues
from mouse_core.objectives import value_reasoning_loss






def test_zero_weight_and_empty_batches_are_finite():
    for n in (0, 2):
        q = torch.zeros(n, 3, 6, requires_grad=True)
        loss, _ = value_reasoning_loss(min_depth=1, q_values=ReasoningActionValues(q[..., :4], q[..., 4:]), target_q_values=ReasoningActionValues(q[..., :4].detach(), q[..., 4:].detach()),
            lengths=torch.ones(n, dtype=torch.long), environment_actions=torch.zeros(n, dtype=torch.long),
            environment_targets=torch.full((n,), float('nan')), weights=torch.zeros(n), think_reward=-.1, gamma_think=1.)
        assert loss.item() == 0
        loss.backward()
        assert torch.isfinite(q.grad).all() and q.grad.count_nonzero() == 0




@pytest.mark.parametrize("gamma_think", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_think_discount_is_rejected(gamma_think):
    with pytest.raises(ValueError, match="gamma_think"):
        value_reasoning_loss(min_depth=1,
            q_values=ReasoningActionValues(torch.zeros(1, 1, 1), torch.zeros(1, 1, 2)), target_q_values=ReasoningActionValues(torch.zeros(1, 1, 1), torch.zeros(1, 1, 2)),
            lengths=torch.ones(1, dtype=torch.long), environment_actions=torch.zeros(1, dtype=torch.long),
            environment_targets=torch.zeros(1), weights=torch.ones(1),
            think_reward=-0., gamma_think=gamma_think,
        )


@pytest.mark.parametrize('cost', [-.7, 0., .7])
def test_final_reasoning_loss_uses_selected_values_and_current_cost_only(cost):
    from mouse_core.objectives import final_reasoning_loss
    q = torch.tensor([2., 3., float('nan')], requires_grad=True)
    target = torch.tensor([8., 9., float('nan')], requires_grad=True)
    weights = torch.tensor([1., 2., 0.], requires_grad=True)
    lengths = torch.tensor([1, 4, 3])
    loss, details = final_reasoning_loss(q_values=q, environment_targets=target,
        lengths=lengths, weights=weights, think_reward=-cost, value_scale=2., value_shift=-1.)
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
        lengths=torch.ones(count, dtype=torch.long), weights=torch.zeros(count), think_reward=-.3)
    assert loss == 0
    loss.backward()
    assert values.grad.count_nonzero() == 0




@pytest.mark.parametrize("reward", [True, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("final", [False, True])
def test_reasoning_reward_requires_finite_number(reward, final):
    from mouse_core.objectives import final_reasoning_loss
    common = dict(lengths=torch.ones(1, dtype=torch.long),
                  environment_targets=torch.zeros(1), weights=torch.ones(1),
                  think_reward=reward)
    with pytest.raises(ValueError, match="think_reward"):
        if final:
            final_reasoning_loss(q_values=torch.zeros(1), **common)
        else:
            value_reasoning_loss(q_values=ReasoningActionValues(torch.zeros(1, 1, 1), torch.zeros(1, 1, 2)),
                target_q_values=ReasoningActionValues(torch.zeros(1, 1, 1), torch.zeros(1, 1, 2)), environment_actions=torch.zeros(1, dtype=torch.long),
                gamma_think=1., min_depth=1, **common)
