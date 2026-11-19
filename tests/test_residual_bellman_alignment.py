"""Compare residual TD targets and gradients against an explicit transition graph."""

import pytest
import torch

from mouse_core.objectives import value_reasoning_loss


@pytest.mark.parametrize("cap", [1, 2, 5, 8])
@pytest.mark.parametrize("gamma_think", [0., .37, 1.])
@pytest.mark.parametrize("value_scale,value_shift", [(1., 0.), (.3, -2.), (3., 1.7)])
def test_ragged_residual_loss_matches_scalar_transition_graph(cap, gamma_think, value_scale, value_shift):
    """Every visited state owns one edge: think to the next depth, or recorded action out."""
    generator = torch.Generator().manual_seed(804 + cap)
    count, environment_choices = 2 * cap + 3, 4
    shape = (count, cap, environment_choices + 1)
    q = torch.randn(shape, generator=generator, dtype=torch.float64)
    target = torch.randn(shape, generator=generator, dtype=torch.float64)
    lengths = 1 + torch.arange(count) % cap
    actions = torch.randint(environment_choices, (count,), generator=generator)
    rewards_and_environment_backups = torch.randn(count, generator=generator, dtype=torch.float64)
    weights = (torch.arange(count) % 3).to(torch.float64)
    cost = .23
    # Think is the largest counterfactual value at every non-cap state, including
    # online stopping states. It must disappear only at the configured hard cap.
    target[..., -1] = 30 + torch.arange(cap, dtype=torch.float64)
    # A current-state lookup instead of next-depth lookup produces a different result.
    target[:, 0] += 1000
    for decision, length in enumerate(lengths.tolist()):
        q[decision, length:] = torch.nan
        target[decision, length:] = torch.nan
    q.requires_grad_()
    target.requires_grad_()
    rewards_and_environment_backups.requires_grad_()
    weights.requires_grad_()

    loss, details = value_reasoning_loss(min_steps=1, 
        q_values=q, target_q_values=target, lengths=lengths,
        environment_actions=actions, environment_targets=rewards_and_environment_backups,
        weights=weights, reasoning_step_cost=cost, gamma_think=gamma_think,
        value_scale=value_scale, value_shift=value_shift,
    )

    # Build the augmented MDP explicitly. No padded state or nonselected action is
    # an edge, and this reference never uses vectorized shifts, gathers, or masks.
    edges = []
    for decision, length in enumerate(lengths.tolist()):
        for depth in range(length):
            if depth + 1 < length:
                action = environment_choices
                successor = depth + 1
                available = range(environment_choices + int(successor + 1 < cap))
                successor_values = [
                    value_scale * float(target[decision, successor, choice].detach()) + value_shift
                    for choice in available
                ]
                expected_target = -cost + gamma_think * max(successor_values)
            else:
                action = int(actions[decision])
                expected_target = float(rewards_and_environment_backups[decision].detach())
            prediction = value_scale * float(q[decision, depth, action].detach()) + value_shift
            weight = float(weights[decision].detach())
            edges.append((decision, depth, action, prediction, expected_target, weight))
            assert float(details['td_target'][decision, depth]) == pytest.approx(expected_target)
            assert float(details['td_prediction'][decision, depth].detach()) == pytest.approx(prediction)
            assert float(details['td_weight'][decision, depth]) == weight
        assert details['td_weight'][decision, length:].count_nonzero() == 0

    denominator = max(1., sum(edge[-1] for edge in edges))
    expected_loss = sum(weight * (prediction - expected_target) ** 2
                        for _, _, _, prediction, expected_target, weight in edges) / denominator
    assert float(loss.detach()) == pytest.approx(expected_loss, rel=1e-12, abs=1e-12)
    loss.backward()
    expected_gradient = torch.zeros_like(q)
    for decision, depth, action, prediction, expected_target, weight in edges:
        expected_gradient[decision, depth, action] = (
            2 * weight * value_scale * (prediction - expected_target) / denominator
        )
    torch.testing.assert_close(q.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    assert target.grad is None
    assert rewards_and_environment_backups.grad is None
    assert weights.grad is None
