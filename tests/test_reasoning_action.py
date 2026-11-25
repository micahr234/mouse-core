"""Independent two-stage probabilities and Bellman targets, including gradients."""
import pytest
import torch
from mouse_core.models import ReasoningActionValues, ReasoningActionHeads
from mouse_core.objectives import value_reasoning_loss


@pytest.mark.parametrize('num_actions', [1, 4, 100, 1000])
@pytest.mark.parametrize('temperature', [0., .5, 1., 2.])
def test_binary_think_probability_does_not_dilute_with_action_count(num_actions, temperature):
    values = ReasoningActionValues(torch.linspace(-100, 100, num_actions)[None], torch.tensor([[1., 2.]]))
    probs = values.policy_logits(torch.tensor([temperature])).exp()
    expected = 1. if temperature == 0 else torch.sigmoid(torch.tensor(1. / temperature)).item()
    assert probs[0, -1].item() == pytest.approx(expected)
    assert probs.sum().item() == pytest.approx(1.)
    assert values.greedy().item() == num_actions  # Even the largest action Q cannot override THINK.


def test_masks_apply_to_binary_gate_and_do_not_permute_environment_actions():
    values = ReasoningActionValues(torch.tensor([[3., 7., 2.]] * 4), torch.tensor([[5., 10.]] * 4))
    masked = values.masked(torch.arange(1, 5), min_depth=2, max_depth=4)
    assert masked.greedy().tolist() == [3, 3, 3, 1]
    probs = masked.policy_logits(torch.ones(4), policy_think_enabled=False).exp()
    assert probs[0, -1] == 1
    assert probs[1:, -1].count_nonzero() == 0
    assert torch.isfinite(values.control_values).all()
    swapped = ReasoningActionValues(values.action_values[:, [2, 0, 1]], values.control_values)
    torch.testing.assert_close(values.policy_logits(torch.ones(4))[:, -1], swapped.policy_logits(torch.ones(4))[:, -1])


@pytest.mark.parametrize('gamma', [0., .4, 1.])
@pytest.mark.parametrize('minimum,cap,lengths', [(1, 1, [1, 1]), (1, 4, [2, 4]), (3, 4, [3, 4]), (3, 3, [3, 3])])
@pytest.mark.parametrize('reward', [-.13, 0., .13])
def test_three_bellman_edges_and_exact_gradient(gamma, minimum, cap, lengths, reward):
    actions = (torch.arange(2 * cap * 4).reshape(2, cap, 4).float() / 17).requires_grad_()
    gate = (torch.arange(2 * cap * 2).reshape(2, cap, 2).float() / 13).requires_grad_()
    target_actions = (actions.detach() + .2).requires_grad_()
    target_gate = gate.detach() + .7
    target_gate[:, :minimum - 1, 0] = 10000.
    target_gate[:, -1, 1] = 20000.
    target_gate.requires_grad_()
    targets = torch.tensor([7., -3.], requires_grad=True)
    weights = torch.tensor([1., 2.], requires_grad=True)
    recorded = [2, 1]
    scale, shift = 2.5, -.2
    loss, details = value_reasoning_loss(q_values=ReasoningActionValues(actions, gate),
        target_q_values=ReasoningActionValues(target_actions, target_gate), lengths=torch.tensor(lengths),
        environment_actions=torch.tensor(recorded), environment_targets=targets, weights=weights,
        think_reward=reward, gamma_think=gamma, min_depth=minimum, value_scale=scale, value_shift=shift)
    denominator = sum(w * (length + 1) for w, length in zip(weights.tolist(), lengths))
    expected = 0.
    action_grad = torch.zeros_like(actions)
    gate_grad = torch.zeros_like(gate)
    for row, length in enumerate(lengths):
        w = float(weights[row].detach()) / denominator
        for position in range(length):
            continuing = position < length - 1
            selected = int(continuing)
            if continuing:
                depth = position + 2
                legal = [1] if depth < minimum else [0] if depth == cap else [0, 1]
                target = reward + gamma * max(scale * float(target_gate[row, position + 1, a].detach()) + shift for a in legal)
            else:
                target = max(scale * float(q) + shift for q in target_actions[row, position].detach())
            error = scale * float(gate[row, position, selected].detach()) + shift - target
            expected += w * error ** 2
            gate_grad[row, position, selected] = 2 * w * scale * error
            assert details['td_target'][row, position].item() == pytest.approx(target, rel=2e-6)
        action = recorded[row]
        error = scale * float(actions[row, length - 1, action].detach()) + shift - float(targets[row].detach())
        expected += w * error ** 2
        action_grad[row, length - 1, action] = 2 * w * scale * error
    assert loss.item() == pytest.approx(expected, rel=2e-6)
    loss.backward()
    torch.testing.assert_close(actions.grad, action_grad)
    torch.testing.assert_close(gate.grad, gate_grad)
    assert target_actions.grad is None and target_gate.grad is None and targets.grad is None and weights.grad is None


def test_both_heads_receive_gradients_and_frozen_head_passes_gradients_to_state():
    head = ReasoningActionHeads(action_head=torch.nn.Linear(8, 4), reasoning_control_head=torch.nn.Linear(8, 2))
    states = torch.randn(2, 3, 8, requires_grad=True)
    head.action_head.requires_grad_(False)
    values = head(states)
    (values.action_values.square().mean() + values.control_values.square().mean()).backward()
    assert states.grad.abs().sum() > 0
    assert all(p.grad is None for p in head.action_head.parameters())
    assert all(p.grad is not None for p in head.reasoning_control_head.parameters())


def test_fixed_budget_sequence_replay_matches_generation():
    from tests.test_sequential_reasoning import components, generate
    from tests.test_residual_reasoning import batch
    parts = components(latent_residual_enabled=True, adaptive_depth=False, latent_feedback_scope='sequence', cap=3)
    embedding, backbone, reasoner, _ = parts
    tokens = batch(((2, 9, 3, 9), (4, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    direct = generate(parts, tokens)
    prepared = reasoner.prepare_replay(tokens, lengths=torch.full((3,), 3, dtype=torch.long),
        embedding=embedding, backbone=backbone, backend='uncached', max_active_requests=2)
    torch.testing.assert_close(reasoner(prepared, embedding=embedding, backbone=backbone),
                               reasoner(direct, embedding=embedding, backbone=backbone))


@pytest.mark.parametrize("perturbation", ["none", "same_token_actions", "same_token_gate", "future_token", "online_actions"])
def test_execute_changes_heads_at_same_token_while_think_changes_tokens(perturbation):
    actions = torch.zeros(1, 3, 2)
    gate = torch.zeros(1, 3, 2)
    target_actions = torch.tensor([[[101., 103.], [7., 11.], [1001., 1003.]]])
    target_gate = torch.tensor([[[401., 403.], [17., 19.], [2001., 2003.]]])
    expected_think, expected_execute = 4.25, 11.
    if perturbation == "same_token_actions":
        target_actions[0, 1, 1] += 100.
        expected_execute += 100.
    elif perturbation == "same_token_gate":
        target_gate[0, 1, 1] += 100.
        expected_think += 25.
    elif perturbation == "future_token":
        target_actions[0, 2] += 10000.
        target_gate[0, 2] += 10000.
    elif perturbation == "online_actions":
        # The unrecorded online action must not become EXECUTE's target.
        actions[0, 1, 1] = 100000.
    for tensor in (actions, gate, target_actions, target_gate):
        tensor.requires_grad_()
    environment_target = torch.tensor([-5.], requires_grad=True)
    loss, details = value_reasoning_loss(
        q_values=ReasoningActionValues(actions, gate),
        target_q_values=ReasoningActionValues(target_actions, target_gate),
        lengths=torch.tensor([2]), environment_actions=torch.tensor([0]),
        environment_targets=environment_target, weights=torch.ones(1),
        think_reward=-.5, gamma_think=.25, min_depth=1,
    )
    torch.testing.assert_close(details["td_target"][0, :2], torch.tensor([expected_think, expected_execute]))
    torch.testing.assert_close(details["action_target"], torch.tensor([-5.]))
    assert loss.item() == pytest.approx((expected_think**2 + expected_execute**2 + 25.) / 3)
    loss.backward()
    expected_gate_gradient = torch.zeros_like(gate)
    expected_gate_gradient[0, 0, 1] = -2 * expected_think / 3
    expected_gate_gradient[0, 1, 0] = -2 * expected_execute / 3
    expected_action_gradient = torch.zeros_like(actions)
    expected_action_gradient[0, 1, 0] = 10 / 3
    torch.testing.assert_close(gate.grad, expected_gate_gradient)
    torch.testing.assert_close(actions.grad, expected_action_gradient)
    assert target_actions.grad is target_gate.grad is environment_target.grad is None
