"""Independent checks of minimum/maximum reasoning action availability."""

from dataclasses import replace

import pytest
import torch

from mouse_core.models import Backbone, Embedding, MultiEmbedding, SequentialReasoner
from mouse_core.objectives import value_reasoning_loss
from tests.test_residual_reasoning import batch


@pytest.mark.parametrize('residual_connections', [False, True])
@pytest.mark.parametrize('minimum,maximum', [(1, 1), (1, 4), (3, 4), (3, 3)])
@pytest.mark.parametrize('prefer_think', [False, True])
@torch.no_grad()
def test_continuous_generation_masks_actions_and_preserves_replay(
    residual_connections, minimum, maximum, prefer_think,
):
    torch.manual_seed(91)
    embedding = MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8)})
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                        head_dim=4, max_position_embeddings=128, dtype=torch.float32)
    reasoner = SequentialReasoner(hidden_dim=8, min_steps=minimum, max_steps=maximum,
                                 residual_connections=residual_connections, wrap_latent=residual_connections, adaptive_depth=True, rms_norm_eps=1e-6)
    values = torch.tensor([3., 1., 2., 0., 10. if prefer_think else -1.])
    calls = []

    def select(available, rows, depths):
        for row, depth, choices in zip(rows.tolist(), depths.tolist(), available, strict=True):
            assert bool(torch.isfinite(choices[:-1]).all()) == (depth >= minimum)
            assert bool(torch.isfinite(choices[-1])) == (depth < maximum)
            probabilities = torch.softmax(choices / .7, dim=-1)
            assert torch.isfinite(probabilities).all()
            assert probabilities.sum().item() == pytest.approx(1.)
            if depth < minimum:
                assert probabilities[:-1].count_nonzero() == 0
                assert probabilities[-1].item() == 1.
            if depth == maximum:
                assert probabilities[-1].item() == 0.
            calls.append((row, depth))
        return available.argmax(-1)

    tokens = batch(((2, 9, 3, 9), (4, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    generated = reasoner.generate(
        tokens, embedding=embedding, backbone=backbone,
        action_values=lambda z: values.expand(len(z), -1), select_actions=select,
        backend='continuous', max_active_requests=2, max_batch_tokens=3,
        cache_capacity_tokens=1024,
    )
    depth = maximum if prefer_think else minimum
    assert generated.lengths.tolist() == [depth] * 3
    assert sorted(calls) == [(row, step) for row in range(3) for step in range(1, depth + 1)]
    assert len(generated.tokens['token_ids']) == len(tokens['token_ids']) + 3 * (depth - 1 + (residual_connections and maximum > 1))
    torch.testing.assert_close(reasoner(generated, embedding=embedding, backbone=backbone),
                               generated.states[:, 1:], atol=3e-5, rtol=3e-5)
    if minimum > 1:
        invalid = replace(generated, lengths=generated.lengths.new_full((3,), minimum - 1))
        with pytest.raises(ValueError, match='Lengths|lengths'):
            reasoner(invalid, embedding=embedding, backbone=backbone)


@pytest.mark.parametrize('gamma', [0., .37, 1.])
@pytest.mark.parametrize('minimum,maximum,lengths', [(3, 4, [3, 4]), (3, 3, [3, 3])])
def test_bellman_backups_use_only_next_states_available_actions(gamma, minimum, maximum, lengths):
    q = (torch.arange(2 * maximum * 5).reshape(2, maximum, 5).float() / 17).requires_grad_()
    target_q = torch.arange(2 * maximum * 5).reshape(2, maximum, 5).float() / 11
    target_q[:, :minimum - 1, :-1] = 10000.  # Environment actions forbidden before minimum.
    target_q[:, -1, -1] = 20000.  # Think forbidden at cap.
    target_q.requires_grad_()
    environment_targets = torch.tensor([7., -3.], requires_grad=True)
    weights = torch.tensor([1., 2.])
    selected_actions = [2, 1]
    scale, shift, cost = 2.5, -.2, .13
    loss, details = value_reasoning_loss(
        q_values=q, target_q_values=target_q, lengths=torch.tensor(lengths),
        environment_actions=torch.tensor(selected_actions), environment_targets=environment_targets,
        weights=weights, reasoning_step_cost=cost, gamma_think=gamma, min_steps=minimum,
        value_scale=scale, value_shift=shift,
    )
    total_weight = sum(w * k for w, k in zip(weights.tolist(), lengths, strict=True))
    expected_loss = 0.
    expected_gradient = torch.zeros_like(q)
    for row, length in enumerate(lengths):
        for position in range(length):
            if position + 1 < length:
                action = 4
                next_depth = position + 2
                available = ([4] if next_depth < minimum else
                             list(range(4)) if next_depth == maximum else list(range(5)))
                target = -cost + gamma * max(scale * float(target_q[row, position + 1, a].detach()) + shift
                                            for a in available)
            else:
                action = selected_actions[row]
                target = float(environment_targets[row].detach())
            error = scale * float(q[row, position, action].detach()) + shift - target
            weight = float(weights[row]) / total_weight
            expected_loss += weight * error ** 2
            expected_gradient[row, position, action] = 2 * weight * scale * error
            assert details['td_target'][row, position].item() == pytest.approx(target, rel=2e-6)
    assert loss.item() == pytest.approx(expected_loss, rel=2e-6)
    loss.backward()
    torch.testing.assert_close(q.grad, expected_gradient)
    assert target_q.grad is None
    assert environment_targets.grad is None


@pytest.mark.parametrize('minimum', [0, True, 1.5, 5])
def test_minimum_must_be_an_integer_within_cap(minimum):
    with pytest.raises(ValueError, match='min_steps'):
        SequentialReasoner(hidden_dim=8, min_steps=minimum, max_steps=4,
                           residual_connections=False, wrap_latent=False, adaptive_depth=True, rms_norm_eps=1e-6)
    with pytest.raises(ValueError, match='min_steps'):
        value_reasoning_loss(q_values=torch.zeros(1, 4, 5), target_q_values=torch.zeros(1, 4, 5),
                             lengths=torch.tensor([4]), environment_actions=torch.tensor([0]),
                             environment_targets=torch.ones(1), weights=torch.ones(1),
                             reasoning_step_cost=.1, gamma_think=1., min_steps=minimum)


def test_loss_rejects_a_trajectory_stopping_before_minimum():
    with pytest.raises(ValueError, match='minimum'):
        value_reasoning_loss(q_values=torch.zeros(1, 4, 5), target_q_values=torch.zeros(1, 4, 5),
                             lengths=torch.tensor([2]), environment_actions=torch.tensor([0]),
                             environment_targets=torch.ones(1), weights=torch.ones(1),
                             reasoning_step_cost=.1, gamma_think=1., min_steps=3)
