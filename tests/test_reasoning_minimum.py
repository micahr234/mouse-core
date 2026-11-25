"""Independent checks of minimum/maximum reasoning action availability."""

from dataclasses import replace

import pytest
import torch
from mouse_core.models import ThinkExecuteValues

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
    reasoner = SequentialReasoner(input_mlp_intermediate_dim=24, input_mlp_output_scale=100., hidden_dim=8, min_steps=minimum, max_steps=maximum,
                                 residual_connections=residual_connections, wrap_latent="step" if residual_connections else "none", adaptive_depth=True, rms_norm_eps=1e-6)
    actions = torch.tensor([3., 1., 2., 0.])
    gate = torch.tensor([3., 10. if prefer_think else -1.])
    calls = []

    def select(available, rows, depths):
        for row, depth, choices in zip(rows.tolist(), depths.tolist(), available.available(), strict=True):
            assert bool(choices[:-1].all()) == (depth >= minimum)
            assert bool(choices[-1]) == (depth < maximum)
            probabilities = available.policy_logits(torch.full((len(rows),), .7))[rows.tolist().index(row)].exp()
            assert torch.isfinite(probabilities).all()
            assert probabilities.sum().item() == pytest.approx(1.)
            if depth < minimum:
                assert probabilities[:-1].count_nonzero() == 0
                assert probabilities[-1].item() == 1.
            if depth == maximum:
                assert probabilities[-1].item() == 0.
            calls.append((row, depth))
        return available.greedy()

    tokens = batch(((2, 9, 3, 9), (4, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    generated = reasoner.generate(
        tokens, embedding=embedding, backbone=backbone,
        action_values=lambda z: ThinkExecuteValues(actions.expand(len(z), -1), gate.expand(len(z), -1)), select_actions=select,
        backend='continuous', max_active_requests=2, max_batch_tokens=3,
        cache_capacity_tokens=1024,
    )
    depth = maximum if prefer_think else minimum
    assert generated.lengths.tolist() == [depth] * 3
    assert sorted(calls) == [(row, step) for row in range(3) for step in range(1, depth + 1)]
    assert len(generated.tokens['token_ids']) == len(tokens['token_ids']) + 3 * (depth - 1)
    torch.testing.assert_close(reasoner(generated, embedding=embedding, backbone=backbone),
                               generated.states[:, 1:], atol=3e-5, rtol=3e-5)
    if minimum > 1:
        invalid = replace(generated, lengths=generated.lengths.new_full((3,), minimum - 1))
        with pytest.raises(ValueError, match='Lengths|lengths'):
            reasoner(invalid, embedding=embedding, backbone=backbone)


@pytest.mark.parametrize('minimum', [0, True, 1.5, 5])
def test_minimum_must_be_an_integer_within_cap(minimum):
    with pytest.raises(ValueError, match='min_steps'):
        SequentialReasoner(input_mlp_intermediate_dim=24, input_mlp_output_scale=100., hidden_dim=8, min_steps=minimum, max_steps=4,
                           residual_connections=False, wrap_latent="none", adaptive_depth=True, rms_norm_eps=1e-6)
    with pytest.raises(ValueError, match='min_steps'):
        value_reasoning_loss(q_values=ThinkExecuteValues(torch.zeros(1, 4, 4), torch.zeros(1, 4, 2)), target_q_values=ThinkExecuteValues(torch.zeros(1, 4, 4), torch.zeros(1, 4, 2)),
                             lengths=torch.tensor([4]), environment_actions=torch.tensor([0]),
                             environment_targets=torch.ones(1), weights=torch.ones(1),
                             reasoning_reward=-.1, gamma_think=1., min_steps=minimum)


def test_loss_rejects_a_trajectory_stopping_before_minimum():
    with pytest.raises(ValueError, match='minimum'):
        value_reasoning_loss(q_values=ThinkExecuteValues(torch.zeros(1, 4, 4), torch.zeros(1, 4, 2)), target_q_values=ThinkExecuteValues(torch.zeros(1, 4, 4), torch.zeros(1, 4, 2)),
                             lengths=torch.tensor([2]), environment_actions=torch.tensor([0]),
                             environment_targets=torch.ones(1), weights=torch.ones(1),
                             reasoning_reward=-.1, gamma_think=1., min_steps=3)
