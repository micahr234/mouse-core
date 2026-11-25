"""Queued complete episodes reuse cache rows without sharing their history."""
import pytest
import torch
from tests.test_residual_reasoning import batch
from tests.test_sequential_reasoning import components, generate


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('wrap', ['none', 'decision', 'sequence'])
@torch.no_grad()
def test_queued_episode_refills_match_uncached_histories(residual, wrap):
    parts = components(latent_residual_enabled=residual, latent_feedback_scope=wrap,
                       adaptive_depth=True, cap=3, real=True)
    # Five episodes share two active slots. They finish at different depths
    # and have different numbers of environment decisions and context tokens.
    tokens = batch(((2, 9, 3, 9), (4, 9), (5, 9, 6, 9, 7, 9),
                    (2, 3, 4, 9), (6, 9, 8, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    lengths = torch.tensor([1, 3, 2, 1, 2, 3, 3, 2, 1])
    cached = generate(parts, tokens, lengths=lengths, backend='continuous')
    reference = generate(parts, tokens, lengths=lengths, backend='uncached')
    assert cached.head_token_ids == reference.head_token_ids
    torch.testing.assert_close(cached.lengths, lengths)
    torch.testing.assert_close(cached.update_positions, reference.update_positions)
    torch.testing.assert_close(cached.states, reference.states, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(cached.tokens['latent_values'], reference.tokens['latent_values'],
                               atol=3e-5, rtol=3e-5)
