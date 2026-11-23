"""Final per-step markers, continuous episode splicing, and objective alignment."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from mouse_core.data import StepTokens, pack_token_batch, head_output_segments, empty_token_batch
from tests.test_sequential_reasoning import components, generate, reference_states


def segmented_batch():
    steps = []
    for i, (ids, heads) in enumerate([
        ([2, 9, 3, 9, 4], [1, 3]), ([6], []),
        ([5, 9, 7, 9, 8], [1, 3]), ([1, 2], []), ([4, 9], [1]),
    ]):
        mask = np.isin(np.arange(len(ids)), heads)
        steps.append(StepTokens(np.array(ids), np.zeros(len(ids)), np.zeros(len(ids), bool), mask,
            objective_fields=dict(action=i, reward=i * .5), autoregressive_mask=mask.copy(),
            tokenizer_names=('text',)))
    return pack_token_batch(steps=steps, sequence_ids=[0, 0, 0, 0, 2], batch_size=3)


def test_step_segments_preserve_separated_heads_passive_steps_tails_and_empty_rows():
    tokens, data = segmented_batch()
    assert tokens['head_mask'].nonzero().flatten().tolist() == [1, 3, 7, 9, 14]
    assert tokens['step_readout_mask'].nonzero().flatten().tolist() == [3, 9, 14]
    assert data['step_ids'].tolist() == [0, 0, 2, 2, 4]
    segments = head_output_segments(tokens)
    assert [s.head_token_ids for s in segments] == [(3, 9), (), (14,)]
    assert [[(s.start, s.stop) for s in e.context_slices] for e in segments] == [
        [(0, 3), (4, 9), (10, 13)], [(13, 13)], [(13, 14), (15, 15)]]
    restored = []
    for episode in segments:
        for i, context in enumerate(episode.context_slices):
            restored.extend(tokens['token_ids'][context].tolist())
            if i < len(episode.head_token_ids):
                restored.append(int(tokens['token_ids'][episode.head_token_ids[i]]))
    assert restored == tokens['token_ids'].tolist()
    assert len(head_output_segments(empty_token_batch(num_sequences=2))) == 2


@pytest.mark.parametrize('residual_connections', [False, True])
@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_multiple_head_tokens_reason_once_per_step_with_literal_history_reference(residual_connections, backend):
    tokens, data = segmented_batch()
    parts = components(residual_connections=residual_connections, wrap_latent="step" if residual_connections else "none", adaptive_depth=True, cap=3, real=True)
    depths = torch.tensor([2, 3, 1])
    trace = generate(parts, tokens, lengths=depths, backend=backend)
    assert trace.head_token_ids == (3, 9, 14)
    assert trace.lengths.tolist() == [2, 3, 1]
    expected_tokens = tokens.clone()
    expected_tokens['head_mask'] = tokens['step_readout_mask'].clone()
    expected = reference_states(parts, expected_tokens, trace)
    torch.testing.assert_close(trace.states[:, 1:], expected, atol=3e-5, rtol=3e-5)
    actual = parts[2](trace, embedding=parts[0], backbone=parts[1])
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    # All non-marker tokens, including earlier readout tokens, remain in order.
    assert trace.tokens['token_ids'][~trace.tokens['latent_mask']].tolist() == tokens['token_ids'][~tokens['step_readout_mask']].tolist()
    assert int(trace.tokens['step_readout_mask'].sum()) == 3
    aligned = trace.align_objective_data(tokens, data)
    assert aligned['step_ids'].tolist() == [0, 2, 4]
    assert data['step_ids'].tolist() == [0, 0, 2, 2, 4]
    torch.testing.assert_close(aligned['action'], data['action'])
    torch.testing.assert_close(aligned['reward'], data['reward'])
    with pytest.raises(ValueError, match='Source head-output rows'):
        trace.align_objective_data(tokens, aligned)
    with pytest.raises(ValueError, match='original head-output'):
        replace(trace, head_token_ids=(0,)).align_objective_data(tokens, data)


def test_missing_or_invalid_step_boundaries_are_rejected():
    tokens, _ = segmented_batch()
    tokens['step_readout_mask'][0] = True
    with pytest.raises(ValueError, match='select head-output'):
        head_output_segments(tokens)
    tokens['step_readout_mask'].zero_()
    with pytest.raises(ValueError, match='explicit step_readout_mask'):
        head_output_segments(tokens)
