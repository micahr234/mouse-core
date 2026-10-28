"""Actions must use the requesting step, never a previous step or padding."""
import numpy as np
import pytest
import torch

from mouse_core.data import StepTokens, pack_token_batch
from mouse_core.models import action_readouts, prepare_decode, select_discrete_action


def _inputs(*rows):
    steps, sequences = [], []
    for sequence, row in enumerate(rows):
        for heads in row:
            steps.append(StepTokens(
                token_ids=np.arange(len(heads)), numeric_values=np.zeros(len(heads)),
                numeric_mask=np.zeros(len(heads), dtype=bool), head_token_mask=np.array(heads, dtype=bool),
            ))
            sequences.append(sequence)
    return pack_token_batch(steps=steps, sequence_ids=sequences, batch_size=len(rows), continuing=None)[0]


def _scores(inputs, cached):
    # Each token has distinct scores; prepare_decode collapses multiple heads
    # per step and left-pads ragged/empty rows exactly as notebook models do.
    features = torch.arange(inputs.L * 3, dtype=torch.float32).reshape(-1, 3)
    if not cached:
        return features[inputs.head_token_ids]
    tensors = inputs.to_tensors('cpu')
    padded, _, heads, _ = prepare_decode(
        embeds=features, tokens=inputs, sequence_ids=tensors['sequence_ids'],
        head_token_ids=tensors['head_token_ids'], head_step_ids=tensors['head_step_ids'],
    )
    return padded.gather(1, heads.unsqueeze(-1).expand(-1, -1, 3))


@pytest.mark.parametrize('cached', [False, True])
def test_current_step_readouts_handle_sparse_heads_padding_and_requested_order(cached):
    inputs = _inputs([[True], [False], [True, True]], [], [[False]], [[True]])
    scores = _scores(inputs, cached)
    actual = action_readouts(scores, inputs, sequence_ids=[3, 0])
    packed = _scores(inputs, False)
    torch.testing.assert_close(actual, packed[[3, 2]])
    actions = select_discrete_action(scores=scores, inputs=inputs, sequence_ids=[3, 0], temperature=0.)
    assert actions.tolist() == [2, 2]
    assert action_readouts(scores, inputs, sequence_ids=[]).shape == (0, 3)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('last_step', [[False], []])
def test_older_readout_cannot_replace_missing_current_step(cached, last_step):
    inputs = _inputs([[True], last_step])
    with pytest.raises(ValueError, match='current step'):
        select_discrete_action(scores=_scores(inputs, cached), inputs=inputs, temperature=0.)


@pytest.mark.parametrize('cached', [False, True])
def test_passive_and_empty_rows_are_rejected_if_requested(cached):
    inputs = _inputs([[True]], [[False]], [])
    scores = _scores(inputs, cached)
    with pytest.raises(ValueError, match='current step'):
        action_readouts(scores, inputs, sequence_ids=[1])
    with pytest.raises(ValueError, match='without a current step'):
        action_readouts(scores, inputs, sequence_ids=[2])


@pytest.mark.parametrize('cached', [False, True])
def test_readout_layout_must_match_input_metadata(cached):
    inputs = _inputs([[True], [True]], [[True]])
    scores = _scores(inputs, cached)[:1]
    with pytest.raises(ValueError, match='scores must'):
        action_readouts(scores, inputs)
