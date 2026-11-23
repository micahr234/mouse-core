"""Joining prepared batches must be equivalent to packing their episodes together."""
import pytest
import torch
from mouse_core.data import Tokenizer, concatenate_token_batches
from tests._event_fixtures import measurement_events
from tests._token_batch_helpers import IntIdTokenizer


def test_concatenation_matches_pack_with_empty_episodes_and_ragged_action_vectors():
    tokenizer = Tokenizer(
        input_fields=[dict(provider='sensor', type='text', input_field='action',
                           format='{action}', head_output="all")],
        objective_fields=[dict(input_field='reward'), dict(input_field='action'), dict(input_field='q')],
        tokenizers={'default': IntIdTokenizer()},
    )
    rows = [measurement_events([dict(action=1, reward=2., q=[1., 2.]),
                                dict(action=3, reward=4., q=[3., 4.])]),
            [], measurement_events([dict(action=2, reward=5., q=[5., 6., 7.])])]
    batches = [tokenizer.pack_rows(rows=rows[:2]), tokenizer.pack_rows(rows=rows[2:])]
    actual = concatenate_token_batches(batches)
    expected = tokenizer.pack_rows(rows=rows)
    for a, b in zip(actual, expected, strict=True):
        assert set(a.keys()) == set(b.keys())
        for key, value in a.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, b[key], equal_nan=True)
            else:
                assert a[key] == b[key]
    assert batches[0][0]['num_sequences'] == 2
    assert batches[1][0]['num_sequences'] == 1


def test_no_batches_is_not_a_valid_concatenation():
    with pytest.raises(ValueError, match='At least one'):
        concatenate_token_batches([])
