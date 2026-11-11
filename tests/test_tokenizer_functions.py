"""User-defined format functions map actions without changing numeric labels."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from mouse_core.data import Tokenizer, TokenizerModalitySpec, load_tokenizer, save_tokenizer


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(char) for char in text]}


def action_name(action):
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError('action must be an integer')
    if not 0 <= action < 4:
        raise ValueError('action must be in [0, 4)')
    return ('A', 'B', 'C', 'D')[action]


def make_tokenizer(**field_options):
    return Tokenizer(tokenizers={'default': CharacterTokenizer()}, input_fields=[
        dict(provider='agent', type='text', input_field='action',
             format='[{action_name(action)}]', format_functions={'action_name': action_name}, **field_options),
        dict(provider='agent', type='token', input_field='action_requested', format='\n', head_output=True),
    ], objective_fields=[dict(input_field='action', missing_value=None)])


def test_function_roundtrip_preserves_events_and_numeric_labels(tmp_path):
    tok = make_tokenizer()
    events = [dict(provider='agent', step_index=i, payload=dict(action=i, action_requested=True)) for i in range(4)]
    original = deepcopy(events)
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'default': CharacterTokenizer()})
    for encoder in (tok, restored):
        steps = encoder.encode(events)
        assert [''.join(map(chr, step.token_ids)) for step in steps] == ['[A]\n', '[B]\n', '[C]\n', '[D]\n']
        assert [step.objective_fields['action'] for step in steps] == [0, 1, 2, 3]
        assert all(step.head_mask.tolist() == [False, False, False, True] for step in steps)
        assert encoder.special_token_ids == {}
        assert encoder(dict(provider='sensor', payload={'action': 1}))['token_ids'].size == 0
        assert encoder(dict(provider='agent', payload={'action': None}))['token_ids'].size == 0
    assert events == original


@pytest.mark.parametrize('value', [np.int64(2), torch.tensor(2), 2])
def test_function_receives_unwrapped_scalar(value):
    assert make_tokenizer()(dict(provider='agent', payload={'action': value}))['token_ids'].tolist() == list(map(ord, '[C]'))


def test_function_runs_after_vector_index_selection():
    piece = make_tokenizer(input_index=1)(dict(provider='agent', payload={'action': [0, 3]}))
    assert piece['token_ids'].tolist() == list(map(ord, '[D]'))


@pytest.mark.parametrize('value', [-1, 4, True, 1.0, '1'])
def test_function_validation_is_reported_with_format_context(value):
    with pytest.raises(ValueError, match='action must'):
        make_tokenizer()(dict(provider='agent', payload={'action': value}))


def test_lookup_option_is_removed():
    with pytest.raises(TypeError, match='lookup'):
        TokenizerModalitySpec(type='text', input_field='action', format='{action}', lookup=['A'])  # type: ignore[call-arg]
