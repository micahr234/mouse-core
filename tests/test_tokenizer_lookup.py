"""Lookup renders action text without changing replay or training labels."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from mouse_core.data import Tokenizer, TokenizerModalitySpec, load_tokenizer, save_tokenizer


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(char) for char in text]}


def make_tokenizer(**field_options):
    return Tokenizer(tokenizer=CharacterTokenizer(), input_fields=[
        dict(type='text', input_field='action', lookup=['A', 'B', 'C', 'D'],
             format='[{field}]', **field_options),
        dict(type='token', input_field='action_requested', format='\n', head_output=True),
    ], objective_fields=[dict(input_field='action', missing_value=None)])


def test_lookup_roundtrip_preserves_events_and_numeric_labels(tmp_path):
    tok = make_tokenizer()
    events = [dict(provider='agent', step_index=i, payload=dict(action=i, action_requested=True))
              for i in range(4)]
    original = deepcopy(events)
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=CharacterTokenizer())
    for encoder in (tok, restored):
        steps = encoder.encode(events)
        assert [''.join(map(chr, step.token_ids)) for step in steps] == ['[A]\n', '[B]\n', '[C]\n', '[D]\n']
        assert [step.objective_fields['action'] for step in steps] == [0, 1, 2, 3]
        assert all(step.head_token_mask.tolist() == [False, False, False, True] for step in steps)
        assert encoder.special_token_ids == {}
        assert encoder(dict(provider='sensor', payload={}))['token_ids'].size == 0
        assert encoder(dict(provider='sensor', payload={'action': None}))['token_ids'].size == 0
    assert events == original


@pytest.mark.parametrize('value', [np.int64(2), torch.tensor(2), 2])
def test_lookup_accepts_integer_scalars(value):
    piece = make_tokenizer()(dict(provider='agent', payload={'action': value}))
    assert piece['token_ids'].tolist() == list(map(ord, '[C]'))


def test_lookup_runs_after_vector_index_selection():
    piece = make_tokenizer(input_index=1)(dict(provider='agent', payload={'action': [0, 3]}))
    assert piece['token_ids'].tolist() == list(map(ord, '[D]'))


@pytest.mark.parametrize('value,error', [(-1, ValueError), (4, ValueError), (True, TypeError),
                                        (1.0, TypeError), ('1', TypeError)])
def test_invalid_lookup_ids_raise(value, error):
    with pytest.raises(error, match='lookup field'):
        make_tokenizer()(dict(provider='agent', payload={'action': value}))


@pytest.mark.parametrize('lookup', [[], 'ABCD', ['A', 1], ['A', '']])
def test_invalid_lookup_tables_raise(lookup):
    with pytest.raises(ValueError, match='lookup'):
        TokenizerModalitySpec(type='text', input_field='action', format='{field}', lookup=lookup)


@pytest.mark.parametrize('kind,field', [('token', 'action'), ('numeric', 'action'), ('image', 'action'), ('text', None)])
def test_lookup_requires_input_text(kind, field):
    with pytest.raises(TypeError, match='lookup requires a text field'):
        TokenizerModalitySpec(type=kind, input_field=field, format='A', lookup=['A'])
