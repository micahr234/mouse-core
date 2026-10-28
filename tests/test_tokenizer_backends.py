"""Native text encoding preserves HF semantics and event/field ordering."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import fields
import importlib

import numpy as np
import pytest
from tokenizers import Tokenizer as Backend
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from mouse_core.data import Tokenizer, load_tokenizer, save_tokenizer
from tests.test_sparse_event_readouts import backend, on_group_start


class WrapperBackend(PreTrainedTokenizerFast):
    """An overridden call forces the normal HF wrapper as a reference."""

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


def configured(*, wrapper=False):
    def source():
        value = backend()
        if wrapper:
            return WrapperBackend(tokenizer_object=value.backend_tokenizer, unk_token='[UNK]')
        return value

    return Tokenizer(tokenizers={'text': source(), 'other': source(), 'image': lambda value: value},
                    input_fields=[
        dict(type='text', tokenizer='text', format='prefix', when=on_group_start),
        dict(type='text', tokenizer='other', provider='agent', input_field='action',
             lookup=['obs', 'action'], format='{field}'),
        dict(type='text', tokenizer='text', provider='sensor', input_field='text', format='{field}'),
        dict(type='numeric', tokenizer='text', input_field='number', format='reward',
             fourier_min=0., fourier_max=2.),
        dict(type='token', tokenizer='text', input_field='request', format='<readout>',
             special=True, head_output=True),
        dict(type='image', tokenizer='image', input_field='image'),
    ], objective_fields=[dict(input_field='action', missing_value=None)])


def events():
    return [event for step in range(3) for event in (
        dict(provider='agent', step_index=step, payload={'action': step % 2}),
        dict(provider='sensor', step_index=step,
             payload={'text': 'obs', 'number': step / 2, 'request': True, 'image': [7, 8]}),
    )]


def assert_steps_equal(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        for field in fields(left):
            a, b = getattr(left, field.name), getattr(right, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b


def test_native_fields_and_steps_match_hf_wrapper_without_output_aliasing(monkeypatch):
    tok, reference = configured(), configured(wrapper=True)
    module = importlib.import_module('mouse_core.data.tokenizer')
    original = module._tokenize_ids
    wrapper_calls = []

    def track(source, text):
        wrapper_calls.append(source)
        return original(source, text)

    monkeypatch.setattr(module, '_tokenize_ids', track)
    for event in events():
        actual, expected = tok(event), reference(event)
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
    expected_steps = reference.encode(events())
    assert_steps_equal(tok.encode(events()), expected_steps)
    assert wrapper_calls and all(isinstance(source, WrapperBackend) for source in wrapper_calls)
    first = tok.encode(events())
    first[0].token_ids[:] = 999
    assert first[0].group_start_token_ids is not None
    first[0].group_start_token_ids[:] = 999
    assert_steps_equal(tok.encode(events()), expected_steps)


def bpe_backend():
    return PreTrainedTokenizerFast(tokenizer_object=Backend(BPE(
        vocab={'[UNK]': 0, 'a': 1, 'b': 2, 'ab': 3}, merges=[('a', 'b')],
        unk_token='[UNK]')), unk_token='[UNK]')


def test_mergeable_fields_and_events_remain_separate_and_ordered():
    source = bpe_backend()
    assert source('ab', add_special_tokens=False)['input_ids'] == [3]
    tok = Tokenizer(tokenizers={'text': source}, input_fields=[
        dict(type='text', input_field='first', format='{field}'),
        dict(type='text', input_field='second', format='{field}'),
    ])
    steps = tok.encode([
        dict(provider='sensor', step_index=0, payload={'second': 'b', 'first': 'a'}),
        dict(provider='sensor', step_index=0, payload={'first': 'a'}),
        dict(provider='sensor', step_index=0, payload={'second': 'b'}),
        dict(provider='sensor', step_index=1, payload={'first': 'ab'}),
    ])
    assert [step.token_ids.tolist() for step in steps] == [[1, 2, 1, 2], [3]]


def test_custom_stateful_callable_runs_in_field_and_event_order():
    calls = []

    def source(text, **kwargs):
        calls.append(text)
        return {'input_ids': [len(calls)]}

    tok = Tokenizer(tokenizers={'custom': source}, input_fields=[
        dict(type='text', format='first'), dict(type='text', format='second')])
    steps = tok.encode(events()[:2])
    assert calls == ['first', 'second', 'first', 'second']
    assert steps[0].token_ids.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize('method', [
    '__call__', '_encode_plus', '_convert_encoding',
    '_get_padding_truncation_strategies', 'set_truncation_and_padding', '_switch_to_input_mode',
])
def test_hf_overrides_and_input_mode_hooks_are_honored(method):
    calls = []
    original = getattr(PreTrainedTokenizerFast, method, None)

    def override(self, *args, **kwargs):
        calls.append(method)
        return None if original is None else original(self, *args, **kwargs)

    custom_type = type('CustomBackend', (PreTrainedTokenizerFast,), {method: override})
    source = custom_type(tokenizer_object=backend().backend_tokenizer, unk_token='[UNK]')
    tok = Tokenizer(tokenizers={'text': source}, input_fields=[
        dict(type='text', format='obs'), dict(type='text', format='action')])
    assert tok(dict(provider='sensor', payload={}))['token_ids'].tolist() == [1, 2]
    assert calls == [method, method]


def test_callback_backend_changes_take_effect_between_fields():
    source = backend()

    def add_word(ctx):
        if not ctx['group_start']:
            source.add_tokens(['new-word'])
        return not ctx['group_start']

    tok = Tokenizer(tokenizers={'text': source}, input_fields=[
        dict(type='text', format='new-word'),
        dict(type='text', format='new-word', when=add_word),
    ])
    assert tok(dict(provider='sensor', payload={}))['token_ids'].tolist() == [0, 5]
    assert tok(dict(provider='sensor', payload={}))['token_ids'].tolist() == [5, 5]


@pytest.mark.parametrize('setting', ['padding', 'truncation', 'split_special', 'native_split_special'])
def test_native_setting_mismatches_use_hf_wrapper(monkeypatch, setting):
    source = backend()
    source.backend_tokenizer.pre_tokenizer = Whitespace()
    source.add_tokens(['<special>'], special_tokens=True)
    tok = Tokenizer(tokenizers={'text': source}, input_fields=[
        dict(type='text', format='obs action <special>')])
    native = source.backend_tokenizer
    if setting == 'padding':
        native.enable_padding(length=8, pad_id=0, pad_token='[UNK]')
    elif setting == 'truncation':
        native.enable_truncation(max_length=1)
    elif setting == 'split_special':
        source.split_special_tokens = True
    else:
        native.encode_special_tokens = True
    expected = deepcopy(source)('obs action <special>', add_special_tokens=False)['input_ids']
    module = importlib.import_module('mouse_core.data.tokenizer')
    original = module._tokenize_ids
    calls = []

    def track(backend, text):
        calls.append(text)
        return original(backend, text)

    monkeypatch.setattr(module, '_tokenize_ids', track)
    for _ in range(2):
        assert tok(dict(provider='sensor', payload={}))['token_ids'].tolist() == expected
    assert calls == ['obs action <special>']
    assert native.padding is native.truncation is None
    assert native.encode_special_tokens == source.split_special_tokens


def test_native_bpe_dropout_changes_are_immediate():
    source = bpe_backend()
    tok = Tokenizer(tokenizers={'text': source}, input_fields=[dict(type='text', format='ab')])
    event = dict(provider='sensor', payload={})
    assert tok(event)['token_ids'].tolist() == [3]
    source.backend_tokenizer.model.dropout = 1.0
    assert tok(event)['token_ids'].tolist() == [1, 2]
    source.backend_tokenizer.model.dropout = None
    assert tok(event)['token_ids'].tolist() == [3]


def test_multiple_backends_copy_and_config_roundtrip(tmp_path):
    tok = configured()
    expected = tok.encode(events())
    clone = deepcopy(tok)
    assert clone.tokenizers['text'] is not tok.tokenizers['text']
    assert_steps_equal(clone.encode(events()), expected)
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers=tok.tokenizers)
    assert_steps_equal(restored.encode(events()), expected)
    event = dict(provider='sensor', payload={'text': 'new-word'})
    assert tok(event)['token_ids'].tolist() == [0]
    tok.tokenizers['text'].add_tokens(['new-word'])
    assert tok(event)['token_ids'].tolist() == [6]
    assert restored(event)['token_ids'].tolist() == [6]
    assert clone(event)['token_ids'].tolist() == [0]


def test_shared_native_backends_keep_concurrent_sequences_identical():
    tok = configured()
    expected = configured(wrapper=True).encode(events())
    with ThreadPoolExecutor(max_workers=4) as workers:
        for actual in workers.map(lambda _: tok.encode(events()), range(16)):
            assert_steps_equal(actual, expected)
