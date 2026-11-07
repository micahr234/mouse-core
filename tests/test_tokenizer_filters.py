"""Provider/field filters and named expressions across encoding entry points."""
import json

import numpy as np
import pytest

from mouse_core.data import Tokenizer, TokenizerModalitySpec, load_tokenizer, save_tokenizer
from tests.test_tokenizer_functions import CharacterTokenizer, action_name


def tokenizer(*fields):
    return Tokenizer(tokenizers={'text': CharacterTokenizer()}, input_fields=list(fields))


def text(tok, payload, provider='sensor'):
    return ''.join(map(chr, tok(dict(provider=provider, payload=payload))['token_ids']))


def test_wildcard_provider_preserves_declaration_order_without_duplicate_emissions():
    tok = tokenizer(dict(type='text', format='a'), dict(type='text', provider='sensor', format='b'),
                    dict(type='text', format='c'), dict(type='text', provider='agent', format='d'))
    assert text(tok, {}) == 'abc'
    assert text(tok, {}, 'agent') == 'acd'
    assert text(tok, {}, 'new-provider') == 'ac'


def test_omitted_input_exposes_payload_names_and_does_not_inject_field_alias():
    tok = tokenizer(dict(type='text', format='{observation}:{reward:.2f}:{field}'))
    assert text(tok, {'observation': 3, 'reward': 1.25, 'field': 'real'}) == '3:1.25:real'
    with pytest.raises(ValueError, match='observation'):
        text(tok, {})


def test_sparse_multi_field_filter_matches_any_present_value():
    tok = tokenizer(dict(type='text', input_field=['terminated', 'truncated'],
                         format="{'END' if terminated or truncated else 'CONTINUE'}"))
    assert text(tok, {}) == ''
    assert text(tok, {'terminated': None, 'truncated': None}) == ''
    assert text(tok, {'terminated': False}) == 'CONTINUE'
    assert text(tok, {'terminated': True}) == 'END'
    assert text(tok, {'truncated': True}) == 'END'
    assert text(tok, {'terminated': True, 'truncated': True}) == 'END'


@pytest.mark.parametrize('value', [False, 0, '', [], {}])
def test_presence_filter_accepts_falsy_and_structured_values(value):
    assert text(tokenizer(dict(type='text', input_field='value', format='present')), {'value': value}) == 'present'


def test_named_multifield_expression_and_indexing_preserve_payload():
    payload = {'observation': [1, 3], 'reward': 2.5}
    tok = tokenizer(dict(type='text', input_field=['observation', 'reward'], format='{observation[1]+reward:.1f}'))
    assert text(tok, payload) == '5.5'
    assert payload == {'observation': [1, 3], 'reward': 2.5}


def test_constant_compute_and_readout_share_filter_and_stream_matches_encode():
    tok = tokenizer(dict(type='text', input_field='action_requested', format=' . .'),
                    dict(type='token', input_field='action_requested', format='\n', head_output=True))
    events = [dict(provider='environment', step_index=i, payload=payload)
              for i, payload in enumerate([{'action_requested': True}, {}, {'action_requested': True}])]
    steps = tok.encode(events)
    assert [s.token_ids.tolist() for s in steps] == [[32, 46, 32, 46, 10], [], [32, 46, 32, 46, 10]]
    assert steps[0].head_token_mask.tolist() == [False] * 4 + [True]
    stream = tok.stream()
    streamed = [step for event in events if (step := stream.push(event)) is not None] + [stream.flush()]
    for a, b in zip(steps, streamed):
        assert b is not None
        np.testing.assert_array_equal(a.token_ids, b.token_ids)
    inputs, objectives = tok.pack_rows(rows=[events])
    assert inputs['token_ids'].tolist() == [32, 46, 32, 46, 10] * 2
    assert objectives['head_step_ids'].tolist() == [0, 2]


def test_multifield_wildcard_functions_roundtrip(tmp_path):
    tok = tokenizer(dict(type='text', input_field=['action', 'reward'], format='{action_name(action)}:{reward:g}',
                         format_functions={'action_name': action_name}))
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'text': CharacterTokenizer()})
    assert text(restored, {'action': 2, 'reward': 1.5}, 'unknown') == 'C:1.5'
    config = json.loads((tmp_path/'tokenizer.json').read_text())
    assert config['input_fields'][0]['input_field'] == ['action', 'reward']
    config['format'] = 'mouse-core-tokenizer-v4'
    (tmp_path/'tokenizer.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='Unsupported tokenizer config'):
        load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'text': CharacterTokenizer()})


@pytest.mark.parametrize('input_field', ['', [], ['x', 'x'], [''], ['x', 3], 4])
def test_invalid_filters_fail_at_construction(input_field):
    with pytest.raises(ValueError, match='input_field'):
        TokenizerModalitySpec(type='text', input_field=input_field, format='x')


@pytest.mark.parametrize('provider', ['', ' ', 1])
def test_invalid_provider_filter(provider):
    with pytest.raises(ValueError, match='provider'):
        TokenizerModalitySpec(type='text', provider=provider, format='x')


@pytest.mark.parametrize('format,expected', [
    ("{sum(values) / len(values):.2f}", "2.00"),
    ("{' '.join(str(x * scale) for x in values)}", "3 6 9"),
    ("{[x * scale for x in values if x > 1]}", "[6, 9]"),
    ("{sorted({x % 2 for x in values})}", "[0, 1]"),
    ("{{x: x * scale for x in values}}", "{x: x * scale for x in values}"),
    ("{ {x: x * scale for x in values} }", "{1: 3, 2: 6, 3: 9}"),
    ("{[[x + y + scale for y in values] for x in values[:1]]}", "[[5, 6, 7]]"),
    ("{(lambda factor: [x * factor for x in values])(scale)}", "[3, 6, 9]"),
    ("{[(lambda: x + scale)() for x in values]}", "[4, 5, 6]"),
    ("{(total := sum(values))}:{(lambda: total * scale)()}", "6:18"),
    ("{values.count(2)}:{'hello'.upper()}", "1:HELLO"),
])
@pytest.mark.parametrize('input_field', [None, ['values', 'scale']])
def test_formats_use_python_expressions_and_lexical_scopes(format, expected, input_field):
    tok = tokenizer(dict(type='text', input_field=input_field, format=format))
    assert text(tok, {'values': [1, 2, 3], 'scale': 3}) == expected


def test_comprehension_functions_roundtrip_and_use_each_events_values(tmp_path):
    tok = tokenizer(dict(type='text', input_field=['actions', 'offset'],
                         format="{' '.join(action_name(x + offset).lower() for x in actions)}",
                         format_functions={'action_name': action_name}))
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'text': CharacterTokenizer()})
    for encoder in (tok, restored):
        assert text(encoder, {'actions': [0, 1], 'offset': 1}) == 'b c'
        assert text(encoder, {'actions': [0, 1], 'offset': 2}) == 'c d'


@pytest.mark.parametrize('format', ['{[x + secret for x in values]}', '{(lambda: secret)()}',
                                    '{[x for x in values]}:{x}'])
def test_python_scope_validation_still_rejects_unselected_names(format):
    with pytest.raises(ValueError, match='not selected'):
        TokenizerModalitySpec(type='text', input_field='values', format=format)


def test_format_namespace_is_fresh_and_does_not_inject_module_globals():
    tok = tokenizer(dict(type='text', format="{locals().get('previous', 'fresh')}:{(previous := value)}"))
    payload = {'value': 3}
    assert text(tok, payload) == 'fresh:3'
    assert text(tok, {'value': 4}) == 'fresh:4'
    assert payload == {'value': 3}
    tok = tokenizer(dict(type='text', format='{math.pi}'))
    with pytest.raises(ValueError, match='math'):
        text(tok, {})


def test_event_bindings_shadow_builtins_as_in_python():
    tok = tokenizer(dict(type='text', input_field='sum', format='{sum.upper()}'))
    assert text(tok, {'sum': 'event value'}) == 'EVENT VALUE'


def test_registered_function_bindings_validate_and_reject_collisions(tmp_path):
    with pytest.raises(TypeError, match='callable'):
        TokenizerModalitySpec(type='text', format='x', format_functions={'fn': 3})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='conflicts'):
        TokenizerModalitySpec(type='text', input_field='action', format='x', format_functions={'action': action_name})
    tok = tokenizer(dict(type='text', format='{action_name(action)}', format_functions={'action_name': action_name}))
    with pytest.raises(ValueError, match='conflict'):
        text(tok, {'action': 0, 'action_name': lambda _: 'spoof'})
    tok = tokenizer(dict(type='text', format='{fn(1)}', format_functions={'fn': lambda _: 'x'}))
    with pytest.raises(TypeError, match='named top-level'):
        save_tokenizer(tokenizer=tok, path=tmp_path)


def test_image_filters_accept_multiple_and_all_inputs():
    backend = lambda value: value
    for selection in (['left', 'right'], None):
        tok = Tokenizer(tokenizers={'image': backend}, input_fields=[dict(type='image', input_field=selection)])
        assert tok(dict(provider='camera', payload={'left': [1, 2], 'right': [3]}))['token_ids'].tolist() == [1, 2, 3]


def test_special_tokens_still_require_literal_formats():
    for fmt in ('{action}', '{1+1}', ''):
        with pytest.raises(ValueError, match='literal'):
            TokenizerModalitySpec(type='token', special=True, format=fmt)


@pytest.mark.parametrize("format,expected", [
    ("{'\\n\\n' if ended else ''}", "\n\n"),
    ("{'a' if ended else \"b\"}", "a"),
    ("literal\\n{'x'}\n", "literal\\nx\n"),
    ("{{literal}} {value!r}", "{literal} 'yes'"),
    ("{value:{width}.2f}", " 1.25"),
])
def test_format_quotes_escapes_and_nested_specs(format, expected):
    payload = dict(ended=True, value=1.25 if "width" in format else "yes", width=5)
    assert text(tokenizer(dict(type="text", format=format)), payload) == expected


def transition_text(action, reward, *, prefix):
    return f'{prefix}{action_name(action)}:{reward:.1f}'


def test_format_function_receives_multiple_inputs_and_keywords():
    tok = tokenizer(dict(type='text', input_field=['action', 'reward'],
                         format="{transition(action, reward, prefix='step:')}",
                         format_functions={'transition': transition_text}))
    assert text(tok, {'action': 1, 'reward': 2.5}) == 'step:B:2.5'


@pytest.mark.parametrize('workers', [0, 2])
def test_format_functions_survive_loader_worker_serialization(workers):
    from mouse_core.data import DataLoader, Datastore

    store = Datastore()
    store.extend([dict(provider='sensor', episode_index=i, step_index=0,
                       payload={'action': i, 'reward': 2.5}) for i in range(4)])
    tok = tokenizer(dict(type='text', input_field=['action', 'reward'],
                         format="{''.join(transition(action, reward, prefix='') for _ in range(1))}",
                         format_functions={'transition': transition_text}, head_output=True))
    with DataLoader(stores=store, events_budget=4, sample_field='episode_index',
                    transform=tok.encode, num_workers=workers, prefetch=1, seed=7) as loader:
        inputs, _ = loader.next_batch()
        rendered = ''.join(map(chr, inputs['token_ids'].tolist()))
        # Sampling is with replacement; each sampled event must still produce
        # one complete function result, in both synchronous and worker modes.
        chunks = [rendered[index:index+5] for index in range(0, len(rendered), 5)]
        assert len(chunks) == 4
        assert all(chunk in {'A:2.5', 'B:2.5', 'C:2.5', 'D:2.5'} for chunk in chunks)
        assert inputs['head_token_mask'].all()
