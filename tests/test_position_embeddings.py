"""Named numeric projections, bounded positions, and explicit provider routing."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer as Backend
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from mouse_core.data import Tokenizer, TokenizerModalitySpec, load_tokenizer, save_tokenizer
from mouse_core.models import Embedding, MultiEmbedding


def group_start(ctx):
    return ctx['group_start']


def tokenizer():
    backend = Backend(WordLevel({'[UNK]': 0, 'a': 1, 'b': 2, 'n': 3, 'head': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    position = dict(position_field='context', position_fourier_min=-1., position_fourier_max=1.)
    return Tokenizer(tokenizers={'text': PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')}, input_fields=[
        dict(type='text', provider='sensor', input_field='prefix', format='{field}', when=group_start, **position),
        dict(type='text', provider='sensor', input_field='reading', format='a {field} b', **position),
        dict(type="token", provider='sensor', input_field='position', numeric_field='position', format='n', fourier_min=0., fourier_max=2., **position),
        dict(type="token", provider='sensor', input_field='angle', numeric_field='angle', format='n', fourier_min=0., fourier_max=2., **position),
        dict(type='token', provider='sensor', input_field='request', format='head', head_output=True),
    ])


def events():
    return [dict(provider='sensor', episode_index=0, step_index=step, payload=payload) for step, payload in [
        (0, dict(prefix='a b', reading='a b', context=-.5, position=1.5, angle=1.5, request=True)),
        (0, dict(reading='b', context=0.)),
        (1, dict(prefix='b', reading='a', context=None, request=True)),
        (2, dict(reading='a', context=.9, request=True)),
    ]]


def test_field_scope_multitoken_and_optional_position():
    tok = tokenizer()
    piece = tok(events()[0])
    assert piece['position_values'].tolist() == [-.5] * 6 + [0]
    assert piece['position_mask'].tolist() == [True] * 6 + [False]
    assert piece['numeric_field_ids'].tolist() == [-1] * 4 + [0, 1, -1]
    assert piece['group_start_position_values'].tolist() == [-.5, -.5]
    assert piece['group_start_position_mask'].all()
    assert not tok(events()[2])['position_mask'].any()
    zero = tok(events()[1])
    assert zero['position_mask'].all() and (zero['position_values'] == 0).all()
    assert tok(dict(provider='sensor', payload={'context': 9}))['token_ids'].size == 0
    assert not tok(dict(provider='different', payload=events()[0]['payload']))['token_ids'].size


def test_stream_pack_prefix_slice_and_saved_field_association(tmp_path):
    tok = tokenizer()
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers=tok.tokenizers)
    assert [s.position_field for s in restored.input_fields] == [s.position_field for s in tok.input_fields]
    packed, packed_objective_data = tok.pack_rows(rows=[events()], continuing=None)
    copied, copied_objective_data = restored.pack_rows(rows=[events()], continuing=None)
    names = ('token_ids', 'position_values', 'position_mask', 'numeric_field_ids')
    assert packed['numeric_field_names'] == copied['numeric_field_names'] == tok.numeric_field_names
    for name in names:
        np.testing.assert_array_equal(packed[name], copied[name])
    first, first_objective_data = tok.pack_rows(rows=[events()[:2]], continuing=None)
    continued, continued_objective_data = tok.pack_rows(rows=[events()[2:]], continuing=[True])
    for name in names:
        np.testing.assert_array_equal(packed[name], np.concatenate([first[name], continued[name]]))
    assert packed['position_values'].dtype == torch.float64
    empty, empty_objective_data = tok.pack_rows(rows=[[]], continuing=None)
    assert empty['numeric_field_names'] == tok.numeric_field_names
    assert empty['position_mask'].numel() == 0
    stream = tok.stream(include_objectives=False)
    steps = []
    for event in events():
        step = stream.push(event)
        if step is not None: steps.append(step)
    steps.append(stream.flush())
    for actual, expected in zip(steps, tok.encode(events(), include_objectives=False), strict=True):
        for name in names:
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '7', [7]])
def test_invalid_position_rejected(bad):
    with pytest.raises(ValueError, match='numeric scalar'):
        tokenizer()(dict(provider='sensor', payload={'reading': 'a', 'context': bad}))


@pytest.mark.parametrize('bounds', [{}, {'position_fourier_min': 0.}, {'position_fourier_max': 1.},
    {'position_fourier_min': 1., 'position_fourier_max': 1.},
    {'position_fourier_min': 2., 'position_fourier_max': 1.},
    {'position_fourier_min': float('-inf'), 'position_fourier_max': 1.},
    {'position_fourier_min': 0., 'position_fourier_max': float('nan')}])
def test_position_requires_valid_explicit_bounds(bounds):
    with pytest.raises(ValueError):
        TokenizerModalitySpec(provider='sensor', type='text', input_field='reading', format='{field}',
                              position_field='context', **bounds)


def test_position_bounds_normalize_independently_without_clipping():
    tok = tokenizer()
    field = replace(tok.input_fields[2], position_fourier_min=-40., position_fourier_max=60.)
    encoded = Tokenizer(tokenizers=tok.tokenizers, input_fields=[field])
    for context, normalized in [(-40., -1.), (10., 0.), (60., 1.), (110., 2.)]:
        result = encoded(dict(provider='sensor', payload={'position': 1.5, 'context': context}))
        assert result['numeric_values'].tolist() == [.5]
        assert result['position_values'].tolist() == [normalized]
    with pytest.raises(ValueError, match='require position_field'):
        replace(field, position_field=None)


@pytest.mark.parametrize('provider', [None, '', ' ', 1])
def test_provider_must_be_specific(provider):
    with pytest.raises(ValueError, match='provider'):
        TokenizerModalitySpec(provider=provider, type='text', format='a')
    with pytest.raises(ValueError, match='provider'):
        tokenizer()(dict(provider=provider, payload={'reading': 'a'}))


def test_provider_cannot_be_omitted():
    with pytest.raises(TypeError, match='provider'):
        Tokenizer(tokenizers=tokenizer().tokenizers, input_fields=[dict(type='text', format='a')])


def test_separate_numeric_fields_share_only_the_position_and_both_learn():
    model = Embedding(vocab_size=6, hidden_dim=8, num_frequencies=3,
                      numeric_field_names=('position', 'angle'), position_num_frequencies=4)
    ids = torch.tensor([1, 1, 1, 1])
    values = torch.tensor([.5, .5, .5, 0.])
    mask = torch.tensor([True, True, True, False])
    routes = torch.tensor([0, 1, 0, -1])
    position = torch.tensor([-.5, -.5, -.5, 0.], dtype=torch.float64)
    args = dict(numeric_values=values, numeric_mask=mask, numeric_field_ids=routes,
                numeric_field_names=('position', 'angle'))
    base = model(ids, **args)
    timed = model(ids, **args, position_values=position, position_mask=mask)
    assert len(model.features.numeric) == 2
    assert model.features.numeric[0].proj.weight is not model.features.numeric[1].proj.weight
    torch.testing.assert_close(base[0], base[2], rtol=0, atol=0)
    assert not torch.allclose(base[0], base[1])
    torch.testing.assert_close(timed[mask], base[mask] + model.features.position(position[mask]))
    torch.testing.assert_close(timed[3], base[3], rtol=0, atol=0)
    timed.square().sum().backward()
    for projection in (*model.features.numeric, model.features.position):
        assert projection.proj.weight.grad.abs().sum() > 0
    endpoints = model.features.position(torch.tensor([-1., 1.], dtype=torch.float64))
    assert not torch.allclose(endpoints[0], endpoints[1])
    with pytest.raises(ValueError, match='identities'):
        model(ids, **{**args, 'numeric_field_names': ('angle', 'position')})


def test_numeric_identity_reused_across_events_and_vector_elements():
    source = tokenizer()
    fields = [dict(type="token", provider='sensor', input_field='position', numeric_field='position', numeric_index=i, input_index=i,
                   format='n', fourier_min=-1., fourier_max=1.) for i in range(2)]
    tok = Tokenizer(tokenizers=source.tokenizers, input_fields=fields)
    assert len(tok.numeric_field_names) == 1
    for value in ([.1, .2], [.8, -.5]):
        piece = tok(dict(provider='sensor', payload={'position': value}))
        assert piece['numeric_field_ids'].tolist() == [0, 0]


def test_one_position_projection_shared_across_named_token_tables():
    model = MultiEmbedding(embeddings={name: Embedding(vocab_size=5, hidden_dim=8) for name in ('a', 'b')},
                           position_num_frequencies=2)
    assert all(table.features.position is None for table in model.embeddings.values())
    ids = torch.tensor([1, 1]); routes = torch.tensor([0, 1]); values = torch.tensor([.2, .2], dtype=torch.float64)
    plain = model(ids, tokenizer_ids=routes, tokenizer_names=('a', 'b'))
    actual = model(ids, tokenizer_ids=routes, tokenizer_names=('a', 'b'),
                   position_values=values, position_mask=torch.tensor([True, True]))
    torch.testing.assert_close(actual, plain + model.features.position(values))
    torch.testing.assert_close(actual[0] - plain[0], actual[1] - plain[1])
    assert sum('position.proj.weight' in name for name, _ in model.named_parameters()) == 1


@pytest.mark.flex_training(default_device=False)
def test_offline_dqn_time_notebook_runs_local_training_and_saves(tmp_path):
    from transformers import Qwen3Config, Qwen3Model
    from mouse_core.models import load_model
    base = tmp_path / 'base'
    backend = Backend(WordLevel({'?': 0, '\n': 1, 'action': 2, 'obs': 3, 'reward': 4, 'same': 5,
                                'sensor': 6, 'reading': 7, '0': 8, '1': 9}, unk_token='?'))
    backend.pre_tokenizer = Whitespace()
    # Keep the request newline as one existing token while splitting prose.
    from tokenizers import AddedToken
    text = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='?')
    text.add_tokens([AddedToken('\n', normalized=False)])
    text.save_pretrained(base)
    Qwen3Model(Qwen3Config(hidden_size=16, vocab_size=len(text), num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                          intermediate_size=32)).save_pretrained(base)
    notebook = Path(__file__).resolve().parents[1] / 'examples/22_train_offline_dqn_time.ipynb'
    ns = {'__name__': '__main__'}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for index, cell in enumerate(json.loads(notebook.read_text())['cells']):
            if cell['cell_type'] != 'code':
                continue
            exec(compile(''.join(cell['source']), f'{notebook}:cell{index}', 'exec'), ns)
            if index == 1:
                ns.update(PRETRAINED=str(base), TOKEN_BUDGET=1024, TRAIN_STEPS=1, NUM_CYCLES=1,
                          OUTPUT_DIR=tmp_path / 'checkpoint', device=torch.device('cuda'))
        assert torch.isfinite(ns['loss'])
        time_projection = ns['encoder'].embedding.features.position.proj
        assert time_projection.weight.grad is not None
        assert time_projection.weight.grad.abs().sum() > 0
        from copy import deepcopy
        restored = deepcopy(ns['modules'])
        with torch.no_grad():
            restored['encoder'].embedding.features.position.proj.weight.zero_()
        load_model(model=restored, repo_id_or_path=str(tmp_path / 'checkpoint'))
        torch.testing.assert_close(restored['encoder'].embedding.features.position.proj.weight,
                                   time_projection.weight)
        saved = load_tokenizer(repo_id_or_path=str(tmp_path / 'checkpoint'))
        assert all(spec.position_field == 'elapsed_seconds' for spec in saved.input_fields)
    finally:
        if 'loader' in ns:
            ns['loader'].close()
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('kind', ['text', 'token', 'image'])
def test_numeric_and_position_additions_cover_every_emitted_token(kind, tmp_path):
    source = tokenizer()
    spec = dict(type=kind, provider='sensor', input_field='content',
                numeric_field='measurement', numeric_index=1, fourier_min=0., fourier_max=4.,
                position_field='context', position_fourier_min=-2., position_fourier_max=2.)
    if kind != 'image':
        spec['format'] = '{field}'
    tok = Tokenizer(tokenizers={'data': (lambda image: image) if kind == 'image' else source.tokenizers['text']},
                    input_fields=[spec])
    content = [1, 2, 3] if kind == 'image' else 'a b a' if kind == 'text' else 'a'
    row = dict(provider='sensor', step_index=0,
               payload=dict(content=content, measurement=[0., 3.], context=1.))
    packed, packed_objective_data = tok.pack_rows(rows=[[row]], continuing=None)
    count = 1 if kind == 'token' else 3
    assert packed['numeric_values'].tolist() == [.5] * count
    assert packed['position_values'].tolist() == [.5] * count
    assert packed['numeric_field_ids'].tolist() == [0] * count
    model = Embedding(vocab_size=6, hidden_dim=8, num_frequencies=2,
                      numeric_field_names=tok.numeric_field_names, position_num_frequencies=3)
    tensors = packed
    kwargs = {key: tensors[key] for key in ('numeric_values', 'numeric_mask', 'numeric_field_ids',
                                           'position_values', 'position_mask')}
    actual = model(tensors['token_ids'], numeric_field_names=packed['numeric_field_names'], **kwargs)
    expected = (model.embed_tokens(tensors['token_ids'])
                + model.features.numeric[0](torch.full((count,), .5))
                + model.features.position(torch.full((count,), .5, dtype=torch.float64)))
    torch.testing.assert_close(actual, expected)
    for missing in ('measurement', 'context'):
        payload = {**row['payload'], missing: None}
        piece = tok({**row, 'payload': payload})
        assert len(piece['token_ids']) == count
        assert piece['numeric_mask'].all() == (missing != 'measurement')
        assert piece['position_mask'].all() == (missing != 'context')
        if missing == 'measurement': assert (piece['numeric_field_ids'] == -1).all()


def test_token_formatting_lookup_and_single_token_constraint():
    tok = Tokenizer(tokenizers=tokenizer().tokenizers, input_fields=[
        dict(type='token', provider='sensor', input_field='label', lookup=['a', 'a b'], format='{field}')])
    assert tok(dict(provider='sensor', payload={'label': 0}))['token_ids'].tolist() == [1]
    with pytest.raises(ValueError, match='expected 1'):
        tok(dict(provider='sensor', payload={'label': 1}))
    with pytest.raises(ValueError, match='placeholders'):
        replace(tok.input_fields[0], special=True)
    with pytest.raises(ValueError, match='unknown tokenizer modality'):
        TokenizerModalitySpec(type='numeric', provider='sensor')


@pytest.mark.parametrize('settings', [{}, {'fourier_min': 0.}, {'fourier_max': 1.},
    {'fourier_min': 1., 'fourier_max': 0.}, {'fourier_min': 0., 'fourier_max': float('inf')}])
def test_numeric_addition_requires_valid_explicit_bounds(settings):
    with pytest.raises(ValueError):
        TokenizerModalitySpec(type='text', provider='sensor', format='a', numeric_field='value', **settings)


def test_same_payload_name_from_different_providers_has_distinct_projection():
    tok = Tokenizer(tokenizers=tokenizer().tokenizers, input_fields=[
        dict(type='token', provider=provider, input_field='value', format='n',
             numeric_field='value', fourier_min=0., fourier_max=1.) for provider in ['a', 'b']])
    assert len(tok.numeric_field_names) == 2
    assert tok(dict(provider='a', payload={'value': .5}))['numeric_field_ids'].tolist() == [0]
    assert tok(dict(provider='b', payload={'value': .5}))['numeric_field_ids'].tolist() == [1]
