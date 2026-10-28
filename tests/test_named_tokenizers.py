"""Named vocabularies retain identity through events, packing, and embeddings."""
import json
from copy import deepcopy
from typing import cast

import numpy as np
import pytest
import torch

from mouse_core.data import DataLoader, EventLog, Tokenizer, load_tokenizer, pack_token_batch, save_tokenizer
from mouse_core.models import (Backbone, Embedding, MultiEmbedding, action_readouts,
                               load_model, prepare_decode, save_model)
from tests.test_sparse_event_readouts import backend, on_group_start


class TextBackend:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        return {'input_ids': [self.value]}


def _tokenizer():
    # Deliberately use the same token ID in both vocabularies.
    return Tokenizer(tokenizers={'text': TextBackend(1), 'actions': TextBackend(1)}, input_fields=[
        dict(type='text', tokenizer='text', format='prefix', when=on_group_start),
        dict(type='text', tokenizer='actions', input_field='action', lookup=['a', 'b'], format='{field}'),
        dict(type='text', tokenizer='text', input_field='observation', format='{field}'),
        dict(type='token', tokenizer='text', input_field='action_requested', format='\n', head_output=True),
    ], objective_fields=[dict(input_field='action', missing_value=None)])


def _events():
    log = EventLog()
    log.event(provider='environment', episode_index=0, step_index=0,
              payload={'observation': 3, 'action_requested': True})
    log.event(provider='agent', episode_index=0, step_index=1, payload={'action': 1})
    log.event(provider='environment', episode_index=0, step_index=1,
              payload={'observation': 4, 'action_requested': True})
    return log


def _embedding():
    return MultiEmbedding(embeddings={name: Embedding(vocab_size=8, hidden_dim=8)
                                      for name in ['text', 'actions']})


def _embed(embedding, inputs):
    t = inputs.to_tensors()
    return embedding(t['token_ids'], tokenizer_ids=t['tokenizer_ids'],
                     tokenizer_names=inputs.tokenizer_names,
                     numeric_values=t['numeric_values'], numeric_mask=t['numeric_mask'])


def test_fields_are_independent_ordered_and_keep_backend_identity_in_prefixes_and_batches():
    tok = _tokenizer()
    event = dict(provider='any', step_index=0, payload={'observation': 9, 'action': 0, 'action_requested': True})
    piece = tok(event)
    assert piece['token_ids'].tolist() == [1, 1, 1]
    assert piece['tokenizer_ids'].tolist() == [0, 1, 1]
    assert piece['group_start_tokenizer_ids'].tolist() == [1]
    assert tok.tokenizers['actions'].calls == ['a']
    assert tok.tokenizers['text'].calls == ['prefix', '9', '\n']
    # Another event does not inherit the first event's action or observation.
    second = tok(dict(provider='environment', payload={'action_requested': True}))
    assert second['tokenizer_ids'].tolist() == [1]
    assert tok.tokenizers['actions'].calls == ['a']
    inputs = tok.pack_rows(rows=[_events(), []], continuing=None)
    assert inputs.tokenizer_names == ('actions', 'text')
    assert inputs.tokenizer_ids is not None
    assert inputs.tokenizer_ids.tolist() == [1, 1, 1, 0, 1, 1]
    assert inputs.head_token_ids.tolist() == [2, 5]
    continuing = tok.pack_rows(rows=[_events()], continuing=[True])
    assert continuing.tokenizer_ids is not None
    assert continuing.tokenizer_ids.tolist() == [1, 1, 0, 1, 1]
    empty = tok.pack_rows(rows=[[]], continuing=None)
    assert empty.tokenizer_names == inputs.tokenizer_names and empty.L == 0
    with DataLoader(stores=_events().to_store(), events_budget=3, batch_size=1,
                    transform=deepcopy(tok).encode, sample_field='episode_index', num_workers=0, seed=0) as loader:
        loaded, labels, group_id = loader.next_batch()
    np.testing.assert_array_equal(loaded.tokenizer_ids, inputs.tokenizer_ids)
    assert loaded.tokenizer_names == inputs.tokenizer_names
    assert labels['action_executed'].tolist() == [False, True]
    assert group_id.tolist() == [0, 0]


def test_colliding_token_ids_use_independent_tables_and_gradients():
    inputs = _tokenizer().pack_rows(rows=[_events()], continuing=None)
    model = _embedding()
    with torch.no_grad():
        cast(Embedding, model.embeddings['actions']).embed_tokens.weight.fill_(2.)
        cast(Embedding, model.embeddings['text']).embed_tokens.weight.fill_(5.)
    values = _embed(model, inputs)
    torch.testing.assert_close(values[:, 0], torch.tensor([5., 5., 5., 2., 5., 5.]))
    model.embeddings['text'].requires_grad_(False)
    values = _embed(model, inputs)
    values.sum().backward()
    assert cast(Embedding, model.embeddings['text']).embed_tokens.weight.grad is None
    gradient = cast(Embedding, model.embeddings['actions']).embed_tokens.weight.grad
    assert gradient is not None
    torch.testing.assert_close(gradient[1], torch.ones(8))
    assert gradient[0].count_nonzero() == 0
    with pytest.raises(ValueError, match='Multiple tokenizers'):
        _embed(Embedding(vocab_size=8, hidden_dim=8), inputs)


def test_numeric_and_image_backends_route_to_their_own_tables():
    tok = Tokenizer(tokenizers={'image': lambda image: image, 'numeric': TextBackend(2)}, input_fields=[
        dict(type='numeric', tokenizer='numeric', input_field='value', format='n', fourier_min=0., fourier_max=2.),
        dict(type='image', tokenizer='image', input_field='image'),
    ])
    inputs = tok.pack_rows(rows=[[dict(provider='sensor', step_index=0, payload={'value': 1.5, 'image': [1, 2]})]], continuing=None)
    model = MultiEmbedding(embeddings={'numeric': Embedding(vocab_size=4, hidden_dim=8, num_frequencies=2),
                                      'image': Embedding(vocab_size=3, hidden_dim=8)})
    actual = _embed(model, inputs)
    expected = torch.cat([
        model.embeddings['numeric'](torch.tensor([2]), numeric_values=torch.tensor([.5]), numeric_mask=torch.tensor([True])),
        model.embeddings['image'](torch.tensor([1, 2])),
    ])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_named_pretrained_sources_and_weights_roundtrip(tmp_path):
    from tests.test_embedding_loading import _source
    sources = {}
    for name in ('text', 'actions'):
        directory = tmp_path / name
        backend().save_pretrained(directory)
        _source('llama').save_pretrained(directory)
        sources[name] = str(directory)
    tok = Tokenizer(tokenizers=sources, input_fields=[
        dict(type='token', tokenizer='text', format='obs'),
        dict(type='token', tokenizer='actions', format='action'),
    ])
    model = MultiEmbedding(embeddings={name: Embedding(pretrained=source) for name, source in sources.items()})
    inputs = tok.pack_rows(rows=[[dict(provider='test', step_index=0, payload={})]], continuing=None)
    for index, name in enumerate(('text', 'actions')):
        expected = cast(Embedding, model.embeddings[name]).embed_tokens.weight[inputs.token_ids[index]]
        torch.testing.assert_close(_embed(model, inputs)[index], expected)
    checkpoint = tmp_path / 'saved'
    save_tokenizer(tokenizer=tok, path=checkpoint)
    save_model(model=model, path=checkpoint)
    restored = load_tokenizer(repo_id_or_path=str(checkpoint))
    assert restored.tokenizer_names == tok.tokenizer_names
    assert restored.tokenizer_sources == sources
    copy = MultiEmbedding(embeddings={name: Embedding(vocab_size=16, hidden_dim=8) for name in reversed(sources)})
    load_model(model=copy, repo_id_or_path=checkpoint)
    torch.testing.assert_close(_embed(copy, inputs), _embed(model, inputs))
    config = json.loads((checkpoint / 'tokenizer.json').read_text())
    assert config['tokenizers'] == sources
    assert [field['tokenizer'] for field in config['input_fields']] == ['text', 'actions']
    overrides = load_tokenizer(repo_id_or_path=str(checkpoint), tokenizers={'text': backend()})
    assert overrides.tokenizer_sources == sources


def test_live_backends_roundtrip_requires_named_overrides_and_preserves_ids(tmp_path):
    tok = _tokenizer()
    save_tokenizer(tokenizer=tok, path=tmp_path)
    with pytest.raises(TypeError, match='callable'):
        load_tokenizer(repo_id_or_path=str(tmp_path))
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'actions': TextBackend(1), 'text': TextBackend(1)})
    expected = tok.pack_rows(rows=[_events()], continuing=None)
    actual = restored.pack_rows(rows=[_events()], continuing=None)
    np.testing.assert_array_equal(actual.token_ids, expected.token_ids)
    np.testing.assert_array_equal(actual.tokenizer_ids, expected.tokenizer_ids)
    with pytest.raises(ValueError, match='saved specification'):
        load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'wrong': TextBackend(1)})


def test_special_token_registration_is_scoped_to_named_backends(tmp_path):
    first, second = backend(), backend()
    second.add_tokens(['extra'])
    tok = Tokenizer(tokenizers={'first': first, 'second': second}, input_fields=[
        dict(type='token', tokenizer=name, format='<new>', special=True) for name in ('first', 'second')])
    assert tok.special_token_ids == {'first': {'<new>': 5}, 'second': {'<new>': 6}}
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'second': second, 'first': first})
    assert restored.special_token_ids == tok.special_token_ids
    with pytest.raises(ValueError, match='Special token IDs differ'):
        load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={'first': backend(), 'second': backend()})


@pytest.mark.parametrize('options', [
    {}, {'tokenizer': TextBackend(1)}, {'pretrained': 'unused'},
    {'image_tokenizer': lambda image: image},
    {'tokenizers': None}, {'tokenizers': 'unused'}, {'tokenizers': TextBackend(1)},
    {'tokenizers': {}}, {'tokenizers': [TextBackend(1)]},
    {'tokenizers': {'text': TextBackend(1)}, 'tokenizer': TextBackend(1)},
    {'tokenizers': {'text': TextBackend(1), 'actions': TextBackend(1)}},
])
def test_invalid_or_ambiguous_backend_configuration_is_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        Tokenizer(input_fields=[dict(type='token', format='x')], **options)


@pytest.mark.parametrize('override', [
    {'tokenizer': TextBackend(1)}, {'pretrained': 'unused'},
    {'image_tokenizer': lambda image: image}, {'tokenizers': TextBackend(1)},
    {'tokenizers': 'unused'}, {'tokenizers': [TextBackend(1)]},
])
def test_load_rejects_single_backend_arguments_before_io(override):
    with pytest.raises(TypeError, match='tokenizer'):
        load_tokenizer(repo_id_or_path='unused/nonexistent', **override)


@pytest.mark.parametrize('backends', [None, {}, 'unused', ['unused']])
def test_load_rejects_saved_specs_without_a_backend_dictionary(tmp_path, backends):
    save_tokenizer(tokenizer=_tokenizer(), path=tmp_path)
    path = tmp_path / 'tokenizer.json'
    config = json.loads(path.read_text())
    if backends is None:
        config.pop('tokenizers')
        config['pretrained'] = 'legacy-model'
    else:
        config['tokenizers'] = backends
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='tokenizers dictionary'):
        load_tokenizer(repo_id_or_path=str(tmp_path))


def test_one_named_backend_or_default_can_be_implicit():
    for backends in ({'text': TextBackend(3)}, {'default': TextBackend(3), 'actions': TextBackend(1)}):
        tok = Tokenizer(tokenizers=backends, input_fields=[dict(type='token', format='x')])
        assert tok(dict(provider='test', payload={}))['token_ids'].tolist() == [3]
    with pytest.raises(ValueError, match='configured tokenizer'):
        Tokenizer(tokenizers={'text': TextBackend(1)}, input_fields=[dict(type='token', tokenizer='typo', format='x')])


def test_routing_mismatches_are_rejected():
    inputs = _tokenizer().pack_rows(rows=[_events()], continuing=None)
    inputs.tokenizer_names = tuple(reversed(inputs.tokenizer_names))
    with pytest.raises(ValueError, match='tokenizer_names'):
        _embed(_embedding(), inputs)
    first = _tokenizer().encode(_events())[0]
    second = _tokenizer().encode(_events())[1]
    second.tokenizer_names = ('text', 'actions')
    with pytest.raises(ValueError, match='tokenizer_names'):
        pack_token_batch(steps=[first, second], continuing=None)
    with pytest.raises(ValueError, match='same hidden_dim'):
        MultiEmbedding(embeddings={'text': Embedding(vocab_size=2, hidden_dim=8),
                                  'actions': Embedding(vocab_size=2, hidden_dim=4)})


def test_named_embeddings_match_cached_and_packed_backbone_readouts():
    tok = _tokenizer()
    embedding = _embedding().eval()
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                        train_kernel='reference', decode_kernel='flex', dtype=torch.float32).eval()
    inputs = tok.pack_rows(rows=[_events()], continuing=None)
    with torch.no_grad():
        features = backbone(_embed(embedding, inputs), sequence_ids=inputs.to_tensors()['sequence_ids'])
        expected = features[inputs.head_token_ids]
        cache = None
        actual = []
        try:
            for events in (_events().events[:1], _events().events[1:]):
                tokens = tok.pack_rows(rows=[events], continuing=None if cache is None else [True])
                t = tokens.to_tensors()
                embeds, lengths, heads, _ = prepare_decode(embeds=_embed(embedding, tokens), tokens=tokens,
                    sequence_ids=t['sequence_ids'], head_token_ids=t['head_token_ids'], head_step_ids=t['head_step_ids'])
                if cache is None:
                    cache = backbone.decode_session(batch_size=1)
                decoded = backbone(embeds, cache=cache, lengths=lengths)
                scores = decoded.gather(1, heads.unsqueeze(-1).expand(-1, -1, 8))
                actual.append(action_readouts(scores, tokens))
        finally:
            if cache is not None:
                cache.close()
    torch.testing.assert_close(torch.cat(actual), expected, atol=2e-5, rtol=2e-5)
