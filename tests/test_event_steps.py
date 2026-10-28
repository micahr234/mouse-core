"""Per-event encoding, incremental step assembly, and whole-step sample windows."""
from functools import partial
import json
from typing import Any

import pytest

from mouse_core.data import (DataLoader, EpochDataLoader, EventLog, Tokenizer,
                             environment_episode_ranges, load_tokenizer, save_tokenizer,
                             step_sample_ranges)
from tests._token_batch_helpers import IntIdTokenizer


def make_tokenizer(**kwargs):
    return Tokenizer(tokenizers={"default": IntIdTokenizer()}, input_fields=[
        dict(type='text', provider='sensor', input_field='value', format='{field}', head_output=True),
    ], **kwargs)



def recorded():
    log = EventLog()
    for episode in range(2):
        for step in range(3):
            log.event(provider='control', episode_index=episode, step_index=step, payload={})
            for value in range(step + 1):
                log.event(provider='sensor', episode_index=episode, step_index=step,
                          payload={'value': value + 1})
    return log


def test_every_event_emits_independently_and_steps_follow_only_markers():
    tok = make_tokenizer()
    event = {'provider': 'sensor', 'payload': {'value': 2}}
    assert tok(event)['token_ids'].tolist() == [2]
    assert tok(event)['head_token_mask'].all()
    assert tok({'provider': 'control', 'payload': None})['token_ids'].size == 0
    steps = tok.encode(recorded())
    assert [step.token_ids.tolist() for step in steps] == [[1], [1, 2], [1, 2, 3]] * 2
    assert all(step.head_token_mask.all() for step in steps)


def test_stream_buffers_a_partial_step_and_tokenizes_each_event_once():
    calls = []
    class CountingTokenizer(Tokenizer):
        def __call__(self, event):
            calls.append(event['payload'])
            return super().__call__(event)
    tok = CountingTokenizer(tokenizers={"default": IntIdTokenizer()}, input_fields=make_tokenizer().input_fields)
    stream = tok.stream()
    events = recorded().events
    assert stream.push(events[0]) is None
    assert stream.push(events[1]) is None
    # No flush between chunks. The first event of the next chunk closes step 0.
    result = [stream.push(events[2])]
    for event in events[3:]:
        completed = stream.push(event)
        if completed is not None:
            result.append(completed)
    result.append(stream.flush())
    assert len(calls) == len(events)
    assert [step.token_ids.tolist() for step in result if step is not None] == [
        [1], [1, 2], [1, 2, 3]] * 2
    assert stream.flush() is None
    with pytest.raises(ValueError, match='flushed step'):
        stream.push(events[-1])


def test_sparse_custom_markers_and_episode_reset_with_same_step_id():
    tok = make_tokenizer(step_field='payload.tick')
    events = [
        dict(provider='control', episode_index=0, payload={'tick': 0}),
        dict(provider='sensor', payload={'value': 1}),
        dict(provider='sensor', episode_index=1, payload={'tick': 0, 'value': 2}),
        dict(provider='sensor', payload={'tick': 3, 'value': 3}),
    ]
    assert [step.token_ids.tolist() for step in tok.encode(events)] == [[1], [2], [3]]
    assert list(step_sample_ranges(events, steps_per_sample=2, step_field='payload.tick')) == [(2, 4)]
    with pytest.raises(ValueError, match='first event'):
        tok.encode(events[1:])
    with pytest.raises(ValueError, match='first event'):
        tok.encode([events[0], dict(provider='sensor', episode_index=1, payload={'value': 2})])


def test_empty_input_step_has_no_readout_and_defaults_are_step_local():
    tok = make_tokenizer(objective_fields=[
        dict(input_field='sensor.value', output_field='label', missing_value=-1),
        dict(input_field='action', missing_value=None),
        dict(input_field='step_index'),
    ])
    log = EventLog()
    log.event(provider='control', episode_index=0, step_index=0, payload={})
    log.event(provider='sensor', episode_index=0, step_index=0, payload={'value': 2, 'action': 1})
    log.event(provider='control', episode_index=0, step_index=1, payload={})
    first, last = tok.encode(log)
    assert first.objective_fields == {'label': 2, 'action': 1, 'step_index': 0}
    assert last.token_ids.tolist() == []
    assert not last.head_token_mask.any()
    assert last.objective_fields == {'label': -1, 'action': None, 'step_index': 1}


def test_training_requires_labels_but_inference_does_not():
    tok = make_tokenizer(objective_fields=[dict(input_field='label')])
    with pytest.raises(KeyError, match='label'):
        tok.encode(recorded())
    packed = tok.pack_rows(rows=[recorded(), []], continuing=None)
    assert packed.N == 6
    assert packed.B == 2


def test_saved_marker_configuration_replays_identically(tmp_path):
    tok = make_tokenizer(step_field='payload.tick', episode_field=None)
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={"default": IntIdTokenizer()})
    assert isinstance(restored, Tokenizer)
    assert restored.step_field == 'payload.tick' and restored.episode_field is None
    events = [dict(provider='sensor', payload={'tick': i, 'value': i + 1}) for i in range(3)]
    assert [step.token_ids.tolist() for step in restored.encode(events)] == [[1], [2], [3]]


def test_tokenizer_requires_event_envelopes():
    tok = make_tokenizer()
    with pytest.raises(TypeError, match='provider/payload event'):
        tok({'value': 1})


def test_image_only_tokenizer_needs_no_text_backend():
    tok = Tokenizer(input_fields=[dict(type='image', input_field='image')],
                    tokenizers={"image": lambda value: value})
    assert tok({'provider': 'camera', 'payload': {'image': [1, 2]}})['token_ids'].tolist() == [1, 2]


def test_row_checkpoint_format_is_rejected(tmp_path):
    (tmp_path / 'tokenizer.json').write_text('{"format": "mouse-core-tokenizer-v1"}')
    with pytest.raises(ValueError, match='Unsupported tokenizer config format'):
        load_tokenizer(repo_id_or_path=str(tmp_path), tokenizers={"default": IntIdTokenizer()})


def test_step_headers_survive_store_dataset_roundtrip():
    log = recorded()
    dataset = log.to_store().to_dataset()
    assert dataset.column_names == ['episode_index', 'step_index', 'provider', 'payload']
    restored = EventLog.from_dataset(dataset)
    assert restored.events == log.events
    assert [step.token_ids.tolist() for step in make_tokenizer().encode(dataset)] == [
        step.token_ids.tolist() for step in make_tokenizer().encode(log)]
    assert list(step_sample_ranges(dataset, steps_per_sample=2)) == [(0, 5), (2, 9), (9, 14), (11, 18)]
    assert list(step_sample_ranges(dataset, steps_per_sample=4)) == []
    assert list(step_sample_ranges(dataset, steps_per_sample=2, include_final_step=False)) == [(0, 5), (2, 9), (9, 14)]


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_invalid_step_window_size(count):
    with pytest.raises(ValueError, match='positive integer'):
        list(step_sample_ranges(recorded(), steps_per_sample=count))


@pytest.mark.parametrize('kind', ['random', 'epoch'])
def test_loader_samples_exact_whole_step_windows(kind):
    tok = make_tokenizer(objective_fields=[dict(input_field='episode_index'), dict(input_field='step_index')])
    kwargs: dict[str, Any] = dict(stores=[recorded().to_store()], transform=tok.encode,
                  sample_ranges=partial(step_sample_ranges, steps_per_sample=2), token_budget=15, seed=3)
    loader = (DataLoader(**kwargs, batch_size=1, num_workers=0) if kind == 'random'
              else EpochDataLoader(**kwargs))
    try:
        for _ in range(2):
            _, labels, groups = loader.next_batch()
            for group in groups.unique_consecutive():
                mask = groups == group
                indices = labels['step_index'][mask].tolist()
                assert len(indices) == 2 and indices[1] == indices[0] + 1
                assert len(labels['episode_index'][mask].unique()) == 1
    finally:
        loader.close()


@pytest.mark.parametrize('window,match', [((1, 5), 'whole steps'), ((0, 4), 'whole steps'),
                                        ((2, 14), 'one episode')])
@pytest.mark.parametrize('kind', ['random', 'epoch'])
def test_both_loaders_reject_partial_step_or_cross_episode_ranges(window, match, kind):
    kwargs: dict[str, Any] = dict(stores=[recorded().to_store()], transform=make_tokenizer().encode,
                  sample_ranges=lambda ds: [window], token_budget=30)
    with pytest.raises(ValueError, match=match):
        if kind == 'random':
            DataLoader(**kwargs, batch_size=1, num_workers=0)
        else:
            EpochDataLoader(**kwargs)


def test_sample_field_step_index_does_not_merge_identical_ids_across_episodes():
    log = EventLog()
    for episode in range(2):
        log.event(provider='sensor', episode_index=episode, step_index=0, payload={'value': 1})
    tok = make_tokenizer(objective_fields=[dict(input_field='episode_index')])
    loader = EpochDataLoader(stores=[log.to_store()], transform=tok.encode,
                             sample_field='step_index', token_budget=10)
    try:
        assert len(loader.samples) == 2
        assert all(len(sample) == 1 for sample in loader.samples)
    finally:
        loader.close()


def test_episode_sampler_excludes_incomplete_tail_but_keeps_terminal_step():
    log = EventLog()
    for episode in range(2):
        log.event(provider='coordinator', episode_index=episode, step_index=0,
                  payload={'episode_start': True})
        log.event(provider='environment', episode_index=episode, step_index=0,
                  payload={'observation': 1})
        log.event(provider='environment', episode_index=episode, step_index=1,
                  payload={'terminated': episode == 0})
    assert list(environment_episode_ranges(log)) == [(0, 3)]


@pytest.mark.parametrize('problem', ['missing_start', 'missing_end', 'duplicate_start', 'after_end', 'reused_index'])
def test_episode_sampler_checks_indexed_episode_boundaries(problem):
    events = [
        dict(provider='coordinator', episode_index=0, payload={'episode_start': True}),
        dict(provider='environment', episode_index=0, payload={'terminated': True}),
        dict(provider='coordinator', episode_index=1, payload={'episode_start': True}),
        dict(provider='environment', episode_index=1, payload={'truncated': True}),
    ]
    if problem == 'missing_start':
        events.pop(0)
    elif problem == 'missing_end':
        events[1]['payload'] = {}
    elif problem == 'duplicate_start':
        events.insert(1, events[0].copy())
    elif problem == 'after_end':
        events.insert(2, dict(provider='environment', episode_index=0, payload={}))
    else:
        events.extend(events[:2])
    with pytest.raises(ValueError):
        list(environment_episode_ranges(events))


def test_raw_json_string_payload_cannot_supply_a_step_marker():
    log = EventLog()
    log.event(provider='control', episode_index=0, payload=json.dumps({'tick': 0}))
    tok = make_tokenizer(step_field='payload.tick')
    for source in (log, log.to_dataset()):
        with pytest.raises(ValueError, match='first event'):
            tok.encode(source)
        with pytest.raises(ValueError, match='first event'):
            list(step_sample_ranges(source, steps_per_sample=1, step_field='payload.tick'))


def test_sparse_step_headers_keep_metadata_labels_identical_after_persistence():
    log = EventLog()
    log.event(provider='control', episode_index=3, step_index=7, payload={})
    log.event(provider='sensor', payload={'value': 1})
    tok = make_tokenizer(objective_fields=[dict(input_field='episode_index'), dict(input_field='step_index')])
    for source in (log, log.to_dataset()):
        step, = tok.encode(source)
        assert step.token_ids.tolist() == [1]
        assert step.objective_fields == {'episode_index': 3, 'step_index': 7}


def on_group_start(ctx):
    return ctx['group_start']


def test_event_prefix_and_readouts_match_full_and_cached_packing():
    tok = Tokenizer(tokenizers={"default": IntIdTokenizer()}, input_fields=[
        dict(type='token', provider='sensor', format='8', when=on_group_start),
        dict(type='text', provider='sensor', input_field='value', format='{field}'),
        dict(type='token', provider='sensor', format='9', head_output=True),
    ])
    events = [dict(provider='sensor', step_index=i, payload={'value': i + 1}) for i in range(3)]
    full = tok.pack_rows(rows=[events], continuing=None)
    first = tok.pack_rows(rows=[events[:1]], continuing=None)
    rest = tok.pack_rows(rows=[events[1:]], continuing=[True])
    assert full.token_ids.tolist() == first.token_ids.tolist() + rest.token_ids.tolist() == [8, 1, 9, 2, 9, 3, 9]
    assert full.head_step_ids.tolist() == [0, 1, 2]
