"""Generic event recording, explicit environment decoding, and live/replay parity."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from datasets import load_from_disk

from mouse_core.data import (EventLog, Tokenizer,
                             EpochDataLoader, EventTransform, environment_decisions, environment_episode_ranges,
                             latest_environment_decision, save_tokenizer, load_tokenizer)
from mouse_core.models import Backbone, Embedding
from tests._model_helpers import simple_model
from tests._token_batch_helpers import IntIdTokenizer


def tokenizer():
    return Tokenizer(tokenizer=IntIdTokenizer(), input_fields=[
        dict(type='numeric', provider='environment', input_field='position', format='1', fourier_min=-10., fourier_max=10.),
        dict(type='numeric', provider='environment', input_field='temperature', format='2', fourier_min=-40., fourier_max=40.),
        dict(type='token', provider='environment', format='4', head_output=True),
        dict(type='numeric', provider='agent', input_field='action', format='3', fourier_min=0., fourier_max=2.),
    ],
        objective_fields=[dict(input_field=name, missing_value=value) for name, value in
                          [('action', None), ('reward', 0.0), ('terminated', False), ('truncated', False)]]
        + [dict(input_field='episode_index', output_field='source_episode_index')])


def collect():
    log = EventLog(name='test')
    for episode in range(2):
        for step, action, result in [
            (0, None, dict(terminated=False, truncated=False, position=0., timestamp='2099-01-01', reward=0.)),
            (1, 1, dict(terminated=False, truncated=False, temperature=21., timestamp='1999-01-01', reward=.25)),
            (2, None, dict(terminated=False, truncated=False, position=1., reward=.75)),
            (3, 0, dict(terminated=True, truncated=False, position=2., reward=2.)),
        ]:
            log.event(provider='coordinator', episode_index=episode, step_index=step, payload={
                **({'episode_start': True} if step == 0 else {}),
            })
            if action is not None:
                log.event(provider='agent', episode_index=episode, step_index=step, payload={'action': action})
            log.event(provider='environment', episode_index=episode, step_index=step, payload={**result, 'action_requested': step in (0, 2)})
    return log


def _events(rows):
    return [event for row in rows for event in json.loads(row['events'])]


def _step(tok, row):
    return tok.encode(_events([row]))[0]


def test_generic_log_accepts_independent_providers_and_arbitrary_json_payloads(tmp_path):
    log = EventLog(name='sensors')
    payloads = [dict(temperature=21., terminated='not a control flag', reward=None),
                [1, 2], 'ready', 3.5, False, None]
    for provider, payload in zip(('sensor', 'camera', 'operator', 'score', 'switch', 'heartbeat'), payloads):
        assert log.event(provider=provider, payload=payload) == {'provider': provider, 'payload': payload}
    dataset = log.to_dataset()
    assert dataset.column_names == ['provider', 'payload']
    dataset.save_to_disk(str(tmp_path / 'generic'))
    restored = EventLog.from_dataset(load_from_disk(str(tmp_path / 'generic')))
    assert restored.events == log.events
    assert restored.to_store().to_dataset().to_list() == dataset.to_list()
    assert restored[-1] == log.last_event
    assert restored[1:3] == log.events[1:3]
    with pytest.raises(ValueError, match='environment, coordinator or agent providers'):
        list(environment_decisions(log))


@pytest.mark.parametrize('provider', ['', None, 4])
def test_generic_log_rejects_invalid_provider_atomically(provider):
    log = EventLog()
    with pytest.raises(ValueError, match='nonempty string'):
        log.event(provider=provider, payload={'data': 1})
    assert len(log) == 0


@pytest.mark.parametrize('payload', [float('nan'), {'value': float('inf')}, object(), {1: 2, '1': 3}])
def test_generic_log_rejects_non_json_payload_atomically(payload):
    log = EventLog()
    with pytest.raises((TypeError, ValueError)):
        log.event(provider='sensor', payload=payload)
    assert len(log) == 0


def test_generic_log_snapshots_inputs_returns_and_reads():
    data = np.array([2.])
    payload = {'data': data, 'reward': None}
    log = EventLog()
    recorded = log.event(provider='sensor', payload=payload)
    data[0] = 99.
    recorded['payload']['data'][0] = 88.
    log.events[0]['payload']['data'][0] = 77.
    log[0]['payload']['data'][0] = 66.
    assert log.last_event == {'provider': 'sensor', 'payload': {'data': [2.], 'reward': None}}


def test_raw_json_string_payload_is_not_mistaken_for_an_environment_object():
    payload = json.dumps(collect()[1]['payload'])
    log = EventLog()
    log.event(**collect()[0])
    log.event(provider='environment', episode_index=0, step_index=0, payload=payload)
    assert log[1]['payload'] == payload
    with pytest.raises(ValueError, match='JSON objects'):
        latest_environment_decision(log)
    for source in (log, log.to_dataset(), log.to_store()[:]):
        with pytest.raises(ValueError, match='JSON objects'):
            list(environment_decisions(source))


def test_sparse_source_log_and_decision_alignment(tmp_path):
    log = collect()
    rows = list(environment_decisions(log))
    assert len(log) == 20 and len(rows) == 8
    assert [row['step_index'] for row in rows] == [0, 1, 2, 3] * 2
    assert [row['reward'] for row in rows] == [0., .25, .75, 2.] * 2
    assert [row.get('action') for row in rows] == [None, 1, None, 0] * 2
    assert [row['action_requested'] for row in rows] == [True, False, True, False] * 2
    assert [row.get('position') for row in rows] == [0., None, 1., 2.] * 2
    assert [row.get('temperature') for row in rows] == [None, 21., None, None] * 2
    log.to_dataset().save_to_disk(str(tmp_path / 'events'))
    restored = EventLog.from_dataset(load_from_disk(str(tmp_path / 'events')))
    assert restored.events == log.events
    assert list(environment_decisions(restored)) == rows
    live = EventLog()
    observed = []
    for event in log:
        live.event(**event)
        if event['provider'] == 'environment':
            observed.append(latest_environment_decision(live))
    assert observed == rows


def test_live_read_does_not_replay_history_or_see_future_agent_fields():
    log = EventLog()
    source = collect().events
    for event in source[:7]:
        log.event(**event)
    before = latest_environment_decision(log)
    log.event(**source[7])
    log.event(provider='agent', episode_index=0, step_index=3, payload={'action': 0, 'old_log_prob': -.2})
    with pytest.raises(ValueError, match='latest event'):
        latest_environment_decision(log)
    log.event(**source[9])
    row = latest_environment_decision(log)
    assert row['old_log_prob'] == -.2 and 'old_log_prob' not in before
    assert list(environment_decisions(log))[2] == before
    class TailOnly:
        def __len__(self): return len(log)
        def __getitem__(self, index):
            assert index in (-1, -2, -3, -4, -5, -6)
            return log[index]
    assert latest_environment_decision(TailOnly()) == row


@pytest.mark.parametrize('reward_fields', [{}, {'reward': 0.0}])
def test_decoder_preserves_reward_presence_without_changing_recorded_payload(reward_fields):
    payload = dict(terminated=False, truncated=False, observation=[0.], **reward_fields)
    original = deepcopy(payload)
    log = EventLog()
    log.event(**collect()[0])
    log.event(provider='environment', episode_index=0, step_index=0, payload=payload)
    decision = latest_environment_decision(log)
    assert log[1]['payload'] == original == payload
    assert ('reward' in decision) == ('reward' in payload)
    decision['observation'][0] = 99.
    assert log[1]['payload']['observation'] == [0.]


@pytest.mark.parametrize('missing', ['episode_index', 'episode_start'])
def test_decoder_requires_explicit_indices_and_controls_but_logger_does_not(missing):
    records = collect().events[:2]
    record = records[0]
    del (record if missing == 'episode_index' else record['payload'])[missing]
    log = EventLog()
    for event in records:
        log.event(**event)
    before = log.events
    with pytest.raises(ValueError, match=missing):
        list(environment_decisions(log))
    with pytest.raises(ValueError, match=missing):
        latest_environment_decision(log)
    assert log.events == before


@pytest.mark.parametrize('invalid', [
    {'episode_start': 1}, {'terminated': 'false'}, {'truncated': None},
    {'action_requested': 'false'}, {'action': 0}, {'reward': '1'}, {'reward': None},
    {'inner_terminated': 1}, {'inner_truncated': 'false'},
    {'terminated': True, 'action_requested': True},
])
def test_decoder_rejects_invalid_environment_payload_without_changing_source(invalid):
    record = collect()[1]
    record['payload'].update(invalid)
    log = EventLog()
    log.event(**record)
    before = log.events
    with pytest.raises(ValueError):
        list(environment_decisions(log))
    assert log.events == before


@pytest.mark.parametrize('mutation', [
    lambda e: e['payload'].__setitem__('step_index', 20),
    lambda e: e['payload'].__setitem__('episode_index', 20),
    lambda e: e['payload'].__setitem__('reward', 20),
    lambda e: e.__setitem__('provider', 'unknown'),
    lambda e: e['payload'].__setitem__('action', None),
])
def test_decoder_rejects_invalid_agent_event(mutation):
    events = collect().events
    mutation(events[3])
    log = EventLog()
    for event in events:
        log.event(**event)
    with pytest.raises(ValueError):
        list(environment_decisions(log))


@pytest.mark.parametrize('bad_order', ['duplicate', 'drop', 'swap', 'early_reset', 'terminal_action',
                                       'duplicate_coordinator', 'premature_agent'])
def test_decoder_rejects_bad_control_order_while_log_preserves_it(bad_order):
    events = collect().events
    if bad_order == 'duplicate': events.insert(4, deepcopy(events[3]))
    if bad_order == 'drop': events.pop(3)
    if bad_order == 'swap': events[3], events[4] = events[4], events[3]
    if bad_order == 'early_reset': events = events[:2] + events[10:]
    if bad_order == 'terminal_action':
        events.insert(10, {'provider': 'agent', 'payload': {'action': 1}})
    if bad_order == 'duplicate_coordinator': events.insert(3, deepcopy(events[2]))
    if bad_order == 'premature_agent': events.insert(1, {'provider': 'agent', 'payload': {'action': 1}})
    log = EventLog()
    for event in events:
        log.event(**event)
    assert log.events == events
    with pytest.raises(ValueError):
        list(environment_decisions(log))


def test_incremental_tokens_and_saved_spec_match_replay(tmp_path):
    log = collect()
    tok = tokenizer()
    save_tokenizer(tokenizer=tok, path=tmp_path)
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=IntIdTokenizer())
    rows = list(environment_decisions(log))[:4]
    full = tok.pack_rows(rows=[_events(rows)], continuing=None)
    first = tok.pack_rows(rows=[_events(rows[:1])], continuing=None)
    rest = loaded.pack_rows(rows=[_events(rows[1:])], continuing=[True])
    for field in ('token_ids', 'numeric_values', 'numeric_mask'):
        np.testing.assert_array_equal(getattr(full, field), np.concatenate([getattr(first, field), getattr(rest, field)]))
    np.testing.assert_array_equal(full.head_step_ids, [0, 1, 2, 3])
    for row in rows:
        tokens = _step(tok, row)
        assert tokens.head_token_mask.sum() == 1
        assert 'source_step_index' not in tokens.objective_fields
        assert tokens.objective_fields['source_episode_index'] == row['episode_index']
    altered = deepcopy(rows[1])
    events = json.loads(altered['events'])
    events[-1]['payload']['timestamp'] = '1700-01-01'
    altered['events'] = json.dumps(events)
    np.testing.assert_array_equal(_step(tok, rows[1]).token_ids, _step(tok, altered).token_ids)
    np.testing.assert_array_equal(_step(tok, rows[1]).numeric_values, _step(tok, altered).numeric_values)


def test_provider_fields_preserve_optional_environment_observations_in_order():
    rows = list(environment_decisions(collect()))
    tokens = _step(tokenizer(), rows[1])
    # The agent precedes its resulting observation. A later passive output gets
    # a separate readout without replaying the previous action or sensor value.
    np.testing.assert_array_equal(tokens.token_ids, [3, 2, 4])
    np.testing.assert_allclose(tokens.numeric_values, [0., .525, 0.])
    np.testing.assert_array_equal(tokens.head_token_mask, [False, False, True])
    following = _step(tokenizer(), rows[2])
    np.testing.assert_array_equal(following.token_ids, [1, 4])
    np.testing.assert_allclose(following.numeric_values, [.1, 0.])


def test_events_without_matching_fields_emit_no_tokens():
    tok = Tokenizer(tokenizer=IntIdTokenizer(), input_fields=[])
    step = _step(tok, list(environment_decisions(collect()))[1])
    assert step.token_ids.size == 0 and not step.head_token_mask.any()


@pytest.mark.parametrize('excluded,expected_ids', [
    (['environment'], [3, 4]),
    (['agent'], [2, 4]),
    (['environment', 'agent'], [4]),
])
def test_empty_provider_fields_explicitly_exclude_payloads(tmp_path, excluded, expected_ids):
    fields = [field for field in tokenizer().input_fields if field.provider not in excluded or field.input_field is None]
    tok = Tokenizer(tokenizer=IntIdTokenizer(), input_fields=fields)
    save_tokenizer(tokenizer=tok, path=tmp_path)
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=IntIdTokenizer())
    tokens = _step(loaded, list(environment_decisions(collect()))[1])
    np.testing.assert_array_equal(tokens.token_ids, expected_ids)
    assert tokens.head_token_mask.sum() == 1


def test_replay_loader_training_and_cached_inference(tmp_path):
    from mouse_core.objectives import DqnObjective, boundary_discount
    torch.manual_seed(4)
    log, tok = collect(), tokenizer()
    log.to_dataset().save_to_disk(str(tmp_path / 'events'))
    replay = EventLog.from_dataset(load_from_disk(str(tmp_path / 'events')))
    loader = EpochDataLoader(stores=[replay.to_store()], transform=tok.encode,
                             token_budget=1000, sample_ranges=environment_episode_ranges, seed=1)
    model = simple_model(backbone=Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                                          train_kernel='reference', decode_kernel='flex', dtype=torch.float32),
                         heads={'q': torch.nn.Linear(8, 2)}, reasoner=None,
                         embedding=Embedding(hidden_dim=8, vocab_size=32, num_frequencies=2))
    inputs, data, group = loader.next_batch()
    assert inputs.head_step_ids.tolist() == list(range(8))
    assert data['action_executed'].tolist() == [False, True, False, True] * 2
    objective = DqnObjective(discount=boundary_discount(gamma_step=.9, gamma_terminated=0., gamma_truncated=1.),
                             reward=None, value=None, temperature=0., cross_group_backups='ignore', gate=None)
    prediction = model(inputs)['predictions']['q']
    losses, _ = objective(predictions=prediction, backup_action_predictions=prediction.detach(),
                          backup_value_predictions=prediction.detach(), objective_data=data,
                          group_id=group)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    before = [p.detach().clone() for p in model.parameters()]
    loss = torch.stack(list(losses.values())).sum()
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    model.eval()
    rows = list(environment_decisions(log))[:4]
    with torch.no_grad():
        full = model(tok.pack_rows(rows=[_events(rows)], continuing=None))['predictions']['q']
        cache = None
        values = []
        try:
            for i, row in enumerate(rows):
                result = model(tok.pack_rows(rows=[_events([row])], continuing=[True] if i else None), cache=cache, use_cache=True)
                cache = result['cache']
                values.append(result['predictions']['q'].reshape(-1, 2))
            torch.testing.assert_close(torch.cat(values), full, atol=2e-5, rtol=2e-5)
        finally:
            if cache is not None: cache.close()
    loader.close()



def test_consecutive_passive_outputs_keep_separate_rewards_and_action_ownership():
    events = collect().events[:7]
    events[3]['payload']['old_log_prob'] = -.2
    log = EventLog()
    for event in events:
        log.event(**event)
    first, second = list(environment_decisions(log))[1:]
    assert first['action'] == 1 and first['old_log_prob'] == -.2
    assert 'action' not in second and 'old_log_prob' not in second
    assert [first['reward'], second['reward']] == [.25, .75]
    assert latest_environment_decision(log) == second
    assert list(environment_decisions(EventLog.from_dataset(log.to_dataset()))) == list(environment_decisions(log))


def test_empty_event_log_roundtrips_with_its_schema():
    log = EventLog.from_dataset(EventLog().to_dataset())
    assert len(log) == 0 and len(log.to_store()) == 0
    with pytest.raises(ValueError, match='no completed environment observation'):
        latest_environment_decision(log)


def test_flat_snapshot_datasets_are_not_accepted_as_event_logs():
    with pytest.raises(ValueError, match='provider/payload'):
        EventLog.from_dataset([{'action': None, 'observation': 1}])


def test_loader_counts_source_events_and_tokenizer_maps_readouts():
    from mouse_core.data import DataLoader
    store = collect().to_store()
    before = store.to_dataset().to_list()
    assert len(store) == 20
    with DataLoader(stores=store, events_budget=10, batch_size=1,
                    sample_ranges=environment_episode_ranges, transform=tokenizer().encode, num_workers=0, seed=3) as loader:
        inputs, data, groups = loader.next_batch()
        assert inputs.N == 4
        assert groups.tolist() == [0, 0, 0, 0]
        assert inputs.head_step_ids.tolist() == list(range(4))
        assert data['action_executed'].tolist() == [False, True, False, True]
        assert data['reward'].tolist() == [0., .25, .75, 2.]
    assert store.to_dataset().to_list() == before


def test_loader_index_overlay_follows_environment_readouts_without_altering_source():
    from mouse_core.data import DataLoader
    store = collect().to_store()
    before = store[:]
    observed = []

    def decode(events):
        observed.append(events)
        return tok.encode(events)

    tok = Tokenizer(tokenizer=IntIdTokenizer(), input_fields=tokenizer().input_fields,
        objective_fields=[dict(input_field='source_offset'),
                          dict(input_field='episode_index', output_field='source_episode_index')],
    )
    with DataLoader(stores=store, events_budget=10, batch_size=1,
                    sample_ranges=lambda dataset: [(10, 20)], index_field='source_offset',
                    transform=decode, num_workers=0, seed=3) as loader:
        _, data, _ = loader.next_batch()
    assert data['source_offset'].tolist() == [11, 14, 16, 19]
    assert 'source_step_index' not in data
    assert data['source_episode_index'].tolist() == [1] * 4
    assert observed
    for events in observed:
        assert [event['source_offset'] for event in events] == list(range(10, 20))
        assert all('source_offset' not in json.loads(event['payload']) for event in events)
    assert store[:] == before


@pytest.mark.parametrize('name', ['reward', 'episode_index', 'action', 'events'])
def test_loader_index_overlay_cannot_overwrite_decoded_fields(name):
    from mouse_core.data import DataLoader
    store = collect().to_store()
    before = store[:]
    with DataLoader(stores=store, events_budget=10, batch_size=1,
                    sample_ranges=environment_episode_ranges, index_field=name,
                    transform=EventTransform(encode=lambda row: _step(tokenizer(), row)), num_workers=0, seed=3) as loader:
        with pytest.raises(ValueError, match='overlays cannot replace'):
            loader.next_batch()
    assert store[:] == before


def test_provider_ownership_and_episode_headers_survive_persistence():
    log = collect()
    for event in log:
        provider, payload = event['provider'], event['payload']
        if provider == 'agent':
            assert set(payload) == {'action'}
        elif provider == 'environment':
            assert not {'episode_index', 'step_index', 'episode_start'} & payload.keys()
        else:
            assert provider == 'coordinator'
            assert set(payload) <= {'episode_start'}
        assert 'episode_index' in event and 'step_index' in event
    rows = list(environment_decisions(log.to_dataset()))
    assert [(row['episode_index'], index % 4, row.get('action')) for index, row in enumerate(rows)] == [
        (episode, step, action) for episode in range(2)
        for step, action in enumerate((None, 1, None, 0))
    ]


def test_mutating_a_readout_cannot_change_subsequent_decoding():
    log = collect()
    expected = list(environment_decisions(log))
    iterator = environment_decisions(log)
    row = next(iterator)
    row.update(episode_index=99, action_requested=False, terminated=True)
    assert list(iterator) == expected[1:]


def test_pending_request_and_agent_wait_for_result_without_replaying_history():
    log = EventLog()
    for event in collect().events[:5]:
        if event['provider'] == 'environment':
            event['payload']['action_requested'] = True
        log.event(**event)
    completed = list(environment_decisions(log))
    log.event(provider='coordinator', episode_index=0, payload={})
    for pending in (None, {'action': 2}):
        if pending is not None:
            log.event(provider='agent', episode_index=0, payload=pending)
        assert list(environment_decisions(log)) == completed
        assert list(environment_decisions(log.to_dataset())) == completed
        with pytest.raises(ValueError, match='latest event'):
            latest_environment_decision(log)
    log.event(provider='environment', episode_index=0, payload={
        'observation': 3, 'reward': 4., 'terminated': True, 'truncated': False,
    })

    class TailOnly:
        def __len__(self): return len(log)
        def __getitem__(self, index):
            assert -6 <= index < 0
            return log[index]

    row = latest_environment_decision(TailOnly())
    assert row == list(environment_decisions(log))[-1]
    assert 'action_requested' not in row and row['terminated']
    assert row['action'] == 2 and row['reward'] == 4.
    assert [event['provider'] for event in json.loads(row['events'])] == [
        'coordinator', 'agent', 'environment',
    ]


def test_sparse_rewards_and_flags_preserve_views_tokens_and_sampling():
    dense = collect()
    sparse = EventLog()
    false_flags = {'episode_start', 'terminated', 'truncated',
                   'inner_terminated', 'inner_truncated'}
    for event in dense:
        event['payload'] = {
            key: value for key, value in event['payload'].items()
            if not (key in false_flags and value is False or key == 'reward' and value == 0)
        }
        sparse.event(**event)
    original = sparse.events
    dense_rows = list(environment_decisions(dense))
    sparse_rows = list(environment_decisions(sparse.to_dataset()))
    tok = tokenizer()
    for dense_row, sparse_row in zip(dense_rows, sparse_rows, strict=True):
        assert {key: value for key, value in dense_row.items() if key != 'events'
                and not (key in false_flags and value is False or key == 'reward' and value == 0)} == {
            key: value for key, value in sparse_row.items() if key != 'events'
        }
        left, right = _step(tok, dense_row), _step(tok, sparse_row)
        np.testing.assert_array_equal(left.token_ids, right.token_ids)
        np.testing.assert_array_equal(left.numeric_values, right.numeric_values)
        assert left.objective_fields == right.objective_fields
    assert list(environment_episode_ranges(sparse.to_dataset())) == [(0, 10), (10, 20)]
    assert latest_environment_decision(sparse) == sparse_rows[-1]
    assert sparse.events == original
    assert sparse[8]['payload']['action'] == 0


def test_episode_header_groups_events_with_logged_steps():
    from mouse_core.data import DataLoader
    store = collect().to_store()
    with DataLoader(stores=store, sample_field='episode_index', events_budget=10,
                    batch_size=1, transform=tokenizer().encode,
                    num_workers=0, seed=3) as loader:
        inputs, labels, groups = loader.next_batch()
    assert inputs.N == 4
    assert groups.tolist() == [0] * 4
    assert inputs.head_step_ids.tolist() == list(range(4))
    assert len(set(labels['source_episode_index'].tolist())) == 1
    assert all('step_index' in event and 'step_index' not in json.loads(event['payload'])
               for event in store[:])


@pytest.mark.parametrize('metadata', [{'episode_index': -1}, {'episode_index': True},
                                      {'episode_index': 1.5}, {'step_index': -1}, {'step_index': True}])
def test_invalid_sampling_headers_are_rejected_before_append(metadata):
    log = EventLog()
    with pytest.raises(ValueError):
        log.event(provider='sensor', payload={}, **metadata)
    assert len(log) == 0


def test_generic_and_indexed_events_keep_their_metadata_through_arrow():
    log = EventLog()
    log.event(provider='sensor', payload='before')
    log.event(provider='sensor', payload='during', episode_index=2)
    assert list(log[-1]) == ['episode_index', 'provider', 'payload']
    assert log.to_dataset().column_names == ['episode_index', 'provider', 'payload']
    assert log.to_store().columns == ['episode_index', 'provider', 'payload']
    assert log.to_store().to_dataset().column_names == ['episode_index', 'provider', 'payload']
    restored = EventLog.from_dataset(log.to_store().to_dataset())
    assert restored.events == log.events


def test_omitting_empty_coordinators_preserves_views_tokens_and_sampling():
    from mouse_core.data import DataLoader
    dense = collect()
    sparse = EventLog()
    live = []
    for event in dense:
        if event['provider'] == 'coordinator' and not event['payload']:
            continue
        sparse.event(**event)
        if event['provider'] == 'environment':
            live.append(latest_environment_decision(sparse))
    replay = list(environment_decisions(sparse.to_dataset()))
    assert live == replay
    assert len(sparse) == 14
    assert [len(json.loads(row['events'])) for row in replay] == [2, 2, 1, 2] * 2
    encode = tokenizer()
    for before, after in zip(environment_decisions(dense), replay, strict=True):
        assert {k: v for k, v in before.items() if k != 'events'} == {
            k: v for k, v in after.items() if k != 'events'
        }
        old, new = _step(encode, before), _step(encode, after)
        np.testing.assert_array_equal(old.token_ids, new.token_ids)
        np.testing.assert_array_equal(old.numeric_values, new.numeric_values)
        assert old.objective_fields == new.objective_fields
    with DataLoader(stores=sparse.to_store(), sample_field='episode_index', events_budget=7,
                    batch_size=1, transform=encode.encode, num_workers=0, seed=3) as loader:
        inputs, labels, _ = loader.next_batch()
    assert inputs.N == 4
    assert labels['action_executed'].tolist() == [False, True, False, True]


@pytest.mark.parametrize('invalid', ['wrong_agent_episode', 'coordinator_after_agent', 'missing_start'])
def test_optional_coordinator_does_not_weaken_order_or_episode_validation(invalid):
    events = [event for event in collect() if event['provider'] != 'coordinator' or event['payload']]
    if invalid == 'wrong_agent_episode':
        events[2]['episode_index'] = 3
    elif invalid == 'coordinator_after_agent':
        events.insert(3, {'provider': 'coordinator', 'episode_index': 0, 'payload': {'component': 0}})
    else:
        events.pop(0)
    log = EventLog()
    for event in events:
        log.event(**event)
    with pytest.raises(ValueError):
        list(environment_decisions(log))
