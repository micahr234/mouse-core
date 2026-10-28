"""Owned recording, validation-only readers, and atomic event transfers."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mouse_core.data import (
    Datastore, EventLog, environment_decisions, latest_environment_decision,
    latest_environment_packet, validate_environment_events,
    validate_latest_environment_step,
)


def episode():
    log = EventLog()
    log.append(provider='coordinator', episode_index=0, step_index=0,
               payload={'episode_start': True})
    log.append(provider='environment', episode_index=0, step_index=0,
               payload={'observation': [0.], 'info': {'map': ['SFG'], 'q_star': [1., 0.]},
                        'action_requested': True})
    log.append(provider='agent', episode_index=0, step_index=1,
               payload={'action': 0, 'note': 'quoted " and newline\n'})
    log.append(provider='environment', episode_index=0, step_index=1,
               payload={'observation': [1.], 'reward': 1., 'terminated': True})
    return log


def test_append_owns_nested_payload_and_cached_export():
    values = np.array([1., 2.])
    payload = {'vector': values, 'nested': {'pair': (True, None)}}
    log = EventLog()
    assert log.append(provider='sensor', payload=payload) is None
    values[0] = 9.
    payload['nested']['pair'] = ('changed',)
    log.events[0]['payload']['vector'][1] = 8.
    expected = {'vector': [1., 2.], 'nested': {'pair': [True, None]}}
    assert log[0]['payload'] == expected
    assert json.loads(log.to_dataset()[0]['payload']) == expected
    assert json.loads(log.to_store()[0][0]['payload']) == expected


@pytest.mark.parametrize('invalid', [
    {'provider': ''}, {'payload': {'bad': float('nan')}}, {'step_index': -1},
])
def test_failed_append_keeps_export_aligned(invalid):
    log = EventLog()
    log.append(provider='sensor', payload={'value': 1})
    with pytest.raises(ValueError):
        log.append(**{'provider': 'sensor', 'payload': {}, **invalid})
    log.append(provider='sensor', payload={'value': 2})
    assert [json.loads(row['payload']) for row in log.to_dataset().to_list()] == [
        {'value': 1}, {'value': 2},
    ]


@pytest.mark.parametrize('source', ['log', 'raw', 'encoded'])
def test_validation_and_packet_match_complete_decoded_source(source):
    log = episode()
    events = {'log': log, 'raw': log.events, 'encoded': log.to_dataset().to_list()}[source]
    expected = list(environment_decisions(log))[-1]
    assert validate_environment_events(events, require_complete=True) is None
    assert validate_latest_environment_step(events) is None
    assert latest_environment_packet(events) == expected['events']
    assert latest_environment_decision(events) == expected


@pytest.mark.parametrize('count', [0, 1, 2, 3])
def test_complete_validation_rejects_empty_or_unfinished_episode(count):
    events = episode().events[:count]
    validate_environment_events(events)
    with pytest.raises(ValueError, match='incomplete episode'):
        validate_environment_events(events, require_complete=True)


@pytest.mark.parametrize('mutation', ['missing_agent', 'wrong_index', 'bad_flag', 'collision'])
def test_validation_only_rejects_same_protocol_errors_as_decoding(mutation):
    events = episode().events
    if mutation == 'missing_agent':
        events.pop(2)
    elif mutation == 'wrong_index':
        events[2]['step_index'] = 99
    elif mutation == 'bad_flag':
        events[-1]['payload']['terminated'] = 'yes'
    else:
        events[2]['payload']['observation'] = 9
    with pytest.raises(ValueError):
        list(environment_decisions(events))
    with pytest.raises(ValueError):
        validate_environment_events(events, require_complete=True)
    with pytest.raises(ValueError):
        validate_latest_environment_step(events)


def test_owned_log_transfer_remaps_only_episode_header_without_aliasing():
    log = episode()
    original = log.to_dataset().to_list()
    store = Datastore()
    store.extend_event_log(log, episode_index=np.int64(7))
    expected = [{**event, 'episode_index': 7} for event in original]
    assert store[:] == expected
    assert log.to_dataset().to_list() == original
    log.events[-1]['payload']['observation'][0] = 99
    store[:][-1]['payload'] = '{}'
    log.append(provider='sensor', payload={'new': True})
    assert store[:] == expected
    assert type(store[0][0]['episode_index']) is int


@pytest.mark.parametrize('index', [-1, True, 1.5, float('nan')])
def test_invalid_transfer_remap_does_not_append(index):
    store = episode().to_store()
    original = store[:]
    with pytest.raises(ValueError):
        store.extend_event_log(episode(), episode_index=index)
    assert store[:] == original


@pytest.mark.parametrize('failure', ['invalid_record', 'iterator_exception'])
def test_raw_batch_failure_rolls_back_all_new_events(failure):
    store = episode().to_store()
    original = store[:]

    def records():
        yield {'provider': 'sensor', 'payload': [1]}
        if failure == 'iterator_exception':
            raise RuntimeError('source failed')
        yield {'provider': '', 'payload': [2]}

    with pytest.raises((ValueError, RuntimeError)):
        store.extend(records())
    assert store[:] == original


def test_raw_and_owned_batches_preserve_arbitrary_json_payloads():
    log = EventLog()
    payloads = ['{"value":2}', None, [1], {3: np.array([4.])}, False]
    for payload in payloads:
        log.append(provider='sensor', payload=payload)
    raw = Datastore()
    raw.extend(iter(log.events))
    owned = Datastore()
    owned.extend_event_log(log)
    assert raw[:] == owned[:] == log.to_dataset().to_list()
    assert EventLog.from_dataset(owned.to_dataset()).events == log.events


def test_provider_subclass_keeps_existing_decode_and_persistence_normalization():
    class Provider(str):
        def __eq__(self, other):
            return False

        __hash__ = str.__hash__

        def __str__(self):
            return 'persisted-provider'

    log = EventLog()
    log.append(provider=Provider('coordinator'), episode_index=0,
               payload={'episode_start': True})
    log.append(provider=Provider('environment'), episode_index=0,
               payload={'terminated': True})
    validate_environment_events(log, require_complete=True)
    validate_latest_environment_step(log)
    assert json.loads(latest_environment_packet(log))[-1]['provider'] == 'environment'
    persisted = log.to_dataset().to_list()
    assert log.to_store()[:] == persisted


def test_owned_transfer_rolls_back_when_provider_normalization_fails():
    class Provider(str):
        def __str__(self):
            raise ValueError('provider conversion failed')

    log = EventLog()
    log.append(provider='sensor', payload=[1])
    log.append(provider=Provider('sensor'), payload=[2])
    store = episode().to_store()
    original = store[:]
    with pytest.raises(ValueError, match='provider conversion failed'):
        store.extend_event_log(log)
    assert store[:] == original


def test_concurrent_appends_return_their_own_snapshot_and_preserve_encodings():
    log = EventLog()

    def produce(worker):
        for index in range(100):
            payload = {'worker': worker, 'values': [index, index + 1]}
            if index % 2:
                result = log.event(provider='sensor', payload=payload)
                assert result['payload'] == payload
                result['payload']['values'][0] = -1
            else:
                log.append(provider='sensor', payload=payload)
            payload['values'][0] = -2

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(produce, range(4)))
    events = log.events
    persisted = log.to_store()[:]
    assert len(events) == len(persisted) == 400
    assert [event['payload'] for event in events] == [
        json.loads(event['payload']) for event in persisted
    ]
    assert {(event['payload']['worker'], tuple(event['payload']['values'])) for event in events} == {
        (worker, (index, index + 1)) for worker in range(4) for index in range(100)
    }
