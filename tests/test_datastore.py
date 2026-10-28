"""Only sparse events cross the persistence boundary."""
import json
import numpy as np
import pytest
from datasets import Dataset, DatasetDict, load_from_disk
from mouse_core.data import Datastore, EventLog, EventTransform, environment_decisions


def stream():
    log = EventLog(name='cartpole')
    log.event(provider='coordinator', episode_index=0, payload={
        'episode_start': True,
    })
    log.event(provider='environment', episode_index=0, payload={
        'terminated': False,
        'truncated': False,
        'observation': [0.0],
        "action_requested": True,
    })
    log.event(provider='coordinator', episode_index=0, payload={
    })
    log.event(provider='agent', episode_index=0, payload={
        'action': np.array([.5]),
    })
    log.event(provider='environment', episode_index=0, payload={
        'terminated': True,
        'truncated': False,
        'observation': [1.0],
        'reward': 2.0,
    })
    return log


def test_empty_named_and_clear():
    store = Datastore(name='cartpole')
    assert repr(store) == "Datastore(name='cartpole', events=0)"
    assert store[:] == []
    assert len(store.to_dataset()) == 0
    store.from_dataset(ds=stream().to_dataset())
    store.clear()
    assert not len(store)


@pytest.mark.parametrize('flat', [{}, {'action': 1, 'observation': 2}, {'events': '[]', 'step_index': 0}])
def test_flat_rows_rejected_on_append_and_load(flat):
    store = Datastore()
    with pytest.raises(ValueError, match='only event logs'):
        store.append(data=flat)
    with pytest.raises(ValueError, match='only event logs'):
        store.from_dataset(ds=Dataset.from_list([flat]))
    assert not len(store)


def test_roundtrip_persists_only_events_and_keeps_vector_shape(tmp_path):
    store = stream().to_store()
    assert len(store) == 5
    ds = store.to_dataset()
    assert ds.column_names == ['episode_index', 'provider', 'payload']
    ds.save_to_disk(str(tmp_path/'events'))
    restored = Datastore()
    restored.from_dataset(ds=load_from_disk(str(tmp_path/'events')))
    assert restored[:] == store[:]
    decisions = list(environment_decisions(restored[:]))
    assert len(decisions) == 2
    assert [d.get('action') for d in decisions] == [None, [.5]]
    assert [d['observation'] for d in decisions] == [[0.], [1.]]
    assert [json.loads(e['payload']).get('action') for e in restored[:]] == [None, None, None, [.5], None]
    assert 'action' not in ds.column_names


@pytest.mark.parametrize('provider', [None, '', 1, True])
def test_invalid_provider_rejected_on_append_and_load(provider):
    event = {'provider': provider, 'payload': {'temperature': 21.5}}
    store = Datastore()
    with pytest.raises(ValueError, match='provider must be a nonempty string'):
        store.append(data=event)
    with pytest.raises(ValueError, match='provider must be a nonempty string'):
        store.from_dataset(ds=[event])
    assert not len(store)


def test_custom_providers_roundtrip_without_environment_protocol():
    store = Datastore()
    for i, provider in enumerate(['thermometer', 'humidity_sensor']):
        store.append(data={'provider': provider, 'payload': {'reading': i}})
    restored = Datastore()
    restored.from_dataset(ds=store.to_dataset())
    assert restored[:] == store[:]
    assert [event['provider'] for event in restored[:]] == ['thermometer', 'humidity_sensor']


def test_missing_provider_rejected_on_append_and_load():
    event = {'payload': {'temperature': 21.5}}
    store = Datastore()
    with pytest.raises(ValueError, match='only event logs'):
        store.append(data=event)
    with pytest.raises(ValueError, match='only event logs'):
        store.from_dataset(ds=Dataset.from_list([event]))
    assert not len(store)


def test_indexing_copies_and_does_not_mutate_source():
    store = stream().to_store()
    assert store[-1] == store[4]
    assert store[[2,0]] == store[2] + store[0]
    assert store[::-1] == list(reversed(store[:]))
    with pytest.raises(IndexError): store[5]
    with pytest.raises(IndexError): store[-6]
    got = store[0]
    got[0]['payload'] = '{}'
    assert store[0] != got
    decisions = list(environment_decisions(store[:]))
    decisions[0]['observation'][0] = 99
    assert list(environment_decisions(store[:]))[0]['observation'] == [0.]


def test_merge_concatenates_without_rewriting_payloads():
    store = stream().to_store()
    original = store[:]
    store.append(data=[stream().to_store(), stream().to_store()])
    assert store[:] == original * 3


@pytest.mark.parametrize('payload', ['hello', 'null', '{"value":2}', None, 2, [None, 1], {'reading': [1.]}])
def test_any_json_payload_roundtrips_without_reinterpreting_strings(payload):
    log = EventLog()
    event = log.event(provider='sensor', payload=payload)
    store = Datastore()
    store.append(data=event)
    restored = EventLog.from_dataset(store.to_dataset())
    assert restored.events == log.events
    store.from_dataset(ds=log.to_dataset())
    assert EventLog.from_dataset(store.to_dataset()).events == log.events * 2


def test_import_failure_is_atomic_and_timestamps_do_not_order():
    store = stream().to_store()
    before = store[:]
    records = stream().to_dataset().to_list()
    records[-1]['payload'] = '{broken'
    with pytest.raises(ValueError): store.from_dataset(ds=records)
    assert store[:] == before
    timed = Datastore()
    for i,e in enumerate(before):
        timed.append(data={**e, 'payload': {**json.loads(e['payload']), 'timestamp': str(2099 - i * 100)}})
    assert len(timed) == 5
    assert "timestamp" not in timed.columns
    with pytest.raises(ValueError, match="only event logs"):
        Datastore().append(data={**before[0], "timestamp": "2099"})
    with pytest.raises(TypeError, match='split explicitly'):
        Datastore().from_dataset(ds=DatasetDict({'train': store.to_dataset()}))


def test_projection_labels_never_leak_into_saved_events():
    store = stream().to_store()
    source = store.to_dataset().to_list()
    encoded = EventTransform(encode=lambda row: {**row, 'advantage': 3.})(store[:])
    assert all(row['advantage'] == 3. for row in encoded)
    assert store.to_dataset().to_list() == source
