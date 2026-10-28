"""Synthetic measurement events for testing sampler mechanics independently of RL."""
import json
from mouse_core.data import Datastore


def measurement_replay(events):
    for event in events:
        yield {**{key: value for key, value in event.items() if key not in ('provider', 'payload')},
               **json.loads(event['payload'])}


def measurement_store(*, name=None):
    return Datastore(name=name)


def append_measurement(store, *, fields):
    # Sampler labels are measurements in this synthetic domain. Production
    # agent/request alignment is tested with real EventLog streams in test_events.
    store.append(data={'provider': 'sensor', 'payload': fields})


def measurement_events(payloads):
    """One marked sensor event per measurement payload."""
    return [{'provider': 'sensor', 'step_index': i, 'payload': payload}
            for i, payload in enumerate(payloads)]


def encode_measurement(tokenizer, payload):
    """Encode one synthetic measurement through the public event API."""
    return tokenizer.encode(measurement_events([payload]))[0]
