"""Ordered provider/payload storage, independent of any event protocol."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict

from mouse_core.data.events import EVENT_FIELDS, INDEX_FIELDS, _dataset, _indices, _json

_REQUIRED = EVENT_FIELDS


def _normalize_value(value: Any) -> Any:
    item = getattr(value, 'item', None)
    return item() if item is not None and getattr(value, 'ndim', None) == 0 else value


class Datastore:
    """An ordered event log. ``len`` and indexing count source events."""
    def __init__(self, *, name: str | None = None):
        self.name = name
        self._events: list[dict] = []

    def __len__(self):
        return len(self._events)

    def __repr__(self):
        name = '' if self.name is None else f'name={self.name!r}, '
        return f'Datastore({name}events={len(self)})'

    @property
    def columns(self):
        return [*(key for key in INDEX_FIELDS if any(key in event for event in self._events)), *_REQUIRED]

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            return deepcopy(self._events[indices])
        return [deepcopy(self._events[int(i)]) for i in np.asarray(indices).ravel()]

    def append(self, *, data):
        """Append a raw event, or concatenate stores without changing payloads."""
        if isinstance(data, Datastore):
            self.from_dataset(ds=data[:])
            return
        if isinstance(data, list):
            if not all(isinstance(store, Datastore) for store in data):
                raise TypeError('append expects an event dict, Datastore, or list of Datastores')
            for store in data:
                self.append(data=store)
            return
        self._append_event(data, encoded=False)

    def _append_event(self, event, *, encoded):
        if not isinstance(event, dict):
            raise TypeError('append expects an event dict')
        if not set(_REQUIRED) <= set(event) <= set(_REQUIRED + INDEX_FIELDS):
            raise ValueError('Datastore accepts only event logs with provider and payload fields')
        if not isinstance(event['provider'], str) or not event['provider']:
            raise ValueError('provider must be a nonempty string')
        payload = event['payload']
        if encoded:
            if not isinstance(payload, str):
                raise ValueError('Persisted event payload must be a JSON-encoded string')
            payload = json.loads(payload)
        indices = _indices({key: value for key, value in event.items() if key not in _REQUIRED})
        self._events.append({**indices, 'provider': event['provider'], 'payload': _json(payload)})

    def from_dataset(self, *, ds):
        """Append persisted records whose payloads are JSON-encoded strings."""
        if isinstance(ds, DatasetDict):
            raise TypeError('Select a dataset split explicitly before loading an event store')
        if isinstance(ds, Dataset) and not set(_REQUIRED) <= set(ds.column_names) <= set(_REQUIRED + INDEX_FIELDS):
            raise ValueError('Datastore accepts only event logs with provider and payload fields')
        # Roll back the entire ingestion on failure.
        length = len(self)
        try:
            for event in ds:
                self._append_event(event, encoded=True)
        except Exception:
            del self._events[length:]
            raise

    def to_dataset(self):
        """The sole persisted format: sparse event envelopes with JSON payloads."""
        return _dataset(self._events, encoded=True)

    def clear(self):
        self._events.clear()
