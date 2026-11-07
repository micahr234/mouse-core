"""Ordered provider/payload storage, independent of any event protocol."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, concatenate_datasets

from mouse_core.data.events import EventLog, EVENT_FIELDS, INDEX_FIELDS, _dataset, _indices, _json

_REQUIRED = EVENT_FIELDS


def _normalize_value(value: Any) -> Any:
    item = getattr(value, 'item', None)
    return item() if item is not None and getattr(value, 'ndim', None) == 0 else value


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('Event payload numbers must be finite')
    return value


def _invalid_constant(text):
    raise ValueError(f'Event payload numbers must be finite, got {text}')


def _persisted_dataset(ds):
    """Validate columns in bounded batches without rebuilding persisted records."""
    if not set(_REQUIRED) <= set(ds.column_names) <= set(_REQUIRED + INDEX_FIELDS):
        raise ValueError('Datastore accepts only event logs with provider and payload fields')
    # Own the Dataset wrapper/formatting, while retaining its Arrow buffers and
    # selection indices. Caller formatting changes must not affect this store.
    ds = ds.with_format(None)
    present = set()
    for batch in ds.iter(batch_size=8192):
        if any(not isinstance(value, str) or not value for value in batch['provider']):
            raise ValueError('provider must be a nonempty string')
        for key in INDEX_FIELDS:
            for value in batch.get(key, ()):
                if value is not None:
                    if type(value) is not int or value < 0:
                        raise ValueError(f'{key} must be a nonnegative integer')
                    present.add(key)
        for payload in batch['payload']:
            if not isinstance(payload, str):
                raise ValueError('Persisted event payload must be a JSON-encoded string')
            # Keep eager rejection of malformed/nonfinite JSON, but retain the
            # existing encoding instead of serializing every payload again.
            json.loads(payload, parse_float=_finite_float, parse_constant=_invalid_constant)
    fields = {key: Value('int64') for key in INDEX_FIELDS if key in present}
    fields.update({key: Value('string') for key in _REQUIRED})
    ds = ds.select_columns(list(fields))
    features = Features(fields)
    return ds if ds.features == features else ds.cast(features)


class Datastore:
    """Ordered events backed by immutable dataset blocks and an append buffer.

    ``len`` and indexing count source events. Imported Dataset buffers stay
    columnar; only selected rows are materialized as Python records.
    """
    def __init__(self, *, name: str | None = None):
        self.name = name
        self._events: list[dict] = []
        self._base: Dataset | None = None
        self._revision = 0
        self._sample_indexes: dict = {}

    def __len__(self):
        return (len(self._base) if self._base is not None else 0) + len(self._events)

    def __repr__(self):
        name = '' if self.name is None else f'name={self.name!r}, '
        return f'Datastore({name}events={len(self)})'

    @property
    def columns(self):
        base = self._base.column_names if self._base is not None else ()
        return [*(key for key in INDEX_FIELDS if key in base or any(key in event for event in self._events)), *_REQUIRED]

    def __getitem__(self, indices):
        if self._base is None:
            if isinstance(indices, slice):
                return deepcopy(self._events[indices])
            return [deepcopy(self._events[int(i)]) for i in np.asarray(indices).ravel()]
        size = len(self)
        if isinstance(indices, slice):
            selected = range(size)[indices]
        else:
            selected = [int(i) for i in np.asarray(indices).ravel()]
            selected = [i + size if i < 0 else i for i in selected]
            if any(i < 0 or i >= size for i in selected):
                raise IndexError('event index out of range')
        if not selected:
            return []
        rows = self.to_dataset().select(selected).to_list()
        # Arrow represents absent optional headers with nulls; store reads keep
        # the sparse event representation used by the append buffer.
        return [{key: value for key, value in row.items()
                 if key not in INDEX_FIELDS or value is not None} for row in rows]

    def append(self, *, data):
        """Append a raw event, or concatenate stores without changing payloads."""
        if isinstance(data, Datastore):
            self.from_dataset(ds=data.to_dataset())
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
        self._revision += 1

    def extend(self, events) -> None:
        """Atomically append raw events, snapshotting and encoding each once."""
        length = len(self._events)
        revision = self._revision
        try:
            for event in events:
                self._append_event(event, encoded=False)
        except Exception:
            del self._events[length:]
            self._revision = revision
            raise

    def extend_event_log(self, log: EventLog, *, episode_index=None) -> None:
        """Append owned log records, reusing their validated immutable JSON.

        An explicit episode_index replaces that envelope field on every event.
        This generic operation does not interpret providers or episode endings.
        """
        if not isinstance(log, EventLog):
            raise TypeError('extend_event_log requires an EventLog')
        replacement = {} if episode_index is None else _indices({'episode_index': episode_index})
        length = len(self._events)
        revision = self._revision
        try:
            for event, payload in log._records:
                provider = event['provider']
                # Match persisted string-column normalization for subclasses;
                # native provider strings and encoded payloads are immutable.
                if type(provider) is not str:
                    provider = str(provider)
                    if not provider:
                        raise ValueError('provider must be a nonempty string')
                self._events.append({**event, **replacement, 'provider': provider, 'payload': payload})
                self._revision += 1
        except Exception:
            del self._events[length:]
            self._revision = revision
            raise

    def from_dataset(self, *, ds):
        """Append persisted events, retaining Dataset buffers and encoded JSON.

        Dataset imports validate records in batches and share their columnar
        storage. Other iterables are copied into the append buffer. Failure
        leaves the store unchanged. JSON whitespace is preserved for datasets.
        """
        if isinstance(ds, DatasetDict):
            raise TypeError('Select a dataset split explicitly before loading an event store')
        if isinstance(ds, Dataset):
            imported = _persisted_dataset(ds)
            if len(imported):
                self._base = (concatenate_datasets([self.to_dataset(), imported])
                              if len(self) else imported)
                self._revision += 1
            return
        # Roll back the entire ingestion on failure.
        length = len(self._events)
        revision = self._revision
        try:
            for event in ds:
                self._append_event(event, encoded=True)
        except Exception:
            del self._events[length:]
            self._revision = revision
            raise

    def to_dataset(self):
        """The sole persisted format: sparse event envelopes with JSON payloads."""
        if self._events:
            appended = _dataset(self._events)
            self._base = (appended if self._base is None else
                          concatenate_datasets([self._base, appended]))
            self._events.clear()
        # A separate wrapper keeps set_format/set_transform on a returned
        # snapshot from mutating the store; Arrow buffers remain shared.
        return _dataset([]) if self._base is None else self._base.with_format(None)

    def clear(self):
        self._revision += 1
        self._sample_indexes.clear()
        self._base = None
        self._events.clear()
