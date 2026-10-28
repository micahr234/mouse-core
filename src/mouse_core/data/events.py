"""Generic ordered event recording."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from typing import Any, SupportsIndex, overload

from datasets import Dataset, Features, Value


def _plain(value) -> Any:
    """JSON-compatible copy, preserving vector shapes including length one."""
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, (str, int)) for k in value) or len({str(k) for k in value}) != len(value):
            raise TypeError("Event object keys must have distinct JSON string representations")
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return deepcopy(value)


def _json(value):
    return json.dumps(_plain(value), separators=(",", ":"), allow_nan=False)


INDEX_FIELDS = ("episode_index", "step_index")
EVENT_FIELDS = ("provider", "payload")


def _indices(values):
    """Validate optional sampling metadata without interpreting event payloads."""
    if not set(values) <= set(INDEX_FIELDS):
        raise ValueError("Sampling metadata supports only episode_index and step_index")
    indices = {}
    for key, value in values.items():
        if value is None:
            continue
        # Native integers already have the persisted JSON representation. Keep
        # normalization for numpy scalars and other supported input wrappers.
        # Very large integers still pass through JSON so its configured digit
        # conversion limit is enforced just as it is for other event fields.
        if type(value) is not int or value.bit_length() > 1024:
            value = json.loads(_json(value))
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        indices[key] = value
    return indices


def _dataset(events):
    fields = {}
    fields.update({key: Value("int64") for key in INDEX_FIELDS
                   if any(key in event for event in events)})
    fields.update({key: Value("string") for key in EVENT_FIELDS})
    columns = {key: [event.get(key) for event in events] for key in fields}
    return Dataset.from_dict(columns, features=Features(fields))


class EventLog:
    """An ordered list of provider/payload records, with no control protocol.

    Providers are nonempty strings. Payloads can be any finite JSON value.
    Optional top-level episode_index and step_index support sampling. Collectors
    stamp every event with its episode and step; step indices may restart in a
    new episode. The logger never infers boundaries from provider names.
    The log snapshots records without interpreting their contents; episode,
    action, observation, and other meanings belong to callers and decoders.
    """
    def __init__(self, *, name: str | None = None):
        self.name = name
        self._records: list[tuple[dict, str]] = []

    def __len__(self):
        return len(self._records)

    @overload
    def __getitem__(self, index: slice) -> list[dict]: ...

    @overload
    def __getitem__(self, index: SupportsIndex) -> dict: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return deepcopy([event for event, _ in self._records[index]])
        return deepcopy(self._records[index][0])

    @property
    def events(self):
        return deepcopy([event for event, _ in self._records])

    @property
    def last_event(self):
        return self[-1]

    def _append(self, *, provider: str, payload, **indices):
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be a nonempty string")
        headers = _indices(indices)
        encoded = _json(payload)
        event = {**headers, "provider": provider, "payload": json.loads(encoded)}
        # One append keeps the owned snapshot and its encoding together even
        # when independent producers share a log.
        self._records.append((event, encoded))
        return event

    def append(self, *, provider: str, payload, **indices) -> None:
        """Append an owned snapshot without constructing a returned copy."""
        self._append(provider=provider, payload=payload, **indices)

    def event(self, *, provider: str, payload, **indices):
        """Append and return a snapshot. Both provider and payload are explicit."""
        return deepcopy(self._append(provider=provider, payload=payload, **indices))

    def to_dataset(self):
        return _dataset([
            {**event, "payload": payload}
            for event, payload in self._records
        ])

    @classmethod
    def from_dataset(cls, dataset, *, name=None):
        if hasattr(dataset, "column_names") and not set(EVENT_FIELDS) <= set(dataset.column_names) <= set(EVENT_FIELDS + INDEX_FIELDS):
            raise ValueError("Expected provider/payload event records with JSON-encoded payloads")
        log = cls(name=name)
        for record in dataset:
            if not set(EVENT_FIELDS) <= set(record) <= set(EVENT_FIELDS + INDEX_FIELDS) or not isinstance(record["payload"], str):
                raise ValueError("Expected provider/payload event records with JSON-encoded payloads")
            log.append(provider=record["provider"], payload=json.loads(record["payload"]),
                      **{key: value for key, value in record.items() if key not in EVENT_FIELDS})
        return log

    def to_store(self):
        """Store source records without interpreting their payloads."""
        from mouse_core.data.datastore import Datastore
        store = Datastore(name=self.name)
        store.extend_event_log(self)
        return store
