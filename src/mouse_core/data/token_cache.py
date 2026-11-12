"""Explicit, disk-backed caches of already-tokenized samples.

Only cache deterministic, complete samples. The caller supplies immutable keys
and a content version covering the tokenizer, projection, objective labels, and
every other dependency of the encoded result. Include random seeds and context
in those identities when relevant; this class never caches transforms implicitly.
Cache files are trusted local pickle files, not an interchange format.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from typing import Any
import weakref

import numpy as np
import torch

from .token_batch import StepTokens


_FORMAT = 1


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a nonempty string')
    value = str.__str__(value)
    if not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _objective_copy(value: Any, active: set[int] | None = None) -> Any:
    """Detach numeric objective values without retaining a producer's graph."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise TypeError('TokenCache objective arrays must contain real numeric values')
        return np.array(value, copy=True, subok=False)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, np.floating):
        return np.array(value, copy=True, subok=False)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        if active is None:
            active = set()
        if id(value) in active:
            raise ValueError('TokenCache objective sequences must not contain cycles')
        active.add(id(value))
        try:
            items = [_objective_copy(item, active) for item in value]
            return items if isinstance(value, list) else tuple(items)
        finally:
            active.remove(id(value))
    raise TypeError('TokenCache objectives must be numeric scalars, arrays, sequences, or None')


def _snapshot(steps: Sequence[StepTokens]) -> list[StepTokens]:
    def optional_array(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else np.array(value, copy=True)
    result = []
    for step in steps:
        if not isinstance(step, StepTokens):
            raise TypeError('TokenCache samples must contain StepTokens')
        if any(not isinstance(name, str) for name in step.objective_fields):
            raise TypeError('TokenCache objective field names must be strings')
        tokenizer_names = tuple(_identity(name, 'tokenizer name') for name in step.tokenizer_names)
        numeric_names = tuple(_identity(name, 'numeric field name') for name in step.numeric_field_names)
        result.append(StepTokens(
            token_ids=np.array(step.token_ids, copy=True),
            numeric_values=np.array(step.numeric_values, copy=True),
            numeric_mask=np.array(step.numeric_mask, copy=True),
            head_mask=np.array(step.head_mask, copy=True),
            tokenizer_ids=optional_array(step.tokenizer_ids),
            numeric_field_ids=optional_array(step.numeric_field_ids),
            position_values=optional_array(step.position_values),
            position_mask=optional_array(step.position_mask),
            objective_fields={str.__str__(name): _objective_copy(value) for name, value in step.objective_fields.items()},
            tokenizer_names=tokenizer_names, numeric_field_names=numeric_names,
        ))
    return result


def _resident_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Account for owned arrays and Python containers, once per object."""
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        # Include any backing allocation instead of undercounting a view.
        if value.base is not None:
            size += _resident_bytes(value.base, seen)
    elif isinstance(value, memoryview):
        # Protocol-5 NumPy unpickling uses memoryview-backed bytearrays. A
        # memoryview's own size excludes its buffer, which may be much larger.
        size += _resident_bytes(value.obj, seen)
    elif isinstance(value, dict):
        size += sum(_resident_bytes(key, seen) + _resident_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_resident_bytes(item, seen) for item in value)
    elif isinstance(value, StepTokens):
        size += _resident_bytes(vars(value), seen)
    return size


def _remove_directory(path: Path, owner_pid: int) -> None:
    # An inherited handle in a forked child must not delete the parent's files.
    if os.getpid() == owner_pid:
        shutil.rmtree(path, ignore_errors=True)


class TokenCache:
    """Owned temporary storage plus a byte-bounded, per-process decoded LRU.

    ``put(key, steps)`` writes an immutable sample atomically; duplicate keys
    raise ``KeyError``. Keys must identify complete source samples, and
    ``content_version`` must change whenever tokenization or labels change.
    ``get`` returns independent mutable ``StepTokens`` suitable for packing or
    augmentation. Changes to input samples, returned samples, or packed tensors
    cannot alter the cache.

    ``max_memory_bytes`` bounds accounted *retained decoded samples*, including
    their arrays and Python containers. It does not bound a caller's returned
    samples, transient serialization/copies, interpreter overhead, or disk use.
    Zero disables decoded retention. Oversized samples are read without retention.

    Construction creates a unique directory under ``directory`` (or the system
    temporary directory). The constructing process owns writes and cleanup.
    Pickling/spawning creates read-only, non-owning handles containing only paths,
    version, and memory budget; warmed samples and producer objects are excluded.
    Keep the owner open until readers/workers have stopped. Closing a reader only
    releases its local memory; closing the owner removes its files. Use a context
    manager or ``close`` for prompt cleanup. Separate reader handles are intended
    for separate workers; concurrent ``put`` calls commit without overwriting,
    but access to one handle's decoded LRU must be externally serialized.
    """

    def __init__(self, *, content_version: str, max_memory_bytes: int,
                 directory: str | Path | None = None):
        self._content_version = _identity(content_version, 'content_version')
        if type(max_memory_bytes) is not int or max_memory_bytes < 0:
            raise ValueError('max_memory_bytes must be a nonnegative integer')
        self._max_memory_bytes = max_memory_bytes
        self._path = Path(tempfile.mkdtemp(prefix='mouse-token-cache-', dir=directory))
        self._owner_pid: int | None = os.getpid()
        self._finalizer = weakref.finalize(self, _remove_directory, self._path, self._owner_pid)
        self._initialize_memory()

    def _initialize_memory(self) -> None:
        self._memory: OrderedDict[str, tuple[list[StepTokens], int]] = OrderedDict()
        self._memory_bytes = 0
        self._closed = False

    @property
    def content_version(self) -> str:
        return self._content_version

    @property
    def max_memory_bytes(self) -> int:
        return self._max_memory_bytes

    @property
    def path(self) -> Path:
        """The uniquely owned cache directory; callers must not modify its files."""
        return self._path

    @property
    def memory_bytes(self) -> int:
        """Accounted bytes retained in this handle's decoded sample LRU."""
        return self._memory_bytes

    @property
    def cached_samples(self) -> int:
        return len(self._memory)

    @property
    def read_only(self) -> bool:
        return self._owner_pid != os.getpid()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError('TokenCache is closed')
        if not self._path.is_dir():
            raise RuntimeError('TokenCache storage is unavailable; the owner must outlive its readers')

    def _file(self, key: str) -> Path:
        key = _identity(key, 'key')
        return self._path / (hashlib.sha256(key.encode()).hexdigest() + '.pickle')

    def __contains__(self, key: str) -> bool:
        self._check_open()
        return self._file(key).is_file()

    def put(self, key: str, steps: Sequence[StepTokens]) -> None:
        self._check_open()
        if self.read_only:
            raise RuntimeError('Only the TokenCache owner may write samples')
        key = _identity(key, 'key')
        target = self._file(key)
        if target.exists():
            raise KeyError(f'TokenCache key already exists: {key!r}')
        snapshot = _snapshot(steps)
        envelope = {'format': _FORMAT, 'content_version': self.content_version, 'key': key, 'steps': snapshot}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._path, prefix='.pending-', delete=False) as stream:
                temporary = Path(stream.name)
                pickle.dump(envelope, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            # An exclusive atomic link publishes a complete file. Unlike replace,
            # a racing duplicate writer cannot overwrite the first committed key.
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise KeyError(f'TokenCache key already exists: {key!r}') from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def get(self, key: str) -> list[StepTokens]:
        self._check_open()
        key = _identity(key, 'key')
        path = self._file(key)
        if key in self._memory:
            self._memory.move_to_end(key)
            return _snapshot(self._memory[key][0])
        try:
            with path.open('rb') as stream:
                envelope = pickle.load(stream)
        except FileNotFoundError as error:
            raise KeyError(key) from error
        if (not isinstance(envelope, dict) or envelope.get('format') != _FORMAT
                or envelope.get('content_version') != self.content_version or envelope.get('key') != key):
            raise ValueError('TokenCache file identity or content version does not match')
        steps = envelope.get('steps')
        if not isinstance(steps, list) or any(not isinstance(step, StepTokens) for step in steps):
            raise ValueError('TokenCache file does not contain a StepTokens sample')
        size = _resident_bytes((key, steps))
        if size > self.max_memory_bytes:
            return steps
        while self._memory and self._memory_bytes + size > self.max_memory_bytes:
            _, (_, removed) = self._memory.popitem(last=False)
            self._memory_bytes -= removed
        self._memory[key] = steps, size
        self._memory_bytes += size
        return _snapshot(steps)

    def __getstate__(self) -> dict[str, Any]:
        self._check_open()
        return {'format': _FORMAT, 'path': str(self._path), 'content_version': self.content_version,
                'max_memory_bytes': self.max_memory_bytes}

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.get('format') != _FORMAT:
            raise ValueError('Unsupported TokenCache handle format')
        self._path = Path(state['path'])
        self._content_version = _identity(state['content_version'], 'content_version')
        budget = state['max_memory_bytes']
        if type(budget) is not int or budget < 0:
            raise ValueError('max_memory_bytes must be a nonnegative integer')
        self._max_memory_bytes = budget
        self._owner_pid = None
        self._finalizer = None
        self._initialize_memory()

    def close(self) -> None:
        if self._closed:
            return
        self._memory.clear()
        self._memory_bytes = 0
        self._closed = True
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self) -> TokenCache:
        self._check_open()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
