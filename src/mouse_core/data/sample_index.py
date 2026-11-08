"""Reusable candidate ranges and shared, lazy sample-check results.

Callbacks are deterministic functions of a sample, must not mutate it, and must
not depend on training state. Changing a callback requires a new callable.
Only status bytes are shared; decoded payloads are never retained here.
"""
from __future__ import annotations

from functools import partial
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.util import Finalize

import numpy as np


class _Identity:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        return isinstance(other, _Identity) and self.value is other.value


def _policy_key(value):
    if isinstance(value, partial):
        return (_policy_key(value.func), tuple(_policy_key(v) for v in value.args),
                tuple(sorted((k, _policy_key(v)) for k, v in (value.keywords or {}).items())))
    if callable(value):
        return _Identity(value)
    try:
        hash(value)
    except TypeError:
        return _Identity(value)
    return (type(value), value)


def _release_memory(memory, owner):
    # multiprocessing finalizers run before interpreter module teardown, even
    # when stores/checks remain reachable. Workers only close their attachment.
    try:
        memory.close()
    finally:
        if owner:
            memory.unlink()


class SampleChecks:
    """One byte per candidate: unknown, accepted, or rejected.

    Parent-owned shared memory survives worker restarts and loader rebuilds.
    Workers attach without owning/unlinking it. Concurrent first checks may
    repeat work, but write the same result; RNG never depends on cache warmth.
    """
    def __init__(self, size, sample_eligible):
        self.sample_eligible = sample_eligible
        self._memory = SharedMemory(create=True, size=max(1, size))
        self.status = np.ndarray((size,), dtype=np.uint8, buffer=self._memory.buf)
        self.status.fill(0)
        self._finalizer = Finalize(self, _release_memory, (self._memory, True), exitpriority=0)

    def __getstate__(self):
        return (self._memory.name, len(self.status), self.sample_eligible)

    def __setstate__(self, state):
        name, size, self.sample_eligible = state
        self._memory = SharedMemory(name=name, track=False)
        self.status = np.ndarray((size,), dtype=np.uint8, buffer=self._memory.buf)
        self._finalizer = Finalize(self, _release_memory, (self._memory, False), exitpriority=0)

    def accepts(self, index, events):
        result = self.status[index]
        if not result:
            # Errors propagate, never become a rejection or a cached success.
            accepted = True if self.sample_eligible is None else self.sample_eligible(events)
            if type(accepted) is not bool:
                raise TypeError("sample_eligible must return a bool")
            result = 1 if accepted else 2
            self.status[index] = result
        return result == 1


def sample_index(store, *, build_ranges, policy, sample_eligible=None):
    """Reuse unchanged indexes; retain check results for unchanged appended ranges."""
    key = tuple(_policy_key(v) for v in (*policy, sample_eligible))
    previous = store._sample_indexes.get(key)
    if previous is not None and previous[0] == store._revision:
        return previous[1:]
    dataset = store.to_dataset()
    ranges = build_ranges(dataset)
    checks = SampleChecks(len(ranges), sample_eligible)
    if previous is not None:
        old = {tuple(span): status for span, status in zip(previous[2], previous[3].status)}
        checks.status[:] = [old.get(tuple(span), 0) for span in ranges]
    result = (store._revision, dataset, ranges, checks)
    store._sample_indexes[key] = result
    return result[1:]
