"""DataLoader — sample ragged windows, apply a per-step transform, pack.

A ``Datastore`` is a flat sequence of arbitrary rows. The loader samples
``B`` sequences, each a contiguous store window of length ``1 .. sequence_length``
(a max), runs ``transform(step)`` on every step, and packs the resulting
:class:`~mouse_core.data.token_batch.StepTokens` into a
:class:`~mouse_core.data.token_batch.TokenBatch` plus a CPU
:class:`~tensordict.TensorDict` of step-level objective columns.

The loader is stage-agnostic: compose augmenter / tokenizer
(or any ``dict → StepTokens`` callable) outside and pass the result as
``transform=``. At the start of each batch fetch, if ``transform`` defines
``reseed()``, it is called as ``reseed(generation=k)`` with the batch index
(so an :class:`~mouse_core.data.augmenter.Augmenter` in the compose pipeline
draws the augmentation set that belongs to batch ``k``).

Determinism
-----------
Batches are numbered ``k = 0, 1, 2, ...`` in the order :meth:`DataLoader.next_batch`
returns them. Batch ``k`` samples its windows from
``SeedSequence(seed, spawn_key=(k,))`` and reseeds the transform with
``generation=k``, so it is a pure function of ``(seed, k, store snapshot)``:
``num_workers`` changes only throughput, never the stream. Workers claim
indices from a shared counter and the consumer hands batches out in index
order (a small reorder buffer absorbs the interleaving). :meth:`DataLoader.refresh`
discards prefetched batches and resumes numbering at the next batch the
consumer has not seen, so the sequence after a refresh depends only on when
(in batches) it was called. ``seed=None`` draws the entropy once at
construction (a fresh stream per loader, still numbered and ordered).

Usage
-----
::

    train_transform = compose(augmenter, tokenizer)
    loader = DataLoader(
        stores=store,
        sequence_length=64,
        batch_size=8,
        transform=train_transform,
    )
    inputs, objective_data = loader.next_batch()
"""

from __future__ import annotations

import queue
import sys
import sysconfig
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tensordict import TensorDict

from mouse_core.data.token_batch import StepTokens, TokenBatch, pack_token_batch

if TYPE_CHECKING:
    from mouse_core.data.datastore import Datastore


StepTransform = Callable[[dict], StepTokens]

_FREE_THREADING_HINT = (
    "DataLoader(num_workers>0) requires a free-threaded CPython build with the "
    "GIL disabled (e.g. Python 3.14t). Install with `uv python install 3.14t` "
    "and create the venv with that interpreter. If imports re-enable the GIL "
    "(Triton still does this), run with `PYTHON_GIL=0` or "
    "`python -Xgil=0`."
)


def _require_free_threading() -> None:
    """Raise unless this process can run CPU-bound worker threads in parallel."""
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError(_FREE_THREADING_HINT)
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if callable(is_gil_enabled) and is_gil_enabled():
        raise RuntimeError(_FREE_THREADING_HINT)


@dataclass(frozen=True)
class _SnapshotConfig:
    """Immutable sampling snapshot shared with worker threads."""

    datasets: tuple[Any, ...]
    ns: tuple[int, ...]
    probs: np.ndarray
    sequence_length: int
    batch_size: int
    index_field: str | None


def _fetch_sequence(
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """Fetch one contiguous window of length ``1 .. sequence_length``."""
    if sum(cfg.ns) == 0:
        raise ValueError("Cannot sample batches: all stores are empty.")

    S_max = cfg.sequence_length
    store_idx = int(rng.choice(len(cfg.datasets), p=cfg.probs))
    ds = cfg.datasets[store_idx]
    n = cfg.ns[store_idx]
    if n < 1:
        raise ValueError("Cannot sample from an empty store.")

    start = int(rng.integers(0, n))
    end = min(start + S_max, n)
    hf_slice = ds[start:end]
    count = end - start
    rows = [{k: hf_slice[k][i] for k in hf_slice} for i in range(count)]
    if cfg.index_field is not None:
        for i, row in enumerate(rows):
            row[cfg.index_field] = start + i
    return rows


def _batch_rng(entropy: int, k: int) -> np.random.Generator:
    """Window-sampling RNG of batch ``k``: ``SeedSequence(entropy, spawn_key=(k,))``."""
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(k,)))


def _fetch_one_batch(
    cfg: _SnapshotConfig,
    entropy: int,
    k: int,
    transform: StepTransform,
) -> tuple[TokenBatch, TensorDict]:
    """Build batch ``k``: reseed the transform to generation ``k``, sample, pack."""
    reseed = getattr(transform, "reseed", None)
    if callable(reseed):
        reseed(generation=k)
    rng = _batch_rng(entropy, k)
    sequences = [_fetch_sequence(cfg, rng) for _ in range(cfg.batch_size)]
    steps: list[StepTokens] = []
    sequence_ids: list[int] = []
    grouping_field: str | None = None
    for b, seq in enumerate(sequences):
        for step in seq:
            packed = transform(step)
            if grouping_field is None:
                grouping_field = packed.grouping_field
            steps.append(packed)
            sequence_ids.append(b)
    return pack_token_batch(
        steps,
        sequence_ids=sequence_ids,
        batch_size=cfg.batch_size,
        grouping_field=grouping_field,
    )


class _WorkerFailure:
    """First exception raised by any worker, delivered out-of-band.

    Errors do not travel through the (bounded) result queue: when the queue is
    full of good batches a queued error could be dropped or only surface after
    the consumer drains everything already prefetched.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.exc: BaseException | None = None

    def record(self, exc: BaseException) -> None:
        with self._lock:
            if self.exc is None:
                self.exc = exc


class _BatchCounter:
    """Hands out batch indices ``k`` to whichever worker asks next.

    One instance per worker generation: a worker that outlives
    ``_stop_workers`` keeps claiming from its own (stale) counter and can
    never punch a hole in the live numbering.
    """

    def __init__(self, start: int) -> None:
        self._lock = threading.Lock()
        self._next = int(start)

    def claim(self) -> int:
        with self._lock:
            k = self._next
            self._next += 1
            return k


def _worker_loop(
    result_queue: queue.Queue,
    stop_event: threading.Event,
    failure: _WorkerFailure,
    counter: _BatchCounter,
    cfg: _SnapshotConfig,
    entropy: int,
    transform: StepTransform,
) -> None:
    """Prefetch loop run inside a worker thread: claim ``k``, build batch ``k``, enqueue ``(k, item)``."""
    while not stop_event.is_set():
        k = counter.claim()
        try:
            item = _fetch_one_batch(cfg, entropy, k, transform)
        except Exception as exc:  # noqa: BLE001
            failure.record(exc)
            return
        while not stop_event.is_set():
            try:
                result_queue.put((k, item), timeout=0.05)
                break
            except queue.Full:
                pass


class DataLoader:
    """Sample ragged windows, map ``transform`` over steps, pack.

    Parameters
    ----------
    stores :
        A single ``Datastore`` or a list of them. Each store is snapshotted
        at construction (and on :meth:`refresh`) via ``Datastore.to_dataset()``.
    sequence_length :
        Maximum length of each contiguous window (in steps).
    batch_size :
        How many such windows per batch.
    transform :
        Required ``dict → StepTokens`` callable applied to every step.
        Compose pipeline stages outside the loader; packing is loader-owned.
    index_field :
        Optional key. When set, each fetched step is stamped with its absolute
        store offset under this name before ``transform`` runs.
    weights / weight_mode / prefetch / num_workers :
        Sampling and worker controls.
    seed :
        Entropy of the batch stream. Batch ``k`` is a pure function of
        ``(seed, k, snapshot)`` for any ``num_workers`` (see module
        docstring). ``None`` draws fresh entropy for this loader.
    """

    def __init__(
        self,
        *,
        stores: Datastore | list[Datastore],
        sequence_length: int,
        batch_size: int,
        transform: StepTransform | None = None,
        index_field: str | None = None,
        weights: list[float] | None = None,
        weight_mode: str = "per_store",
        prefetch: int = 4,
        num_workers: int = 1,
        seed: int | None = None,
    ) -> None:
        from mouse_core.data.datastore import Datastore as _DS

        self._stop: threading.Event | None = None
        self._result_queue: queue.Queue | None = None
        self._workers: list[threading.Thread] = []
        self._failure = _WorkerFailure()
        # Batch numbering: the consumer returns _next_k next and parks
        # out-of-order worker arrivals in _reorder. Workers claim indices from
        # a counter created per _start_workers call, starting at _next_k.
        self._next_k = 0
        self._reorder: dict[int, tuple[TokenBatch, TensorDict]] = {}

        if isinstance(stores, _DS):
            stores = [stores]
        if not stores or not all(isinstance(s, _DS) for s in stores):
            raise TypeError("DataLoader requires a Datastore or a non-empty list of Datastores.")
        if transform is None or not callable(transform):
            raise TypeError(
                "DataLoader requires transform= "
                "(callable dict → StepTokens, e.g. compose(...))."
            )
        if weight_mode not in ("per_store", "per_step"):
            raise ValueError(f"weight_mode must be 'per_store' or 'per_step', got {weight_mode!r}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if prefetch < 1:
            raise ValueError(f"prefetch must be >= 1, got {prefetch}.")
        if weights is not None:
            if len(weights) != len(stores):
                raise ValueError(
                    f"weights length ({len(weights)}) must match number of stores ({len(stores)})."
                )
            if any(w <= 0 for w in weights):
                raise ValueError("All weights must be positive.")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
        if num_workers > 0:
            _require_free_threading()

        self.stores = stores
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.weight_mode = weight_mode
        self.seed = seed
        self.transform = transform
        self.index_field = index_field
        self._num_workers = num_workers
        self._prefetch = prefetch
        self._weights: np.ndarray = (
            np.ones(len(stores)) if weights is None else np.asarray(weights, dtype=float)
        )
        self._entropy: int = (
            int(seed) if seed is not None else int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
        )

        self._datasets: list = []
        self._ns: list[int] = []
        self._probs: np.ndarray = np.empty(0)
        self._resnapshot_stores()

        if num_workers > 0:
            self._start_workers()

    @property
    def total_batches(self) -> int:
        """Approximate total non-overlapping max-windows across all stores."""
        total_windows = sum(n // self.sequence_length for n in self._ns)
        return max(0, (total_windows + self.batch_size - 1) // self.batch_size)

    def refresh(self) -> None:
        """Drop prefetched batches and re-snapshot all stores.

        Numbering resumes at the next batch the consumer has not received,
        so batches after the refresh are rebuilt against the new snapshot
        with their original indices.
        """
        if self._num_workers > 0:
            self._stop_workers()
        self._reorder.clear()
        self._resnapshot_stores()
        if self._num_workers > 0:
            self._start_workers()

    def next_batch(self) -> tuple[TokenBatch, TensorDict]:
        """Return ``(inputs, objective_data)`` for the next batch index.

        ``inputs`` is the packed :class:`TokenBatch`. ``objective_data``
        is a CPU :class:`~tensordict.TensorDict` of tokenizer
        ``objective_fields`` (plus ``sequence_id`` and the grouping column).
        """
        k = self._next_k
        if self._num_workers == 0:
            item = _fetch_one_batch(self._snapshot_config(), self._entropy, k, self.transform)
            self._next_k = k + 1
            return item
        assert self._result_queue is not None
        while k not in self._reorder:
            if self._failure.exc is not None:
                raise RuntimeError("A prefetch worker raised an exception.") from self._failure.exc
            try:
                got_k, got_item = self._result_queue.get(timeout=0.05)
            except queue.Empty:
                if self._failure.exc is None and not any(w.is_alive() for w in self._workers):
                    raise RuntimeError("All prefetch workers stopped unexpectedly.")
                continue
            self._reorder[got_k] = got_item
        self._next_k = k + 1
        return self._reorder.pop(k)

    def close(self) -> None:
        """Stop background workers and drain the queue."""
        self._stop_workers()

    def __enter__(self) -> DataLoader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        store_info = ", ".join(
            f"{s.name or '?'}({n})" for s, n in zip(self.stores, self._ns)
        )
        return (
            f"DataLoader(stores=[{store_info}], S_max={self.sequence_length}, "
            f"B={self.batch_size}, seed={self.seed})"
        )

    def _snapshot_config(self) -> _SnapshotConfig:
        return _SnapshotConfig(
            datasets=tuple(self._datasets),
            ns=tuple(self._ns),
            probs=self._probs.copy(),
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            index_field=self.index_field,
        )

    def _start_workers(self) -> None:
        assert self._num_workers > 0
        self._failure = _WorkerFailure()
        self._result_queue = queue.Queue(maxsize=self._prefetch)
        self._stop = threading.Event()
        cfg = self._snapshot_config()
        counter = _BatchCounter(self._next_k)
        self._workers = []
        for i in range(self._num_workers):
            thread = threading.Thread(
                target=_worker_loop,
                args=(
                    self._result_queue,
                    self._stop,
                    self._failure,
                    counter,
                    cfg,
                    self._entropy,
                    self.transform,
                ),
                daemon=True,
                name=f"DataLoader-{i}",
            )
            thread.start()
            self._workers.append(thread)

    def _stop_workers(self) -> None:
        if self._stop is None:
            self._workers = []
            self._result_queue = None
            return
        self._stop.set()
        if self._result_queue is not None:
            while True:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break
        for w in self._workers:
            w.join(timeout=2.0)
        self._workers = []
        self._stop = None
        self._result_queue = None

    def _resnapshot_stores(self) -> None:
        self._datasets = [s.to_dataset() for s in self.stores]
        self._ns = [len(ds) for ds in self._datasets]

        w = self._weights.copy()
        ns = np.array(self._ns, dtype=float)
        if self.weight_mode == "per_step":
            w = w * ns
        else:
            w = w * (ns > 0)
        if w.sum() == 0:
            self._probs = np.ones(len(self.stores)) / len(self.stores)
        else:
            self._probs = w / w.sum()
