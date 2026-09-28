"""DataLoader — sample ragged windows, apply a per-step transform, pack.

A ``Datastore`` is a flat sequence of arbitrary rows. The loader samples
``B`` sequences, each a contiguous store window of length ``1 .. sequence_length``
(a max), runs ``transform(step)`` on every step, and packs the resulting
:class:`~mouse_core.data.token_batch.StepTokens` into a
:class:`~mouse_core.data.token_batch.TokenBatch` plus a CPU
``dict[str, Tensor]`` of step-level objective columns.

``sample_start=None`` (the default) may begin a window at any store
offset. ``sample_start`` may also be a callable
``cols → bool ndarray``: ``cols`` maps column name → 1-d array over the
whole store, and ``True`` marks legal start rows (the matching row
itself). Notebooks often define a ``full_task_start`` callable inline
so windows open only on ``episode_index == 0`` and ``step_index == 0``.
The window then runs forward for
``min(sequence_length, steps remaining in the store)`` — the same
ragged suffix as unrestricted sampling — unless ``sample_end`` is
set.

``sample_end=None`` (the default) never truncates early for a field.
``sample_end`` may be a callable ``cols → bool ndarray`` over the
candidate window rows; the window includes the first ``True`` row then
stops. Notebooks often define a ``full_task_end`` callable inline
(``task_done != 0``). If ``sample_end`` is set but no matching row
appears before ``sequence_length`` or the store end for a chosen
start, that draw is discarded and another start is sampled (not a
silent truncate). Exhaustion — every candidate start incomplete, or
too many consecutive misses — raises ``ValueError``. Every yielded
window with ``sample_end`` set must end on a match; a broken
invariant raises.

The loader is stage-agnostic: compose augmenter / tokenizer
(or any ``dict → StepTokens`` callable) outside and pass the result as
``transform=``. Before each sampled sequence ``b`` of batch ``k``, if
``transform`` defines ``reseed()``, it is called as
``reseed(generation=k * batch_size + b)`` so an
:class:`~mouse_core.data.augmenter.Augmenter` in the compose pipeline
draws a starting seed unique to that window. Steps that share a
``seed_field`` value inside one window still share permute/scale/shift
draws; the same index on two windows does not.

Determinism
-----------
Batches are numbered ``k = 0, 1, 2, ...`` in the order :meth:`DataLoader.next_batch`
returns them. Batch ``k`` samples its windows from
``SeedSequence(seed, spawn_key=(k,))`` and reseeds the transform once per
sequence with ``generation=k * batch_size + b``, so it is a pure function
of ``(seed, k, store snapshot)``:
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

    def full_task_start(cols):
        return (np.asarray(cols["episode_index"]) == 0) & (
            np.asarray(cols["step_index"]) == 0
        )

    def full_task_end(cols):
        return np.asarray(cols["task_done"]) != 0

    train_transform = compose(stages=(augmenter, tokenizer))
    loader = DataLoader(
        stores=store,
        sequence_length=64,
        batch_size=8,
        transform=train_transform,
        sample_start=full_task_start,
        sample_end=full_task_end,
        num_workers=0,
    )
    inputs, objective_data = loader.next_batch()
"""

from __future__ import annotations

import queue
import sys
import sysconfig
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mouse_core.data.datastore import _normalize_value
from mouse_core.data.token_batch import StepTokens, TokenBatch, pack_token_batch

if TYPE_CHECKING:
    import torch

    from mouse_core.data.datastore import Datastore


StepTransform = Callable[[dict], StepTokens]
SampleFn = Callable[[Mapping[str, Any]], np.ndarray]

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
    start_indices: tuple[np.ndarray, ...] | None
    end_fn: SampleFn | None


class _ColumnView(Mapping[str, Any]):
    """Lazy column access over a HF dataset (or slice) for sample callables."""

    def __init__(self, *, ds: Any, offset: int = 0, n: int | None = None) -> None:
        self._ds = ds
        self._offset = int(offset)
        self._n = int(len(ds) if n is None else n)
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._cache:
            if key not in self._ds.column_names:
                raise KeyError(
                    f"sample condition requires column {key!r}; "
                    f"store has {sorted(self._ds.column_names)}"
                )
            column = self._ds[self._offset : self._offset + self._n][key]
            self._cache[key] = _column_array(column=column, n=self._n, field=key)
        return self._cache[key]

    def __iter__(self):
        return iter(self._ds.column_names)

    def __len__(self) -> int:
        return len(self._ds.column_names)


def _column_array(*, column: Any, n: int, field: str) -> np.ndarray:
    """Column as a length-``n`` numpy array (object columns unwrapped)."""
    values = np.asarray(column)
    if values.dtype != object and values.shape == (n,):
        return values
    raw = list(column)
    if len(raw) != n:
        raise ValueError(
            f"{field} length ({len(raw)}) does not match the store ({n})."
        )
    out = np.empty(n, dtype=object)
    for i, value in enumerate(raw):
        out[i] = _normalize_value(value)
    # Prefer a numeric array when every cell is a plain number.
    try:
        numeric = np.asarray(out.tolist())
    except (TypeError, ValueError):
        return out
    if numeric.shape == (n,) and numeric.dtype != object:
        return numeric
    return out


def _eval_sample_mask(
    *, fn: SampleFn, cols: Mapping[str, Any], n: int, role: str
) -> np.ndarray:
    """Run ``fn(cols)`` and require a length-``n`` bool mask."""
    mask = fn(cols)
    arr = np.asarray(mask)
    if arr.shape != (n,):
        raise ValueError(
            f"sample_{role} callable must return a boolean array of length {n}, "
            f"got shape {arr.shape}."
        )
    if arr.dtype != bool:
        arr = arr.astype(bool, copy=False)
    return arr


def _start_indices(*, ds: Any, sample_start: SampleFn) -> np.ndarray:
    """Offsets where a sample may start under ``sample_start``."""
    n = len(ds)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    cols = _ColumnView(ds=ds, offset=0, n=n)
    mask = _eval_sample_mask(fn=sample_start, cols=cols, n=n, role="start")
    return np.flatnonzero(mask).astype(np.int64, copy=False)


# Cap consecutive incomplete ``sample_end`` draws (safety against huge
# unrestricted start spaces). Distinct-start exhaustion can raise sooner.
_SAMPLE_END_MAX_RETRIES = 1024


def _sample_end_label(end_fn: SampleFn) -> str:
    name = getattr(end_fn, "__qualname__", None) or getattr(
        end_fn, "__name__", repr(end_fn)
    )
    return f"sample_end={name}"


def _window_end(
    *,
    start: int,
    n: int,
    s_max: int,
    ds: Any,
    end_fn: SampleFn | None,
) -> int | None:
    """Exclusive end index for a window starting at ``start``.

    When ``end_fn`` is set, the first matching row must appear at or
    before ``min(start + s_max, n) - 1``; otherwise return ``None`` so
    the caller can discard this start and resample.
    """
    end = min(start + s_max, n)
    if end_fn is None or start >= end:
        return end
    cols = _ColumnView(ds=ds, offset=start, n=end - start)
    mask = _eval_sample_mask(fn=end_fn, cols=cols, n=end - start, role="end")
    hits = np.flatnonzero(mask)
    if len(hits) == 0:
        return None
    return start + int(hits[0]) + 1


def _all_sample_starts_exhausted(
    *,
    cfg: _SnapshotConfig,
    incomplete: set[tuple[int, int]],
) -> bool:
    """True when every sampleable ``(store_idx, start)`` is known incomplete."""
    any_candidate = False
    for store_idx, n in enumerate(cfg.ns):
        if n < 1:
            continue
        if cfg.start_indices is None:
            candidates = range(n)
        else:
            starts = cfg.start_indices[store_idx]
            if len(starts) == 0:
                continue
            candidates = (int(s) for s in starts)
        for start in candidates:
            any_candidate = True
            if (store_idx, start) not in incomplete:
                return False
    return any_candidate


def _require_window_ends_on_sample_end(
    *,
    ds: Any,
    start: int,
    end: int,
    end_fn: SampleFn,
) -> None:
    """Raise if a yielded ``sample_end`` window does not end on a match."""
    count = end - start
    if count < 1:
        raise ValueError(
            "DataLoader invariant violated: empty window with sample_end set "
            f"({_sample_end_label(end_fn)}; start={start}, exclusive_end={end})."
        )
    cols = _ColumnView(ds=ds, offset=start, n=count)
    mask = _eval_sample_mask(fn=end_fn, cols=cols, n=count, role="end")
    if not bool(mask[-1]):
        raise ValueError(
            "DataLoader invariant violated: yielded window does not end on "
            f"a sample_end match ({_sample_end_label(end_fn)}; "
            f"start={start}, exclusive_end={end})."
        )


def _pick_store_and_start(
    *,
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
) -> tuple[int, Any, int, int]:
    """Choose ``(store_idx, ds, n, start)`` for one window draw."""
    store_idx = int(rng.choice(len(cfg.datasets), p=cfg.probs))
    ds = cfg.datasets[store_idx]
    n = cfg.ns[store_idx]
    if n < 1:
        raise ValueError("Cannot sample from an empty store.")
    if cfg.start_indices is None:
        start = int(rng.integers(0, n))
    else:
        starts = cfg.start_indices[store_idx]
        if len(starts) == 0:
            raise ValueError(
                "Cannot sample starts from a store with no sample_start boundaries."
            )
        start = int(starts[int(rng.integers(0, len(starts)))])
    return store_idx, ds, n, start


def _rows_from_slice(
    *,
    ds: Any,
    start: int,
    end: int,
    index_field: str | None,
) -> list[dict]:
    """Materialize ``ds[start:end]`` as a list of row dicts."""
    hf_slice = ds[start:end]
    count = end - start
    rows = [
        {k: _normalize_value(hf_slice[k][i]) for k in hf_slice} for i in range(count)
    ]
    if index_field is not None:
        for i, row in enumerate(rows):
            row[index_field] = start + i
    return rows


def _fetch_sequence(
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
) -> list[dict]:
    """Fetch one contiguous window of length ``1 .. sequence_length``.

    With ``sample_end`` set, incomplete starts (no end match within the
    allowed range) are discarded and another start is drawn. Raises when
    every candidate start is incomplete or after
    ``_SAMPLE_END_MAX_RETRIES`` consecutive misses.
    """
    if sum(cfg.ns) == 0:
        raise ValueError("Cannot sample batches: all stores are empty.")

    S_max = cfg.sequence_length
    end_fn = cfg.end_fn
    if end_fn is None:
        _, ds, n, start = _pick_store_and_start(cfg=cfg, rng=rng)
        end = _window_end(start=start, n=n, s_max=S_max, ds=ds, end_fn=None)
        assert end is not None
        return _rows_from_slice(
            ds=ds, start=start, end=end, index_field=cfg.index_field
        )

    incomplete: set[tuple[int, int]] = set()
    for _ in range(_SAMPLE_END_MAX_RETRIES):
        store_idx, ds, n, start = _pick_store_and_start(cfg=cfg, rng=rng)
        key = (store_idx, start)
        if key in incomplete:
            if _all_sample_starts_exhausted(cfg=cfg, incomplete=incomplete):
                break
            continue
        end = _window_end(
            start=start,
            n=n,
            s_max=S_max,
            ds=ds,
            end_fn=end_fn,
        )
        if end is None:
            incomplete.add(key)
            if _all_sample_starts_exhausted(cfg=cfg, incomplete=incomplete):
                break
            continue
        _require_window_ends_on_sample_end(
            ds=ds, start=start, end=end, end_fn=end_fn
        )
        return _rows_from_slice(
            ds=ds, start=start, end=end, index_field=cfg.index_field
        )

    if incomplete and _all_sample_starts_exhausted(cfg=cfg, incomplete=incomplete):
        raise ValueError(
            "No complete sample_end window in the store(s): "
            f"{_sample_end_label(end_fn)}; "
            f"tried {len(incomplete)} distinct incomplete start(s); "
            f"sequence_length={S_max}."
        )
    raise ValueError(
        "Could not sample a complete sample_end window after "
        f"{_SAMPLE_END_MAX_RETRIES} attempts: {_sample_end_label(end_fn)}; "
        f"incomplete_starts_seen={len(incomplete)}, sequence_length={S_max}."
    )


def _batch_rng(entropy: int, k: int) -> np.random.Generator:
    """Window-sampling RNG of batch ``k``: ``SeedSequence(entropy, spawn_key=(k,))``."""
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(k,)))


def _sequence_generation(*, batch_index: int, sequence_index: int, batch_size: int) -> int:
    """Unique augmenter generation for sequence ``b`` of batch ``k``."""
    return batch_index * batch_size + sequence_index


def _fetch_one_batch(
    cfg: _SnapshotConfig,
    entropy: int,
    k: int,
    transform: StepTransform,
) -> tuple[TokenBatch, dict[str, torch.Tensor]]:
    """Build batch ``k``: sample windows, reseed each sequence, pack."""
    rng = _batch_rng(entropy, k)
    sequences = [_fetch_sequence(cfg, rng) for _ in range(cfg.batch_size)]
    reseed = getattr(transform, "reseed", None)
    steps: list[StepTokens] = []
    sequence_ids: list[int] = []
    grouping_field: str | None = None
    for b, seq in enumerate(sequences):
        if callable(reseed):
            reseed(
                generation=_sequence_generation(
                    batch_index=k,
                    sequence_index=b,
                    batch_size=cfg.batch_size,
                )
            )
        for step in seq:
            packed = transform(step)
            if grouping_field is None:
                grouping_field = packed.grouping_field
            steps.append(packed)
            sequence_ids.append(b)
    return pack_token_batch(
        steps=steps,
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
    sample_start :
        When ``None`` (default), a window may start at any store offset.
        When a callable ``cols → bool ndarray``, every window starts on a
        row where the mask is ``True`` (the matching row itself).
        ``cols`` maps column name → 1-d array over the store. Without
        ``sample_end``, the window still runs forward until
        ``sequence_length`` or the store ends.
    sample_end :
        When ``None`` (default), the window stops only at ``sequence_length``
        or the store end. When a callable ``cols → bool ndarray`` over the
        candidate window, the window includes the first ``True`` row then
        stops. Incomplete starts (no match before ``sequence_length`` /
        store end) are skipped and another start is drawn. Exhaustion or a
        yielded window that somehow lacks an end match raises
        ``ValueError``. A short matching segment is a shorter window; rows
        are not padded.
    batch_size :
        How many such windows per batch.
    transform :
        Required ``dict → StepTokens`` callable applied to every step.
        Compose pipeline stages outside the loader; packing is loader-owned.
    index_field :
        Optional key. When set, each fetched step is stamped with its absolute
        store offset under this name before ``transform`` runs.
    weights / weight_mode / prefetch / num_workers :
        Sampling and worker controls. ``num_workers`` is required: ``0`` is
        in-process; ``> 0`` needs free-threaded CPython with the GIL off.
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
        transform: StepTransform,
        sample_start: SampleFn | None = None,
        sample_end: SampleFn | None = None,
        index_field: str | None = None,
        weights: list[float] | None = None,
        weight_mode: str = "per_store",
        prefetch: int = 4,
        num_workers: int,
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
        self._reorder: dict[int, tuple[TokenBatch, dict[str, torch.Tensor]]] = {}

        if isinstance(stores, _DS):
            stores = [stores]
        if not stores or not all(isinstance(s, _DS) for s in stores):
            raise TypeError("DataLoader requires a Datastore or a non-empty list of Datastores.")
        if not callable(transform):
            raise TypeError(
                "DataLoader requires transform= "
                "(callable dict → StepTokens, e.g. compose(...))."
            )
        if weight_mode not in ("per_store", "per_step"):
            raise ValueError(f"weight_mode must be 'per_store' or 'per_step', got {weight_mode!r}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}.")
        if sample_start is not None and not callable(sample_start):
            raise TypeError(
                "sample_start must be a callable cols → bool ndarray, or None, "
                f"got {type(sample_start).__name__}."
            )
        if sample_end is not None and not callable(sample_end):
            raise TypeError(
                "sample_end must be a callable cols → bool ndarray, or None, "
                f"got {type(sample_end).__name__}."
            )
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
        self.sample_start = sample_start
        self.sample_end = sample_end
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
        self._start_indices: tuple[np.ndarray, ...] | None = None
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

    def next_batch(self) -> tuple[TokenBatch, dict[str, torch.Tensor]]:
        """Return ``(inputs, objective_data)`` for the next batch index.

        ``inputs`` is the packed :class:`TokenBatch`. ``objective_data``
        is a CPU ``dict[str, Tensor]`` of tokenizer ``objective_fields``
        (plus ``sequence_id`` and the grouping column).
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
        if getattr(self, "_workers", None) is None:
            return
        self.close()

    def __repr__(self) -> str:
        store_info = ", ".join(
            f"{s.name or '?'}({n})" for s, n in zip(self.stores, self._ns)
        )
        return (
            f"DataLoader(stores=[{store_info}], S_max={self.sequence_length}, "
            f"B={self.batch_size}, sample_start={self.sample_start!r}, "
            f"sample_end={self.sample_end!r}, seed={self.seed})"
        )

    def _snapshot_config(self) -> _SnapshotConfig:
        return _SnapshotConfig(
            datasets=tuple(self._datasets),
            ns=tuple(self._ns),
            probs=self._probs.copy(),
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            index_field=self.index_field,
            start_indices=self._start_indices,
            end_fn=self.sample_end,
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
        if self.sample_start is None:
            self._start_indices = None
        else:
            self._start_indices = tuple(
                _start_indices(ds=ds, sample_start=self.sample_start) for ds in self._datasets
            )

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
