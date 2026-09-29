"""DataLoader — sample contiguous runs, apply a per-step transform, pack.

A ``Datastore`` is a flat sequence of arbitrary rows. Pass exactly one
of ``samples_budget`` or ``token_budget``, plus ``batch_size``:

* ``samples_budget`` — ``batch_size`` fills. That count does not depend
  on the budget. Each fill adds whole samples until the next one would
  pass that many steps.
* ``token_budget`` — the same fills, counting packed tokens instead of
  steps.

``sample_field`` names a scalar column. A sample is a maximal contiguous
run of rows that share one value in that column: the run starts where
the value changes (or at the first row) and ends at the next change.
Each sample the loader keeps is one group. Every sample in a batch
gets a distinct ``group_id``, and every step of that sample shares it.
Attention and the TD / PPO / GRPO run stop where ``group_id`` changes,
so they do not cross the field. A sample that does not fit is left
out. The first sample of a fill is left out too when it alone is over
the budget, and that raises. Examples pass ``batch_size=1`` (one fill)
and ``sample_field="task_index"``.

Token ``group_ids`` and the step ``group_id`` tensor use those same
ids, ``0 .. B - 1``, one contiguous block each. ``group_id`` is
returned beside ``objective_data``, not inside it.

The loader is stage-agnostic: compose augmenter / tokenizer
(or any ``dict → StepTokens`` callable) outside and pass the result as
``transform=``. Before each sampled segment of batch ``k``, if
``transform`` defines ``reseed()``, it is called with a generation
unique to that segment. The generation is
``k * batch_size * (budget + 1) + fill * (budget + 1) + attempt``,
where ``budget`` is ``samples_budget`` or ``token_budget``. Each
segment costs at least one unit, so a fill draws at most ``budget + 1``
generations (the segments that fit, plus one that does not) and the
next batch does not collide. An
:class:`~mouse_core.data.augmenter.Augmenter` in the compose pipeline
draws a starting seed unique to that sequence.
Steps that share a ``seed_field`` value inside one window still share
permute/scale/shift draws; the same index on two windows does not.

Determinism
-----------
Batches are numbered ``k = 0, 1, 2, ...`` in the order :meth:`DataLoader.next_batch`
returns them. Batch ``k`` samples its windows from
``SeedSequence(seed, spawn_key=(k,))`` and reseeds the transform once per
sequence with a generation unique to ``(k, sequence)``, so it is a pure function
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

    train_transform = compose(stages=(augmenter, tokenizer))
    loader = DataLoader(
        stores=store,
        token_budget=4096,
        batch_size=1,
        transform=train_transform,
        sample_field="task_index",
        num_workers=0,
    )
    inputs, objective_data, group_id = loader.next_batch()
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
    samples_budget: int | None
    batch_size: int | None
    token_budget: int | None
    index_field: str | None
    sample_field: str
    runs: tuple[np.ndarray, ...]


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


def _sample_runs(*, ds: Any, sample_field: str) -> np.ndarray:
    """``[n_runs, 2]`` int64 ``(start, exclusive_end)`` of equal ``sample_field`` values.

    A run is every contiguous row that shares one scalar value. The first
    row opens a run; each later row whose value differs from the previous
    row opens the next one.
    """
    n = len(ds)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if sample_field not in ds.column_names:
        raise KeyError(
            f"sample_field {sample_field!r} is not a column; "
            f"store has {sorted(ds.column_names)}"
        )
    values = np.asarray(_ColumnView(ds=ds, offset=0, n=n)[sample_field])
    if values.shape != (n,):
        raise ValueError(
            f"sample_field {sample_field!r} must be a scalar column of length {n}, "
            f"got shape {tuple(values.shape)}."
        )
    if values.dtype == object and any(
        isinstance(value, (list, tuple, dict, np.ndarray)) for value in values
    ):
        raise ValueError(
            f"sample_field {sample_field!r} must be a scalar column of length {n}, "
            f"got nested values."
        )
    if n == 1:
        return np.array([[0, 1]], dtype=np.int64)
    change = np.asarray(values[1:] != values[:-1], dtype=bool)
    boundary = np.flatnonzero(change).astype(np.int64, copy=False) + 1
    starts = np.concatenate([np.zeros(1, dtype=np.int64), boundary])
    ends = np.concatenate([boundary, np.array([n], dtype=np.int64)])
    return np.stack([starts, ends], axis=1)


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
) -> tuple[Any, int, int]:
    """Choose one sample. Return ``(dataset, start, exclusive_end)``."""
    if sum(cfg.ns) == 0:
        raise ValueError("Cannot sample batches: all stores are empty.")
    store_idx = int(rng.choice(len(cfg.datasets), p=cfg.probs))
    ds = cfg.datasets[store_idx]
    runs = cfg.runs[store_idx]
    if len(runs) == 0:
        raise ValueError(
            f"Cannot sample {cfg.sample_field!r} from an empty store."
        )
    pick = int(rng.integers(0, len(runs)))
    return ds, int(runs[pick, 0]), int(runs[pick, 1])


def _batch_rng(entropy: int, k: int) -> np.random.Generator:
    """Window-sampling RNG of batch ``k``: ``SeedSequence(entropy, spawn_key=(k,))``."""
    return np.random.default_rng(np.random.SeedSequence(entropy, spawn_key=(k,)))


def _sequence_generation(*, batch_index: int, sequence_index: int, stride: int) -> int:
    """Unique augmenter generation for one sampled segment of batch ``k``.

    ``stride`` is ``batch_size * (budget + 1)``. The budget is
    ``samples_budget`` or ``token_budget``. Each segment costs at least
    one unit, so a fill draws at most ``budget + 1`` generations and the
    next batch does not collide.
    """
    return batch_index * stride + sequence_index


def _step_token_count(step: StepTokens, *, started: bool) -> int:
    """Packed tokens this step adds inside its own sample.

    Includes group-start tokens on the first step of a fresh sequence,
    matching :func:`~mouse_core.data.token_batch.pack_token_batch`.
    """
    extra = 0
    if step.group_start_ids is not None and not started:
        extra = int(step.group_start_ids.shape[0])
    return extra + int(step.T)


def _take_window_steps(
    *,
    ds: Any,
    start: int,
    end: int,
    index_field: str | None,
    transform: StepTransform,
) -> list[StepTokens]:
    """Every step of ``[start, end)``. The window is not trimmed."""
    kept: list[StepTokens] = []
    i = start
    while i < end:
        chunk_end = min(end, i + 32)
        chunk = _rows_from_slice(
            ds=ds, start=i, end=chunk_end, index_field=index_field
        )
        kept.extend(transform(row) for row in chunk)
        i = chunk_end
    return kept


def _segment_token_count(steps: list[StepTokens]) -> int:
    """Packed tokens of one sample, including its group-start tokens."""
    tokens = 0
    started = False
    for step in steps:
        tokens += _step_token_count(step, started=started)
        started = True
    return tokens


def _append_window(
    *,
    steps: list[StepTokens],
    group_ids: list[int],
    window: list[StepTokens],
    group_id: int,
) -> None:
    if not window:
        return
    steps.extend(window)
    group_ids.extend([group_id] * len(window))


def _budget_of(cfg: _SnapshotConfig) -> tuple[int, str, str]:
    """``(limit, parameter name, unit)`` for the active budget."""
    if cfg.token_budget is not None:
        return cfg.token_budget, "token_budget", "packed tokens"
    assert cfg.samples_budget is not None
    return cfg.samples_budget, "samples_budget", "steps"


def _segment_cost(steps: list[StepTokens], *, count_tokens: bool) -> int:
    """Steps in the segment, or packed tokens when ``count_tokens``."""
    if count_tokens:
        return _segment_token_count(steps)
    return len(steps)


def _fetch_budget_batch(
    cfg: _SnapshotConfig,
    entropy: int,
    k: int,
    transform: StepTransform,
) -> tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor]:
    """Fill batch ``k`` ``batch_size`` times up to the active budget.

    Each fill samples whole segments. The next segment is measured, and
    if it does not fit entirely it is left out and that fill stops. A
    fill whose first segment does not fit raises. ``token_budget``
    measures packed tokens. ``samples_budget`` measures steps. Each
    kept segment gets the next ``group_id``, unique in this batch.
    """
    assert cfg.batch_size is not None
    budget, name, unit = _budget_of(cfg)
    count_tokens = cfg.token_budget is not None
    rng = _batch_rng(entropy, k)
    reseed = getattr(transform, "reseed", None)
    steps: list[StepTokens] = []
    group_ids: list[int] = []
    n_group = 0
    # One rejected segment after the ones that fit in a fill. Each
    # segment costs at least one unit, so a fill stays within this width.
    per_fill = budget + 1
    stride = cfg.batch_size * per_fill
    for fill in range(cfg.batch_size):
        used = 0
        added = 0
        attempt = 0
        while True:
            ds, start, end = _fetch_sequence(cfg, rng)
            if callable(reseed):
                reseed(
                    generation=_sequence_generation(
                        batch_index=k,
                        sequence_index=fill * per_fill + attempt,
                        stride=stride,
                    )
                )
            attempt += 1
            window = _take_window_steps(
                ds=ds,
                start=start,
                end=end,
                index_field=cfg.index_field,
                transform=transform,
            )
            cost = _segment_cost(window, count_tokens=count_tokens)
            if cost < 1:
                raise ValueError(
                    f"Sampled segment produced no {unit} "
                    f"(start={start}, exclusive_end={end})."
                )
            if used + cost > budget:
                if added == 0:
                    raise ValueError(
                        f"Sampled segment is {cost} {unit} and does not fit "
                        f"in {name}={budget}."
                    )
                break
            _append_window(
                steps=steps,
                group_ids=group_ids,
                window=window,
                group_id=n_group,
            )
            used += cost
            n_group += 1
            added += 1
    return pack_token_batch(
        steps=steps,
        group_ids=group_ids,
        batch_size=n_group,
        continuing=None,
    )


def _fetch_one_batch(
    cfg: _SnapshotConfig,
    entropy: int,
    k: int,
    transform: StepTransform,
) -> tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor]:
    """Build batch ``k``."""
    return _fetch_budget_batch(cfg, entropy, k, transform)


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
    """Sample contiguous field-runs, map ``transform`` over steps, pack.

    Parameters
    ----------
    stores :
        A single ``Datastore`` or a list of them. Each store is snapshotted
        at construction (and on :meth:`refresh`) via ``Datastore.to_dataset()``.
    samples_budget :
        Steps per fill. Whole samples are added until the next sample
        would pass the budget. A sample that does not fit is left out.
        The first sample of a fill is left out too when it alone is over
        the budget, and that raises. Pass this or ``token_budget``, not
        both. Requires ``sample_field`` and ``batch_size``.
    sample_field :
        Scalar column that defines a sample. A sample is every contiguous
        row that shares one value in this column. Each sample is one
        sequence, so attention and loss do not cross a value change.
    batch_size :
        How many times the budget is filled in the returned batch.
        Independent of either budget. Examples pass ``1``.
    token_budget :
        Packed tokens per fill (step tokens plus each sample's
        group-start tokens). Whole samples are added until the next
        sample would pass the budget. A sample that does not fit is left
        out. The first sample of a fill is left out too when it alone is
        over the budget, and that raises. Pass this or ``samples_budget``,
        not both. Requires ``sample_field`` and ``batch_size``.
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
        transform: StepTransform,
        samples_budget: int | None = None,
        token_budget: int | None = None,
        batch_size: int | None = None,
        sample_field: str,
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
        if (samples_budget is None) == (token_budget is None):
            raise ValueError(
                "Pass exactly one of samples_budget or token_budget."
            )
        if samples_budget is not None and samples_budget < 1:
            raise ValueError(f"samples_budget must be >= 1, got {samples_budget}.")
        if token_budget is not None and token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}.")
        if batch_size is None:
            raise ValueError("batch_size is required.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if not isinstance(sample_field, str) or not sample_field:
            raise ValueError(
                "sample_field must be a non-empty column name, "
                f"got {sample_field!r}."
            )
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
        self.samples_budget = samples_budget
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.sample_field = sample_field
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
        self._runs: tuple[np.ndarray, ...] = ()
        self._resnapshot_stores()

        if num_workers > 0:
            self._start_workers()

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

    def next_batch(
        self,
    ) -> tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor]:
        """Return ``(inputs, objective_data, group_id)`` for the next batch.

        ``inputs`` is the packed :class:`TokenBatch`. ``objective_data``
        is a CPU ``dict[str, Tensor]`` of tokenizer ``objective_fields``.
        ``group_id`` is int64 ``[N]``, one id per step. It is not a key
        in ``objective_data``.
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
            f"DataLoader(stores=[{store_info}], samples_budget={self.samples_budget}, "
            f"token_budget={self.token_budget}, B={self.batch_size}, "
            f"sample_field={self.sample_field!r}, seed={self.seed})"
        )

    def _snapshot_config(self) -> _SnapshotConfig:
        return _SnapshotConfig(
            datasets=tuple(self._datasets),
            ns=tuple(self._ns),
            probs=self._probs.copy(),
            samples_budget=self.samples_budget,
            batch_size=self.batch_size,
            token_budget=self.token_budget,
            index_field=self.index_field,
            sample_field=self.sample_field,
            runs=self._runs,
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
        self._runs = tuple(
            _sample_runs(ds=ds, sample_field=self.sample_field) for ds in self._datasets
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
