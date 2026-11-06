"""Sample complete event sequences into one event or token budget per batch.

Each sampled segment is an independent sequence. Token-level ``sequence_ids`` live in inputs; step-level
``step_sequence_ids`` and prediction-level ``head_step_ids`` live in objective_data.
The loader returns
``(inputs, objective_data)``. A budget limits the whole returned batch.

``num_workers=0`` prepares synchronously. Positive worker counts use persistent
spawned processes and bounded, reusable shared-memory result slots. Transforms
(including notebook-local functions) are serialized once per worker generation.
Returned arrays are owned by the consumer and survive refresh/close.

Batch k uses SeedSequence(seed, spawn_key=(k,)); each sampled segment is reseeded
with k * (budget + 1) + attempt. Worker count changes throughput, not the stream.
Refresh restarts workers against new store snapshots at the next unseen batch.
"""

from __future__ import annotations
from tensordict import TensorDict

import json
from functools import partial
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mouse_core.data.process_batches import ProcessBatches
from mouse_core.data.datastore import _normalize_value
from mouse_core.data.step_sampling import _step_spans, _validate_step_ranges
from mouse_core.data.token_batch import StepTokens, pack_token_batch

if TYPE_CHECKING:
    import torch

    from mouse_core.data.datastore import Datastore


EventSequenceTransform = Callable[[list[dict]], list[StepTokens]]

@dataclass(frozen=True)
class _SnapshotConfig:
    """Immutable sampling snapshot sent once to each worker process."""

    datasets: tuple[Any, ...]
    ns: tuple[int, ...]
    probs: np.ndarray
    events_budget: int | None
    token_budget: int | None
    index_field: str | None
    sample_field: str | None
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
            # Project before slicing so boundary scans do not materialize the
            # payload (often by far the largest column) or other unused fields.
            column = self._ds[key][self._offset : self._offset + self._n]
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
    if sample_field.startswith("payload."):
        key = sample_field.removeprefix("payload.")
        try:
            values = np.empty(n, dtype=object)
            values[:] = [json.loads(payload)[key] for payload in ds["payload"]]
        except KeyError as exc:
            raise KeyError(f"sample_field {sample_field!r} is missing from an event payload") from exc
    else:
        if sample_field not in ds.column_names:
            raise KeyError(f"sample_field {sample_field!r} is not a column; store has {sorted(ds.column_names)}")
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


def _event_sample_ranges(*, ds, sample_field, sample_ranges, step_field=None,
                         episode_field: str | None = "episode_index") -> np.ndarray:
    """Validate source-event windows chosen by a domain's sampling policy."""
    # Old unindexed stores retain their explicit domain sampling policy. Newly
    # collected indexed datasets always enforce step boundaries, even for a
    # caller-supplied range function.
    if step_field is None and "step_index" in ds.column_names:
        step_field = "step_index"
    ranges: np.ndarray
    if sample_ranges is None:
        if step_field is not None and sample_field == step_field:
            ranges = np.asarray([(start, end) for start, end, _ in
                                 _step_spans(ds, step_field=step_field, episode_field=episode_field)])
        else:
            ranges = _sample_runs(ds=ds, sample_field=sample_field)
    else:
        ranges = np.asarray(list(sample_ranges(ds)))
    if ranges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or ranges.dtype.kind not in "iu":
        raise ValueError("sample_ranges must return integer (start, exclusive_end) event offsets")
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] > len(ds)) or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ValueError("sample_ranges must select nonempty windows within the event log")
    if step_field is not None:
        _validate_step_ranges(ds, ranges, step_field=step_field, episode_field=episode_field)
    return ranges.astype(np.int64)


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
            if index_field in row:
                raise ValueError("Event overlays cannot replace source fields")
            row[index_field] = start + i
    return rows


def _fetch_sequence(
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
) -> tuple[Any, int, int]:
    """Choose one sample. Return ``(dataset, start, exclusive_end)``."""
    if sum(cfg.ns) == 0:
        raise ValueError("Cannot sample batches: no eligible samples in the stores.")
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

    ``stride`` is ``budget + 1``. The budget is
    ``events_budget`` or ``token_budget``. Each segment costs at least
    one unit, so a fill draws at most ``budget + 1`` generations and the
    next batch does not collide.
    """
    return batch_index * stride + sequence_index


def _tokenize_event_window(
    *,
    ds: Any,
    start: int,
    end: int,
    index_field: str | None,
    transform: EventSequenceTransform,
) -> list[StepTokens]:
    """Tokenize one complete ordered event sample, without trimming it."""
    events = _rows_from_slice(ds=ds, start=start, end=end, index_field=index_field)
    encoded = transform(events)
    if not isinstance(encoded, list) or not encoded or not all(isinstance(item, StepTokens) for item in encoded):
        raise TypeError("Event transform must return a nonempty list of StepTokens readouts")
    return encoded


def _segment_token_count(steps: list[StepTokens]) -> int:
    """All tokens emitted by one complete sample."""
    return sum(step.T for step in steps)


def _append_window(
    *,
    steps: list[StepTokens],
    sequence_ids: list[int],
    window: list[StepTokens],
    sequence_id: int,
) -> None:
    if not window:
        return
    steps.extend(window)
    sequence_ids.extend([sequence_id] * len(window))


def _budget_of(cfg: _SnapshotConfig) -> tuple[int, str, str]:
    """``(limit, parameter name, unit)`` for the active budget."""
    if cfg.token_budget is not None:
        return cfg.token_budget, "token_budget", "packed tokens"
    assert cfg.events_budget is not None
    return cfg.events_budget, "events_budget", "source events"


def _fetch_budget_batch(
    cfg: _SnapshotConfig,
    entropy: int,
    k: int,
    transform: EventSequenceTransform,
) -> tuple[TensorDict, TensorDict]:
    """Fill one budget with whole, independently sampled sequences."""
    budget, name, unit = _budget_of(cfg)
    count_tokens = cfg.token_budget is not None
    rng = _batch_rng(entropy, k)
    reseed = getattr(transform, "reseed", None)
    steps: list[StepTokens] = []
    sequence_ids: list[int] = []
    n_sequences = 0
    used = 0
    added = 0
    attempt = 0
    while True:
        ds, start, end = _fetch_sequence(cfg, rng)
        if callable(reseed):
            reseed(
                generation=_sequence_generation(
                    batch_index=k,
                    sequence_index=attempt,
                    stride=budget + 1,
                )
            )
        attempt += 1
        # Event counts are known before decoding or tokenizing.
        if not count_tokens and used + (end - start) > budget:
            if added == 0:
                raise ValueError(
                    f"Sampled segment is {end - start} {unit} and does not fit "
                    f"in {name}={budget}."
                )
            break
        window = _tokenize_event_window(
            ds=ds,
            start=start,
            end=end,
            index_field=cfg.index_field,
            transform=transform,
        )
        cost = _segment_token_count(window) if count_tokens else end - start
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
            sequence_ids=sequence_ids,
            window=window,
            sequence_id=n_sequences,
        )
        used += cost
        n_sequences += 1
        added += 1
    return pack_token_batch(
        steps=steps,
        sequence_ids=sequence_ids,
        batch_size=n_sequences,
    )


def _fetch_one_batch(
    cfg: _SnapshotConfig,
    entropy: int,
    k: int,
    transform: EventSequenceTransform,
) -> tuple[TensorDict, TensorDict]:
    """Build batch ``k``."""
    return _fetch_budget_batch(cfg, entropy, k, transform)


class DataLoader:
    """Sample contiguous event sequences, tokenize each whole sample, pack.

    Parameters
    ----------
    stores :
        A single ``Datastore`` or a list of them. Each store is snapshotted
        at construction (and on :meth:`refresh`) via ``Datastore.to_dataset()``.
    events_budget :
        Source events per batch. Whole samples are added until the next sample
        would pass the budget. A sample that does not fit is left out.
        The first sample of a batch is left out too when it alone is over
        the budget, and that raises. Pass this or ``token_budget``, not
        both. Requires a sample boundary policy.
    sample_field :
        Scalar column that defines a sample. A sample is every contiguous
        row that shares one value in this column. Each sample is one
        sequence, so attention and loss do not cross a value change.
    sample_ranges :
        Alternative callable receiving the raw event dataset and returning
        (start, exclusive_end) event offsets. Supports overlapping windows;
        indexed datasets require whole steps within one episode. Use
        ``environment_episode_ranges`` for complete episodes or
        ``partial(step_sample_ranges, steps_per_sample=N)`` for N whole steps.
        Pass this or sample_field.
    step_field / episode_field :
        Boundary fields for validating sample ranges, including ``payload.key``
        paths. The standard ``step_index`` column is detected automatically;
        set ``step_field`` explicitly for custom markers. ``episode_field``
        defaults to ``episode_index``. Use the same fields as Tokenizer.
    token_budget :
        Packed tokens per batch, including episode prompts emitted by the
        tokenizer. Whole samples are added until the next
        sample would pass the budget. A sample that does not fit is left
        out. The first sample of a batch is left out too when it alone is
        over the budget, and that raises. Pass this or ``events_budget``,
        not both. Requires a sample boundary policy.
    transform :
        Required ``list[event] → list[StepTokens]`` callable applied once per sample.
        Compose pipeline stages outside the loader; packing is loader-owned.
    index_field :
        Optional key. When set, each source event is stamped with its absolute
        store offset under this name before ``transform`` runs.
    weights / weight_mode / prefetch / num_workers :
        Sampling and worker controls. ``num_workers`` is required: ``0`` is
        in-process; ``> 0`` uses spawned processes with shared-memory batch transfer.
    seed :
        Entropy of the batch stream. Batch ``k`` is a pure function of
        ``(seed, k, snapshot)`` for any ``num_workers`` (see module
        docstring). ``None`` draws fresh entropy for this loader.
    """

    def __init__(
        self,
        *,
        stores: Datastore | list[Datastore],
        transform: EventSequenceTransform,
        events_budget: int | None = None,
        token_budget: int | None = None,
        sample_field: str | None = None,
        sample_ranges: Callable | None = None,
        step_field: str | None = None,
        episode_field: str | None = "episode_index",
        index_field: str | None = None,
        weights: list[float] | None = None,
        weight_mode: str = "per_store",
        prefetch: int = 4,
        num_workers: int,
        seed: int | None = None,
    ) -> None:
        from mouse_core.data.datastore import Datastore as _DS

        self._pool: ProcessBatches | None = None
        self._closed = False
        self._next_k = 0

        if isinstance(stores, _DS):
            stores = [stores]
        if not stores or not all(isinstance(s, _DS) for s in stores):
            raise TypeError("DataLoader requires a Datastore or a non-empty list of Datastores.")
        if not callable(transform):
            raise TypeError(
                "DataLoader requires transform= "
                "(callable list[event] → list[StepTokens], e.g. compose(...))."
            )
        if weight_mode not in ("per_store", "per_event"):
            raise ValueError(f"weight_mode must be 'per_store' or 'per_event', got {weight_mode!r}")
        if (events_budget is None) == (token_budget is None):
            raise ValueError(
                "Pass exactly one of events_budget or token_budget."
            )
        if events_budget is not None and events_budget < 1:
            raise ValueError(f"events_budget must be >= 1, got {events_budget}.")
        if token_budget is not None and token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}.")
        if (sample_field is None) == (sample_ranges is None):
            raise ValueError("Pass exactly one of sample_field or sample_ranges")
        if sample_field is not None and (not isinstance(sample_field, str) or not sample_field):
            raise ValueError("sample_field must be a non-empty column name")
        if sample_ranges is not None and not callable(sample_ranges):
            raise TypeError("sample_ranges must be callable")
        if index_field in ("provider", "payload"):
            raise ValueError("index_field cannot overwrite provider or payload")
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

        self.stores = stores
        self.events_budget = events_budget
        self.token_budget = token_budget
        self.sample_field = sample_field
        self.sample_ranges = sample_ranges
        self.step_field = step_field
        self.episode_field = episode_field
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
        if self._closed:
            raise RuntimeError("DataLoader is closed")
        self._stop_workers()
        self._resnapshot_stores()
        if self._num_workers > 0:
            self._start_workers()

    def next_batch(
        self,
    ) -> tuple[TensorDict, TensorDict]:
        """Return packed inputs and CPU objective tensors for the next batch."""
        if self._closed:
            raise RuntimeError("DataLoader is closed")
        if self._pool is None:
            item = _fetch_one_batch(self._snapshot_config(), self._entropy, self._next_k, self.transform)
        else:
            item = self._pool.next_batch()
        self._next_k += 1
        return item

    def close(self) -> None:
        """Stop workers and release shared memory; already returned batches survive."""
        self._closed = True
        self._stop_workers()

    def __enter__(self) -> DataLoader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_pool"):
            self.close()

    def __repr__(self) -> str:
        store_info = ", ".join(
            f"{s.name or '?'}({n})" for s, n in zip(self.stores, self._ns)
        )
        return (
            f"DataLoader(stores=[{store_info}], events_budget={self.events_budget}, "
            f"token_budget={self.token_budget}, "
            f"sample_field={self.sample_field!r}, seed={self.seed})"
        )

    def _snapshot_config(self) -> _SnapshotConfig:
        return _SnapshotConfig(
            datasets=tuple(self._datasets),
            ns=tuple(self._ns),
            probs=self._probs.copy(),
            events_budget=self.events_budget,
            token_budget=self.token_budget,
            index_field=self.index_field,
            sample_field=self.sample_field,
            runs=self._runs,
        )

    def _start_workers(self) -> None:
        build = partial(_fetch_one_batch, self._snapshot_config(), self._entropy,
                        transform=self.transform)
        self._pool = ProcessBatches(build, workers=self._num_workers,
                                    prefetch=self._prefetch, start=self._next_k)

    def _stop_workers(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _resnapshot_stores(self) -> None:
        self._datasets = [s.to_dataset() for s in self.stores]
        self._runs = tuple(
            _event_sample_ranges(ds=ds, sample_field=self.sample_field, sample_ranges=self.sample_ranges,
                                 step_field=self.step_field, episode_field=self.episode_field)
            for ds in self._datasets
        )

        self._ns = [len(ds) if len(runs) else 0 for ds, runs in zip(self._datasets, self._runs)]

        w = self._weights.copy()
        ns = np.array(self._ns, dtype=float)
        if self.weight_mode == "per_event":
            w = w * ns
        else:
            w = w * (ns > 0)
        if w.sum() == 0:
            self._probs = np.ones(len(self.stores)) / len(self.stores)
        else:
            self._probs = w / w.sum()
