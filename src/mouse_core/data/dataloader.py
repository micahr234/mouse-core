"""DataLoader — sample ragged windows, apply a per-step transform, pack.

A ``Datastore`` is a flat sequence of arbitrary rows. The loader samples
``B`` sequences, each a contiguous store window of length ``1 .. sequence_length``
(a max), runs ``transform(step)`` on every step, and packs the resulting
:class:`~mouse_core.data.token_batch.StepTokens` into a
:class:`~mouse_core.data.token_batch.TokenBatch` plus a CPU
:class:`~tensordict.TensorDict` of step-level objective columns.

The loader is stage-agnostic: compose augmenter / selector / tokenizer
(or any ``dict → StepTokens`` callable) outside and pass the result as
``transform=``. At the start of each batch fetch, if ``transform`` defines
``reseed()``, it is called once (so an :class:`~mouse_core.data.augmenter.Augmenter`
in the compose pipeline draws a new augmentation set per batch).

Usage
-----
::

    train_transform = compose(augmenter, selector, tokenizer)
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
from typing import TYPE_CHECKING, Any, cast

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


def _fetch_one_batch(
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
    transform: StepTransform,
) -> tuple[TokenBatch, TensorDict]:
    reseed = getattr(transform, "reseed", None)
    if callable(reseed):
        reseed()
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


def _worker_loop(
    result_queue: queue.Queue,
    stop_event: threading.Event,
    cfg: _SnapshotConfig,
    sample_seed: Any,
    transform: StepTransform,
) -> None:
    """Prefetch loop run inside a worker thread."""
    rng = np.random.default_rng(seed=sample_seed)
    while not stop_event.is_set():
        try:
            item = _fetch_one_batch(cfg, rng, transform)
            while not stop_event.is_set():
                try:
                    result_queue.put(("ok", item), timeout=0.05)
                    break
                except queue.Full:
                    pass
        except Exception as exc:  # noqa: BLE001
            try:
                result_queue.put(("err", exc), timeout=1.0)
            except Exception:  # noqa: BLE001
                pass
            return


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
    weights / weight_mode / prefetch / num_workers / seed :
        Sampling and worker controls (unchanged semantics).
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
        self._sync_rng: np.random.Generator | None = None
        self._sync_transform: StepTransform | None = None
        self._worker_error: BaseException | None = None

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
        self._seed_seq: np.random.SeedSequence | None = (
            np.random.SeedSequence(seed) if seed is not None else None
        )

        self._datasets: list = []
        self._ns: list[int] = []
        self._probs: np.ndarray = np.empty(0)
        self._resnapshot_stores()

        if num_workers == 0:
            self._sync_rng = np.random.default_rng(seed=seed)
            self._sync_transform = transform
        else:
            self._start_workers()

    @property
    def total_batches(self) -> int:
        """Approximate total non-overlapping max-windows across all stores."""
        total_windows = sum(n // self.sequence_length for n in self._ns)
        return max(0, (total_windows + self.batch_size - 1) // self.batch_size)

    def refresh(self) -> None:
        """Drop prefetched batches and re-snapshot all stores."""
        if self._num_workers > 0:
            self._stop_workers()
        self._resnapshot_stores()
        if self._num_workers > 0:
            self._start_workers()

    def next_batch(self) -> tuple[TokenBatch, TensorDict]:
        """Return ``(inputs, objective_data)``.

        ``inputs`` is the packed :class:`TokenBatch`. ``objective_data``
        is a CPU :class:`~tensordict.TensorDict` of tokenizer
        ``objective_fields`` (plus ``sequence_id`` and the grouping column).
        """
        if self._sync_rng is not None:
            cfg = self._snapshot_config()
            assert self._sync_transform is not None
            return _fetch_one_batch(cfg, self._sync_rng, self._sync_transform)
        assert self._result_queue is not None
        while True:
            if self._worker_error is not None:
                raise RuntimeError("A prefetch worker raised an exception.") from self._worker_error
            try:
                kind, payload = self._result_queue.get(timeout=0.05)
            except queue.Empty:
                if not any(w.is_alive() for w in self._workers):
                    raise RuntimeError("All prefetch workers stopped unexpectedly.")
                continue
            if kind == "err":
                self._worker_error = cast(BaseException, payload)
                raise RuntimeError("A prefetch worker raised an exception.") from self._worker_error
            return payload

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

    def _worker_seeds(self) -> list[Any]:
        n = self._num_workers
        if self._seed_seq is None:
            return [None] * n
        return list(self._seed_seq.spawn(n))

    def _start_workers(self) -> None:
        assert self._num_workers > 0
        self._worker_error = None
        self._result_queue = queue.Queue(maxsize=self._prefetch)
        self._stop = threading.Event()
        cfg = self._snapshot_config()
        sample_seeds = self._worker_seeds()
        self._workers = []
        for i in range(self._num_workers):
            thread = threading.Thread(
                target=_worker_loop,
                args=(
                    self._result_queue,
                    self._stop,
                    cfg,
                    sample_seeds[i],
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
