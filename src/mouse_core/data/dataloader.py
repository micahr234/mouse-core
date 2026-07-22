"""DataLoader — sample steps, optionally augment, then build a TokenBatch.

A ``Datastore`` is a flat sequence of arbitrary rows. The loader samples
contiguous windows of ``S`` steps for ``B`` sequences, runs an optional
augmenter on the raw ``[B][S]`` dicts, then calls an encoder ``preparer`` to
produce a flat concatenated :class:`~mouse_core.models.embedding.token_batch.TokenBatch`
(no padding). Workers run sample → augment → prepare so the training process
mostly feeds the GPU.

Usage
-----
::

    loader = DataLoader(
        store,
        sequence_length=64,
        batch_size=8,
        preparer=model.encoder.make_preparer(),
    )
    token_batch = loader.next_batch()
    predictions, objective_data, _ = model(token_batch)
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

if TYPE_CHECKING:
    from mouse_core.data.datastore import Datastore
    from mouse_core.models.embedding.token_batch import TokenBatch


BatchAugmenter = Callable[[list[list[dict]]], list[list[dict]]]
BatchPreparer = Callable[..., Any]  # (batch, segment_ids) -> TokenBatch

_FREE_THREADING_HINT = (
    "DataLoader(num_workers>0) requires a free-threaded CPython build with the "
    "GIL disabled (e.g. Python 3.14t). Install with `uv python install 3.14t` "
    "and create the venv with that interpreter. If imports re-enable the GIL "
    "(common with older `tokenizers` wheels), run with `PYTHON_GIL=0` or "
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
    pack: bool
    pad: bool


def _fetch_sequence(
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
) -> tuple[list[dict], list[int]]:
    """Fetch exactly ``sequence_length`` steps and their segment IDs."""
    if sum(cfg.ns) == 0:
        raise ValueError("Cannot sample batches: all stores are empty.")

    S = cfg.sequence_length
    steps: list[dict] = []
    segment_ids: list[int] = []
    seg = 0

    while len(steps) < S:
        store_idx = int(rng.choice(len(cfg.datasets), p=cfg.probs))
        ds = cfg.datasets[store_idx]
        n = cfg.ns[store_idx]

        if cfg.pack or cfg.pad:
            start = int(rng.integers(0, n))
        else:
            start = int(rng.integers(0, n - S + 1))

        end = min(start + (S - len(steps)), n)
        hf_slice = ds[start:end]
        count = end - start
        steps.extend({k: hf_slice[k][i] for k in hf_slice} for i in range(count))
        segment_ids.extend([seg] * count)

        if cfg.pad:
            # One real slice, then right-pad; do not stitch further segments.
            break

        if not cfg.pack:
            # Without packing the single slice is always full; exit immediately.
            break

        seg += 1

    if cfg.pad and len(steps) < S:
        if not steps:
            raise ValueError("Cannot pad an empty sequence: sampled zero real steps.")
        pad_row = dict(steps[-1])
        next_seg = seg + 1
        while len(steps) < S:
            steps.append(dict(pad_row))
            segment_ids.append(next_seg)
            next_seg += 1

    return steps[:S], segment_ids[:S]


def _fetch_one_batch(
    cfg: _SnapshotConfig,
    rng: np.random.Generator,
    augmenter: BatchAugmenter | None,
    preparer: BatchPreparer | None,
) -> Any:
    pairs = [_fetch_sequence(cfg, rng) for _ in range(cfg.batch_size)]
    batch = [steps for steps, _ in pairs]
    segment_ids = [ids for _, ids in pairs]
    if augmenter is not None:
        batch = augmenter(batch)
    if preparer is None:
        return batch, segment_ids
    return preparer(batch, segment_ids)


def _worker_loop(
    result_queue: queue.Queue,
    stop_event: threading.Event,
    cfg: _SnapshotConfig,
    sample_seed: Any,
    augmenter: BatchAugmenter | None,
    preparer: BatchPreparer | None,
) -> None:
    """Prefetch loop run inside a worker thread."""
    rng = np.random.default_rng(seed=sample_seed)
    while not stop_event.is_set():
        try:
            item = _fetch_one_batch(cfg, rng, augmenter, preparer)
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


def _fork_augmenter(augmenter: BatchAugmenter | None, *, seed: int | None) -> BatchAugmenter | None:
    if augmenter is None:
        return None
    fork = getattr(augmenter, "fork", None)
    if callable(fork):
        return cast(BatchAugmenter, fork(seed=seed))
    return augmenter


class DataLoader:
    """Produces batches of raw step records plus per-step segment IDs.

    Each store holds a flat sequence of arbitrary rows. The loader samples
    contiguous windows and returns them as Python dicts without any encoding.
    Columns are untouched; the model's encoder decides what to extract and
    how to tensorize it. Segment IDs are returned separately from the row
    dicts and label which independently sampled pack slice each step came from.

    When multiple stores are provided, each sequence in a batch independently
    draws its store according to the computed per-store probabilities, giving
    smooth within-batch mixing.

    Background workers (when enabled) run in threads on a free-threaded
    CPython build so slice and augmentation work can use multiple CPU cores
    while sharing the HF Dataset snapshot and prepared batches in-process.

    Parameters
    ----------
    stores :
        A single ``Datastore`` or a list of them. Each store is snapshotted
        at construction (and on :meth:`refresh`) via ``Datastore.to_dataset()``.
        Call :meth:`refresh` after appending new rows so sampling sees the
        latest data.
    sequence_length :
        Length of each contiguous slice (in steps).
    batch_size :
        How many such slices per batch.
    weights :
        One positive float per store controlling the relative probability of
        drawing from each store. ``None`` means uniform. Interpreted
        differently depending on ``weight_mode``.
    weight_mode :
        How ``weights`` interact with store sizes.

        ``"per_store"`` (default) — the probability of drawing from store *i*
        is ``weight[i] / sum(weights)``.  Store size is ignored; all stores
        are equally likely when ``weights`` is ``None``.

        ``"per_step"`` — the probability is proportional to
        ``weight[i] * len(store_i)``.  Larger stores are drawn more often;
        ``weights`` act as multipliers on top of the size-proportional
        baseline.
    pack :
        If ``True``, a sequence can start at any position in a store,
        including the last few steps. When the initial window is shorter than
        ``sequence_length``, additional independently-sampled segments (drawn
        from the same store distribution) are appended until the sequence is
        full. ``next_batch`` returns parallel ``segment_ids``: within each
        sequence the first pack slice is ``0``, the next ``1``, and so on.
        The model uses these IDs for attention/RoPE isolation and TD
        objectives skip pairs whose adjacent steps have different IDs. Empty
        stores are allowed at construction and on :meth:`refresh`;
        :meth:`next_batch` raises if every store is still empty. Mutually
        exclusive with ``pad``.
    pad :
        If ``True``, a sequence can start at any position in a store. When the
        remaining suffix is shorter than ``sequence_length``, the window is
        right-padded by repeating the last real row until length ``S``. Each
        padded step gets its own ``segment_id`` (continuing after the real
        slice's id) so attention/RoPE isolate pads and TD objectives skip
        pad transitions. Empty stores are allowed at construction and on
        :meth:`refresh`; :meth:`next_batch` raises if every store is still
        empty. Mutually exclusive with ``pack``.

        If both ``pack`` and ``pad`` are ``False`` (default), every sequence
        comes from a single in-store window of exactly ``sequence_length``
        steps (all ``segment_ids`` are ``0``); stores shorter than
        ``sequence_length`` are rejected at construction.
    prefetch :
        How many batches to keep pre-fetched in the background queue.
    num_workers :
        Background worker threads (0 = synchronous). ``num_workers > 0``
        requires a free-threaded CPython build with the GIL disabled.
    seed :
        Seed for the loader's internal NumPy RNG. ``None`` (default) uses
        unpredictable entropy. When set, each worker's sampling RNG and its
        forked augmenter receive independent child seeds derived via
        ``numpy.random.SeedSequence(seed).spawn(...)``, so multi-worker
        sampling is reproducible for a given ``seed`` and no two streams
        (across workers, or between sampling and augmentation) coincide.
        Each :meth:`refresh` that restarts workers draws a fresh spawn wave
        from the same ``SeedSequence`` so streams do not replay after refresh.
    augmenter :
        Optional callable applied to each sampled raw ``[B][S]`` batch **before**
        the preparer. With background workers, augmentation runs in the worker.
        If the callable exposes ``fork(seed=...)``, each worker receives an
        independent copy with its own RNG.
    preparer :
        Callable ``(batch, segment_ids) -> TokenBatch`` from
        ``encoder.make_preparer()``. Required for training: turns raw steps into
        a flat concatenated token stream. When ``None``, :meth:`next_batch`
        returns ``(batch, segment_ids)`` for debugging only.
    """

    def __init__(
        self,
        stores: Datastore | list[Datastore],
        sequence_length: int,
        batch_size: int,
        *,
        weights: list[float] | None = None,
        weight_mode: str = "per_store",
        pack: bool = False,
        pad: bool = False,
        prefetch: int = 4,
        num_workers: int = 1,
        seed: int | None = None,
        augmenter: BatchAugmenter | None = None,
        preparer: BatchPreparer | None = None,
    ) -> None:
        from mouse_core.data.datastore import Datastore as _DS

        # Set teardown attrs early so __del__/close are safe even if later validation fails.
        self._stop: threading.Event | None = None
        self._result_queue: queue.Queue | None = None
        self._workers: list[threading.Thread] = []
        self._sync_rng: np.random.Generator | None = None
        self._sync_augmenter: BatchAugmenter | None = None
        self._worker_error: BaseException | None = None

        if isinstance(stores, _DS):
            stores = [stores]
        if not stores or not all(isinstance(s, _DS) for s in stores):
            raise TypeError("DataLoader requires a Datastore or a non-empty list of Datastores.")
        if weight_mode not in ("per_store", "per_step"):
            raise ValueError(f"weight_mode must be 'per_store' or 'per_step', got {weight_mode!r}")
        if pack and pad:
            raise ValueError("pack and pad are mutually exclusive; set at most one of them.")
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
        self.pack = pack
        self.pad = pad
        self.seed = seed
        self.augmenter = augmenter
        self.preparer = preparer
        self._num_workers = num_workers
        self._prefetch = prefetch
        self._sync_preparer: BatchPreparer | None = None
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
            self._sync_augmenter = augmenter
            self._sync_preparer = preparer
        else:
            self._start_workers()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def total_batches(self) -> int:
        """Approximate total non-overlapping windows across all stores."""
        total_windows = sum(n // self.sequence_length for n in self._ns)
        return max(0, (total_windows + self.batch_size - 1) // self.batch_size)

    def refresh(self) -> None:
        """Drop prefetched batches and re-snapshot all stores.

        Call after appending new rows to the underlying ``Datastore`` objects
        so :meth:`next_batch` samples from the latest data. When
        ``weight_mode="per_step"``, store-size weights are recomputed too.
        With background workers, threads are restarted against the new
        snapshot so they never read a stale shared object graph.
        """
        if self._num_workers > 0:
            self._stop_workers()
        self._resnapshot_stores()
        if self._num_workers > 0:
            self._start_workers()

    def set_preparer(self, preparer: BatchPreparer | None) -> None:
        """Attach or replace the encoder preparer (restarts workers if needed)."""
        self.preparer = preparer
        if self._num_workers == 0:
            self._sync_preparer = preparer
            return
        self._stop_workers()
        self._start_workers()

    def next_batch(self) -> Any:
        """Return the next :class:`TokenBatch` (or raw batch if no preparer).

        With ``preparer=`` (normal training), returns a flat concatenated
        ``TokenBatch``. Without a preparer, returns ``(batch, segment_ids)`` as
        ``list[list[dict]]`` / ``list[list[int]]`` for debugging.
        """
        if self._sync_rng is not None:
            cfg = self._snapshot_config()
            return _fetch_one_batch(
                cfg, self._sync_rng, self._sync_augmenter, self._sync_preparer
            )
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
            f"DataLoader(stores=[{store_info}], S={self.sequence_length}, "
            f"B={self.batch_size}, pack={self.pack}, pad={self.pad}, seed={self.seed})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _snapshot_config(self) -> _SnapshotConfig:
        return _SnapshotConfig(
            datasets=tuple(self._datasets),
            ns=tuple(self._ns),
            probs=self._probs.copy(),
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            pack=self.pack,
            pad=self.pad,
        )

    def _worker_seeds(self) -> tuple[list[Any], list[int | None]]:
        n = self._num_workers
        if self._seed_seq is None:
            return [None] * n, [None] * n
        # Independent child seeds per worker for sampling (even slots) and
        # augmentation (odd slots): seed arithmetic like ``seed + i`` makes
        # the two streams within a worker identical and lets nearby seeds
        # share worker streams across loaders. Each call advances the
        # SeedSequence spawn counter so refresh waves do not replay.
        children = self._seed_seq.spawn(2 * n)
        sample_seeds: list[Any] = list(children[0::2])
        augment_seeds: list[int | None] = [
            int(child.generate_state(1)[0]) for child in children[1::2]
        ]
        return sample_seeds, augment_seeds

    def _start_workers(self) -> None:
        assert self._num_workers > 0
        self._worker_error = None
        self._result_queue = queue.Queue(maxsize=self._prefetch)
        self._stop = threading.Event()
        cfg = self._snapshot_config()
        sample_seeds, augment_seeds = self._worker_seeds()
        self._workers = []
        for i in range(self._num_workers):
            thread = threading.Thread(
                target=_worker_loop,
                args=(
                    self._result_queue,
                    self._stop,
                    cfg,
                    sample_seeds[i],
                    _fork_augmenter(self.augmenter, seed=augment_seeds[i]),
                    self.preparer,
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

        allow_short = self.pack or self.pad
        if not allow_short:
            for i, (n, s) in enumerate(zip(self._ns, self.stores)):
                if n == 0:
                    raise ValueError(f"Store {i} ({s!r}) is empty.")
                if n < self.sequence_length:
                    name = s.name or f"index {i}"
                    raise ValueError(
                        f"Store {name!r} has {n} steps but sequence_length={self.sequence_length}. "
                        "Use pack=True or pad=True to allow shorter stores."
                    )
        w = self._weights.copy()
        ns = np.array(self._ns, dtype=float)
        if self.weight_mode == "per_step":
            w = w * ns
        elif allow_short:
            w = w * (ns > 0)
        if w.sum() == 0:
            self._probs = np.ones(len(self.stores)) / len(self.stores)
        else:
            self._probs = w / w.sum()
