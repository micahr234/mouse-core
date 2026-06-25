"""DataLoader — batch producer for sequential step data from one or more Datastores.

A ``Datastore`` is a flat sequence of arbitrary rows.

The ``DataLoader`` repeatedly samples contiguous windows of ``S`` rows and
returns them as a ``list[list[dict]]`` of shape ``[B][S]``. Each element is a
plain Python dict exactly as it was stored, unless an optional batch augmenter
rewrites the raw rows before return (no encoding or tensorisation happens
here). The model (its encoder) is responsible for extracting and converting
the fields it needs.

When background workers are enabled the slice and augmentation work happens in
the background so ``next_batch()`` is usually immediate.

Usage
-----
Single store::

    store = Datastore()
    store.from_dataset(ds)

    loader = DataLoader(store, sequence_length=64, batch_size=8)
    batch = loader.next_batch()   # list[list[dict]] shape [B][S]
    loader.close()

Multiple stores with weights::

    loader = DataLoader(
        [store_a, store_b],
        sequence_length=64,
        batch_size=8,
        weights=[2.0, 1.0],
        weight_mode="per_store",
    )

With packing (sequences may span store boundaries)::

    loader = DataLoader(stores, sequence_length=64, batch_size=8, pack=True)
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mouse_core.data.datastore import Datastore


BatchAugmenter = Callable[[list[list[dict]]], list[list[dict]]]


class DataLoader:
    """Produces ``list[list[dict]]`` batches of raw step records from one or more Datastores.

    Each store holds a flat sequence of arbitrary rows. The loader samples
    contiguous windows and returns them as Python dicts without any encoding.
    Columns are untouched; the model's encoder decides what to extract and
    how to tensorize it.

    When multiple stores are provided, each sequence in a batch independently
    draws its store according to the computed per-store probabilities, giving
    smooth within-batch mixing.

    Background workers (when enabled) do the work asynchronously.

    Parameters
    ----------
    stores :
        A single ``Datastore`` or a list of them. Pure-append stores are
        supported and will be snapshotted internally; appends after
        construction are not observed by the loader.
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
        full. If ``False`` (default), every sequence comes from a single
        in-store window of exactly ``sequence_length`` steps; stores shorter
        than ``sequence_length`` are rejected at construction.
    prefetch :
        How many batches to keep pre-fetched in the background queue.
    num_workers :
        Background worker threads (0 = synchronous).
    augmenter :
        Optional callable applied to each sampled batch. With background workers,
        augmentation runs in the worker before the batch is put into the prefetch
        queue. If the callable exposes ``fork(seed=...)``, each worker receives an
        independent copy.
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
        prefetch: int = 4,
        num_workers: int = 1,
        augmenter: BatchAugmenter | None = None,
    ) -> None:
        from mouse_core.data.datastore import Datastore as _DS

        # Set teardown attrs early so __del__/close are safe even if later validation fails.
        self._stop = None
        self._result_queue = None
        self._workers = []
        self._sync_rng = None
        self._worker_error = None

        if isinstance(stores, _DS):
            stores = [stores]
        if not stores or not all(isinstance(s, _DS) for s in stores):
            raise TypeError("DataLoader requires a Datastore or a non-empty list of Datastores.")
        if weight_mode not in ("per_store", "per_step"):
            raise ValueError(f"weight_mode must be 'per_store' or 'per_step', got {weight_mode!r}")
        if weights is not None:
            if len(weights) != len(stores):
                raise ValueError(
                    f"weights length ({len(weights)}) must match number of stores ({len(stores)})."
                )
            if any(w <= 0 for w in weights):
                raise ValueError("All weights must be positive.")

        self.stores = stores
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.weight_mode = weight_mode
        self.pack = pack
        self.augmenter = augmenter

        # Snapshot each store as an Arrow Dataset for fast slicing.
        self._datasets = [
            s._source if s._source is not None else s.to_dataset()
            for s in stores
        ]
        self._ns: list[int] = [len(ds) for ds in self._datasets]

        if not pack:
            for i, (n, s) in enumerate(zip(self._ns, stores)):
                if n == 0:
                    raise ValueError(f"Store {i} ({s!r}) is empty.")
                if n < sequence_length:
                    name = s.name or f"index {i}"
                    raise ValueError(
                        f"Store {name!r} has {n} steps but sequence_length={sequence_length}. "
                        "Use pack=True to allow sequences that span store boundaries."
                    )
        else:
            for i, (n, s) in enumerate(zip(self._ns, stores)):
                if n == 0:
                    raise ValueError(f"Store {i} ({s!r}) is empty.")

        # Compute sampling probabilities.
        w = np.ones(len(stores)) if weights is None else np.asarray(weights, dtype=float)
        if weight_mode == "per_step":
            w = w * np.array(self._ns, dtype=float)
        self._probs: np.ndarray = w / w.sum()

        if num_workers == 0:
            self._sync_rng: np.random.Generator | None = np.random.default_rng(seed=0)
            self._sync_augmenter: BatchAugmenter | None = augmenter
            self._result_queue: queue.Queue[list[list[dict]]] | None = None
            self._stop: threading.Event | None = None
            self._worker_error: BaseException | None = None
            self._workers: list[threading.Thread] = []
        else:
            self._sync_rng = None
            self._sync_augmenter = None
            self._result_queue = queue.Queue(maxsize=prefetch)
            self._stop = threading.Event()
            self._worker_error = None
            self._workers = [
                threading.Thread(
                    target=self._worker_loop,
                    args=(
                        np.random.default_rng(seed=i),
                        _fork_augmenter(augmenter, seed=i),
                    ),
                    daemon=True,
                    name=f"DataLoader-{i}",
                )
                for i in range(num_workers)
            ]
            for w in self._workers:
                w.start()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def total_batches(self) -> int:
        """Approximate total non-overlapping windows across all stores."""
        total_windows = sum(n // self.sequence_length for n in self._ns)
        return max(0, (total_windows + self.batch_size - 1) // self.batch_size)

    def next_batch(self) -> list[list[dict]]:
        """Return the next batch of raw rows, blocking until one is ready.

        Returns:
            ``list[list[dict]]`` of shape ``[B][S]``.  Each inner dict is a
            step record exactly as stored in the Datastore.
        """
        if self._sync_rng is not None:
            return self._fetch_one_batch(self._sync_rng, self._sync_augmenter)
        assert self._result_queue is not None
        while True:
            if self._worker_error is not None:
                raise RuntimeError("A prefetch worker raised an exception.") from self._worker_error
            try:
                return self._result_queue.get(timeout=0.05)
            except queue.Empty:
                if not any(w.is_alive() for w in self._workers):
                    raise RuntimeError("All prefetch workers stopped unexpectedly.")

    def close(self) -> None:
        """Stop background workers and drain the queue."""
        if self._stop is None:
            return
        self._stop.set()
        assert self._result_queue is not None
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        for w in self._workers:
            w.join(timeout=2.0)

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
            f"B={self.batch_size}, pack={self.pack})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_sequence(self, rng: np.random.Generator) -> list[dict]:
        """Fetch exactly ``sequence_length`` steps for one sequence slot.

        Picks a store according to ``_probs``, draws a random start, and
        slices up to ``sequence_length`` steps.  When ``pack=True`` and the
        slice hits the end of the store, additional independently-sampled
        segments are appended until the sequence is full.  When ``pack=False``
        the start is guaranteed to leave a full window.
        """
        S = self.sequence_length
        steps: list[dict] = []

        while len(steps) < S:
            store_idx = int(rng.choice(len(self._datasets), p=self._probs))
            ds = self._datasets[store_idx]
            n = self._ns[store_idx]

            if self.pack:
                start = int(rng.integers(0, n))
            else:
                start = int(rng.integers(0, n - S + 1))

            end = min(start + (S - len(steps)), n)
            hf_slice = ds[start:end]
            count = end - start
            steps.extend(
                {k: hf_slice[k][i] for k in hf_slice} for i in range(count)
            )

            if not self.pack:
                # Without packing the single slice is always full; exit immediately.
                break

        return steps[:S]

    def _fetch_one_batch(
        self,
        rng: np.random.Generator,
        augmenter: BatchAugmenter | None,
    ) -> list[list[dict]]:
        batch = [self._fetch_sequence(rng) for _ in range(self.batch_size)]
        if augmenter is not None:
            batch = augmenter(batch)
        return batch

    def _worker_loop(self, rng: np.random.Generator, augmenter: BatchAugmenter | None) -> None:
        assert self._stop is not None and self._result_queue is not None
        while not self._stop.is_set():
            try:
                batch = self._fetch_one_batch(rng, augmenter)
                while not self._stop.is_set():
                    try:
                        self._result_queue.put(batch, timeout=0.05)
                        break
                    except queue.Full:
                        pass
            except Exception as exc:  # noqa: BLE001
                self._worker_error = exc
                return


def _fork_augmenter(augmenter: BatchAugmenter | None, *, seed: int) -> BatchAugmenter | None:
    if augmenter is None:
        return None
    fork = getattr(augmenter, "fork", None)
    if callable(fork):
        return fork(seed=seed)
    return augmenter
