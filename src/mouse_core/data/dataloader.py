"""DataLoader — batch producer for sequential step data from a Datastore.

A ``Datastore`` is a flat sequence of arbitrary rows.

The ``DataLoader`` repeatedly samples contiguous windows of ``S`` rows from
the store and returns them as a ``list[list[dict]]`` of shape ``[B][S]``.
Each element is a plain Python dict exactly as it was stored (no encoding or
tensorisation happens here). The model (its encoder) is responsible for
extracting and converting the fields it needs.

When background workers are enabled the slice work happens in the background
so ``next_batch()`` is usually immediate.

Usage
-----
::

    store = Datastore()
    store.from_dataset(ds)   # or populate via append()

    loader = DataLoader(
        store,
        sequence_length=64,
        batch_size=8,
    )
    td = loader.next_batch()   # TensorDict[B, S]
    loader.close()
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mouse_core.data.datastore import Datastore


class DataLoader:
    """Produces ``list[list[dict]]`` batches of raw step records from a ``Datastore``.

    The store holds a flat sequence of arbitrary rows. This loader samples
    contiguous windows and returns them as Python dicts without any encoding.
    Columns are untouched; the model's encoder decides what to extract and
    how to tensorize it.

    Background workers (when enabled) do the work asynchronously.

    The feature dimensions are *not* passed here; they live in your model
    embedder configuration. The embedders adapt to the shapes that arrive
    from data.

    Stores populated only via ``append`` are supported (a snapshot is taken at
    construction time).

    Parameters
    ----------
    store :
        ``Datastore`` (loaded via ``from_dataset`` or populated via ``append``).
        A pure-append store is supported and will be snapshotted internally for
        iteration; appends after construction are not observed by the loader.
    sequence_length :
        Length of each contiguous slice (in steps).
    batch_size :
        How many such slices per batch.
    sampling :
        Policy for choosing the starting positions of the slices.
    prefetch : int
        How many batches to keep pre-encoded.
    num_workers : int
        Background workers (0 = synchronous).
    """

    def __init__(
        self,
        store: Datastore,
        sequence_length: int,
        batch_size: int,
        sampling: str = "random",
        prefetch: int = 4,
        num_workers: int = 1,
    ) -> None:
        from mouse_core.data.datastore import Datastore as _DS

        # Set teardown attrs early so __del__/close are safe even if later validation fails
        self._stop = None
        self._result_queue = None
        self._workers = []
        self._sync_rng = None
        self._worker_error = None

        if not isinstance(store, _DS):
            raise TypeError("DataLoader requires a Datastore instance.")
        if len(store) == 0:
            raise ValueError("DataLoader requires a non-empty Datastore.")
        if sampling not in ("batch", "random", "sequential", "last"):
            raise ValueError(f"sampling must be one of batch/random/sequential/last, got {sampling!r}")

        self.store = store
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.sampling = sampling

        # Use the Arrow Dataset for fast slicing. Prefer the loaded source when present
        # (the append buffer is not visible to the loader until the user calls from_dataset).
        # For pure-append stores we snapshot once for convenience.
        self._dataset = store._source if store._source is not None else store.to_dataset()
        self._n = len(self._dataset)

        # Epoch-order state for sequential / batch / synchronous modes.
        self._lock = threading.Lock()
        self._next_window: int = 0
        self._window_order: np.ndarray = self._new_epoch_order()

        n_workers = num_workers if sampling in ("random", "last") else 1

        if n_workers == 0:
            # Synchronous mode: fetch directly on the calling thread.
            self._sync_rng: np.random.Generator | None = np.random.default_rng(seed=0)
            self._result_queue: queue.Queue[list[list[dict]]] | None = None
            self._stop: threading.Event | None = None
            self._worker_error: BaseException | None = None
            self._workers: list[threading.Thread] = []
        else:
            self._sync_rng = None
            self._result_queue = queue.Queue(maxsize=prefetch)
            self._stop = threading.Event()
            self._worker_error = None
            self._workers = [
                threading.Thread(
                    target=self._worker_loop,
                    args=(np.random.default_rng(seed=i),),
                    daemon=True,
                    name=f"DataLoader-{i}",
                )
                for i in range(n_workers)
            ]
            for w in self._workers:
                w.start()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def total_batches(self) -> int:
        """Approximate non-overlapping windows in the dataset."""
        num_windows = self._n // self.sequence_length
        return max(0, (num_windows + self.batch_size - 1) // self.batch_size)

    def next_batch(self) -> list[list[dict]]:
        """Return the next batch of raw rows, blocking until one is ready.

        Returns:
            ``list[list[dict]]`` of shape ``[B][S]``.  Each inner dict is a
            step record exactly as stored in the Datastore.
        """
        if self._sync_rng is not None:
            return self._fetch_one_batch(self._sync_rng)
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
            return  # synchronous mode — nothing to tear down
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
        return (
            f"DataLoader(n={self._n}, S={self.sequence_length}, "
            f"B={self.batch_size}, sampling={self.sampling!r})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _new_epoch_order(self) -> np.ndarray:
        num_windows = max(self._n // self.sequence_length, 0)
        return (
            np.arange(num_windows)
            if self.sampling == "sequential"
            else np.random.permutation(num_windows)
        )

    def _sample_starts(self, rng: np.random.Generator) -> np.ndarray:
        """Return B start indices.  Each sequence is always dataset[start:start+S]."""
        N, S, B = self._n, self.sequence_length, self.batch_size

        if self.sampling == "random":
            return rng.integers(0, N - S + 1, size=B)

        if self.sampling == "last":
            num_windows = N // S
            if num_windows < B:
                raise ValueError(
                    f"Dataset has {num_windows} windows but batch_size={B}; "
                    "need at least batch_size windows for sampling='last'."
                )
            return np.arange(num_windows - B, num_windows) * S

        with self._lock:
            num_windows = N // S
            if len(self._window_order) != num_windows:
                self._window_order = self._new_epoch_order()
                self._next_window = 0
            if self._next_window >= num_windows:
                self._next_window = 0
                self._window_order = self._new_epoch_order()
            start = self._next_window
            end = min(start + B, num_windows)
            windows = self._window_order[start:end].copy()
            self._next_window = end

        return windows * S

    def _fetch_one_batch(self, rng: np.random.Generator) -> list[list[dict]]:
        starts = self._sample_starts(rng)
        S = self.sequence_length
        batch: list[list[dict]] = []
        for s in starts:
            hf_slice = self._dataset[int(s) : int(s) + S]
            seq = [{k: hf_slice[k][i] for k in hf_slice} for i in range(S)]
            batch.append(seq)
        return batch

    def _worker_loop(self, rng: np.random.Generator) -> None:
        assert self._stop is not None and self._result_queue is not None
        while not self._stop.is_set():
            try:
                td = self._fetch_one_batch(rng)

                while not self._stop.is_set():
                    try:
                        self._result_queue.put(td, timeout=0.05)
                        break
                    except queue.Full:
                        pass

            except Exception as exc:  # noqa: BLE001
                self._worker_error = exc
                return
