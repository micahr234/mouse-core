"""PrefetchBatchifier — background-thread batch prefetching from a DatasetStore.

``num_workers`` daemon threads continuously fetch batches from the Arrow-backed
Dataset, encode them, and park them in a bounded queue.  ``next_batch()`` pops
from that queue — instant once the queue is warm.

When ``num_workers=0`` the class operates in **synchronous mode**: no threads
or queue are created, and ``next_batch()`` fetches directly on the calling
thread.  This is useful for debugging (no background threads competing for the
debugger) and for environments where threading is undesirable.

The full dataset is never materialised in RAM; only
``prefetch × batch_size × sequence_length`` steps are held at any time.

Concurrency
-----------
Arrow reads and NumPy operations both release the GIL, so ``threading`` gives
genuine parallelism for data-loading work.  Multiple workers help for
``random``/``last`` sampling.  ``sequential``/``batch`` are always single-worker
to preserve epoch order.

Usage
-----
::
    store = DatasetStore(...)
    store.from_dataset(ds)

    with PrefetchBatchifier(store, sequence_length=64, batch_size=8) as bf:
        td = bf.next_batch()   # TensorDict, instant once warm
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDict


def to_tensor_dict(raw: np.ndarray) -> TensorDict:
    """Convert a structured numpy step array to a TensorDict (zero-copy via torch.from_numpy)."""
    assert raw.dtype.names is not None
    return TensorDict(
        {name: torch.from_numpy(np.ascontiguousarray(raw[name])) for name in raw.dtype.names},
        batch_size=raw.shape,
    )

if TYPE_CHECKING:
    from mouse.data.dataset_store import DatasetStore


class PrefetchBatchifier:
    """Background-thread batchifier for a ``DatasetStore`` source.

    Parameters
    ----------
    store :
        ``DatasetStore`` with ``from_dataset`` already called.
    sequence_length :
        Number of consecutive steps per sequence.
    batch_size :
        Number of sequences per batch.
    sampling :
        How start indices are chosen; see ``_sample_starts``.
    prefetch : int
        Pre-encoded batches to keep ready.  Higher values smooth over slow
        Arrow reads; each batch costs ``batch_size × sequence_length`` encoded
        steps in memory.
    num_workers : int
        Background threads.  ``sequential``/``batch`` modes ignore this and
        always use one worker to preserve epoch order.  Pass ``0`` to skip
        threading entirely and fetch synchronously on the calling thread.
    pin_memory : bool
        If True, workers call ``.pin_memory()`` on each batch before queuing
        it.  Pinned CPU tensors enable DMA-backed, non-blocking H2D copies on
        the main thread.  Only effective when the training device is CUDA.
        Ignored in synchronous mode (``num_workers=0``).
    """

    def __init__(
        self,
        store: DatasetStore,
        sequence_length: int,
        batch_size: int,
        sampling: str = "random",
        prefetch: int = 4,
        num_workers: int = 1,
        pin_memory: bool = False,
    ) -> None:
        from mouse.data.dataset_store import DatasetStore as _DS

        if not isinstance(store, _DS) or store._source is None:
            raise TypeError(
                "PrefetchBatchifier requires a DatasetStore with a loaded source dataset. "
                "Call store.from_dataset(ds) first."
            )
        if sampling not in ("batch", "random", "sequential", "last"):
            raise ValueError(f"sampling must be one of batch/random/sequential/last, got {sampling!r}")

        self.store = store
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.sampling = sampling

        self._dataset = store._source
        self._n = len(self._dataset)

        # Epoch-order state for sequential / batch / synchronous modes.
        self._lock = threading.Lock()
        self._next_window: int = 0
        self._window_order: np.ndarray = self._new_epoch_order()

        self._pin_memory = pin_memory

        n_workers = num_workers if sampling in ("random", "last") else 1

        if n_workers == 0:
            # Synchronous mode: fetch directly on the calling thread.
            self._sync_rng: np.random.Generator | None = np.random.default_rng(seed=0)
            self._result_queue: queue.Queue[TensorDict] | None = None
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
                    name=f"PrefetchBatchifier-{i}",
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

    def next_batch(self) -> TensorDict:
        """Return the next pre-encoded batch, blocking until one is ready."""
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

    def __enter__(self) -> PrefetchBatchifier:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"PrefetchBatchifier(n={self._n}, S={self.sequence_length}, "
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

    def _fetch_one_batch(self, rng: np.random.Generator) -> TensorDict:
        starts = self._sample_starts(rng)
        S = self.sequence_length
        seqs = [self._dataset[int(s) : int(s) + S] for s in starts]
        merged = {k: [v for seq in seqs for v in seq[k]] for k in seqs[0]}
        encoded = self.store.encode_hf_rows(merged)
        td = to_tensor_dict(encoded.reshape(len(starts), S))
        if self._pin_memory:
            td = td.pin_memory()
        return td

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
