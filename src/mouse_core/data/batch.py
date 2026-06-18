"""PrefetchBatchifier — background-thread batch prefetching from a DatasetStore.

A ``DatasetStore`` holds a flat sequential stream of steps (conventionally
``MouseEnvRecord`` rows: mouse-env results + the paired action).

Training sequences are therefore just contiguous slices of that sequence.
``PrefetchBatchifier`` repeatedly takes slices, encodes the mouse-env contract
fields (via ``encode_hf_rows``) into the flat tensors the model pipeline needs,
and yields ready ``TensorDict[B, S]`` batches.

Workers (when enabled) do the slice+encode work in the background so
``next_batch()`` is usually immediate.

Usage
-----
::
    store = DatasetStore()
    store.from_dataset(ds)   # ds from load_dataset (config, split, ...)

    bf = PrefetchBatchifier(
        store,
        sequence_length=64,
        batch_size=8,
        max_action_dim=...,   # model sizes
    )
    td = bf.next_batch()   # TensorDict[B, S]
    bf.close()
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np
from tensordict import TensorDict

if TYPE_CHECKING:
    from mouse_core.data.dataset_store import DatasetStore


class PrefetchBatchifier:
    """Background batch prefetcher over a sequential ``DatasetStore``.

    Because storage is sequential, every training sequence is a contiguous
    slice of the store.  This class just keeps a few such encoded slices ready.

    Parameters
    ----------
    store :
        ``DatasetStore`` after you have called ``from_dataset``. Rows are
        expected to follow the mouse-env contract (see ``MouseEnvRecord``).
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
    pin_memory : bool
        Pin the CPU tensors (CUDA only).
    max_action_dim :
        Width for discrete action index and q_star (model side).
    max_action_continuous_dim :
        Continuous action vector size for batches (0 disables).
    max_obs_continuous_dim, max_obs_discrete_dim, max_obs_image_pixels :
        Observation feature sizes for the projected batch tensors.
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
        *,
        max_action_dim: int = 1000,
        max_action_continuous_dim: int = 0,
        max_obs_continuous_dim: int = 0,
        max_obs_discrete_dim: int = 0,
        max_obs_image_pixels: int = 0,
    ) -> None:
        from mouse_core.data.dataset_store import DatasetStore as _DS

        # Set teardown attrs early so __del__/close are safe even if later validation fails
        self._stop = None
        self._result_queue = None
        self._workers = []
        self._sync_rng = None
        self._worker_error = None

        if not isinstance(store, _DS) or store._source is None:
            raise TypeError(
                "PrefetchBatchifier requires a DatasetStore that has data loaded via from_dataset(). "
                "After pure append, do store.from_dataset(store.to_dataset()) first, or load via from_dataset."
            )
        if sampling not in ("batch", "random", "sequential", "last"):
            raise ValueError(f"sampling must be one of batch/random/sequential/last, got {sampling!r}")

        self.store = store
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.sampling = sampling

        self._dataset = store._source
        self._n = len(self._dataset)

        # Target shapes for encode (owned here, not by the store).
        self._max_action_dim = int(max_action_dim)
        self._max_action_continuous_dim = int(max_action_continuous_dim)
        self._max_obs_continuous_dim = int(max_obs_continuous_dim)
        self._max_obs_discrete_dim = int(max_obs_discrete_dim)
        self._max_obs_image_pixels = int(max_obs_image_pixels)

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
        td = self.store.encode_hf_rows(
            merged,
            max_action_dim=self._max_action_dim,
            max_action_continuous_dim=self._max_action_continuous_dim,
            max_obs_continuous_dim=self._max_obs_continuous_dim,
            max_obs_discrete_dim=self._max_obs_discrete_dim,
            max_obs_image_pixels=self._max_obs_image_pixels,
        ).reshape(len(starts), S)
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
