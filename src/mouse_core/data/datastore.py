"""Datastore — ordered sequence of arbitrary step records backed by a Hugging Face Dataset.

A thin sequential container. It stores rows (plain dicts) in the order they
were appended or loaded. There is no required schema or "contract" — a row can
contain whatever fields your data source or collection produces.

Two operations are fast:

- ``append(one_row)`` for cheap live collection
- contiguous slices for training sequences

You select and load data with the real ``datasets.load_dataset`` (configs,
splits, globs, etc.) and hand the resulting Dataset to ``from_dataset``.

The store's main added value is mixing loaded history with new appends and
feeding slices to ``DataLoader`` (or ``encode_hf_rows``) so they can be turned
into model batches. All other I/O is standard Hugging Face Dataset / DatasetDict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import datasets
from datasets import Dataset, concatenate_datasets

datasets.disable_progress_bar()


def _normalize_value(value: Any) -> Any:
    """Unwrap 0-dim arrays/tensors to plain Python scalars.

    Environments (e.g. mouse-gym) emit step fields as 0-dim NumPy arrays.
    ``Dataset.from_list`` would serialize those as 1-element lists, breaking
    scalar consumers downstream, so scalars are unwrapped once at append time.
    """
    item = getattr(value, "item", None)
    if item is not None and getattr(value, "ndim", None) == 0:
        return item()
    return value


def _hf_batch_to_rows(batch: dict[str, Any]) -> list[dict]:
    """Convert an HF batch (dict-of-lists) to a list of row dicts."""
    if not batch:
        return []
    n = len(next(iter(batch.values())))
    return [{k: batch[k][i] for k in batch} for i in range(n)]


class Datastore:
    """Ordered sequence of arbitrary rows, backed by a Hugging Face Dataset.

    The store does not care what is inside the rows. Each row is just a dict
    you give it. It only provides fast append and fast contiguous slicing.

    ``name`` is optional metadata. Named stores can be pushed as individual
    Hugging Face dataset configs via ``push_stores_to_hub``; unnamed stores
    remain anonymous.

    No encoding or tensorisation happens here. ``DataLoader.next_batch()``
    returns plain row dicts plus parallel segment IDs; the model's encoder
    extracts and converts what it needs from the rows.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name

        # Source segment — HF Dataset stored by reference, never mutated.
        self._source: Dataset | None = None

        # Buf segment — raw row dicts for ``append`` (rollout path).
        self._rows: list[dict] = []

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    @property
    def _src_len(self) -> int:
        return len(self._source) if self._source is not None else 0

    @property
    def _buf_len(self) -> int:
        return len(self._rows)

    def __len__(self) -> int:
        return self._src_len + self._buf_len

    def __repr__(self) -> str:
        cols = self.columns
        col_str = f", columns={cols}" if cols else ""
        if self.name is not None:
            return f"Datastore(name={self.name!r}, steps={len(self)}{col_str})"
        return f"Datastore(steps={len(self)}{col_str})"

    @property
    def columns(self) -> list[str]:
        """Return the column names present in this store, or an empty list if the store is empty."""
        if self._source is not None:
            return list(self._source.column_names)
        if self._rows:
            return list(self._rows[0].keys())
        return []

    def __getitem__(self, indices: Any) -> list[dict]:
        """Return raw step records for the given indices as a list of dicts.

        Most training code goes through DataLoader instead.
        """
        src_len = self._src_len
        idx = np.asarray(indices).ravel()

        if self._buf_len == 0:
            return _hf_batch_to_rows(self._source[idx.tolist()])  # type: ignore[index]

        if src_len == 0:
            return [self._rows[int(i)] for i in idx]

        # Mixed: collect from both segments, preserving original order.
        src_mask = idx < src_len
        buf_mask = ~src_mask

        result: list[dict] = [{}] * len(idx)

        if src_mask.any():
            src_rows = _hf_batch_to_rows(
                self._source[idx[src_mask].tolist()]  # type: ignore[index]
            )
            for k, pos in enumerate(np.where(src_mask)[0]):
                result[pos] = src_rows[k]

        if buf_mask.any():
            for k, pos in enumerate(np.where(buf_mask)[0]):
                result[pos] = self._rows[int(idx[pos]) - src_len]

        return result

    # ------------------------------------------------------------------
    # Append  (rollout / test path)
    # ------------------------------------------------------------------

    def append(self, data: dict[str, Any] | Datastore | list[Datastore]) -> None:
        """Append a row, another store, or a list of stores.

        Row append is the fast path for collection loops. The row can contain
        any keys and nested values; nothing is validated or transformed here.

        Store append extends this store's sequence with the appended store's
        rows, preserving order.
        """
        if isinstance(data, Datastore):
            self.from_dataset(data.to_dataset())
            return
        if isinstance(data, list):
            if not all(isinstance(store, Datastore) for store in data):
                raise TypeError("append expects a row dict, Datastore, or list of Datastore objects.")
            for store in data:
                self.append(store)
            return
        if not isinstance(data, dict):
            raise TypeError("append expects a row dict, Datastore, or list of Datastore objects.")
        if not data:
            raise ValueError("Row cannot be empty.")
        self._rows.append({k: _normalize_value(v) for k, v in data.items()})

    # ------------------------------------------------------------------
    # HuggingFace Dataset I/O
    # ------------------------------------------------------------------

    def from_dataset(self, ds: "Dataset | datasets.DatasetDict") -> None:
        """Ingest an already-loaded Hugging Face ``Dataset`` into the store.

        All selection (which config/subset, which split, globs, etc.) is done
        before calling this method, either with ``load_stores_from_hub`` or
        directly with ``datasets.load_dataset`` for custom loading workflows.
        This method just takes ownership of the rows so they can be mixed with
        later appends and fed to the batch encoder.

        If you pass a DatasetDict its contents are concatenated (the store is
        a flat sequence). Use separate stores if you want to keep logical
        separation.

        Calling repeatedly extends the history.

        Examples::

            stores = load_stores_from_hub(
                "my-rollouts",
                split="train",
            )
        """
        if isinstance(ds, datasets.DatasetDict):
            # Concatenate whatever splits are present, in the dict's iteration order.
            parts = [d for d in ds.values() if len(d) > 0]
            if not parts:
                return
            ds = concatenate_datasets(parts)

        if len(ds) == 0:
            return
        if self._buf_len > 0:
            current = self.to_dataset()
            self._source = concatenate_datasets([current, ds]) if len(current) > 0 else ds
            self._rows.clear()
            return
        self._source = ds if self._source is None else concatenate_datasets([self._source, ds])

    def to_dataset(self) -> Dataset:
        """Return a HuggingFace Dataset of all steps.

        Source-only: returns the reference directly (zero-copy).
        Buf-only: builds a Dataset from the persisted raw rows.
        Both: concatenates source and buf datasets.
        """
        buf_ds = self._build_dataset(self._rows) if self._buf_len > 0 else None
        if self._source is None and buf_ds is None:
            return Dataset.from_list([])
        if self._source is None:
            return buf_ds  # type: ignore[return-value]
        if buf_ds is None:
            return self._source
        return concatenate_datasets([self._source, buf_ds])

    @staticmethod
    def _build_dataset(rows: list[dict]) -> Dataset:
        return Dataset.from_list(rows)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset the store."""
        self._source = None
        self._rows.clear()
