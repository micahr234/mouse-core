"""DatasetStore — ordered sequence of arbitrary step records backed by a Hugging Face Dataset.

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
import torch
from tensordict import TensorDict

import datasets
from datasets import Dataset, concatenate_datasets

datasets.disable_progress_bar()

# Modality sub-keys used by the current default encoder when present in
# nested "observation" / "action" columns. Any sub-key name is valid in
# source data; only these names are interpreted by encode_hf_rows.
ACTION_KEY_DISCRETE = "discrete"
ACTION_KEY_CONTINUOUS = "continuous"

OBS_KEY_DISCRETE = "discrete"
OBS_KEY_CONTINUOUS = "continuous"
OBS_KEY_IMAGE = "image"


def _rows_to_hf_dict(rows: list[dict]) -> dict[str, list]:
    """Convert a list of row dicts to a dict-of-lists (HF batch format)."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: [r.get(k) for r in rows] for k in keys}


def _interleave_tds(
    n: int,
    src_td: TensorDict | None,
    src_pos: np.ndarray,
    buf_td: TensorDict | None,
    buf_pos: np.ndarray,
) -> TensorDict:
    """Interleave two TensorDicts at specified positions into a single TensorDict[n]."""
    keys: set[str] = set()
    if src_td is not None:
        keys |= {str(k) for k in src_td.keys()}
    if buf_td is not None:
        keys |= {str(k) for k in buf_td.keys()}

    result: dict[str, torch.Tensor] = {}
    for key in keys:
        ref = (
            src_td[key]
            if src_td is not None and key in src_td.keys()
            else buf_td[key]  # type: ignore[index]
        )
        tail = ref.shape[1:]
        t = torch.zeros(n, *tail, dtype=ref.dtype)
        if src_td is not None and key in src_td.keys():
            t[src_pos] = src_td[key]
        if buf_td is not None and key in buf_td.keys():
            t[buf_pos] = buf_td[key]
        result[key] = t

    return TensorDict(result, batch_size=[n])


class DatasetStore:
    """Ordered sequence of arbitrary rows, backed by a Hugging Face Dataset.

    The store does not care what is inside the rows. Each row is just a dict
    you give it. It only provides fast append and fast contiguous slicing.

    Field extraction and tensorization happen in ``DataLoader`` / ``encode_hf_rows``;
    the resulting shapes come from the data (with per-slice max for vectors).
    Model embedders are sized for the capacities they need and adapt on input.
    """

    def __init__(self) -> None:

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
        return f"DatasetStore(steps={len(self)})"

    def __getitem__(self, indices: Any, **encode_kwargs) -> TensorDict:
        """Return encoded step records for the given indices as a TensorDict[N].

        Extra kwargs are ignored (dimensions are derived from the data itself).
        Most training code goes through DataLoader instead.
        """
        src_len = self._src_len
        idx = np.asarray(indices).ravel()

        if self._buf_len == 0:
            return self.encode_hf_rows(self._source[idx.tolist()], **encode_kwargs)  # type: ignore[index]

        if src_len == 0:
            rows = [self._rows[int(i)] for i in idx]
            return self.encode_hf_rows(_rows_to_hf_dict(rows), **encode_kwargs)

        # Mixed
        src_mask = idx < src_len
        buf_mask = ~src_mask
        src_positions = np.where(src_mask)[0]
        buf_positions = np.where(buf_mask)[0]

        assert self._source is not None
        src_td = (
            self.encode_hf_rows(self._source[idx[src_mask].tolist()], **encode_kwargs)
            if src_mask.any() else None
        )
        buf_td = None
        if buf_mask.any():
            buf_rows = [self._rows[int(i) - src_len] for i in idx[buf_mask]]
            buf_td = self.encode_hf_rows(_rows_to_hf_dict(buf_rows), **encode_kwargs)

        return _interleave_tds(len(idx), src_td, src_positions, buf_td, buf_positions)

    # ------------------------------------------------------------------
    # Append  (rollout / test path)
    # ------------------------------------------------------------------

    def append(self, data: dict[str, Any]) -> None:
        """Append a single row to the in-memory buffer.

        This is the fast path for collection loops.  The row can contain any
        keys and nested values; nothing is validated or transformed here.
        """
        if not data:
            raise ValueError("Row cannot be empty.")
        self._rows.append(dict(data))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _modality_column(struct_rows: list[Any], key: str) -> list[Any]:
        """Pull one modality (``key``) out of a struct column.

        ``action`` and ``observation`` are always dicts; ``key`` may be any
        sub-key name. Rows lacking the sub-key yield ``None``.
        """
        return [r.get(key) for r in struct_rows]

    @staticmethod
    def _vector_buffer(col: list[Any], dtype: Any) -> np.ndarray | None:
        """Pack a modality column of per-row vectors into ``[n, D]``, zero-filled.

        D is the largest length present among non-None values in *this* column.
        Shorter values are right-padded with zeros; absent (None) rows are zero.
        Returns ``None`` when the modality is absent from every row.
        """
        present = [v for v in col if v is not None]
        if not present:
            return None
        dims = [np.asarray(v, dtype=dtype).ravel().size for v in present]
        dim = max(dims) if dims else 0
        if dim == 0:
            return None
        buf = np.zeros((len(col), dim), dtype=dtype)
        for i, v in enumerate(col):
            if v is not None:
                arr = np.asarray(v, dtype=dtype).ravel()
                d = min(arr.size, dim)
                buf[i, :d] = arr[:d]
        return buf

    def encode_hf_rows(
        self,
        rows: dict[str, Any],
        **kwargs: Any,
    ) -> TensorDict:
        """Project a batch of rows into a TensorDict for the model.

        This (default) encoder extracts conventional fields the models are
        wired to consume and stacks them as tensors.

        Vector fields (continuous obs/action, images, q_star) have their width
        taken from the data: D is the max length present in *this* slice; rows
        lacking the value or having shorter are zero (or -inf for q_star) padded
        on the right to that D. No external target sizes are required.

        Any other content in the rows is ignored for the batch but stays in
        the underlying data. The shapes the model receives come from your data;
        configure your model's embedders with capacities large enough for them.
        """
        # Legacy size kwargs (max_*_dim etc.) are ignored; dimensions are data-driven.
        for legacy in ("max_action_dim", "max_action_continuous_dim",
                       "max_obs_continuous_dim", "max_obs_discrete_dim",
                       "max_obs_image_pixels"):
            kwargs.pop(legacy, None)

        cols = set(rows.keys())
        n = len(rows[next(iter(cols))]) if cols else 0
        tensors: dict[str, torch.Tensor] = {}

        # Always emit discrete action index (0 placeholder for continuous-only rows).
        disc_actions = self._modality_column(rows["action"], ACTION_KEY_DISCRETE)
        action_arr = np.array([0 if a is None else a for a in disc_actions], dtype=np.int64)
        tensors["action"] = torch.from_numpy(action_arr)
        tensors["reward"] = torch.from_numpy(np.asarray(rows["reward"], dtype=np.float32))
        tensors["done"]   = torch.from_numpy(np.asarray(rows["done"],   dtype=np.int64))

        # Continuous action (if any row in *this* slice has it)
        buf = self._vector_buffer(
            self._modality_column(rows["action"], ACTION_KEY_CONTINUOUS),
            np.float32,
        )
        if buf is not None:
            tensors["action_continuous"] = torch.from_numpy(buf)

        # Optional scalar fields
        if "reward_episodic" in cols:
            tensors["xformed_reward"] = torch.from_numpy(
                np.asarray(rows["reward_episodic"], dtype=np.float32)
            )
        if "time" in cols:
            tensors["time"] = torch.from_numpy(
                np.asarray(rows["time"], dtype=np.int64)
            )

        # q_star — width comes from the data in this slice.
        if "q_star" in cols:
            q_list = rows["q_star"]
            ref = next((np.asarray(v, dtype=np.float32).ravel() for v in q_list if v is not None), None)
            if ref is not None:
                q_dim = ref.size
                q_buf = np.full((n, q_dim), -np.inf, dtype=np.float32)
                for i, v in enumerate(q_list):
                    if v is not None:
                        qa = np.asarray(v, dtype=np.float32).ravel()
                        d = min(qa.size, q_dim)
                        q_buf[i, :d] = qa[:d]
                tensors["q_star"] = torch.from_numpy(q_buf)

        # Observation modalities — emitted only when at least one row in the slice has the sub-modality.
        if "observation" in cols:
            obs_rows = rows["observation"]

            buf = self._vector_buffer(
                self._modality_column(obs_rows, OBS_KEY_CONTINUOUS),
                np.float64,
            )
            if buf is not None:
                tensors["obs_continuous"] = torch.from_numpy(buf)

            disc = self._modality_column(obs_rows, OBS_KEY_DISCRETE)
            if any(d is not None for d in disc):
                tensors["obs_discrete"] = torch.from_numpy(
                    np.array([0 if d is None else d for d in disc], dtype=np.int64)
                )

            buf = self._vector_buffer(
                self._modality_column(obs_rows, OBS_KEY_IMAGE),
                np.int64,
            )
            if buf is not None:
                tensors["obs_image"] = torch.from_numpy(buf)

        return TensorDict(tensors, batch_size=[n])

    # ------------------------------------------------------------------
    # HuggingFace Dataset I/O
    # ------------------------------------------------------------------

    def from_dataset(self, ds: "Dataset | datasets.DatasetDict") -> None:
        """Ingest an already-loaded Hugging Face ``Dataset`` into the store.

        All selection (which config/subset, which split, globs, etc.) is done
        with the real ``datasets.load_dataset``. This method just takes ownership
        of the rows so they can be mixed with later appends and fed to the
        batch encoder.

        If you pass a DatasetDict its contents are concatenated (the store is
        a flat sequence). Use separate stores if you want to keep logical
        separation.

        Calling repeatedly extends the history.

        Examples::

            ds = load_dataset("user/my-rollouts", "cartpole_expert", split="train")
            store.from_dataset(ds)

            ds2 = load_dataset("user/my-rollouts", "lunar_random", split="train")
            store.from_dataset(ds2)
        """
        if isinstance(ds, datasets.DatasetDict):
            # Concatenate whatever splits are present, in the dict's iteration order.
            parts = [d for d in ds.values() if len(d) > 0]
            if not parts:
                return
            ds = concatenate_datasets(parts)

        if len(ds) == 0:
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

    @classmethod
    def merge_stores_to_dataset(cls, stores: list[DatasetStore]) -> Dataset:
        """Concatenate multiple DatasetStores into one HF Dataset."""
        parts = [p for s in stores if len(p := s.to_dataset()) > 0]
        return concatenate_datasets(parts) if parts else Dataset.from_list([])

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
