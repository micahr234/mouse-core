"""DatasetStore — HuggingFace Dataset-backed step buffer.

Primary storage is a ``datasets.Dataset`` assigned by reference from
``from_dataset`` (O(1) — no encoding pass at load time).  A list buffer
handles ``append`` so rollout steps can be collected without materialising
the full dataset.

Sampling for training is handled by ``PrefetchBatchifier``, which reads
directly from the Dataset via ``encode_hf_rows`` — keeping only the active
prefetch queue in RAM rather than materialising the entire dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

import datasets
from datasets import Dataset, concatenate_datasets

datasets.disable_progress_bar()

# Typed-dict keys shared with the mouse-env rollout contract. ``action`` and
# ``observation`` columns are HuggingFace struct features whose sub-keys select
# the modality (see https://github.com/micahr234/mouse-env docs/guide.md).
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
    """HuggingFace Dataset-backed step buffer.

    Parameters
    ----------
    max_action_dim :
        Maximum number of discrete actions; used to clip q_star columns.
    max_action_continuous_dim :
        Number of continuous action dimensions to retain; 0 = no continuous action.
    max_obs_continuous_dim :
        Number of continuous observation dimensions to retain; 0 = no continuous obs.
    max_obs_discrete_dim :
        Number of discrete observation dimensions; 0 = no discrete obs.
    max_obs_image_pixels :
        Number of pixels per image observation; 0 = no image obs.
    """

    def __init__(
        self,
        max_action_dim: int = 1000,
        max_action_continuous_dim: int = 0,
        max_obs_continuous_dim: int = 0,
        max_obs_discrete_dim: int = 0,
        max_obs_image_pixels: int = 0,
    ) -> None:
        if int(max_action_dim) <= 0:
            raise ValueError(f"max_action_dim must be positive, got {max_action_dim}")
        self._max_action_dim = int(max_action_dim)
        self._max_action_continuous_dim = int(max_action_continuous_dim)
        self._max_obs_continuous_dim = int(max_obs_continuous_dim)
        self._max_obs_discrete_dim = int(max_obs_discrete_dim)
        self._max_obs_image_pixels = int(max_obs_image_pixels)

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

    def __getitem__(self, indices: Any) -> TensorDict:
        """Return encoded step records for the given indices as a TensorDict[N]."""
        src_len = self._src_len
        idx = np.asarray(indices).ravel()

        if self._buf_len == 0:
            return self.encode_hf_rows(self._source[idx.tolist()])  # type: ignore[index]

        if src_len == 0:
            rows = [self._rows[int(i)] for i in idx]
            return self.encode_hf_rows(_rows_to_hf_dict(rows))

        # Mixed: some indices from source, some from buf.
        src_mask = idx < src_len
        buf_mask = ~src_mask
        src_positions = np.where(src_mask)[0]
        buf_positions = np.where(buf_mask)[0]

        assert self._source is not None  # src_len != 0 in the mixed branch
        src_td = (
            self.encode_hf_rows(self._source[idx[src_mask].tolist()])
            if src_mask.any() else None
        )
        buf_td = None
        if buf_mask.any():
            buf_rows = [self._rows[int(i) - src_len] for i in idx[buf_mask]]
            buf_td = self.encode_hf_rows(_rows_to_hf_dict(buf_rows))

        return _interleave_tds(len(idx), src_td, src_positions, buf_td, buf_positions)

    # ------------------------------------------------------------------
    # Append  (rollout / test path)
    # ------------------------------------------------------------------

    def append(self, data: dict[str, Any]) -> None:
        """Append one transition to the rollout buffer."""
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
    def _vector_buffer(col: list[Any], dim: int, dtype: Any) -> np.ndarray | None:
        """Pack a modality column of per-row vectors into ``[n, dim]``, zero-filled.

        Each row's value is raveled and written into a zero row, truncated to
        ``dim``. Rows whose value is ``None`` (modality absent for that step)
        stay zero. Returns ``None`` when the modality is absent from every row.
        """
        if not any(v is not None for v in col):
            return None
        buf = np.zeros((len(col), dim), dtype=dtype)
        for i, v in enumerate(col):
            if v is not None:
                arr = np.asarray(v, dtype=dtype).ravel()
                d = min(arr.size, dim)
                buf[i, :d] = arr[:d]
        return buf

    def encode_hf_rows(self, rows: dict[str, Any]) -> TensorDict:
        """Encode a HF Dataset batch (dict-of-lists) into a TensorDict[N].

        Columns follow the mouse-env rollout contract. ``action`` and
        ``observation`` are always dict (struct) columns; any sub-key name is
        valid. This encoder maps the modality sub-keys the model understands
        (``discrete`` / ``continuous`` / ``image``) onto flat tensors and leaves
        every other sub-key untouched in the stored dataset. The discrete
        ``action`` index is always emitted (0 placeholder for continuous-only
        rows); the continuous action vector is emitted as ``action_continuous``
        when ``max_action_continuous_dim > 0`` and present in the rows. Other
        optional fields (``reward_episodic``, ``q_star``, ``time``, observation
        modalities) are only included as keys when present in ``rows`` — no
        zero-fill fallbacks, no silent substitutions.
        """
        cols = set(rows.keys())
        n = len(rows[next(iter(cols))]) if cols else 0
        tensors: dict[str, torch.Tensor] = {}

        # Required fields. ``action`` is a struct column whose sub-key selects the
        # modality (``discrete`` index and/or ``continuous`` vector). The discrete
        # index is always emitted; 0 placeholder where a step has no discrete action.
        disc_actions = self._modality_column(rows["action"], ACTION_KEY_DISCRETE)
        action_arr = np.array([0 if a is None else a for a in disc_actions], dtype=np.int64)
        tensors["action"] = torch.from_numpy(action_arr)
        tensors["reward"] = torch.from_numpy(np.asarray(rows["reward"], dtype=np.float32))
        tensors["done"]   = torch.from_numpy(np.asarray(rows["done"],   dtype=np.int64))

        if self._max_action_continuous_dim > 0:
            buf = self._vector_buffer(
                self._modality_column(rows["action"], ACTION_KEY_CONTINUOUS),
                self._max_action_continuous_dim, np.float32,
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

        # q_star — only if data has it and at least the first entry is not None
        if "q_star" in cols:
            q_list = rows["q_star"]
            if q_list[0] is not None:
                q_dim = self._max_action_dim
                q_buf = np.full((n, q_dim), -np.inf, dtype=np.float32)
                if all(v is not None for v in q_list):
                    q_arr = np.asarray(q_list, dtype=np.float32)
                    qdim = min(q_arr.shape[-1], q_dim)
                    q_buf[:, :qdim] = q_arr[:, :qdim]
                else:
                    for i, v in enumerate(q_list):
                        if v is not None:
                            qa = np.asarray(v, dtype=np.float32).ravel()
                            qdim = min(qa.size, q_dim)
                            q_buf[i, :qdim] = qa[:qdim]
                tensors["q_star"] = torch.from_numpy(q_buf)

        # Observation modalities live under the ``observation`` struct column.
        if "observation" in cols:
            obs_rows = rows["observation"]

            if self._max_obs_continuous_dim > 0:
                buf = self._vector_buffer(
                    self._modality_column(obs_rows, OBS_KEY_CONTINUOUS),
                    self._max_obs_continuous_dim, np.float64,
                )
                if buf is not None:
                    tensors["obs_continuous"] = torch.from_numpy(buf)

            if self._max_obs_discrete_dim > 0:
                disc = self._modality_column(obs_rows, OBS_KEY_DISCRETE)
                if any(d is not None for d in disc):
                    # 0 placeholder for rows whose observation has no discrete modality.
                    tensors["obs_discrete"] = torch.from_numpy(
                        np.array([0 if d is None else d for d in disc], dtype=np.int64)
                    )

            if self._max_obs_image_pixels > 0:
                buf = self._vector_buffer(
                    self._modality_column(obs_rows, OBS_KEY_IMAGE),
                    self._max_obs_image_pixels, np.int64,
                )
                if buf is not None:
                    tensors["obs_image"] = torch.from_numpy(buf)

        return TensorDict(tensors, batch_size=[n])

    # ------------------------------------------------------------------
    # HuggingFace Dataset I/O
    # ------------------------------------------------------------------

    def from_dataset(
        self,
        ds: "Dataset | datasets.DatasetDict",
        splits: "list[str] | None" = None,
        split_pattern: "str | list[str] | None" = None,
    ) -> None:
        """Load data from a HuggingFace ``Dataset`` or ``DatasetDict``.

        When *ds* is a ``Dataset`` the rows are appended to the source segment
        (zero-copy reference).  When *ds* is a ``DatasetDict``, the splits to
        load are selected by *splits* (exact names), *split_pattern* (one or
        more glob patterns), or — if neither is provided — all splits.  Calling
        ``from_dataset`` more than once concatenates onto what is already loaded.

        Args:
            ds: A ``Dataset`` or ``DatasetDict`` (e.g. from ``load_dataset``).
            splits: Exact split names to include.  Raises ``KeyError`` if a
                name is not present.  Mutually exclusive with *split_pattern*.
            split_pattern: Glob pattern string or list of glob pattern strings.
                Every split whose name matches any pattern is included.
                Uses ``fnmatch`` rules — ``*`` matches anything, ``?`` matches
                one character.  E.g. ``"train_*"`` loads ``train_frozenlake``,
                ``train_lunar``, …  Raises ``KeyError`` if nothing matches.

        Examples::

            # Single split
            store.from_dataset(load_dataset("org/ds", split="train"))

            # All splits from a DatasetDict
            store.from_dataset(load_dataset("org/ds"))

            # Selected splits by exact name
            store.from_dataset(load_dataset("org/ds"), splits=["train", "test"])

            # Glob patterns — all train_ and eval_ splits
            store.from_dataset(load_dataset("org/ds"), split_pattern=["train_*", "eval_*"])

            # Single pattern
            store.from_dataset(load_dataset("org/ds"), split_pattern="test_*")
        """
        import fnmatch

        if isinstance(ds, datasets.DatasetDict):
            if splits is not None and split_pattern is not None:
                raise ValueError("Provide splits or split_pattern, not both.")

            if splits is not None:
                keys = splits
                for key in keys:
                    if key not in ds:
                        raise KeyError(
                            f"Split {key!r} not found in DatasetDict. "
                            f"Available splits: {list(ds.keys())}"
                        )
            elif split_pattern is not None:
                patterns = [split_pattern] if isinstance(split_pattern, str) else list(split_pattern)
                keys = [k for k in ds.keys() if any(fnmatch.fnmatch(str(k), p) for p in patterns)]
                if not keys:
                    raise KeyError(
                        f"No splits match pattern(s) {patterns!r}. "
                        f"Available splits: {list(ds.keys())}"
                    )
            else:
                keys = list(ds.keys())

            for key in keys:
                self.from_dataset(ds[key])
            return

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
