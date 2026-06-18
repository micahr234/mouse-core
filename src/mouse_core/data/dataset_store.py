"""DatasetStore — ordered sequence of step records backed by Hugging Face Dataset.

A thin sequential container. Rows are stored in insertion order. The
recommended row layout is ``MouseEnvRecord`` (the structure produced by
mouse-env rollouts plus the paired action that produced each step). The store
itself accepts any row shape; only the encoder and downstream pipeline
conventionally interpret the mouse-env contract fields.

Two operations are fast by design:

- ``append(one_row)`` — cheap for live collection from an environment
- contiguous slices — for large training sequences of length S

Loading and selection use the real Hugging Face loader (including YAML-defined
configs/subsets and splits per the repository structure guide). You call
``load_dataset(repo, config_name, split=...)`` (or any of its rich selection
features), then pass the result to ``from_dataset``.

The store only adds the ability to mix loaded history with new appends and to
produce model-ready ``TensorDict`` batches via ``PrefetchBatchifier`` (or
``encode_hf_rows``).

All other operations (push, versioning, splitting, filtering) are ordinary
Hugging Face ``Dataset`` / ``DatasetDict`` calls.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch
from tensordict import TensorDict

import datasets
from datasets import Dataset, concatenate_datasets

datasets.disable_progress_bar()

# Modality sub-keys used by the encoder when present in struct columns.
# Any sub-key name is valid in source data; only these are interpreted.
ACTION_KEY_DISCRETE = "discrete"
ACTION_KEY_CONTINUOUS = "continuous"

OBS_KEY_DISCRETE = "discrete"
OBS_KEY_CONTINUOUS = "continuous"
OBS_KEY_IMAGE = "image"


class MouseEnvRecord(TypedDict, total=False):
    """Canonical step record produced by mouse-env (plus the paired action).

    This layout is the native / recommended format for rows stored in a
    ``DatasetStore`` when working with the mouse ecosystem.

    - ``observation`` and ``action`` are dicts containing modality sub-keys
      (``discrete``, ``continuous``, ``image`` as applicable).
    - Top-level scalars and metadata come directly from mouse-env's
      ``RolloutResult`` (``reward``, ``done``, ``time``, ``group_id``,
      ``episode_index``, ``reward_episodic``, ``q_star``, ``ns_params``).
    - The ``action`` at a row is the one that produced the observation/reward/done
      for that step.

    The store itself is schema-agnostic and will hold anything, but the
    batch encoder, collection examples, documentation, and objectives all
    assume (and document) this contract.
    """
    time: int
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    done: int
    group_id: str
    episode_index: int
    reward_episodic: float
    q_star: Any | None
    ns_params: dict[str, Any] | None


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
    """Ordered sequence of step records, backed by a Hugging Face Dataset.

    Rows are typically ``MouseEnvRecord`` instances (mouse-env contract +
    paired action). The store is intentionally schema-agnostic and will
    preserve any extra columns or structures you add.

    Optimized paths:
    - ``append`` for fast single-record collection
    - contiguous slices for fast contiguous training sequences

    Dimension parameters for encoding live in ``PrefetchBatchifier`` /
    ``encode_hf_rows`` (model/batch concerns), not the store.
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

        Pass encode kwargs (max_action_dim etc.) to control the projection
        shapes. Most training code goes through PrefetchBatchifier instead.
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

    def encode_hf_rows(
        self,
        rows: dict[str, Any],
        *,
        max_action_dim: int = 1000,
        max_action_continuous_dim: int = 0,
        max_obs_continuous_dim: int = 0,
        max_obs_discrete_dim: int = 0,
        max_obs_image_pixels: int = 0,
    ) -> TensorDict:
        """Project stored rows (MouseEnvRecord layout) into a TensorDict.

        Input rows are expected in the mouse-env rollout contract:
        nested ``observation`` and ``action`` dicts (with discrete/continuous/image
        sub-keys) plus top-level scalars (reward, done, time, reward_episodic,
        q_star, group_id, episode_index, ...). Any other columns are ignored
        for the batch but preserved in the underlying dataset.

        The dimension arguments are *target* sizes from your model, not stored
        properties of the data. Missing modalities are zero-filled to the
        requested width; q_star uses -inf padding for invalid slots.
        """
        cols = set(rows.keys())
        n = len(rows[next(iter(cols))]) if cols else 0
        tensors: dict[str, torch.Tensor] = {}

        # Always emit discrete action index (0 placeholder for continuous-only rows).
        disc_actions = self._modality_column(rows["action"], ACTION_KEY_DISCRETE)
        action_arr = np.array([0 if a is None else a for a in disc_actions], dtype=np.int64)
        tensors["action"] = torch.from_numpy(action_arr)
        tensors["reward"] = torch.from_numpy(np.asarray(rows["reward"], dtype=np.float32))
        tensors["done"]   = torch.from_numpy(np.asarray(rows["done"],   dtype=np.int64))

        if max_action_continuous_dim > 0:
            buf = self._vector_buffer(
                self._modality_column(rows["action"], ACTION_KEY_CONTINUOUS),
                max_action_continuous_dim, np.float32,
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

        # q_star — width controlled by max_action_dim
        if "q_star" in cols:
            q_list = rows["q_star"]
            if q_list[0] is not None:
                q_dim = max_action_dim
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

        # Observation modalities
        if "observation" in cols:
            obs_rows = rows["observation"]

            if max_obs_continuous_dim > 0:
                buf = self._vector_buffer(
                    self._modality_column(obs_rows, OBS_KEY_CONTINUOUS),
                    max_obs_continuous_dim, np.float64,
                )
                if buf is not None:
                    tensors["obs_continuous"] = torch.from_numpy(buf)

            if max_obs_discrete_dim > 0:
                disc = self._modality_column(obs_rows, OBS_KEY_DISCRETE)
                if any(d is not None for d in disc):
                    tensors["obs_discrete"] = torch.from_numpy(
                        np.array([0 if d is None else d for d in disc], dtype=np.int64)
                    )

            if max_obs_image_pixels > 0:
                buf = self._vector_buffer(
                    self._modality_column(obs_rows, OBS_KEY_IMAGE),
                    max_obs_image_pixels, np.int64,
                )
                if buf is not None:
                    tensors["obs_image"] = torch.from_numpy(buf)

        return TensorDict(tensors, batch_size=[n])

    # ------------------------------------------------------------------
    # HuggingFace Dataset I/O
    # ------------------------------------------------------------------

    def from_dataset(self, ds: "Dataset | datasets.DatasetDict") -> None:
        """Ingest an already-loaded Hugging Face ``Dataset`` (MouseEnvRecord rows).

        All rich selection (configs/subsets as bins, splits, globs, YAML-defined
        repository structure, streaming, etc.) is performed with the real
        ``datasets.load_dataset``. ``from_dataset`` is a thin hand-off that lets
        the selected rows participate in append mixing and batch encoding.

        Pass a ``DatasetDict`` only if you want its splits concatenated into one
        flat sequence. For separate logical bins, load each (config, split) into
        its own store.

        Calling repeatedly extends the sequential history.

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
