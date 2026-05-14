"""DatasetStore — HuggingFace Dataset-backed step buffer.

Primary storage is a ``datasets.Dataset`` assigned by reference from
``from_dataset`` (O(1) — no encoding pass at load time).  A small numpy ring
buffer handles ``append`` so rollout reads stay O(1).

Sampling for training is handled by ``PrefetchBatchifier``, which reads
directly from the Dataset via ``encode_hf_rows`` — keeping only the active
prefetch queue in RAM rather than materialising the entire dataset.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import datasets
from datasets import Dataset, concatenate_datasets

datasets.disable_progress_bar()

_INITIAL_CAP = 1024


def _build_step_dtype(
    max_action_dim: int,
    max_obs_continuous_dim: int = 0,
    max_obs_discrete_dim: int = 0,
    max_obs_image_pixels: int = 0,
) -> np.dtype:
    """Build the structured numpy dtype for one environment transition.

    Fields:
        action          int64    — discrete action taken
        reward          float32  — scaled transition reward
        xformed_reward  float32  — episodically-corrected reward (DQN loss only)
        done            int64    — 0=not done, 1=terminal, 2=truncated
        q_star          float32  (max_action_dim,) — per-action Q* annotation
        time            int64    — episode_step from env; -1 if unavailable
        obs_continuous  float64  (max_obs_continuous_dim,) — continuous obs vector
        obs_discrete    int64    — discrete state index
        obs_image       int64    (max_obs_image_pixels,) — flattened pixel values 0-255
    """
    q_dim = int(max_action_dim)
    if q_dim <= 0:
        raise ValueError(f"max_action_dim must be positive, got {q_dim}")
    cont_dim = int(max_obs_continuous_dim)
    disc_dim = int(max_obs_discrete_dim)
    img_dim = int(max_obs_image_pixels)

    fields: list[tuple] = [
        ("action",         np.int64),
        ("reward",         np.float32),
        ("xformed_reward", np.float32),
        ("done",           np.int64),
        ("q_star",         np.float32, (q_dim,)),
        ("time",           np.int64),
    ]
    if cont_dim > 0:
        fields += [("obs_continuous", np.float64, (cont_dim,))]
    if disc_dim > 0:
        fields += [("obs_discrete",   np.int64)]
    if img_dim > 0:
        fields += [("obs_image",      np.int64,   (img_dim,))]
    return np.dtype(fields)


class DatasetStore:
    """HuggingFace Dataset-backed step buffer.

    Parameters
    ----------
    max_action_dim, max_obs_continuous_dim, max_obs_discrete_dim,
    max_obs_image_pixels :
        Define the ``step_dtype`` used when encoding rows.
    """

    def __init__(
        self,
        max_action_dim: int = 1000,
        max_obs_continuous_dim: int = 0,
        max_obs_discrete_dim: int = 0,
        max_obs_image_pixels: int = 0,
    ) -> None:
        self._step_dtype = _build_step_dtype(
            max_action_dim=max_action_dim,
            max_obs_continuous_dim=max_obs_continuous_dim,
            max_obs_discrete_dim=max_obs_discrete_dim,
            max_obs_image_pixels=max_obs_image_pixels,
        )

        # Source segment — HF Dataset stored by reference, never mutated.
        self._source: Dataset | None = None

        # Buf segment — growable numpy array for ``append`` (rollout path).
        self._buf: np.ndarray = np.empty(_INITIAL_CAP, dtype=self._step_dtype)
        self._buf_len: int = 0

        # Raw row dicts kept for ``to_dataset`` export (buf segment only).
        self._rows: list[dict] = []

    # ------------------------------------------------------------------
    # Core protocol
    # ------------------------------------------------------------------

    @property
    def _src_len(self) -> int:
        return len(self._source) if self._source is not None else 0

    def __len__(self) -> int:
        return self._src_len + self._buf_len

    def __repr__(self) -> str:
        return f"DatasetStore(steps={len(self)})"

    def _buf_view(self) -> np.ndarray:
        return self._buf[: self._buf_len]

    def __getitem__(self, indices: Any) -> np.ndarray:
        """Return encoded step records for the given indices."""
        src_len = self._src_len
        if src_len == 0:
            return self._buf_view()[indices].copy()
        if self._buf_len == 0:
            return self.encode_hf_rows(self._source[np.asarray(indices).tolist()])
        idx = np.asarray(indices)
        src_mask = idx < src_len
        result = np.empty(idx.shape, dtype=self._step_dtype)
        if src_mask.any():
            result[src_mask] = self.encode_hf_rows(self._source[idx[src_mask].tolist()])
        if (~src_mask).any():
            result[~src_mask] = self._buf_view()[idx[~src_mask] - src_len].copy()
        return result

    # ------------------------------------------------------------------
    # Append  (rollout / test path)
    # ------------------------------------------------------------------

    def _ensure_buf_capacity(self, n: int) -> None:
        needed = self._buf_len + n
        if needed > len(self._buf):
            new_cap = max(len(self._buf) * 2, needed)
            new_buf = np.empty(new_cap, dtype=self._step_dtype)
            new_buf[: self._buf_len] = self._buf[: self._buf_len]
            self._buf = new_buf

    def append(self, data: dict[str, Any]) -> None:
        """Append one transition (encodes eagerly into the numpy buf)."""
        if not data:
            raise ValueError("Row cannot be empty.")
        self._ensure_buf_capacity(1)
        self._buf[self._buf_len] = self.encode_hf_rows({k: [v] for k, v in data.items()})[0]
        self._buf_len += 1
        self._rows.append(dict(data))

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_uint8(value: Any) -> np.ndarray:
        """Convert an HF Dataset observation_image value to a uint8 ndarray."""
        if value is None:
            raise ValueError("observation_image is None")
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.uint8)
        try:
            import PIL.Image
        except ImportError:
            raise ImportError("PIL required to load observation_image from dataset") from None
        if hasattr(value, "size") and not isinstance(value, dict):
            return np.asarray(value, dtype=np.uint8)
        if isinstance(value, dict):
            raw = value.get("bytes") or value.get("path")
            if raw is None:
                raise ValueError("observation_image dict has no 'bytes' or 'path'")
            img = PIL.Image.open(BytesIO(raw) if isinstance(raw, bytes) else raw)
            return np.asarray(img, dtype=np.uint8)
        return np.asarray(value, dtype=np.uint8)

    def encode_hf_rows(self, rows: dict[str, Any]) -> np.ndarray:
        """Encode a HF Dataset batch (dict-of-lists) into a 1-D step array.

        Used by ``PrefetchBatchifier`` to encode batches fetched directly from
        the Arrow-backed Dataset without materialising the full dataset in RAM.

        Vectorised: each column is converted to numpy in one C-level call and
        bulk-assigned into the output buffer.  Falls back to per-row PIL decode
        only for ``observation_image``.
        """
        cols = set(rows.keys())
        n = len(rows[next(iter(cols))]) if cols else 0
        buf = np.zeros(n, dtype=self._step_dtype)

        buf["action"] = np.asarray(rows["action"], dtype=np.int64)
        reward_arr = np.asarray(rows["reward"], dtype=np.float32)
        buf["reward"] = reward_arr
        buf["xformed_reward"] = (
            np.asarray(rows["xformed_reward"], dtype=np.float32)
            if "xformed_reward" in cols
            else reward_arr
        )
        buf["done"] = np.asarray(rows["done"], dtype=np.int64)
        buf["time"] = (
            np.asarray(rows["episode_step"], dtype=np.int64)
            if "episode_step" in cols
            else np.full(n, -1, dtype=np.int64)
        )

        buf["q_star"] = -np.inf
        if "metadata_q_star" in cols:
            q_list = rows["metadata_q_star"]
            if q_list[0] is not None:
                if all(v is not None for v in q_list):
                    q_arr = np.asarray(q_list, dtype=np.float32)
                    qdim = min(q_arr.shape[-1], buf["q_star"].shape[-1])
                    buf["q_star"][:, :qdim] = q_arr[:, :qdim]
                else:
                    for i, v in enumerate(q_list):
                        if v is not None:
                            qa = np.asarray(v, dtype=np.float32).ravel()
                            qdim = buf["q_star"].shape[-1]
                            buf["q_star"][i, : min(qa.size, qdim)] = qa[:qdim]

        if "observation" in cols and "obs_continuous" in self._step_dtype.names:
            obs_list = rows["observation"]
            if obs_list[0] is not None:
                obs = np.asarray(obs_list, dtype=np.float64)
                odim = min(obs.shape[-1], buf["obs_continuous"].shape[-1])
                buf["obs_continuous"][:, :odim] = obs[:, :odim]

        if "observation_discrete" in cols and "obs_discrete" in self._step_dtype.names:
            obs_list = rows["observation_discrete"]
            if obs_list[0] is not None:
                obs = np.asarray(obs_list, dtype=np.int64)
                buf["obs_discrete"][:] = obs.ravel()[:len(buf["obs_discrete"])]

        if "observation_image" in cols and "obs_image" in self._step_dtype.names:
            img_dim = buf["obs_image"].shape[-1]
            for i, img_val in enumerate(rows["observation_image"]):
                if img_val is not None:
                    arr = self._image_to_uint8(img_val).ravel().astype(np.int64)
                    buf["obs_image"][i, : min(arr.size, img_dim)] = arr[:img_dim]

        return buf

    # ------------------------------------------------------------------
    # HuggingFace Dataset I/O
    # ------------------------------------------------------------------

    def from_dataset(self, ds: Dataset) -> None:
        """Assign a HuggingFace Dataset as the source segment — O(1).

        When called multiple times the datasets are concatenated.
        """
        if len(ds) == 0:
            return
        self._source = ds if self._source is None else concatenate_datasets([self._source, ds])

    def to_dataset(self) -> Dataset:
        """Return a HuggingFace Dataset of all steps.

        Source-only: returns the reference directly (zero-copy).
        Buf-only: builds a Dataset from the persisted raw rows.
        Both: concatenates source and buf datasets.
        """
        if self._buf_len > 0 and len(self._rows) != self._buf_len:
            raise RuntimeError(
                f"to_dataset() requires all appended rows persisted "
                f"({len(self._rows)} persisted vs {self._buf_len} in buf)."
            )
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
        ds = Dataset.from_list(rows)
        if "observation_image" not in ds.column_names:
            return ds
        from datasets.features import Image as HFImage

        image_feature = HFImage()

        def _encode_img(example: dict[str, Any]) -> dict[str, Any]:
            arr = np.asarray(example["observation_image"])
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.asarray(arr, dtype=np.uint8)
            example["observation_image"] = image_feature.encode_example(arr)
            return example

        ds = ds.map(_encode_img, desc=None)
        return ds.cast_column("observation_image", HFImage())

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset the store without freeing the underlying buffer."""
        self._source = None
        self._buf_len = 0
        self._rows.clear()
