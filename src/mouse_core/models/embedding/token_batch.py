"""Unified TokenBatch — flat concatenated token stream with parallel id arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def step_counts_from_sequence_id(
    sequence_id: np.ndarray | None,
    B: int,
) -> np.ndarray:
    """Per-sequence step counts ``[B]`` from flat ``sequence_id`` ``[N]``.

    Missing IDs (empty decode rows) become zeros when ``minlength=B``.
    """
    if B <= 0:
        return np.zeros(0, dtype=np.int64)
    if sequence_id is None:
        return np.zeros(B, dtype=np.int64)
    sid = np.asarray(sequence_id, dtype=np.int64).reshape(-1)
    if sid.size == 0:
        return np.zeros(B, dtype=np.int64)
    return np.bincount(sid, minlength=B).astype(np.int64)[:B]


@dataclass
class TokenBatch:
    """One concatenated token stream (no padding) plus parallel id arrays.

    Length ``L`` is the total number of tokens across all sequences and steps.
    Step-level fields use ``N = len(prediction_indices)`` (ragged windows allowed).
    Per-sequence step counts are derived from ``col_values["sequence_id"]`` + ``B``
    (see :meth:`step_counts`); they are not stored separately.

    Attributes:
        token_types: ``[L]`` int64 — which embedder path / table.
        token_ids: ``[L]`` int64 — integer payload (discrete / text / image / learnable).
        scalars: ``[L]`` float32 — float payload for Fourier-typed positions.
        sequence_ids: ``[L]`` int64 — which of the ``B`` sequences each token belongs to.
        step_ids: ``[L]`` int64 — local step index within its sequence (``0 .. len_b-1``).
        prediction_indices: ``[N]`` int64 — index into ``0 .. L-1`` of each step's
            prediction token (last token of that step's span), in flat step order
            (sequence 0's steps, then sequence 1's, …).
        col_values: step-level arrays shaped ``[N]`` or ``[N, dim]`` for objectives
            (includes ``sequence_id`` ``[N]`` when ``N > 0``).
        B: Number of sequences (kept so empty rows can have zero step count).
    """

    token_types: np.ndarray
    token_ids: np.ndarray
    scalars: np.ndarray
    sequence_ids: np.ndarray
    step_ids: np.ndarray
    prediction_indices: np.ndarray
    col_values: dict[str, np.ndarray] = field(default_factory=dict)
    B: int = 0

    def __post_init__(self) -> None:
        L = int(np.asarray(self.token_types).shape[0])
        for name in (
            "token_types",
            "token_ids",
            "scalars",
            "sequence_ids",
            "step_ids",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (L,):
                raise ValueError(f"{name} must have shape [{L}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        pred = np.asarray(self.prediction_indices, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "prediction_indices", pred)
        n = int(pred.shape[0])
        counts = self.step_counts()
        if int(counts.sum()) != n:
            raise ValueError(
                f"prediction_indices length [{n}] must equal sum of step counts "
                f"from sequence_id [{int(counts.sum())}] (B={self.B})"
            )
        sid = self.col_values.get("sequence_id")
        if n > 0:
            if sid is None:
                raise ValueError("col_values must include sequence_id when N > 0")
            sid_arr = np.asarray(sid, dtype=np.int64).reshape(-1)
            if sid_arr.shape != (n,):
                raise ValueError(
                    f"sequence_id must have shape [{n}], got {sid_arr.shape}"
                )

    @property
    def L(self) -> int:
        return int(self.token_types.shape[0])

    @property
    def N(self) -> int:
        return int(self.prediction_indices.shape[0])

    @property
    def S(self) -> int:
        """Max steps in the batch (convenience; rows may be shorter)."""
        if self.B <= 0:
            return 0
        counts = self.step_counts()
        return int(counts.max()) if counts.size else 0

    def step_counts(self) -> np.ndarray:
        """Steps per sequence ``[B]``, derived from ``col_values["sequence_id"]``."""
        return step_counts_from_sequence_id(
            self.col_values.get("sequence_id"), self.B
        )

    def to_tensors(self, device: torch.device | str | None = None) -> dict[str, Any]:
        """Move arrays to torch tensors on ``device`` (CPU if None)."""
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _long(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)

        def _float(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)

        col: dict[str, torch.Tensor] = {}
        for k, v in self.col_values.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                col[k] = torch.from_numpy(arr.astype(np.float32)).to(dev)
            else:
                col[k] = torch.from_numpy(arr.astype(np.int64)).to(dev)

        return {
            "token_types": _long(self.token_types),
            "token_ids": _long(self.token_ids),
            "scalars": _float(self.scalars),
            "sequence_ids": _long(self.sequence_ids),
            "step_ids": _long(self.step_ids),
            "prediction_indices": _long(self.prediction_indices),
            "col_values": col,
            "B": self.B,
        }


def empty_token_batch(B: int = 0) -> TokenBatch:
    """Empty batch (L=0, N=0); all ``B`` sequences have zero steps."""
    return TokenBatch(
        token_types=np.zeros(0, dtype=np.int64),
        token_ids=np.zeros(0, dtype=np.int64),
        scalars=np.zeros(0, dtype=np.float32),
        sequence_ids=np.zeros(0, dtype=np.int64),
        step_ids=np.zeros(0, dtype=np.int64),
        prediction_indices=np.zeros(0, dtype=np.int64),
        col_values={},
        B=B,
    )
