"""Unified TokenBatch — flat concatenated token stream with parallel id arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class TokenBatch:
    """One concatenated token stream (no padding) plus parallel id arrays.

    Length ``L`` is the total number of tokens across all sequences and steps.
    Step-level fields use ``N = B * S`` (fixed step windows from the DataLoader).

    Attributes:
        token_types: ``[L]`` int64 — which embedder path / table.
        token_ids: ``[L]`` int64 — integer payload (discrete / text / image / learnable).
        scalars: ``[L]`` float32 — float payload for Fourier-typed positions.
        sequence_ids: ``[L]`` int64 — which of the ``B`` sequences each token belongs to.
        step_ids: ``[L]`` int64 — flat step index in ``0 .. N-1`` (``b * S + s``).
        segment_ids: ``[L]`` int64 — pack/pad seam id expanded to tokens.
        step_token_indices: ``[N]`` int64 — index into ``0 .. L-1`` of each step's
            prediction token (last token of that step's span).
        col_values: step-level arrays shaped ``[B, S]`` or ``[B, S, dim]`` for objectives.
        B: Number of sequences.
        S: Steps per sequence.
    """

    token_types: np.ndarray
    token_ids: np.ndarray
    scalars: np.ndarray
    sequence_ids: np.ndarray
    step_ids: np.ndarray
    segment_ids: np.ndarray
    step_token_indices: np.ndarray
    col_values: dict[str, np.ndarray] = field(default_factory=dict)
    B: int = 0
    S: int = 0

    def __post_init__(self) -> None:
        L = int(np.asarray(self.token_types).shape[0])
        for name in (
            "token_types",
            "token_ids",
            "scalars",
            "sequence_ids",
            "step_ids",
            "segment_ids",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (L,):
                raise ValueError(f"{name} must have shape [{L}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        n = self.B * self.S
        sti = np.asarray(self.step_token_indices)
        if sti.shape != (n,):
            raise ValueError(
                f"step_token_indices must have shape [{n}] (B*S), got {sti.shape}"
            )
        object.__setattr__(self, "step_token_indices", sti)

    @property
    def L(self) -> int:
        return int(self.token_types.shape[0])

    @property
    def N(self) -> int:
        return self.B * self.S

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
            "segment_ids": _long(self.segment_ids),
            "step_token_indices": _long(self.step_token_indices),
            "col_values": col,
            "B": self.B,
            "S": self.S,
        }


def empty_token_batch(B: int = 0, S: int = 0) -> TokenBatch:
    """Empty batch (L=0)."""
    return TokenBatch(
        token_types=np.zeros(0, dtype=np.int64),
        token_ids=np.zeros(0, dtype=np.int64),
        scalars=np.zeros(0, dtype=np.float32),
        sequence_ids=np.zeros(0, dtype=np.int64),
        step_ids=np.zeros(0, dtype=np.int64),
        segment_ids=np.zeros(0, dtype=np.int64),
        step_token_indices=np.zeros(B * S, dtype=np.int64),
        col_values={},
        B=B,
        S=S,
    )
