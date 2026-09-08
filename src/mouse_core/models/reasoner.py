"""Coconut-style continuous latent reasoning.

A :class:`LatentReasoner` generates ``num_thoughts`` latent "thought" tokens
at one sampled step per sequence during the training forward: the input
embedding of thought ``r`` is the adapter applied to the backbone's output
hidden state at the previous position. The latents are inserted immediately
before the burst step's *first* prediction token (its action prompt), so
every prediction token of that step — and every later token in the same
``(sequence, grouping)`` run — attends to the thoughts. Generation happens
on the autograd tape, so TD errors backpropagate through the latent chain
into the backbone.

Use :func:`sample_reasoning_splits` to pick one burst step per sequence and
pass the result to ``Model.forward(batch, reasoning=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from mouse_core.data.token_batch import TokenBatch


class LatentReasoner(nn.Module):
    """Adapter that maps backbone hidden states to latent thought embeddings.

    ``forward`` takes the backbone output at the previous position ``[B, D]``
    and returns the input embedding of the next latent thought ``[B, D]``
    (LayerNorm + Linear). Attach to a model with ``Model(reasoner=...)``;
    it follows the encoder/backbone compute dtype under ``Model.to``.

    Args:
        hidden_dim: Backbone hidden dimension ``D``.
        num_thoughts: Number of latent thoughts ``R`` generated per burst.
    """

    def __init__(self, *, hidden_dim: int, num_thoughts: int) -> None:
        super().__init__()
        if int(num_thoughts) < 1:
            raise ValueError(f"num_thoughts must be >= 1, got {num_thoughts}.")
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        self.hidden_dim = int(hidden_dim)
        self.num_thoughts = int(num_thoughts)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Next latent input embedding from the previous hidden state."""
        return self.proj(self.norm(h))


def sample_reasoning_splits(
    batch: TokenBatch,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick one burst step per sequence for ``Model.forward(reasoning=...)``.

    Returns ``[B]`` int64 local step indices; ``-1`` skips the burst for that
    sequence. A step is eligible when the *next* step exists and shares its
    grouping (same run), so the TD pair out of the burst step carries loss
    weight and the latents receive gradient.
    """
    rng = generator if generator is not None else np.random.default_rng()
    counts = batch.step_counts()
    first_rows = _first_prediction_rows(batch)
    step_groups = np.asarray(batch.grouping_ids, dtype=np.int64)[
        np.asarray(batch.prediction_indices, dtype=np.int64)[first_rows]
    ]
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    splits = np.full(batch.B, -1, dtype=np.int64)
    for b in range(batch.B):
        groups = step_groups[offsets[b] : offsets[b + 1]]
        eligible = np.flatnonzero(groups[:-1] == groups[1:])
        if eligible.size:
            splits[b] = int(eligible[int(rng.integers(0, eligible.size))])
    return splits


def _first_prediction_rows(batch: TokenBatch) -> np.ndarray:
    """Row index into ``prediction_indices`` of each step's first prediction token."""
    psteps = np.asarray(batch.prediction_steps, dtype=np.int64)
    if psteps.size == 0:
        return np.zeros(0, dtype=np.int64)
    first = np.ones(psteps.size, dtype=bool)
    first[1:] = psteps[1:] != psteps[:-1]
    return np.flatnonzero(first)


@dataclass
class _InsertionPlan:
    """Host-side index bookkeeping for one batch of latent bursts.

    ``token_positions[i]`` is the extended-stream position of original token
    ``i``; ``latent_positions`` is burst-major ``[nb * R]``. Latents for burst
    ``j`` sit immediately before that burst step's first prediction token,
    which (like every later token) shifts right by ``R`` per earlier insertion.
    """

    num_thoughts: int
    burst_rows: np.ndarray  # [nb] sequence indices with a burst
    prefix_starts: np.ndarray  # [nb] first token index of each burst sequence
    anchors: np.ndarray  # [nb] token index of each burst step's first prediction token
    latent_groups: np.ndarray  # [nb] grouping id assigned to the latents
    token_positions: np.ndarray  # [L] extended position of each original token
    latent_positions: np.ndarray  # [nb * R] extended positions of the latents
    ext_sequence_ids: np.ndarray  # [L_ext]
    ext_grouping_ids: np.ndarray  # [L_ext]
    ext_prediction_indices: np.ndarray  # [P]
    ext_length: int


def _validate_splits(splits: np.ndarray, counts: np.ndarray) -> None:
    if splits.shape != counts.shape:
        raise ValueError(
            f"reasoning splits must have shape [{counts.shape[0]}] (one local "
            f"step index or -1 per sequence), got {tuple(splits.shape)}."
        )
    bad = (splits < -1) | (splits >= counts)
    if bool(bad.any()):
        b = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"reasoning split {int(splits[b])} out of range for sequence {b} "
            f"with {int(counts[b])} steps (expected -1 or 0..{int(counts[b]) - 1})."
        )


def _plan_insertions(
    batch: TokenBatch,
    splits: np.ndarray,
    num_thoughts: int,
) -> _InsertionPlan | None:
    """Extended-stream layout for the given per-sequence burst steps.

    Returns ``None`` when no sequence hosts a burst (all splits ``-1``).
    """
    counts = batch.step_counts()
    splits = np.asarray(splits, dtype=np.int64).reshape(-1)
    _validate_splits(splits, counts)

    burst_rows = np.flatnonzero(splits >= 0)
    if burst_rows.size == 0:
        return None

    R = int(num_thoughts)
    L = batch.L
    pred = np.asarray(batch.prediction_indices, dtype=np.int64)
    seq = np.asarray(batch.sequence_ids, dtype=np.int64)
    group = np.asarray(batch.grouping_ids, dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])

    # Anchor = the burst step's *first* prediction token (its action prompt),
    # so every prediction token of the step attends to the latents. Latents
    # are inserted immediately before it. Anchors are strictly increasing
    # because sequences occupy contiguous, ordered token blocks.
    first_rows = _first_prediction_rows(batch)
    anchors = pred[first_rows[offsets[burst_rows] + splits[burst_rows]]]
    prefix_starts = np.searchsorted(seq, burst_rows, side="left")
    if bool(np.any(anchors == prefix_starts)):
        b = int(burst_rows[int(np.flatnonzero(anchors == prefix_starts)[0])])
        raise ValueError(
            f"burst step in sequence {b} has no tokens before its prediction "
            "token; latent generation needs at least one preceding token."
        )

    nb = int(burst_rows.size)
    ext_length = L + R * nb
    token_positions = np.arange(L, dtype=np.int64) + R * np.searchsorted(
        anchors, np.arange(L, dtype=np.int64), side="right"
    )
    latent_positions = (
        (anchors + R * np.arange(nb, dtype=np.int64))[:, None]
        + np.arange(R, dtype=np.int64)[None, :]
    ).reshape(-1)
    latent_groups = group[anchors]

    ext_sequence_ids = np.zeros(ext_length, dtype=np.int64)
    ext_sequence_ids[token_positions] = seq
    ext_sequence_ids[latent_positions] = np.repeat(burst_rows, R)
    ext_grouping_ids = np.zeros(ext_length, dtype=np.int64)
    ext_grouping_ids[token_positions] = group
    ext_grouping_ids[latent_positions] = np.repeat(latent_groups, R)
    ext_prediction_indices = pred + R * np.searchsorted(anchors, pred, side="right")

    return _InsertionPlan(
        num_thoughts=R,
        burst_rows=burst_rows,
        prefix_starts=prefix_starts,
        anchors=anchors,
        latent_groups=latent_groups,
        token_positions=token_positions,
        latent_positions=latent_positions,
        ext_sequence_ids=ext_sequence_ids,
        ext_grouping_ids=ext_grouping_ids,
        ext_prediction_indices=ext_prediction_indices,
        ext_length=ext_length,
    )
