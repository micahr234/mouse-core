"""Coconut-style continuous latent reasoning.

A :class:`LatentReasoner` generates ``num_thoughts`` latent "thought" tokens
at one sampled step per sequence during the training forward: the input
embedding of thought ``r`` is the adapter applied to the backbone's output
hidden state at the previous position. The latents are inserted immediately
before the burst step's *first* head-output token (its action prompt), so
every head-output token of that step — and every later token in the same
``(sequence, grouping)`` run — attends to the thoughts. Generation happens
on the autograd tape, so TD errors backpropagate through the latent chain
into the backbone.

Use :func:`sample_reasoning_splits` to pick one burst step per sequence and
pass the result to ``Model.forward(batch, reasoning=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from mouse_core.data.token_batch import TokenBatch


class LatentReasoner(nn.Module):
    """Adapter that maps backbone hidden states to latent thought embeddings.

    ``forward`` takes the backbone output at the previous position ``[B, D]``
    and returns the input embedding of the next latent thought ``[B, D]``
    (LayerNorm + Linear). Register it as a module on your model;
    it stays float32 under ``Model.to`` and casts the backbone output to its
    own dtype.

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
        h = h.to(dtype=self.proj.weight.dtype)
        h = self.norm(h)
        embeddings = self.proj(h)
        return embeddings

    def insert(
        self, *, embeds: torch.Tensor, tokens: TokenBatch,
        sequence_ids: torch.Tensor, head_token_ids: torch.Tensor,
        splits: Sequence[int] | np.ndarray,
        backbone: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert latent embeddings using the caller's explicit backbone function.

        Returns embeddings, sequence IDs, and shifted head-token IDs. The model
        still runs the final backbone, projection, and prediction calls.
        """
        if embeds.shape[-1] != self.hidden_dim:
            raise ValueError("hidden_dim mismatch between reasoner and embeddings.")
        plan = _plan_insertions(tokens, np.asarray(splits, dtype=np.int64).reshape(-1), self.num_thoughts)
        if plan is None:
            return embeds, sequence_ids, head_token_ids
        embeds, sequence_ids, head_token_ids = self._generate_latents(
            embeds=embeds, sequence_ids=sequence_ids, plan=plan, backbone=backbone,
        )
        return embeds, sequence_ids, head_token_ids

    def _generate_latents(
        self,
        *,
        embeds: torch.Tensor,
        sequence_ids: torch.Tensor,
        plan: _InsertionPlan,
        backbone: Callable[..., torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate latent thoughts on the autograd tape and assemble the extended stream.

        Runs ``R`` extra backbone passes over the growing per-burst prefixes
        (all bursts in lockstep; attention isolation makes dropping the other
        sequences exact). Thought ``r``'s input embedding is the reasoner
        adapter applied to the backbone output at the previous position, so
        gradients flow through the whole latent chain. Returns
        the extended embeddings, sequence IDs, and head-token IDs.
        """
        R = plan.num_thoughts
        device = embeds.device
        nb = int(plan.burst_rows.size)

        prefix_bounds = list(
            zip(plan.prefix_starts.tolist(), plan.anchors.tolist())
        )
        thoughts: list[torch.Tensor] = []  # thought r → [nb, D]
        for r in range(R):
            parts: list[torch.Tensor] = []
            seq_parts: list[torch.Tensor] = []
            last_positions: list[int] = []
            offset = 0
            for j, (start, anchor) in enumerate(prefix_bounds):
                parts.append(embeds[start:anchor])
                seq_parts.append(sequence_ids[start:anchor])
                if r > 0:
                    parts.append(torch.stack([thoughts[q][j] for q in range(r)]))
                    seq_parts.append(
                        sequence_ids.new_full((r,), int(plan.burst_rows[j]))
                    )
                block = (anchor - start) + r
                last_positions.append(offset + block - 1)
                offset += block
            gen_out = backbone(
                torch.cat(parts),
                sequence_ids=torch.cat(seq_parts),
            )
            h_last = gen_out[
                torch.as_tensor(last_positions, device=device)
            ]
            thoughts.append(self(h_last).to(dtype=embeds.dtype))

        latent_embeds = torch.stack(thoughts, dim=1).reshape(nb * R, embeds.shape[-1])
        token_indices = torch.as_tensor(plan.token_positions, device=device)
        latent_indices = torch.as_tensor(plan.latent_positions, device=device)
        ext_embeds = (
            embeds.new_zeros(plan.ext_length, embeds.shape[-1])
            .index_copy(0, token_indices, embeds)
            .index_copy(0, latent_indices, latent_embeds)
        )
        ext_sequence_ids = torch.as_tensor(plan.ext_sequence_ids, device=device)
        ext_head_token_ids = torch.as_tensor(
            plan.ext_head_token_ids, device=device
        )
        return (
            ext_embeds,
            ext_sequence_ids,
            ext_head_token_ids,
        )


def sample_reasoning_splits(
    *,
    batch: TokenBatch,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Pick one burst step per sequence for ``Model.forward(reasoning=...)``.

    Returns ``[B]`` int64 local step indices; ``-1`` skips the burst for that
    sequence. A step is eligible when the next step exists in the same sequence, so
    the TD pair out of the burst step carries loss weight and the latents
    receive gradient.
    """
    rng = generator if generator is not None else np.random.default_rng()
    counts = batch.step_counts()
    splits = np.full(batch.B, -1, dtype=np.int64)
    offset = 0
    for b in range(batch.B):
        n = int(counts[b])
        eligible = np.unique(batch.head_step_ids[(batch.head_step_ids >= offset) &
                                                 (batch.head_step_ids < offset + n - 1)]) - offset
        if eligible.size:
            splits[b] = int(rng.choice(eligible))
        offset += n
    return splits


def _first_head_output_rows(batch: TokenBatch) -> np.ndarray:
    """Row index into ``head_token_ids`` of each step's first head-output token."""
    head_step_ids = np.asarray(batch.head_step_ids, dtype=np.int64)
    if head_step_ids.size == 0:
        return np.full(batch.N, -1, dtype=np.int64)
    first = np.ones(head_step_ids.size, dtype=bool)
    first[1:] = head_step_ids[1:] != head_step_ids[:-1]
    rows = np.full(batch.N, -1, dtype=np.int64)
    rows[head_step_ids[first]] = np.flatnonzero(first)
    return rows


@dataclass
class _InsertionPlan:
    """Host-side index bookkeeping for one batch of latent bursts.

    ``token_positions[i]`` is the extended-stream position of original token
    ``i``; ``latent_positions`` is burst-major ``[nb * R]``. Latents for burst
    ``j`` sit immediately before that burst step's first head-output token,
    which (like every later token) shifts right by ``R`` per earlier insertion.
    """

    num_thoughts: int
    burst_rows: np.ndarray  # [nb] sequence indices with a burst
    prefix_starts: np.ndarray  # [nb] first token index of each burst sequence
    anchors: np.ndarray  # [nb] token index of each burst step's first head-output token
    token_positions: np.ndarray  # [L] extended position of each original token
    latent_positions: np.ndarray  # [nb * R] extended positions of the latents
    ext_sequence_ids: np.ndarray  # [L_ext]
    ext_head_token_ids: np.ndarray  # [P]
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
    pred = np.asarray(batch.head_token_ids, dtype=np.int64)
    seq = np.asarray(batch.sequence_ids, dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])

    # Anchor = the burst step's *first* head-output token (its action prompt),
    # so every head-output token of the step attends to the latents. Latents
    # are inserted immediately before it. Anchors are strictly increasing
    # because sequences occupy contiguous, ordered token blocks.
    first_rows = _first_head_output_rows(batch)
    selected_rows = first_rows[offsets[burst_rows] + splits[burst_rows]]
    if np.any(selected_rows < 0):
        raise ValueError("A reasoning split requires an action-request readout")
    anchors = pred[selected_rows]
    prefix_starts = np.searchsorted(seq, burst_rows, side="left")
    if bool(np.any(anchors == prefix_starts)):
        b = int(burst_rows[int(np.flatnonzero(anchors == prefix_starts)[0])])
        raise ValueError(
            f"burst step in sequence {b} has no tokens before its head-output "
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

    ext_sequence_ids = np.zeros(ext_length, dtype=np.int64)
    ext_sequence_ids[token_positions] = seq
    ext_sequence_ids[latent_positions] = np.repeat(burst_rows, R)
    ext_head_token_ids = pred + R * np.searchsorted(anchors, pred, side="right")

    return _InsertionPlan(
        num_thoughts=R,
        burst_rows=burst_rows,
        prefix_starts=prefix_starts,
        anchors=anchors,
        token_positions=token_positions,
        latent_positions=latent_positions,
        ext_sequence_ids=ext_sequence_ids,
        ext_head_token_ids=ext_head_token_ids,
        ext_length=ext_length,
    )
