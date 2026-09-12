"""Decode-layout helpers for rectangular (cached) token batches."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def left_align_content(
    embeds: torch.Tensor,
    head_output_indices: torch.Tensor,
    token_lengths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift right-padded rows so content sits in trailing columns (FlexDecode).

    ``token_lengths[b]`` is the number of real tokens in row ``b`` (0 for an
    idle decode row). Length is never inferred from ``head_output_indices``:
    an empty row is all zeros there, which would look like length 1.

    Returns ``(aligned_embeds, head_output_indices)`` with indices adjusted for
    the aligned layout (what ``Model`` pools head outputs from).
    """
    B, L, _D = embeds.shape
    if len(token_lengths) != B:
        raise ValueError(
            f"token_lengths length ({len(token_lengths)}) must match batch {B}"
        )
    aligned = embeds.new_zeros(embeds.shape)
    aligned_indices = head_output_indices.clone()
    for b in range(B):
        rl = int(token_lengths[b])
        if rl == 0:
            continue
        if rl > L:
            raise ValueError(
                f"token_lengths[{b}]={rl} exceeds embed length {L}"
            )
        offset = L - rl
        aligned[b, offset:] = embeds[b, :rl]
        aligned_indices[b] = head_output_indices[b] + offset
    return aligned, aligned_indices
