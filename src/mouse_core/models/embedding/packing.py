"""Decode-layout helpers for rectangular (cached) token batches."""

from __future__ import annotations

import torch


def left_align_content(
    embeds: torch.Tensor,
    head_output_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift right-padded rows so content sits in trailing columns (FlexDecode).

    Returns ``(aligned_embeds, head_output_indices)`` with indices adjusted for
    the aligned layout (what ``Model`` pools head outputs from).
    """
    B, L, _D = embeds.shape
    row_lens = head_output_indices[:, -1] + 1
    aligned = embeds.new_zeros(embeds.shape)
    aligned_indices = head_output_indices.clone()
    for b in range(B):
        rl = int(row_lens[b].item())
        if rl == 0:
            continue
        offset = L - rl
        aligned[b, offset:] = embeds[b, :rl]
        aligned_indices[b] = head_output_indices[b] + offset
    return aligned, aligned_indices
