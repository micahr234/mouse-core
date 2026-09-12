"""Tests for decode-layout packing helpers."""
from __future__ import annotations
import torch
from mouse_core.models.embedding.packing import left_align_content

def test_left_align_content_shifts_indices() -> None:
    embeds = torch.zeros(1, 5, 2)
    embeds[0, 0] = 1.0
    embeds[0, 1] = 2.0
    embeds[0, 2] = 3.0
    head_output_indices = torch.tensor([[2]])
    aligned, aligned_indices = left_align_content(embeds, head_output_indices, [3])
    assert aligned_indices.tolist() == [[4]]
    assert torch.allclose(aligned[0, 2], torch.tensor([1.0, 1.0]))
    assert torch.allclose(aligned[0, 3], torch.tensor([2.0, 2.0]))
    assert torch.allclose(aligned[0, 4], torch.tensor([3.0, 3.0]))
    assert torch.allclose(aligned[0, :2], torch.zeros(2, 2))


def test_left_align_content_empty_row_stays_zero() -> None:
    embeds = torch.zeros(2, 4, 2)
    embeds[0, 0] = 1.0
    embeds[0, 1] = 2.0
    # Empty row: indices are the all-zero pad; length must come from token_lengths.
    head_output_indices = torch.tensor([[1, 0], [0, 0]])
    aligned, aligned_indices = left_align_content(
        embeds, head_output_indices, [2, 0]
    )
    assert aligned_indices[0].tolist() == [3, 2]
    assert aligned_indices[1].tolist() == [0, 0]
    assert torch.allclose(aligned[0, 2], torch.tensor([1.0, 1.0]))
    assert torch.allclose(aligned[0, 3], torch.tensor([2.0, 2.0]))
    assert torch.allclose(aligned[1], torch.zeros(4, 2))
