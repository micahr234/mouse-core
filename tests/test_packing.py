"""Tests for ragged packing helpers."""

from __future__ import annotations

import torch

from mouse_core.models.embedding.packing import (
    counts_from_step_token_indices,
    pack_and_pad_rows,
    real_token_lengths_from_indices,
    token_pad_mask,
)


def test_counts_from_indices() -> None:
    indices = torch.tensor([[1, 4, 6]])  # counts 2, 3, 2
    counts = counts_from_step_token_indices(indices)
    assert counts.tolist() == [[2, 3, 2]]


def test_token_pad_mask() -> None:
    indices = torch.tensor([[2, 4], [1, 2]])
    mask = token_pad_mask(indices, length=5)
    assert mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]


def test_real_token_lengths_trailing_steps() -> None:
    # S=3 steps, counts 2,2,2 → indices 1,3,5
    indices = torch.tensor([[1, 3, 5]])
    assert real_token_lengths_from_indices(indices, num_real_steps=[3]) == [6]
    assert real_token_lengths_from_indices(indices, num_real_steps=[1]) == [2]
    assert real_token_lengths_from_indices(indices, num_real_steps=[2]) == [4]


def test_pack_and_pad_rows() -> None:
    D = 3
    row0 = [torch.ones(1, D), torch.ones(2, D) * 2]
    row1 = [torch.ones(2, D) * 3]
    # different S not allowed — pad to same S in callers
    row1 = [torch.ones(2, D) * 3, torch.ones(1, D) * 4]
    embeds, indices = pack_and_pad_rows(
        [row0, row1],
        hidden_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert embeds.shape == (2, 3, D)
    assert indices.tolist() == [[0, 2], [1, 2]]
