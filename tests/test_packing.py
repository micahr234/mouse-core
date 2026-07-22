"""Tests for ragged packing helpers."""

from __future__ import annotations

import torch

from mouse_core.models.embedding.packing import (
    counts_from_step_token_indices,
    pack_and_pad_rows,
    pack_live_step_tokens,
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


def test_pack_live_step_tokens_compacts_holes() -> None:
    # Concat-style: 3 slots/step; middle slot absent on step 0.
    B, S, Tslot, D = 1, 2, 3, 2
    buf = torch.zeros(B, S, Tslot, D)
    buf[0, 0, 0] = 1.0
    buf[0, 0, 2] = 2.0
    buf[0, 1, 0] = 3.0
    buf[0, 1, 1] = 4.0
    buf[0, 1, 2] = 5.0
    live = torch.tensor([[[True, False, True], [True, True, True]]])
    counts = torch.tensor([[2, 3]])
    embeds, indices = pack_live_step_tokens(buf, live, counts)
    assert embeds.shape == (1, 5, D)
    assert indices.tolist() == [[1, 4]]
    assert torch.allclose(embeds[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(embeds[0, 1], torch.tensor([2.0, 2.0]))
    assert torch.allclose(embeds[0, 2], torch.tensor([3.0, 3.0]))
