"""Cached decoding preserves ragged rows and tokenizer-selected head positions."""
from __future__ import annotations
from _token_batch_fixture import token_batch_fixture

import numpy as np
import pytest
import torch

from mouse_core.data.token_batch import empty_token_batch
from mouse_core.models import prepare_decode


def test_prepare_decode_pads_ragged_rows_and_keeps_every_marked_head() -> None:
    tokens = token_batch_fixture(
        token_ids=np.arange(6), numeric_values=np.zeros(6),
        numeric_mask=np.zeros(6, dtype=bool),
        sequence_ids=np.array([0, 0, 0, 0, 2, 2]),
        head_positions=np.array([0, 1, 3, 4, 5]),
         num_sequences=3,
    )
    tensors = tokens.to("cpu")
    embeds = torch.arange(1., 7.).unsqueeze(-1).requires_grad_()
    padded, lengths, head_ids, valid = prepare_decode(
        embeds=embeds, tokens=tokens, sequence_ids=tensors["sequence_ids"],
        head_token_mask=tensors["head_token_mask"],
    )
    assert lengths == [4, 0, 2]
    assert padded.squeeze(-1).tolist() == [[1., 2., 3., 4.], [0., 0., 0., 0.], [0., 0., 5., 6.]]
    assert head_ids.tolist() == [[0, 1, 3], [0, 0, 0], [2, 2, 3]]
    assert valid.tolist() == [[True, True, True], [False, False, False], [False, True, True]]
    selected = padded.gather(1, head_ids.unsqueeze(-1)).squeeze(-1)
    assert selected[valid].tolist() == [1., 2., 4., 5., 6.]
    padded.sum().backward()
    torch.testing.assert_close(embeds.grad, torch.ones_like(embeds))


def test_prepare_decode_rejects_an_entirely_idle_batch() -> None:
    tokens = empty_token_batch(num_sequences=2)
    tensors = tokens.to("cpu")
    with pytest.raises(ValueError, match="at least one non-empty row"):
        prepare_decode(
            embeds=torch.empty(0, 4), tokens=tokens, sequence_ids=tensors["sequence_ids"],
            head_token_mask=tensors["head_token_mask"],
        )
