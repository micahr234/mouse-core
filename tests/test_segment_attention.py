"""Tests for pack-segment attention masks and RoPE position ids."""

from __future__ import annotations

import torch

from mouse_core.models.base import (
    Model,
    _expand_segment_ids_to_tokens,
    _segment_causal_attention_mask,
    _segment_position_ids,
)
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads.dqn import DiscreteActionValueHead


def test_expand_segment_ids_to_tokens() -> None:
    ids = torch.tensor([[0, 0, 1]])
    counts = torch.tensor([[2, 2, 2]])
    expanded = _expand_segment_ids_to_tokens(ids, counts)
    assert expanded.tolist() == [[0, 0, 0, 0, 1, 1]]


def test_expand_segment_ids_uneven_counts() -> None:
    ids = torch.tensor([[0, 1, 1]])
    counts = torch.tensor([[1, 3, 2]])
    expanded = _expand_segment_ids_to_tokens(ids, counts)
    assert expanded.tolist() == [[0, 1, 1, 1, 1, 1]]


def test_segment_position_ids_reset_per_segment() -> None:
    # Token-level ids: segment 0 (3 tokens), segment 1 (2 tokens), segment 2 (1 token)
    segment_token_ids = torch.tensor([[0, 0, 0, 1, 1, 2]])
    pos = _segment_position_ids(segment_token_ids)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0]]


def test_segment_causal_attention_mask_blocks_cross_segment() -> None:
    segment_token_ids = torch.tensor([[0, 0, 1, 1]])
    mask = _segment_causal_attention_mask(segment_token_ids, dtype=torch.float32)
    assert mask.shape == (1, 1, 4, 4)
    # Within segment 0: causal
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 1, 0] == 0.0
    assert mask[0, 0, 1, 1] == 0.0
    assert mask[0, 0, 0, 1] < 0.0  # future
    # Cross-segment blocked even when causal would allow
    assert mask[0, 0, 2, 0] < 0.0
    assert mask[0, 0, 2, 1] < 0.0
    assert mask[0, 0, 3, 0] < 0.0
    # Within segment 1: causal
    assert mask[0, 0, 2, 2] == 0.0
    assert mask[0, 0, 3, 2] == 0.0
    assert mask[0, 0, 3, 3] == 0.0


def test_model_forward_injects_segment_ids_and_runs() -> None:
    encoder = NumericEmbedder(
        hidden_dim=8,
        modalities=[{"field": "action", "type": "discrete", "vocab_size": 4}],
    )
    backbone = IdentityBackbone(hidden_dim=8)
    model = Model(
        encoder=encoder,
        backbone=backbone,
        heads=DiscreteActionValueHead(
            in_features=8,
            out_features=4,
            hidden_dim=8,
            num_layers=1,
        ),
    )
    batch = [[{"action": i % 4} for i in range(3)]]
    segment_ids = [[0, 0, 1]]

    predictions, objective_data, _ = model(batch, segment_ids=segment_ids)

    assert "segment_id" in objective_data.keys()
    assert objective_data["segment_id"].tolist() == [[0, 0, 1]]
    assert predictions["action_value"].shape == (1, 3, 4)

    # TokenBatch path: flat embeds through identity
    tb = encoder.prepare(batch, segment_ids)
    assert tb.L == 3
    assert list(tb.segment_ids) == [0, 0, 1]
    preds2, _, _ = model(tb)
    assert preds2["action_value"].shape == (1, 3, 4)
