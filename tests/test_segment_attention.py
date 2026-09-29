from __future__ import annotations

"""Tests for sequence-isolated attention masks and RoPE position ids."""

import torch

from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.base import Model, _flat_sequence_causal_mask, _flat_sequence_position_ids
from mouse_core.models.heads import RegressionHead
from tests._token_batch_helpers import batch_to_packed, token_tokenizer


def test_flat_sequence_position_ids_reset_per_sequence() -> None:
    group_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    pos = _flat_sequence_position_ids(group_ids=group_ids)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0]]


def test_flat_sequence_position_ids_count_within_one_sequence() -> None:
    group_ids = torch.zeros(6, dtype=torch.long)
    pos = _flat_sequence_position_ids(group_ids=group_ids)
    assert pos.tolist() == [[0, 1, 2, 3, 4, 5]]


def test_flat_sequence_causal_mask_blocks_cross_sequence() -> None:
    group_ids = torch.tensor([0, 0, 1, 1])
    mask = _flat_sequence_causal_mask(
        dtype=torch.float32, group_ids=group_ids
    )
    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 0] == 0.0
    assert mask[0, 0, 1, 0] == 0.0
    assert mask[0, 0, 1, 1] == 0.0
    assert mask[0, 0, 0, 1] < 0.0
    assert mask[0, 0, 2, 0] < 0.0
    assert mask[0, 0, 2, 1] < 0.0
    assert mask[0, 0, 3, 0] < 0.0
    assert mask[0, 0, 2, 2] == 0.0
    assert mask[0, 0, 3, 2] == 0.0
    assert mask[0, 0, 3, 3] == 0.0


def test_flat_sequence_causal_mask_allows_earlier_tokens_in_the_same_sequence() -> None:
    group_ids = torch.zeros(4, dtype=torch.long)
    mask = _flat_sequence_causal_mask(
        dtype=torch.float32, group_ids=group_ids
    )
    assert mask[0, 0, 1, 0] == 0.0
    assert mask[0, 0, 2, 0] == 0.0
    assert mask[0, 0, 3, 2] == 0.0
    assert mask[0, 0, 0, 2] < 0.0


def test_model_forward_injects_group_id_and_runs_flat() -> None:
    backbone = IdentityBackbone(hidden_dim=8, vocab_size=32)
    model = Model(
        backbone=backbone,
        heads=(head := RegressionHead(
            in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True,
            propagate_gradient=1.0,
        )),
        action_source="action_value",
        reasoner=None,
    )
    batch = [[{"action": i % 4} for i in range(3)], [{"action": 1}, {"action": 2}]]
    tb, objective_data, group_id = batch_to_packed(token_tokenizer("action"), batch)
    predictions = model(tb).predictions
    assert "group_id" not in objective_data.keys()
    assert group_id.tolist() == [0, 0, 0, 1, 1]
    assert predictions["action_value"].shape == (5, 4)
    assert tb.N == 5
    assert list(tb.step_counts()) == [3, 2]
    preds2 = model(tb).predictions
    assert preds2["action_value"].shape == (5, 4)
