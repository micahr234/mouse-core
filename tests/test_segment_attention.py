from __future__ import annotations

"""Tests for sequence/grouping-id-isolated attention masks and RoPE position ids."""

import torch

from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.base import Model, _flat_sequence_causal_mask, _flat_sequence_position_ids
from mouse_core.models.heads import RegressionHead
from tests._token_batch_helpers import batch_to_packed, token_tokenizer


def test_flat_sequence_position_ids_reset_per_sequence() -> None:
    sequence_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    grouping_ids = torch.zeros_like(sequence_ids)
    pos = _flat_sequence_position_ids(sequence_ids=sequence_ids, grouping_ids=grouping_ids)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0]]


def test_flat_sequence_position_ids_reset_per_grouping_id() -> None:
    sequence_ids = torch.zeros(6, dtype=torch.long)
    grouping_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    pos = _flat_sequence_position_ids(sequence_ids=sequence_ids, grouping_ids=grouping_ids)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0]]


def test_flat_sequence_causal_mask_blocks_cross_sequence() -> None:
    sequence_ids = torch.tensor([0, 0, 1, 1])
    grouping_ids = torch.zeros_like(sequence_ids)
    mask = _flat_sequence_causal_mask(
        dtype=torch.float32, sequence_ids=sequence_ids, grouping_ids=grouping_ids
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


def test_flat_sequence_causal_mask_blocks_cross_grouping_id() -> None:
    sequence_ids = torch.zeros(4, dtype=torch.long)
    grouping_ids = torch.tensor([0, 0, 1, 1])
    mask = _flat_sequence_causal_mask(
        dtype=torch.float32, sequence_ids=sequence_ids, grouping_ids=grouping_ids
    )
    assert mask[0, 0, 1, 0] == 0.0
    assert mask[0, 0, 2, 0] < 0.0
    assert mask[0, 0, 2, 1] < 0.0
    assert mask[0, 0, 3, 2] == 0.0


def test_model_forward_injects_sequence_id_and_runs_flat() -> None:
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
    tb, objective_data = batch_to_packed(token_tokenizer("action"), batch)
    predictions = model(tb).predictions
    assert "sequence_id" in objective_data.keys()
    assert objective_data["sequence_id"].tolist() == [0, 0, 0, 1, 1]
    assert objective_data["grouping_id"].tolist() == [0, 0, 0, 0, 0]
    assert predictions["action_value"].shape == (5, 4)
    assert tb.N == 5
    assert list(tb.step_counts()) == [3, 2]
    assert list(tb.grouping_ids) == [0] * tb.L
    preds2 = model(tb).predictions
    assert preds2["action_value"].shape == (5, 4)


def test_prepare_derives_grouping_ids_from_field() -> None:
    batch = [
        [
            {"action": 0, "episode_done": 0, "task_done": 0, "task_index": 0},
            {"action": 1, "episode_done": 1, "task_done": 2, "task_index": 0},
            {"action": 2, "episode_done": 0, "task_done": 0, "task_index": 1},
            {"action": 3, "episode_done": 1, "task_done": 0, "task_index": 1},
            {"action": 0, "episode_done": 2, "task_done": 2, "task_index": 1},
            {"action": 1, "episode_done": 0, "task_done": 0, "task_index": 2},
        ]
    ]
    tb, objective_data = batch_to_packed(
        token_tokenizer("action", "episode_done", grouping_field="task_index"),
        batch,
        grouping_field="task_index",
    )
    assert objective_data["task_index"].tolist() == [0, 0, 1, 1, 1, 2]
    assert list(tb.grouping_ids) == [0] * 4 + [1] * 6 + [2] * 2


def test_missing_grouping_field_stamps_zero() -> None:
    batch = [[{"action": 0, "episode_done": 1, "task_done": 2}, {"action": 1, "episode_done": 0, "task_done": 0}]]
    tb, objective_data = batch_to_packed(token_tokenizer("action", "episode_done"), batch)
    assert objective_data["grouping_id"].tolist() == [0, 0]
    assert list(tb.grouping_ids) == [0] * tb.L
