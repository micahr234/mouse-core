"""Tests for sequence-isolated attention masks and RoPE position ids."""
from __future__ import annotations
import torch
from mouse_core.models.base import Model, _flat_sequence_causal_mask, _flat_sequence_position_ids
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads.dqn import DiscreteActionValueHead

def test_flat_sequence_position_ids_reset_per_sequence() -> None:
    sequence_ids = torch.tensor([0, 0, 0, 1, 1, 2])
    pos = _flat_sequence_position_ids(sequence_ids=sequence_ids)
    assert pos.tolist() == [[0, 1, 2, 0, 1, 0]]

def test_flat_sequence_causal_mask_blocks_cross_sequence() -> None:
    sequence_ids = torch.tensor([0, 0, 1, 1])
    mask = _flat_sequence_causal_mask(dtype=torch.float32, sequence_ids=sequence_ids)
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

def test_model_forward_injects_sequence_id_and_runs_flat() -> None:
    encoder = NumericEmbedder(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}])
    backbone = IdentityBackbone(hidden_dim=8)
    model = Model(encoder=encoder, backbone=backbone, heads=DiscreteActionValueHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1))
    batch = [[{'action': i % 4} for i in range(3)], [{'action': 1}, {'action': 2}]]
    predictions, objective_data, _ = model(batch)
    assert 'sequence_id' in objective_data.keys()
    assert objective_data['sequence_id'].tolist() == [0, 0, 0, 1, 1]
    assert predictions['action_value'].shape == (5, 4)
    tb = encoder.prepare(batch)
    assert tb.N == 5
    assert list(tb.step_counts()) == [3, 2]
    preds2, _, _ = model(tb)
    assert preds2['action_value'].shape == (5, 4)
