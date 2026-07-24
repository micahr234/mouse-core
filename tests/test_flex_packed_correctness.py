"""Verify Flex packed attention / RoPE changes match prior semantics."""
from __future__ import annotations
from typing import cast
import numpy as np
import pytest
import torch
from mouse_core.models.backbone.flex_train import flex_packed_forward
from mouse_core.models.backbone.qwen3 import Qwen3Backbone
from mouse_core.models.base import Model, _flat_sequence_causal_mask, _flat_sequence_position_ids
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads.dqn import DiscreteActionValueHead

def _legacy_flat_sequence_position_ids(sequence_ids: torch.Tensor) -> torch.Tensor:
    """Pre-optimization RoPE positions (uses host .item() for n_runs)."""
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    if L == 0:
        return torch.zeros(1, 0, dtype=torch.long, device=device)
    same = sequence_ids[1:] == sequence_ids[:-1]
    run_ids = torch.zeros(L, dtype=torch.long, device=device)
    run_ids[1:] = (~same).cumsum(dim=0)
    arange = torch.arange(L, device=device)
    first = torch.ones(L, dtype=torch.bool, device=device)
    first[1:] = run_ids[1:] != run_ids[:-1]
    starts = torch.zeros(int(run_ids[-1].item()) + 1, dtype=torch.long, device=device)
    starts[run_ids[first]] = arange[first]
    return (arange - starts[run_ids]).unsqueeze(0)

@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_flat_sequence_position_ids_match_legacy(device: str) -> None:
    """New cummax RoPE resets must match the old scatter algorithm exactly."""
    rng = np.random.default_rng(0)
    for L in (0, 1, 7, 128, 257):
        if L == 0:
            seq = torch.zeros(0, dtype=torch.long, device=device)
        else:
            seq = torch.zeros(L, dtype=torch.long, device=device)
            s_id = 0
            for i in range(L):
                if i > 0 and rng.random() < 0.15:
                    s_id += 1
                seq[i] = s_id
        got = _flat_sequence_position_ids(sequence_ids=seq)
        ref = _legacy_flat_sequence_position_ids(seq)
        assert torch.equal(got, ref), f'mismatch at L={L} device={device}'

def test_flex_train_position_ids_match_legacy_on_device() -> None:
    """Reproduce flex_train's inlined cummax formula vs legacy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L = 200
    rng = np.random.default_rng(1)
    seq = torch.zeros(L, dtype=torch.long, device=device)
    s_id = 0
    for i in range(L):
        if i > 0 and rng.random() < 0.1:
            s_id += 1
        seq[i] = s_id
    arange = torch.arange(L, device=device)
    new_run = torch.ones(L, dtype=torch.bool, device=device)
    new_run[1:] = seq[1:] != seq[:-1]
    markers = torch.where(new_run, arange, torch.full_like(arange, -1))
    got = arange - torch.cummax(markers, dim=0).values
    ref = _legacy_flat_sequence_position_ids(seq).squeeze(0)
    assert torch.equal(got, ref)

def test_prepare_sequence_id_col_matches_step_counts() -> None:
    encoder = NumericEmbedder(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'field': 'reward', 'type': 'rff'}, {'type': 'learnable', 'tokens': 1}])
    batch = [[{'action': s % 4, 'reward': float(s)} for s in range(5)], [{'action': 1, 'reward': 0.0}, {'action': 2, 'reward': 1.0}, {'action': 3, 'reward': 2.0}]]
    tb = encoder.prepare(batch)
    assert list(tb.step_counts()) == [5, 3]
    assert tb.col_values['sequence_id'].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert tb.prediction_indices.shape == (8,)
    assert set((int(x) for x in tb.step_ids[tb.sequence_ids == 0])) == {0, 1, 2, 3, 4}
    assert set((int(x) for x in tb.step_ids[tb.sequence_ids == 1])) == {0, 1, 2}

@pytest.mark.skipif(not torch.cuda.is_available(), reason='Flex packed path is CUDA-only')
def test_flex_packed_matches_sdpa_document_mask() -> None:
    """Flex packed forward ≈ SDPA with the same dense sequence mask."""
    torch.manual_seed(0)
    device = torch.device('cuda')
    backbone = Qwen3Backbone(hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4).to(device=device, dtype=torch.float32)
    transformer = backbone.model
    assert transformer is not None
    L = 12
    embeds = torch.randn(L, backbone.hidden_dim, device=device, dtype=torch.float32)
    sequence_ids = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long)
    q = torch.arange(L, device=device)
    kv = torch.arange(L, device=device)
    allow = (kv.unsqueeze(0) <= q.unsqueeze(1)) & (sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(0))
    assert not bool(allow[9, 0].item())
    assert bool(allow[6, 0].item())
    with torch.no_grad():
        h_flex = cast(torch.Tensor, flex_packed_forward(output_hidden_states=False, model=transformer, embeds=embeds, sequence_ids=sequence_ids))
        attention_mask = _flat_sequence_causal_mask(dtype=embeds.dtype, sequence_ids=sequence_ids)
        position_ids = _flat_sequence_position_ids(sequence_ids=sequence_ids)
        h_sdpa = cast(torch.Tensor, backbone(embeds.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)).squeeze(0)
    assert h_flex.shape == h_sdpa.shape == (L, backbone.hidden_dim)
    max_abs = (h_flex - h_sdpa).abs().max().item()
    assert max_abs < 0.002, f'flex vs sdpa max abs diff {max_abs}'

@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA Flex path')
def test_model_flex_forward_stable_under_sequence_isolation() -> None:
    """End-to-end: packed Model on CUDA isolates attention across sequences."""
    torch.manual_seed(2)
    device = torch.device('cuda')
    backbone = Qwen3Backbone(hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4)
    encoder = NumericEmbedder(hidden_dim=backbone.hidden_dim, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 4}, {'type': 'learnable', 'tokens': 1}])
    head = DiscreteActionValueHead(in_features=backbone.hidden_dim, out_features=4, hidden_dim=backbone.hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head).to(device=device, dtype=torch.float32).eval()
    batch = [[{'action': i % 4} for i in range(3)], [{'action': i % 4} for i in range(3)]]
    tb = encoder.prepare(batch)
    with torch.no_grad():
        preds0, od0, _ = model(tb)
        batch_corrupt = [[{'action': 3} for _ in range(3)], [{'action': i % 4} for i in range(3)]]
        tb_c = encoder.prepare(batch_corrupt)
        preds1, _, _ = model(tb_c)
    q0 = preds0['action_value']
    q1 = preds1['action_value']
    assert torch.allclose(q0[3:], q1[3:], atol=1e-05, rtol=1e-05)
    assert not torch.allclose(q0[:3], q1[:3], atol=1e-05, rtol=1e-05)
    assert od0['sequence_id'].tolist() == [0, 0, 0, 1, 1, 1]
