"""Verify Flex packed attention / RoPE changes match prior semantics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.models.backbone.flex_train import flex_packed_forward
from mouse_core.models.backbone.qwen3 import Qwen3Backbone
from mouse_core.models.base import (
    Model,
    _flat_segment_causal_mask,
    _flat_segment_position_ids,
)
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads.dqn import DiscreteActionValueHead


def _legacy_flat_segment_position_ids(
    sequence_ids: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Pre-optimization RoPE positions (uses host .item() for n_runs)."""
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    if L == 0:
        return torch.zeros(1, 0, dtype=torch.long, device=device)
    same = (sequence_ids[1:] == sequence_ids[:-1]) & (segment_ids[1:] == segment_ids[:-1])
    run_ids = torch.zeros(L, dtype=torch.long, device=device)
    run_ids[1:] = (~same).cumsum(dim=0)
    arange = torch.arange(L, device=device)
    first = torch.ones(L, dtype=torch.bool, device=device)
    first[1:] = run_ids[1:] != run_ids[:-1]
    starts = torch.zeros(int(run_ids[-1].item()) + 1, dtype=torch.long, device=device)
    starts[run_ids[first]] = arange[first]
    return (arange - starts[run_ids]).unsqueeze(0)


def _legacy_segment_id_from_step_loop(
    seg_tok: torch.Tensor,
    step_ids: torch.Tensor,
    B: int,
    S: int,
) -> torch.Tensor:
    """Pre-optimization B*S boolean-mask rebuild of step-level segment_id."""
    out = torch.zeros(B, S, device=seg_tok.device, dtype=torch.long)
    L = seg_tok.shape[0]
    if L == 0 or S == 0:
        return out
    for flat_step in range(B * S):
        mask = step_ids == flat_step
        if mask.any():
            out[flat_step // S, flat_step % S] = seg_tok[mask][0]
    return out


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_flat_segment_position_ids_match_legacy(device: str) -> None:
    """New cummax RoPE resets must match the old scatter algorithm exactly."""
    rng = np.random.default_rng(0)
    for L in (0, 1, 7, 128, 257):
        if L == 0:
            seq = torch.zeros(0, dtype=torch.long, device=device)
            seg = torch.zeros(0, dtype=torch.long, device=device)
        else:
            # Random pack seams: new sequence and/or new segment runs.
            seq = torch.zeros(L, dtype=torch.long, device=device)
            seg = torch.zeros(L, dtype=torch.long, device=device)
            s_id, g_id = 0, 0
            for i in range(L):
                if i > 0 and rng.random() < 0.15:
                    s_id += 1
                    g_id = 0
                elif i > 0 and rng.random() < 0.25:
                    g_id += 1
                seq[i] = s_id
                seg[i] = g_id
        got = _flat_segment_position_ids(seq, seg)
        ref = _legacy_flat_segment_position_ids(seq, seg)
        assert torch.equal(got, ref), f"mismatch at L={L} device={device}"


def test_flex_train_position_ids_match_legacy_on_device() -> None:
    """Reproduce flex_train's inlined cummax formula vs legacy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L = 200
    rng = np.random.default_rng(1)
    seq = torch.zeros(L, dtype=torch.long, device=device)
    seg = torch.zeros(L, dtype=torch.long, device=device)
    s_id = g_id = 0
    for i in range(L):
        if i > 0 and rng.random() < 0.1:
            s_id += 1
            g_id = 0
        elif i > 0 and rng.random() < 0.2:
            g_id += 1
        seq[i] = s_id
        seg[i] = g_id

    arange = torch.arange(L, device=device)
    new_run = torch.ones(L, dtype=torch.bool, device=device)
    new_run[1:] = (seq[1:] != seq[:-1]) | (seg[1:] != seg[:-1])
    markers = torch.where(new_run, arange, torch.full_like(arange, -1))
    got = arange - torch.cummax(markers, dim=0).values
    ref = _legacy_flat_segment_position_ids(seq, seg).squeeze(0)
    assert torch.equal(got, ref)


def test_segment_id_gather_matches_legacy_loop() -> None:
    """``seg_tok[step_token_indices]`` must match the old per-step mask loop."""
    encoder = NumericEmbedder(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
            {"type": "learnable", "tokens": 1},
        ],
    )
    B, S = 3, 5
    batch = [
        [{"action": (b + s) % 4, "reward": float(b - s)} for s in range(S)]
        for b in range(B)
    ]
    segment_ids = [[0, 0, 1, 1, 2] for _ in range(B)]
    segment_ids[1] = [0, 1, 1, 1, 1]
    tb = encoder.prepare(batch, segment_ids)
    t = tb.to_tensors("cpu")
    sti = t["step_token_indices"].view(B, S)
    got = t["segment_ids"][sti]
    ref = _legacy_segment_id_from_step_loop(t["segment_ids"], t["step_ids"], B, S)
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flex packed path is CUDA-only")
def test_flex_packed_matches_sdpa_document_mask() -> None:
    """Flex packed forward ≈ SDPA with the same dense document/segment mask.

    Uses float32 so Flex runs the unfused path (closer to SDPA math) and we can
    demand a tight absolute tolerance.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    backbone = Qwen3Backbone(
        hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4
    ).to(device=device, dtype=torch.float32)
    transformer = backbone.model
    assert transformer is not None

    # Two sequences packed with an interior segment seam.
    # seq0: tokens 0..4 (seg 0), 5..7 (seg 1); seq1: tokens 8..11 (seg 0)
    L = 12
    embeds = torch.randn(L, backbone.hidden_dim, device=device, dtype=torch.float32)
    sequence_ids = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long
    )
    segment_ids = torch.tensor(
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], device=device, dtype=torch.long
    )

    # Sanity: Flex mask_mod allow-set equals dense allow mask.
    q = torch.arange(L, device=device)
    kv = torch.arange(L, device=device)
    allow = (
        (kv.unsqueeze(0) <= q.unsqueeze(1))
        & (sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(0))
        & (segment_ids.unsqueeze(1) == segment_ids.unsqueeze(0))
    )
    # Cross-segment blocked within seq0.
    assert not bool(allow[6, 0].item())
    assert bool(allow[6, 5].item())
    # Cross-sequence blocked.
    assert not bool(allow[9, 0].item())

    with torch.no_grad():
        h_flex = flex_packed_forward(
            transformer, embeds, sequence_ids, segment_ids, output_hidden_states=False
        )
        attention_mask = _flat_segment_causal_mask(
            sequence_ids, segment_ids, dtype=embeds.dtype
        )
        position_ids = _flat_segment_position_ids(sequence_ids, segment_ids)
        h_sdpa = backbone(
            embeds.unsqueeze(0),
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).squeeze(0)

    assert h_flex.shape == h_sdpa.shape == (L, backbone.hidden_dim)
    # Unfused Flex vs SDPA should agree closely in fp32.
    max_abs = (h_flex - h_sdpa).abs().max().item()
    assert max_abs < 2e-3, f"flex vs sdpa max abs diff {max_abs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA Flex path")
def test_model_flex_forward_stable_under_segment_seams() -> None:
    """End-to-end: packed Model on CUDA runs and respects segment isolation.

    Compares Q-values at a seam: corrupting tokens from a *previous* segment
    must not change predictions for the current segment (Flex + RoPE isolation).
    """
    torch.manual_seed(2)
    device = torch.device("cuda")
    backbone = Qwen3Backbone(
        hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4
    )
    encoder = NumericEmbedder(
        hidden_dim=backbone.hidden_dim,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"type": "learnable", "tokens": 1},
        ],
    )
    head = DiscreteActionValueHead(
        in_features=backbone.hidden_dim,
        out_features=4,
        hidden_dim=backbone.hidden_dim,
        num_layers=1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=head).to(
        device=device, dtype=torch.float32
    ).eval()

    # One sequence, two segments of 3 steps each.
    batch = [[{"action": i % 4} for i in range(6)]]
    segment_ids = [[0, 0, 0, 1, 1, 1]]
    tb = encoder.prepare(batch, segment_ids)

    with torch.no_grad():
        preds0, od0, _ = model(tb)
        # Corrupt early-segment raw actions; rebuild TokenBatch.
        batch_corrupt = [[{"action": 3 if i < 3 else i % 4} for i in range(6)]]
        tb_c = encoder.prepare(batch_corrupt, segment_ids)
        preds1, _, _ = model(tb_c)

    q0 = preds0["action_value"][0]
    q1 = preds1["action_value"][0]
    # Steps in segment 1 (indices 3..5) must be unchanged.
    assert torch.allclose(q0[3:], q1[3:], atol=1e-5, rtol=1e-5)
    # Steps in segment 0 should change (corruption landed there).
    assert not torch.allclose(q0[:3], q1[:3], atol=1e-5, rtol=1e-5)
    assert od0["segment_id"].tolist() == [[0, 0, 0, 1, 1, 1]]
