from __future__ import annotations

"""Verify Flex packed attention / RoPE changes match prior semantics."""
from typing import cast
import numpy as np
import pytest
import torch
from mouse_core.models.backbone.flex_train import flex_packed_forward
from mouse_core.models.backbone.qwen3 import Qwen3Backbone
from mouse_core.models.base import Model, _flat_sequence_causal_mask, _flat_sequence_position_ids
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder

def _legacy_flat_sequence_position_ids(
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pre-optimization RoPE positions (uses host .item() for n_runs)."""
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    if L == 0:
        return torch.zeros(1, 0, dtype=torch.long, device=device)
    if grouping_ids is None:
        grouping_ids = torch.zeros_like(sequence_ids)
    same = (sequence_ids[1:] == sequence_ids[:-1]) & (grouping_ids[1:] == grouping_ids[:-1])
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
            task = torch.zeros(0, dtype=torch.long, device=device)
        else:
            seq = torch.zeros(L, dtype=torch.long, device=device)
            task = torch.zeros(L, dtype=torch.long, device=device)
            s_id = 0
            t_id = 0
            for i in range(L):
                if i > 0 and rng.random() < 0.15:
                    s_id += 1
                    t_id = 0
                elif i > 0 and rng.random() < 0.1:
                    t_id += 1
                seq[i] = s_id
                task[i] = t_id
        got = _flat_sequence_position_ids(sequence_ids=seq, grouping_ids=task)
        ref = _legacy_flat_sequence_position_ids(seq, task)
        assert torch.equal(got, ref), f'mismatch at L={L} device={device}'

@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_packed_rope_positions_match_brute_force_with_recurring_ids(device: str) -> None:
    """Shared train-path rule == count of earlier same-(sequence, grouping) tokens."""
    from mouse_core.models.backbone.flex_decode import packed_rope_positions

    rng = np.random.default_rng(1)
    for L in (1, 50, 200, 300):
        seq = torch.as_tensor(np.sort(rng.integers(0, 4, size=L)), device=device)
        grp = torch.as_tensor(rng.integers(0, 3, size=L), device=device)  # ids recur freely
        got = packed_rope_positions(sequence_ids=seq, grouping_ids=grp)
        same = (seq[:, None] == seq[None, :]) & (grp[:, None] == grp[None, :])
        earlier = torch.arange(L, device=device)[None, :] < torch.arange(L, device=device)[:, None]
        ref = (same & earlier).sum(-1)
        assert torch.equal(got, ref), f'mismatch at L={L} device={device}'
        assert torch.equal(_flat_sequence_position_ids(sequence_ids=seq, grouping_ids=grp).squeeze(0), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='compiled Flex path is CUDA bf16 only')
@pytest.mark.parametrize('L', [12, 300], ids=['L<block', 'L>block'])
def test_flex_packed_bf16_matches_fp32_sdpa_within_bf16_noise(L: int) -> None:
    """Compiled bf16 Flex train forward (both BLOCK_SIZE regimes, recurring ids) ≈ fp32 SDPA."""
    torch.manual_seed(0)
    device = torch.device('cuda')
    bb32 = Qwen3Backbone(hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4).to(device)
    bb16 = Qwen3Backbone(hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4).to(device)
    bb16.load_state_dict(bb32.state_dict())
    bb16.to(torch.bfloat16)
    embeds = torch.randn(L, 64, device=device)
    sequence_ids = torch.arange(L, device=device) * 3 // L
    grouping_ids = (torch.arange(L, device=device) // 7) % 2
    with torch.no_grad():
        h_flex = cast(torch.Tensor, flex_packed_forward(
            model=bb16.model, embeds=embeds.to(torch.bfloat16),
            sequence_ids=sequence_ids, grouping_ids=grouping_ids,
        )).float()
        attention_mask = _flat_sequence_causal_mask(dtype=torch.float32, sequence_ids=sequence_ids, grouping_ids=grouping_ids)
        position_ids = _flat_sequence_position_ids(sequence_ids=sequence_ids, grouping_ids=grouping_ids)
        h_ref = cast(torch.Tensor, bb32(embeds.unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)).squeeze(0)
        h_ref16 = cast(torch.Tensor, bb16(
            embeds.to(torch.bfloat16).unsqueeze(0),
            attention_mask=attention_mask.to(torch.bfloat16), position_ids=position_ids,
        )).squeeze(0).float()
    flex_err = (h_flex - h_ref).abs().max().item()
    bf16_floor = (h_ref16 - h_ref).abs().max().item()
    assert flex_err <= 2.0 * bf16_floor + 1e-3, f'flex err {flex_err} vs bf16 SDPA floor {bf16_floor}'

def test_prepare_sequence_id_col_matches_step_counts() -> None:
    encoder = NumericEmbedder(hidden_dim=8, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}, {"type": 'fourier', "field": "reward"}, {'type': 'learnable', 'tokens': 1}])
    batch = [[{'action': s % 4, 'reward': float(s)} for s in range(5)], [{'action': 1, 'reward': 0.0}, {'action': 2, 'reward': 1.0}, {'action': 3, 'reward': 2.0}]]
    tb, objective_data = batch_to_packed(_tok(encoder), batch)
    assert list(tb.step_counts()) == [5, 3]
    assert objective_data['sequence_id'].tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert objective_data['grouping_id'].tolist() == [0] * 8
    assert tb.prediction_indices.shape == (8,)
    assert list(tb.sequence_ids[tb.prediction_indices]).count(0) == 5
    assert list(tb.sequence_ids[tb.prediction_indices]).count(1) == 3

@pytest.mark.skipif(not torch.cuda.is_available(), reason='Flex packed path is CUDA-only')
def test_flex_packed_matches_sdpa_document_mask() -> None:
    """Flex packed forward ≈ SDPA with the same dense sequence/task mask."""
    torch.manual_seed(0)
    device = torch.device('cuda')
    backbone = Qwen3Backbone(hidden_dim=64, num_layers=2, num_heads=4, num_key_value_heads=4).to(device=device, dtype=torch.float32)
    transformer = backbone.model
    assert transformer is not None
    L = 12
    embeds = torch.randn(L, backbone.hidden_dim, device=device, dtype=torch.float32)
    sequence_ids = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long)
    grouping_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], device=device, dtype=torch.long)
    q = torch.arange(L, device=device)
    kv = torch.arange(L, device=device)
    allow = (
        (kv.unsqueeze(0) <= q.unsqueeze(1))
        & (sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(0))
        & (grouping_ids.unsqueeze(1) == grouping_ids.unsqueeze(0))
    )
    assert not bool(allow[9, 0].item())
    assert not bool(allow[5, 0].item())  # cross-task within seq 0
    assert bool(allow[3, 0].item())
    assert bool(allow[6, 4].item())
    with torch.no_grad():
        h_flex = cast(
            torch.Tensor,
            flex_packed_forward(
                output_hidden_states=False,
                model=transformer,
                embeds=embeds,
                sequence_ids=sequence_ids,
                grouping_ids=grouping_ids,
            ),
        )
        attention_mask = _flat_sequence_causal_mask(
            dtype=embeds.dtype, sequence_ids=sequence_ids, grouping_ids=grouping_ids
        )
        position_ids = _flat_sequence_position_ids(
            sequence_ids=sequence_ids, grouping_ids=grouping_ids
        )
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
    encoder = NumericEmbedder(hidden_dim=backbone.hidden_dim, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}, {'type': 'learnable', 'tokens': 1}])
    head = DiscreteActionValueHead(in_features=backbone.hidden_dim, out_features=4, hidden_dim=backbone.hidden_dim, num_layers=1)
    model = Model(encoder=encoder, backbone=backbone, heads=head).to(device=device, dtype=torch.float32).eval()
    batch = [[{'action': i % 4} for i in range(3)], [{'action': i % 4} for i in range(3)]]
    tb = batch_to_token_batch(_tok(encoder), batch)
    with torch.no_grad():
        preds0, _ = model(tb)
        batch_corrupt = [[{'action': 3} for _ in range(3)], [{'action': i % 4} for i in range(3)]]
        tb_c = batch_to_token_batch(_tok(encoder), batch_corrupt)
        preds1, _ = model(tb_c)
    q0 = preds0['action_value']
    q1 = preds1['action_value']
    assert torch.allclose(q0[3:], q1[3:], atol=1e-05, rtol=1e-05)
    assert not torch.allclose(q0[:3], q1[:3], atol=1e-05, rtol=1e-05)
    assert tb.N == 6
    assert list(tb.sequence_ids[tb.prediction_indices]) == [0, 0, 0, 1, 1, 1]

def test_model_train_isolates_tasks_within_sequence() -> None:
    """Packed train forward on a two-task window matches a single-task suffix forward."""
    torch.manual_seed(11)
    backbone = Qwen3Backbone(hidden_dim=32, num_layers=2, num_heads=4, num_key_value_heads=4)
    encoder = NumericEmbedder(
        hidden_dim=backbone.hidden_dim,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4},
            {"type": 'discrete', "field": "episode_done", "vocab_size": 3},
            {'type': 'learnable', 'tokens': 1},
        ],
    )
    head = DiscreteActionValueHead(
        in_features=backbone.hidden_dim,
        out_features=4,
        hidden_dim=backbone.hidden_dim,
        num_layers=1,
    )
    model = Model(encoder=encoder, backbone=backbone, heads=head).eval()
    task0 = [
        {'action': 0, 'episode_done': 0, 'task_done': 0, 'task_index': 0},
        {'action': 1, 'episode_done': 0, 'task_done': 0, 'task_index': 0},
        {'action': 2, 'episode_done': 1, 'task_done': 2, 'task_index': 0},
    ]
    task1 = [
        {'action': 3, 'episode_done': 0, 'task_done': 0, 'task_index': 1},
        {'action': 1, 'episode_done': 0, 'task_done': 0, 'task_index': 1},
    ]
    with torch.no_grad():
        tb_both, od = batch_to_packed(
            _tok(encoder, grouping_field="task_index"),
            [task0 + task1],
            grouping_field="task_index",
        )
        tb_t1 = batch_to_token_batch(
            _tok(encoder, grouping_field="task_index"),
            [task1],
            grouping_field="task_index",
        )
        preds_both, _ = model(tb_both)
        preds_t1, _ = model(tb_t1)
    assert od['task_index'].tolist() == [0, 0, 0, 1, 1]
    # Current-task suffix predictions match a fresh single-task forward.
    assert torch.allclose(
        preds_both['action_value'][3:],
        preds_t1['action_value'],
        atol=1e-05,
        rtol=1e-05,
    )
