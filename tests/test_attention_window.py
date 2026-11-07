"""Attention-distance semantics and circular KV storage against dense oracles."""
from __future__ import annotations

import pytest
import torch
from transformers.generation.continuous_batching.cache_allocators.sliding_attention import SlidingAttentionCacheAllocator

from mouse_core.models import Backbone
from mouse_core.models.backbone import ContinuousDecodeSession
from mouse_core.models.backbone.packed_train import packed_forward

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))]


def make_model(distance, architecture='qwen3', device='cpu'):
    torch.manual_seed(19)
    b = Backbone(architecture=architecture, dtype=torch.float32, hidden_dim=32,
                 num_layers=2, num_heads=2, num_key_value_heads=1,
                 max_attention_distance_tokens=distance).to(device).eval()
    with torch.no_grad():
        for name, p in b.named_parameters():
            if name.endswith(('q_proj.weight', 'k_proj.weight')):
                p.mul_(8)
            if name.endswith(('v_proj.weight', 'o_proj.weight')):
                p.mul_(3)
    return b


def dense_reference(model, embeds, sequence_ids, distance):
    # Independent O(L^2) oracle: distances count previous tokens of the SAME
    # sequence, even when outer episodes are interleaved in the flat input.
    same = sequence_ids[:, None] == sequence_ids[None, :]
    lower = torch.ones_like(same).tril()
    positions = (same & lower).sum(-1) - 1
    delta = positions[:, None] - positions[None, :]
    allowed = same & lower
    if distance is not None:
        allowed &= delta <= distance
    mask = torch.zeros_like(delta, dtype=embeds.dtype).masked_fill(~allowed, torch.finfo(embeds.dtype).min)
    return model.model(inputs_embeds=embeds[None], attention_mask=mask[None, None],
                       position_ids=positions[None], use_cache=False).last_hidden_state[0]


@pytest.mark.parametrize('distance', [0, 3, 16, 1000])
@pytest.mark.parametrize('architecture', ['qwen3', 'llama'])
@torch.no_grad()
def test_packed_window_matches_dense_oracle_for_interleaved_episodes(distance, architecture):
    model = make_model(distance, architecture)
    embeds = torch.randn(25, 32)
    sequence_ids = torch.tensor([0, 3, 0, 0, 3] * 5)
    actual = model(embeds, sequence_ids=sequence_ids)
    expected = dense_reference(model, embeds, sequence_ids, distance)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('distance', [0, 3, 16])
@pytest.mark.parametrize('device', DEVICES)
@torch.no_grad()
def test_ring_wraps_without_growth_and_matches_full_windowed_history(distance, device):
    model = make_model(distance, device=device)
    span = distance + 1
    page_tokens = ((span + 15) // 16) * 16
    session = model.decode_session(2, max_batch_tokens=23, cache_capacity_tokens=2 * page_tokens)
    allocator = session._cache.layer_to_allocator[0]
    assert isinstance(allocator, SlidingAttentionCacheAllocator)
    initial_pointer = session._cache.cache_tensor.data_ptr()
    initial_bytes = session._cache.cache_tensor.numel()
    histories = [torch.empty(0, 32, device=device) for _ in range(2)]
    for lengths, reset in [([8, 3], []), ([1, 0], []), ([21, 7], []), ([2, 19], [1]), ([17, 1], [])]:
        if reset:
            session.reset_rows(reset)
            for row in reset:
                histories[row] = histories[row][:0]
        x = torch.randn(2, max(lengths), 32, device=device)
        actual = session.forward(embeds=x, lengths=lengths)
        for row, count in enumerate(lengths):
            if count:
                histories[row] = torch.cat([histories[row], x[row, -count:]])
                seq = torch.zeros(len(histories[row]), device=device, dtype=torch.long)
                expected = dense_reference(model, histories[row], seq, distance)
                torch.testing.assert_close(actual[row, -count:], expected[-count:], rtol=1e-4, atol=1e-4)
            assert len(allocator.block_table[str(row)]) <= page_tokens // 16
        assert session.lengths == tuple(len(x) for x in histories)
        assert session._cache.cache_tensor.data_ptr() == initial_pointer
        assert session._cache.cache_tensor.numel() == initial_bytes
    session.close()


@torch.no_grad()
def test_long_window_is_equivalent_to_full_episode_attention():
    windowed = make_model(1000)
    x = torch.randn(2, 35, 32)
    expected = torch.stack([dense_reference(windowed, row, torch.zeros(35, dtype=torch.long), None) for row in x])
    torch.testing.assert_close(windowed(x), expected)
    session = windowed.decode_session(2, max_batch_tokens=7, cache_capacity_tokens=128)
    chunks = [session.forward(embeds=x[:, :11], lengths=[11, 11]),
              session.forward(embeds=x[:, 11:], lengths=[24, 24])]
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected)
    session.close()


@torch.no_grad()
def test_decode_session_selects_hf_ring_and_reserves_one_window_per_row():
    model = make_model(16)
    session = model.decode_session(3)
    assert isinstance(session, ContinuousDecodeSession)
    assert session.cache_capacity_tokens == 3 * 32  # 17 slots, rounded to 16-token pages
    assert session._cache.layer_to_allocator[0].sliding_window == 17
    session.close()


@torch.no_grad()
def test_ring_preserves_hidden_history_instead_of_recomputing_a_truncated_prefix():
    model = make_model(3)
    x = torch.randn(1, 10, 32)
    session = model.decode_session(1)
    session.forward(embeds=x[:, :8], lengths=[8])
    actual = session.forward(embeds=x[:, 8:], lengths=[2])
    expected = model(x)
    truncated = model(x[:, -4:], position_ids=torch.arange(6, 10)[None])
    torch.testing.assert_close(actual, expected[:, -2:])
    assert not torch.allclose(actual[:, -1], truncated[:, -1], rtol=1e-5, atol=1e-5)
    session.close()


@torch.no_grad()
def test_window_changes_require_new_cache_even_after_reset():
    model = make_model(3)
    session = model.decode_session(1)
    model.model.config.sliding_window = 9
    with pytest.raises(RuntimeError, match='window changed'):
        session.forward(embeds=torch.randn(1, 1, 32), lengths=[1])
    with pytest.raises(RuntimeError, match='attention window changed'):
        session.reset_rows()
    with pytest.raises(RuntimeError, match='closed'):
        session.step()


@pytest.mark.parametrize('distance', [-1, 1.5, True, '16', None])
def test_invalid_distance_is_rejected(distance):
    with pytest.raises(ValueError, match='max_attention_distance_tokens'):
        make_model(distance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_window_training_gradients_and_checkpoint_rebind_match_dense_reference():
    model = make_model(3, device='cuda')
    other = make_model(11, device='cuda')
    embeds = torch.randn(19, 32, device='cuda', requires_grad=True)
    seq = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], device='cuda')

    def run(kind):
        model.zero_grad(set_to_none=True)
        embeds.grad = None
        if kind == 'dense':
            result = dense_reference(model, embeds, seq, 3)
        else:
            result = packed_forward(model=model.model, embeds=embeds, sequence_ids=seq,
                                    checkpoint=kind == 'checkpoint')
        # Change both the packed segment layout AND the window before backward.
        with torch.no_grad():
            other(torch.randn(23, 32, device='cuda'), sequence_ids=torch.zeros(23, dtype=torch.long, device='cuda'))
        result.square().mean().backward()
        return result.detach(), [embeds.grad.clone()] + [p.grad.clone() for p in model.parameters() if p.grad is not None]

    expected, gradients = run('dense')
    for kind in ('plain', 'checkpoint'):
        actual, actual_gradients = run(kind)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        for got, want in zip(actual_gradients, gradients, strict=True):
            torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-5)
