"""Full-episode attention and shared KV storage against dense oracles."""
from __future__ import annotations

from transformers.generation import ContinuousBatchingConfig

PAGE_SIZE = ContinuousBatchingConfig().page_size

import pytest
import torch
from transformers.generation.continuous_batching.cache_allocators.full_attention import FullAttentionCacheAllocator

from mouse_core.models import Backbone
from mouse_core.models.backbone import ContinuousDecodeSession
from mouse_core.models.backbone.packed_train import packed_forward

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))]


def make_model(architecture='qwen3', device='cpu'):
    torch.manual_seed(19)
    b = Backbone(architecture=architecture, dtype=torch.float32, hidden_dim=32,
                 num_layers=2, num_heads=2, num_key_value_heads=1).to(device).eval()
    with torch.no_grad():
        for name, p in b.named_parameters():
            if name.endswith(('q_proj.weight', 'k_proj.weight')):
                p.mul_(8)
            if name.endswith(('v_proj.weight', 'o_proj.weight')):
                p.mul_(3)
    return b


def dense_reference(model, embeds, sequence_ids):
    # Independent O(L^2) oracle: distances count previous tokens of the SAME
    # sequence, even when outer episodes are interleaved in the flat input.
    same = sequence_ids[:, None] == sequence_ids[None, :]
    lower = torch.ones_like(same).tril()
    positions = (same & lower).sum(-1) - 1
    delta = positions[:, None] - positions[None, :]
    allowed = same & lower
    mask = torch.zeros_like(delta, dtype=embeds.dtype).masked_fill(~allowed, torch.finfo(embeds.dtype).min)
    return model.model(inputs_embeds=embeds[None], attention_mask=mask[None, None],
                       position_ids=positions[None], use_cache=False).last_hidden_state[0]


@pytest.mark.parametrize('architecture', ['qwen3', 'llama'])
@torch.no_grad()
def test_full_episode_mask_matches_dense_oracle_for_interleaved_episodes(architecture):
    model = make_model(architecture)
    embeds = torch.randn(25, 32)
    sequence_ids = torch.tensor([0, 3, 0, 0, 3] * 5)
    actual = model(embeds, sequence_ids=sequence_ids)
    expected = dense_reference(model, embeds, sequence_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('device', DEVICES)
@torch.no_grad()
def test_rows_borrow_shared_capacity_and_reset_reuses_pool(device):
    model = make_model(device=device)
    session = model.decode_session(2, max_batch_tokens=127, cache_capacity_tokens=3 * PAGE_SIZE)
    initial_pointer = session._cache.cache_tensor.data_ptr()
    x = torch.randn(2, 2 * PAGE_SIZE, 32, device=device)
    # One episode can use more than half the shared pool; there is no per-row limit.
    got = session.forward(embeds=x, lengths=[2 * PAGE_SIZE, 1])
    expected = dense_reference(model, x[0], torch.zeros(2 * PAGE_SIZE, device=device, dtype=torch.long))
    torch.testing.assert_close(got[0], expected, rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError, match='KV capacity exceeded'):
        session.forward(embeds=x[:, :1], lengths=[1, 0])
    assert session.lengths == (2 * PAGE_SIZE, 1)
    assert not session.has_pending
    session.reset_rows([1])
    extra = x[:, :1]
    got = session.forward(embeds=extra, lengths=[1, 0])
    history = torch.cat([x[0], extra[0]])
    expected = dense_reference(model, history, torch.zeros(len(history), device=device, dtype=torch.long))
    torch.testing.assert_close(got[0], expected[-1:], rtol=1e-4, atol=1e-4)
    assert session._cache.cache_tensor.data_ptr() == initial_pointer
    session.reset_rows([0])
    got = session.forward(embeds=extra, lengths=[1, 1])
    for row in range(2):
        expected = dense_reference(model, extra[row], torch.zeros(1, device=device, dtype=torch.long))
        torch.testing.assert_close(got[row], expected)
    session.close()


@torch.no_grad()
def test_dense_and_chunked_inputs_match_full_episode_attention():
    model = make_model()
    x = torch.randn(2, 35, 32)
    expected = torch.stack([dense_reference(model, row, torch.zeros(35, dtype=torch.long)) for row in x])
    torch.testing.assert_close(model(x), expected)
    session = model.decode_session(2, max_batch_tokens=7, cache_capacity_tokens=2 * PAGE_SIZE)
    chunks = [session.forward(embeds=x[:, :11], lengths=[11, 11]),
              session.forward(embeds=x[:, 11:], lengths=[24, 24])]
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected)
    session.close()


@torch.no_grad()
def test_decode_session_uses_hf_full_attention_pages():
    model = make_model()
    session = model.decode_session(3, cache_capacity_tokens=3 * PAGE_SIZE)
    assert isinstance(session, ContinuousDecodeSession)
    assert session._cache.layer_to_allocator[0].tokens_per_page == PAGE_SIZE
    assert type(session._cache.layer_to_allocator[0]) is FullAttentionCacheAllocator
    assert getattr(model.model.config, 'sliding_window', None) is None
    session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_full_episode_training_gradients_and_checkpoint_rebind_match_dense_reference():
    model = make_model(device='cuda')
    other = make_model(device='cuda')
    embeds = torch.randn(19, 32, device='cuda', requires_grad=True)
    seq = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], device='cuda')

    def run(kind):
        model.zero_grad(set_to_none=True)
        embeds.grad = None
        if kind == 'dense':
            result = dense_reference(model, embeds, seq)
        else:
            result = packed_forward(model=model.model, embeds=embeds, sequence_ids=seq,
                                    checkpoint=kind == 'checkpoint')
        # Change the packed segment layout before checkpoint recomputation.
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
