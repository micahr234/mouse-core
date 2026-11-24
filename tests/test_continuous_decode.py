"""Real HF scheduling/cache + Mouse Flex inference, without downloaded models."""
from __future__ import annotations
from typing import Any, cast
from mouse_core.models.backbone.transformer import Architecture

import pytest
import torch
from torch import nn
from transformers.generation import ContinuousBatchingConfig
from transformers.generation.continuous_batching.cache import PagedAttentionCache
from transformers.generation.continuous_batching.scheduler import FIFOScheduler

from mouse_core.models import Backbone, DecodeCache
from mouse_core.models.backbone import ContinuousDecodeSession


PAGE_SIZE = ContinuousBatchingConfig().page_size

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def backbone(architecture: Architecture="qwen3", device="cpu", autocast_dtype=None):
    torch.manual_seed(81)
    model = Backbone(architecture=architecture, dtype=torch.float32, hidden_dim=32,
                     num_layers=2, num_heads=2, num_key_value_heads=1,
                     autocast_dtype=autocast_dtype,).to(device).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("q_proj.weight", "k_proj.weight")):
                param.mul_(8)
    return model


@torch.no_grad()
def reference(model, tokens, *, layers=False):
    return model(tokens, sequence_ids=torch.zeros(len(tokens), device=tokens.device, dtype=torch.long),
                 output_hidden_states=layers)


@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@pytest.mark.parametrize("budget", [3, 32])
@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_incremental_ragged_chunking_matches_training_mask(architecture, budget, device):
    model = backbone(architecture, device)
    cache = model.decode_session(3, max_batch_tokens=budget, cache_capacity_tokens=3 * PAGE_SIZE)
    assert isinstance(cache._cache, PagedAttentionCache)
    assert isinstance(cache._scheduler, FIFOScheduler)
    histories = [torch.empty(0, 32, device=device) for _ in range(3)]
    # Pause rows, prefill while others decode, and reset one episode.
    for lengths, reset in [([17, 2, 0], []), ([1, 0, 9], []), ([2, 8, 1], [1]), ([0, 0, 0], [])]:
        if reset:
            cache.reset_rows(reset)
            for row in reset:
                histories[row] = histories[row][:0]
        inputs = torch.randn(3, max(lengths), 32, device=device)
        actual, layers = model(inputs, cache=cache, lengths=lengths, output_hidden_states=True)
        for row, count in enumerate(lengths):
            if count:
                histories[row] = torch.cat([histories[row], inputs[row, -count:]])
                expected, expected_layers = reference(model, histories[row], layers=True)
                torch.testing.assert_close(actual[row, -count:], expected[-count:], atol=2e-5, rtol=2e-5)
                for got, want in zip(layers, expected_layers, strict=True):
                    torch.testing.assert_close(got[row, -count:], want[-count:], atol=2e-5, rtol=2e-5)
            assert torch.count_nonzero(actual[row, :actual.shape[1] - count]) == 0
        assert cache.lengths == tuple(len(h) for h in histories)
    DecodeCache((cache,)).close()
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.step()


@torch.no_grad()
def test_ready_rows_arrive_during_prefill_and_reuse_finished_slots():
    model = backbone()
    cache = model.decode_session(3, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    first = torch.randn(10, 32)
    second = torch.randn(1, 32)
    cache.submit(0, first)
    assert cache.step() == {}
    cache.submit(1, second)
    with pytest.raises(ValueError, match="unfinished"):
        cache.submit(0, second)
    assert cache.step() == {}
    result = cache.step()
    assert set(result) == {0, 1}
    torch.testing.assert_close(result[0], reference(model, first))
    torch.testing.assert_close(result[1], reference(model, second))
    # Idle episode 0 remains in the allocator when episode 1 is replaced.
    assert cache._cache is not None
    allocator = cache._cache.layer_to_allocator[0]
    blocks_before = list(allocator.block_table['0'])
    cache.reset_rows([1])
    cache.submit(1, first[:2])
    cache.submit(0, second)
    result = cache.step()
    torch.testing.assert_close(result[0], reference(model, torch.cat([first, second]))[-1:])
    torch.testing.assert_close(result[1], reference(model, first[:2]))
    assert allocator.block_table['0'] == blocks_before
    assert cache.step() == {}
    cache.close()


@torch.no_grad()
def test_cancel_partial_prefill_releases_capacity_and_restarts_positions():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    cache.submit(0, torch.randn(16, 32))
    cache.step()
    cache.reset_rows([0])
    assert not cache.has_pending
    assert cache.lengths == (0, 0)
    x = torch.randn(32, 32)
    cache.submit(1, x)
    result = {}
    while cache.has_pending:
        result.update(cache.step())
    torch.testing.assert_close(result[1], reference(model, x))
    cache.close()


@torch.no_grad()
def test_capacity_never_evicts_live_history_and_rejected_batch_is_atomic():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=2 * PAGE_SIZE)
    x = torch.randn(2, 1, 32)
    cache.forward(embeds=x, lengths=[1, 1])
    with pytest.raises(ValueError, match="KV capacity exceeded"):
        cache.forward(embeds=torch.randn(2, PAGE_SIZE, 32), lengths=[1, PAGE_SIZE])
    assert cache.lengths == (1, 1)
    assert not cache.has_pending
    y = torch.randn(2, PAGE_SIZE - 1, 32)
    result = cache.forward(embeds=y, lengths=[PAGE_SIZE - 1, PAGE_SIZE - 1])
    torch.testing.assert_close(result[0], reference(model, torch.cat([x[0], y[0]]))[-(PAGE_SIZE - 1):])
    cache.reset_rows([1])
    z = torch.randn(1, 32)
    cache.submit(0, z)
    got = cache.step()[0]
    torch.testing.assert_close(got, reference(model, torch.cat([x[0], y[0], z]))[-1:])
    cache.close()


@torch.no_grad()
def test_pending_chunks_reserve_capacity():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    cache.submit(0, torch.randn(2 * PAGE_SIZE, 32))
    cache.step()
    with pytest.raises(ValueError, match="KV capacity exceeded"):
        cache.submit(1, torch.randn(1, 32))
    cache.close()


@torch.no_grad()
def test_weight_versions_require_explicit_full_reset():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=2 * PAGE_SIZE)
    x = torch.randn(2, 1, 32)
    cache.forward(embeds=x, lengths=[1, 1])
    next(model.parameters()).add_(0.1)
    with pytest.raises(RuntimeError, match="weights changed"):
        cache.forward(embeds=x, lengths=[1, 1])
    with pytest.raises(RuntimeError, match="weights changed"):
        cache.reset_rows([0])
    cache.reset_rows()
    got = cache.forward(embeds=x, lengths=[1, 1])
    torch.testing.assert_close(got[0], reference(model, x[0]))
    cache.close()


@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
@torch.no_grad()
def test_hidden_states_feed_multiple_existing_rl_heads(architecture):
    model = backbone(architecture)
    cache = model.decode_session(1, max_batch_tokens=3, cache_capacity_tokens=2 * PAGE_SIZE)
    heads = nn.ModuleDict({'q': nn.Linear(32, 4), 'policy': nn.Linear(32, 4), 'value': nn.Linear(32, 1)})
    x = torch.randn(1, 13, 32)  # Can already include numeric or compute embeddings.
    h = model(x, cache=cache, lengths=[13])
    expected = reference(model, x[0])
    for head in heads.values():
        torch.testing.assert_close(head(h[0, -1]), head(expected[-1]))
    assert not h.requires_grad
    cache.close()


@pytest.mark.parametrize('kwargs, message', [
    ({'batch_size': 0}, 'batch_size'), ({'max_batch_tokens': 0}, 'max_batch_tokens'),
    ({'cache_capacity_tokens': PAGE_SIZE + 1}, f'multiple of {PAGE_SIZE}'), ({'cache_capacity_tokens': True}, 'positive integer'),
])
def test_invalid_capacity_configuration(kwargs, message):
    opts: dict[str, Any] = dict(batch_size=2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    opts.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ContinuousDecodeSession(cast(nn.Module, backbone().model), **opts)


def test_cache_memory_error_explains_inference_pool_and_preserves_cause(monkeypatch):
    from transformers.generation.continuous_batching.cache import PagedAttentionMemoryHandler

    # Exercise HF's actual footprint rejection without allocating a large cache
    # or depending on available GPU memory.
    monkeypatch.setattr(PagedAttentionMemoryHandler, 'get_available_memory', lambda self: 0)
    with pytest.raises(MemoryError, match='Cannot allocate the continuous decode cache') as raised:
        backbone(autocast_dtype=torch.bfloat16).decode_session(
            2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    message = str(raised.value)
    for detail in (f'cache_capacity_tokens={2 * PAGE_SIZE}', 'max_batch_tokens=4',
                   'batch_size=2', 'cache_dtype=torch.bfloat16', 'training batch size'):
        assert detail in message
    assert isinstance(raised.value.__cause__, MemoryError)
    assert 'Memory footprint' in str(raised.value.__cause__)


def test_live_mask_cannot_be_changed_by_cache_configuration():
    model = backbone()
    model.model.config.layer_types = ["full_attention", "sliding_attention"]
    with pytest.raises(ValueError, match="full attention"):
        model.decode_session(1, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)


def test_invalid_inputs_do_not_poison_session():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    for row, x in [(2, torch.randn(1, 32)), (0, torch.randn(0, 32)), (0, torch.randn(1, 31))]:
        with pytest.raises(ValueError):
            cache.submit(row, x)
    with pytest.raises(ValueError, match='per-row lengths'):
        cache.forward(embeds=torch.randn(2, 1, 32), lengths=[2, 1])
    with pytest.raises(RuntimeError, match='ambient'):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            cache.submit(0, torch.randn(1, 32))
    model.train()
    with pytest.raises(RuntimeError, match='eval'):
        cache.submit(0, torch.randn(1, 32))
    model.eval()
    cache.submit(0, torch.randn(1, 32))
    with pytest.raises(RuntimeError, match='pending'):
        cache.forward(embeds=torch.randn(2, 1, 32), lengths=[1, 1])
    cache.close()


def test_failed_forward_closes_partially_written_cache(monkeypatch):
    model = backbone()
    cache = model.decode_session(1, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    cache.submit(0, torch.randn(2, 32))
    def fail(*args, **kwargs):
        raise RuntimeError('injected decoder failure')
    monkeypatch.setattr(model.model.layers[1].mlp, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        cache.step()
    with pytest.raises(RuntimeError, match='closed'):
        cache.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_cuda_mixed_precision():
    model = backbone(device='cuda', autocast_dtype=torch.bfloat16)
    cache = model.decode_session(2, max_batch_tokens=16, cache_capacity_tokens=2 * PAGE_SIZE)
    x = torch.randn(2, 25, 32, device='cuda')
    result = cache.forward(embeds=x, lengths=[25, 7])
    expected = reference(model, x[0])
    torch.testing.assert_close(result[0], expected, atol=4e-3, rtol=4e-3)
    assert cache._cache is not None
    assert cache._cache.dtype == torch.bfloat16
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_cuda_bf16_base_with_fp32_lora():
    from mouse_core.models import LoRAConfig
    torch.manual_seed(72)
    model = Backbone(
        architecture="qwen3", dtype=torch.bfloat16, hidden_dim=32,
        num_layers=2, num_heads=2, num_key_value_heads=1,
        lora=LoRAConfig(rank=4, alpha=8),
    ).cuda().eval()
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.normal_(std=0.02)
    cache = model.decode_session(2, max_batch_tokens=7, cache_capacity_tokens=2 * PAGE_SIZE)
    x = torch.randn(2, 17, 32, device="cuda", dtype=torch.bfloat16)
    result = cache.forward(embeds=x, lengths=[17, 1])
    expected = reference(model, x[0])
    torch.testing.assert_close(result[0], expected, atol=0.04, rtol=0.02)
    assert cache._cache is not None
    assert cache._cache.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n)
    cache.close()


@pytest.mark.parametrize("tokens", [1, PAGE_SIZE + 1, 2 * PAGE_SIZE + 1])
def test_physical_addresses_match_hf_allocator(tokens):
    """Match HF's address oracle across interleaved full-episode pages."""
    import numpy as np
    from mouse_core.models.backbone.continuous_decode import _physical_indices
    session = backbone().decode_session(2, cache_capacity_tokens=4 * PAGE_SIZE)
    assert session._cache is not None
    allocator = session._cache.layer_to_allocator[0]
    try:
        # Use the scheduler to reserve sectors and interleave each row's pages.
        first = min(tokens, allocator.tokens_per_page)
        session.forward(embeds=torch.randn(2, first, 32), lengths=[first, first])
        if tokens > first:
            session.forward(embeds=torch.randn(2, tokens - first, 32), lengths=[tokens - first, 0])
        for past in sorted({0, min(1, tokens - 1), tokens - 1}):
            for count in sorted({1, tokens - past}):
                read = _physical_indices(allocator, '0', np.arange(past + count, dtype=np.int64))
                assert read.tolist() == allocator.get_read_indices('0', past, count)
                write = _physical_indices(allocator, '0', np.arange(past, past + count, dtype=np.int64))
                assert write.tolist() == allocator.get_write_indices('0', past, count)
    finally:
        session.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens_per_row", [32, 66, 96])
@torch.no_grad()
def test_cuda_batched_gqa_decode_matches_full_episodes(tokens_per_row):
    # Exercise both ordinary decode and GQA query tiles larger than the mask
    # block, now within each environment row rather than a packed global row.
    model = backbone(device="cuda", autocast_dtype=torch.bfloat16)
    cache = model.decode_session(3, max_batch_tokens=3 * tokens_per_row, cache_capacity_tokens=6 * PAGE_SIZE)
    history = torch.empty(3, 0, 32, device="cuda")
    for _ in range(3):
        x = torch.randn(3, tokens_per_row, 32, device="cuda")
        history = torch.cat([history, x], dim=1)
        got = cache.forward(embeds=x, lengths=[tokens_per_row] * 3)
        for row in range(3):
            expected = reference(model, history[row])[-tokens_per_row:]
            torch.testing.assert_close(got[row], expected, atol=4e-3, rtol=4e-3)
    cache.close()


@torch.no_grad()
def test_unpadded_chunks_match_padded_forward_with_deferred_history():
    model = backbone()
    sessions = [model.decode_session(3, max_batch_tokens=4, cache_capacity_tokens=3 * PAGE_SIZE)
                for _ in range(2)]
    padded, ragged = sessions
    try:
        for lengths, reset in [([7, 2, 1], []), ([1, 0, 5], []), ([3, 1, 2], [1])]:
            for session in sessions:
                if reset:
                    session.reset_rows(reset)
            x = torch.randn(3, max(lengths), 32)
            # Deliberately reverse insertion order: scheduling must remain by row.
            chunks = {row: x[row, -lengths[row]:] for row in (2, 1, 0) if lengths[row]}
            actual = ragged.forward_chunks(chunks, output_hidden_states=True)
            expected, expected_layers = padded.forward(embeds=x, lengths=lengths, output_hidden_states=True)
            for row, (out, layers) in actual.items():
                torch.testing.assert_close(out, expected[row, -lengths[row]:], atol=0, rtol=0)
                for got, want in zip(layers, expected_layers, strict=True):
                    torch.testing.assert_close(got, want[row, -lengths[row]:], atol=0, rtol=0)
            assert padded.lengths == ragged.lengths
            suffix = torch.randn(2, 32)
            for session in sessions:
                session.defer(0, suffix)
        assert ragged.forward_chunks({}) == {}
    finally:
        for session in sessions:
            session.close()


@torch.no_grad()
def test_unpadded_invalid_batch_is_rejected_before_any_submission():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    try:
        with pytest.raises(ValueError, match='nonempty'):
            cache.forward_chunks({0: torch.randn(1, 32), 1: torch.empty(0, 32)})
        with pytest.raises(ValueError, match='KV capacity'):
            cache.forward_chunks({0: torch.randn(PAGE_SIZE, 32), 1: torch.randn(PAGE_SIZE + 1, 32)})
        assert not cache.has_pending and cache.lengths == (0, 0)
        cache.submit(0, torch.randn(1, 32))
        with pytest.raises(RuntimeError, match='Drain pending'):
            cache.forward_chunks({1: torch.randn(1, 32)})
    finally:
        cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_capture_stream_reuses_library_warmup_and_preserves_cache(monkeypatch):
    import mouse_core.models.backbone.continuous_decode as decode
    monkeypatch.setattr(decode, '_GRAPH_REUSE_THRESHOLD', 2)
    monkeypatch.setattr(decode, '_MAX_DECODE_GRAPHS', 2)
    model = Backbone(architecture='qwen3', dtype=torch.float32, hidden_dim=128,
                     num_layers=2, num_heads=4, num_key_value_heads=2).cuda().eval()
    cache = model.decode_session(2, max_batch_tokens=32, cache_capacity_tokens=2 * PAGE_SIZE)
    original = cache._inputs
    calls = 0
    def counted(*args):
        nonlocal calls
        calls += 1
        return original(*args)
    cache._inputs = counted
    history = [torch.empty(0, 128, device='cuda') for _ in range(2)]
    captures = []
    saved_outputs = []
    try:
        for width, expected_calls in zip((1, 1, 1, 3, 3, 3, 2, 2, 2, 1),
                                          (2, 4, 0, 2, 2, 0, 2, 2, 0, 2), strict=True):
            inputs = torch.randn(2, width, 128, device='cuda')
            before = calls
            outputs = cache.forward_chunks({i: inputs[i] for i in range(2)})
            assert calls - before == expected_calls
            for row in range(2):
                history[row] = torch.cat((history[row], inputs[row]))
                torch.testing.assert_close(outputs[row], reference(model, history[row])[-width:],
                                           atol=4e-5, rtol=4e-5)
            saved_outputs.append((outputs[0], outputs[0].clone()))
            if cache._capture_stream is not None:
                captures.append(cache._capture_stream)
        assert len(cache._graphs) == 2
        assert all(stream is captures[0] for stream in captures)
        for output, snapshot in saved_outputs:
            torch.testing.assert_close(output, snapshot, atol=0, rtol=0)
        cache.reset_rows()
        assert not cache._graphs
        assert cache._capture_stream is captures[0]
    finally:
        cache.close()
    assert cache._capture_stream is None


@torch.no_grad()
def test_deferred_batch_is_atomic_owns_inputs_and_preserves_history():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=2 * PAGE_SIZE)
    prefix = {0: torch.randn(2, 32), 1: torch.randn(3, 32)}
    snapshots = {row: value.clone() for row, value in prefix.items()}
    try:
        cache.defer_chunks(dict(reversed(list(prefix.items()))))
        for value in prefix.values():
            value.zero_()
        for invalid, message in (({0: torch.randn(1, 32), 1: torch.empty(0, 32)}, 'nonempty'),
                                 ({0: torch.randn(PAGE_SIZE, 32), 1: torch.randn(1, 32)}, 'KV capacity')):
            with pytest.raises(ValueError, match=message):
                cache.defer_chunks(invalid)
            assert cache.lengths == (2, 3)
            for row in prefix:
                torch.testing.assert_close(cache._deferred[row], snapshots[row], atol=0, rtol=0)
        query = {row: torch.randn(1, 32) for row in prefix}
        result = cache.forward_chunks(query)
        for row in prefix:
            expected = reference(model, torch.cat((snapshots[row], query[row])))[-1:]
            torch.testing.assert_close(result[row], expected)
        next(model.parameters()).add_(.1)
        with pytest.raises(RuntimeError, match='weights changed'):
            cache.defer_chunks(query)
        cache.reset_rows()
        cache.defer_chunks(query)
        assert cache.lengths == (1, 1)
    finally:
        cache.close()
