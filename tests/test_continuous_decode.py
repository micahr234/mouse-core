"""Real HF scheduling/cache + Mouse Flex inference, without downloaded models."""
from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.generation.continuous_batching.cache import PagedAttentionCache
from transformers.generation.continuous_batching.scheduler import FIFOScheduler

from mouse_core.models import Backbone, DecodeCache
from mouse_core.models.backbone import ContinuousDecodeSession


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def backbone(architecture="qwen3", device="cpu", autocast_dtype=None, distance=16383):
    torch.manual_seed(81)
    model = Backbone(architecture=architecture, dtype=torch.float32, hidden_dim=32,
                     num_layers=2, num_heads=2, num_key_value_heads=1,
                     train_autocast_dtype=autocast_dtype, decode_autocast_dtype=autocast_dtype,
                     max_attention_distance_tokens=distance).to(device).eval()
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
    cache = model.decode_session(3, max_batch_tokens=budget, cache_capacity_tokens=256)
    assert isinstance(cache._cache, PagedAttentionCache)
    assert isinstance(cache._scheduler, FIFOScheduler)
    histories = [torch.empty(0, 32, device=device) for _ in range(3)]
    # Cross KV pages, pause rows, prefill while others decode, and reset one episode.
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
    cache = model.decode_session(3, max_batch_tokens=4, cache_capacity_tokens=128)
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
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=32)
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
    cache = model.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=32)
    x = torch.randn(2, 1, 32)
    cache.forward(embeds=x, lengths=[1, 1])
    with pytest.raises(ValueError, match="KV capacity exceeded"):
        cache.forward(embeds=torch.randn(2, 16, 32), lengths=[1, 16])
    assert cache.lengths == (1, 1)
    assert not cache.has_pending
    y = torch.randn(2, 15, 32)
    result = cache.forward(embeds=y, lengths=[15, 15])
    torch.testing.assert_close(result[0], reference(model, torch.cat([x[0], y[0]]))[-15:])
    cache.reset_rows([1])
    z = torch.randn(1, 32)
    cache.submit(0, z)
    got = cache.step()[0]
    torch.testing.assert_close(got, reference(model, torch.cat([x[0], y[0], z]))[-1:])
    cache.close()


@torch.no_grad()
def test_pending_chunks_reserve_capacity():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=32)
    cache.submit(0, torch.randn(32, 32))
    cache.step()
    with pytest.raises(ValueError, match="KV capacity exceeded"):
        cache.submit(1, torch.randn(1, 32))
    cache.close()


@torch.no_grad()
def test_weight_versions_require_explicit_full_reset():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=32)
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
    cache = model.decode_session(1, max_batch_tokens=3, cache_capacity_tokens=32)
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
    ({'cache_capacity_tokens': 17}, 'multiple of 16'), ({'cache_capacity_tokens': True}, 'positive integer'),
])
def test_invalid_capacity_configuration(kwargs, message):
    opts = dict(batch_size=2, max_batch_tokens=4, cache_capacity_tokens=32)
    opts.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ContinuousDecodeSession(backbone().model, **opts)


def test_live_mask_cannot_be_changed_by_cache_configuration():
    model = backbone()
    model.model.config.layer_types = ["full_attention", "sliding_attention"]
    with pytest.raises(ValueError, match="same sliding"):
        model.decode_session(1, max_batch_tokens=4, cache_capacity_tokens=32)


def test_invalid_inputs_do_not_poison_session():
    model = backbone()
    cache = model.decode_session(2, max_batch_tokens=4, cache_capacity_tokens=32)
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
    cache = model.decode_session(1, max_batch_tokens=4, cache_capacity_tokens=32)
    cache.submit(0, torch.randn(2, 32))
    def fail(*args, **kwargs):
        raise RuntimeError('injected decoder failure')
    monkeypatch.setattr(model.model.layers[1].mlp, 'forward', fail)
    with pytest.raises(RuntimeError, match='injected'):
        cache.step()
    with pytest.raises(RuntimeError, match='closed'):
        cache.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize("distance", [16383, 3])
@torch.no_grad()
def test_cuda_mixed_precision(distance):
    model = backbone(device='cuda', autocast_dtype=torch.bfloat16, distance=distance)
    cache = model.decode_session(2, max_batch_tokens=16, cache_capacity_tokens=128)
    x = torch.randn(2, 25, 32, device='cuda')
    result = cache.forward(embeds=x, lengths=[25, 7])
    expected = reference(model, x[0])
    torch.testing.assert_close(result[0], expected, atol=4e-3, rtol=4e-3)
    assert cache._cache.dtype == torch.bfloat16
    assert result.dtype == torch.float32
    cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("distance", [16383, 3])
@torch.no_grad()
def test_cuda_bf16_base_with_fp32_lora(distance):
    from mouse_core.models import LoRAConfig
    torch.manual_seed(72)
    model = Backbone(
        architecture="qwen3", dtype=torch.bfloat16, hidden_dim=32,
        num_layers=2, num_heads=2, num_key_value_heads=1,
        lora=LoRAConfig(rank=4, alpha=8),
        max_attention_distance_tokens=distance,
    ).cuda().eval()
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.normal_(std=0.02)
    cache = model.decode_session(2, max_batch_tokens=7, cache_capacity_tokens=128)
    x = torch.randn(2, 17, 32, device="cuda", dtype=torch.bfloat16)
    result = cache.forward(embeds=x, lengths=[17, 1])
    expected = reference(model, x[0])
    torch.testing.assert_close(result[0], expected, atol=0.04, rtol=0.02)
    assert cache._cache.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if "lora_" in n)
    cache.close()
