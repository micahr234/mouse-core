"""Decode performance paths preserve history, outputs and training parameters."""
import pytest
import torch
from test_continuous_decode import backbone, reference, PAGE_SIZE

@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
@torch.no_grad()
def test_deferred_history_preserves_positions_layers_and_reset(device):
    model = backbone(device=device)
    cache = model.decode_session(2, max_batch_tokens=3, cache_capacity_tokens=4 * PAGE_SIZE)
    histories = [torch.empty(0, 32, device=device) for _ in range(2)]
    retained = []
    for step in range(4):
        for row in range(2):
            prefix = torch.randn(row + 1, 32, device=device)
            cache.defer(row, prefix)
            histories[row] = torch.cat((histories[row], prefix))
        assert cache.lengths == tuple(map(len, histories))
        x = torch.randn(2, 4, 32, device=device)
        out, layers = cache.forward(embeds=x, lengths=[4, 4], output_hidden_states=True)
        for row in range(2):
            histories[row] = torch.cat((histories[row], x[row]))
            expected, expected_layers = reference(model, histories[row], layers=True)
            torch.testing.assert_close(out[row], expected[-4:], atol=2e-5, rtol=2e-5)
            for actual, expected in zip(layers, expected_layers, strict=True):
                torch.testing.assert_close(actual[row], expected[-4:], atol=2e-5, rtol=2e-5)
        retained.append((out, out.clone()))
    for out, copy in retained:
        torch.testing.assert_close(out, copy)
    cache.defer(0, torch.randn(1, 32, device=device))
    cache.reset_rows([0])
    assert 0 not in cache._deferred
    cache.reset_rows()
    assert cache.lengths == (0, 0)
    assert not cache._deferred
    cache.close()

@torch.no_grad()
def test_deferred_capacity_and_changed_weights():
    model = backbone()
    cache = model.decode_session(1, max_batch_tokens=3, cache_capacity_tokens=PAGE_SIZE)
    cache.defer(0, torch.randn(PAGE_SIZE, 32))
    with pytest.raises(ValueError, match='capacity'):
        cache.forward(embeds=torch.randn(1, 1, 32), lengths=[1])
    next(model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match='weights changed'):
        cache.defer(0, torch.randn(1, 32))
    cache.reset_rows()
    cache.defer(0, torch.randn(4, 32))
    cache.submit(0, torch.randn(1, 32))
    assert cache.lengths == (4,)
    assert cache.step() == {}
    assert cache.lengths == (4,)
    cache.step()
    assert cache.lengths == (5,)
    cache.close()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_graph_replay_updates_metadata_and_inputs_and_keeps_outputs():
    model = backbone(device='cuda', autocast_dtype=torch.bfloat16)
    cache = model.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=4 * PAGE_SIZE)
    histories = [torch.empty(0, 32, device='cuda') for _ in range(2)]
    saved = []
    for step in range(14):
        if step == 10:
            cache.reset_rows([1])
            histories[1] = histories[1][:0]
        x = torch.randn(2, 3, 32, device='cuda')
        out, layers = cache.forward(embeds=x, lengths=[3, 3], output_hidden_states=True)
        for row in range(2):
            histories[row] = torch.cat((histories[row], x[row]))
            expected, expected_layers = reference(model, histories[row], layers=True)
            torch.testing.assert_close(out[row], expected[-3:], atol=2e-2, rtol=2e-2)
            for actual, expected_layer in zip(layers, expected_layers, strict=True):
                torch.testing.assert_close(actual[row], expected_layer[-3:], atol=2e-2, rtol=2e-2)
        saved.append((out, out.clone()))
    assert cache._graphs
    for out, before in saved:
        torch.testing.assert_close(out, before)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for param in model.parameters():
        param.add_(.01)
    cache.reset_rows()
    assert not cache._graphs
    x = torch.randn(2, 3, 32, device='cuda')
    out = cache.forward(embeds=x, lengths=[3, 3])
    for row in range(2):
        torch.testing.assert_close(out[row], reference(model, x[row]), atol=2e-2, rtol=2e-2)
    cache.close()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.no_grad()
def test_autocast_decode_keeps_training_parameters_and_lora_unchanged():
    from mouse_core.models import Backbone, LoRAConfig
    torch.manual_seed(51)
    model = Backbone(architecture='qwen3', dtype=torch.float32, hidden_dim=32,
                     num_layers=2, num_heads=2, num_key_value_heads=1,
                     lora=LoRAConfig(rank=4, alpha=8),
                     train_autocast_dtype=torch.bfloat16,
                     decode_autocast_dtype=torch.bfloat16).cuda().eval()
    for name, p in model.named_parameters():
        if 'lora_B' in name:
            p.normal_(std=.02)
    originals = {name: (p.clone(), p.requires_grad, p.data_ptr()) for name, p in model.named_parameters()}
    cache = model.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=2 * PAGE_SIZE)
    x = torch.randn(2, 15, 32, device='cuda')
    out = cache.forward(embeds=x, lengths=[15, 7])
    for row, count in enumerate([15, 7]):
        torch.testing.assert_close(out[row, -count:], reference(model, x[row, -count:]), atol=.04, rtol=.02)
    for name, p in model.named_parameters():
        value, trainable, pointer = originals[name]
        torch.testing.assert_close(p, value, atol=0, rtol=0)
        assert p.requires_grad == trainable and p.data_ptr() == pointer
        assert p.dtype == torch.float32
    cache.close()


@pytest.mark.parametrize('change', ['parameter', 'child', 'add_parameter', 'remove_parameter'])
@torch.no_grad()
def test_cache_rejects_changed_parameter_registration(change):
    from copy import deepcopy
    model = backbone()
    cache = model.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=2 * PAGE_SIZE)
    try:
        x = torch.randn(1, 1, 32)
        cache.forward(embeds=x, lengths=[1])
        attention = cache.model.layers[0].self_attn
        if change == 'parameter':
            attention.q_proj.weight = torch.nn.Parameter(attention.q_proj.weight.clone())
        elif change == 'child':
            attention.q_proj = deepcopy(attention.q_proj)
        elif change == 'add_parameter':
            attention.register_parameter('extra', torch.nn.Parameter(torch.zeros(1)))
        else:
            attention.q_proj.register_parameter('weight', None)
        with pytest.raises(RuntimeError, match='weights changed'):
            cache.forward(embeds=x, lengths=[1])
    finally:
        cache.close()
