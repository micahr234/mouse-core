"""Cached decoding must reject changed parameters without rescanning LoRA groups."""
import pytest
import torch
from mouse_core.models import Backbone, LoRAConfig


@pytest.mark.parametrize('operation', ['forward', 'step'])
@pytest.mark.parametrize('change', ['dtype', 'parameter', 'child', 'weight'])
@torch.no_grad()
def test_cache_rejects_changed_parameters_or_modules(change, operation):
    backbone = Backbone(architecture='qwen3', dtype=torch.float32, hidden_dim=16,
        num_layers=1, num_heads=2, num_key_value_heads=1, head_dim=8,
        lora=LoRAConfig(rank=2)).eval()
    cache = backbone.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=256)
    try:
        x = torch.randn(1, 1, 16)
        backbone(x, cache=cache, lengths=[1])
        if operation == 'step':
            cache.submit(0, x[0])
        layer = backbone.model.layers[0].self_attn.q_proj
        if change == 'dtype':
            layer.base.weight.data = layer.base.weight.data.to(torch.float64)
        elif change == 'parameter':
            layer.lora_A.weight = torch.nn.Parameter(layer.lora_A.weight.detach().clone())
        elif change == 'child':
            layer.lora_A = torch.nn.Linear(16, 2, bias=False)
        else:
            layer.lora_B.weight.add_(.1)
        with pytest.raises(RuntimeError, match='weights changed'):
            if operation == 'step':
                cache.step()
            else:
                backbone(x, cache=cache, lengths=[1])
    finally:
        cache.close()
