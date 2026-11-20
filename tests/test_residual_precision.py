"""Independent transformer residual precision across packed and cached execution."""
from contextlib import closing
import pytest
import torch
from torch import nn
from mouse_core.models import Backbone, LoRAConfig
from mouse_core.models.backbone.packed_train import _attention_output


def make_backbone(residual_dtype, *, dtype=torch.bfloat16, autocast=None, lora=False):
    torch.manual_seed(317)
    return Backbone(architecture="qwen3", dtype=dtype, residual_dtype=residual_dtype,
                    autocast_dtype=autocast,
                    hidden_dim=32, num_layers=2, num_heads=2, num_key_value_heads=1,
                    intermediate_size=64, lora=LoRAConfig(rank=2, alpha=2.) if lora else None).eval()


@pytest.mark.parametrize("residual_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("base_dtype,autocast", [(torch.bfloat16, None),
                                              (torch.bfloat16, torch.bfloat16),
                                              (torch.float32, torch.bfloat16)])
@torch.no_grad()
def test_residual_precision_packed_dense_cached_and_deferred(residual_dtype, base_dtype, autocast):
    model = make_backbone(residual_dtype, dtype=base_dtype, autocast=autocast, lora=True)
    x = torch.randn(6, 32)
    packed, hidden = model(x, sequence_ids=torch.zeros(6, dtype=torch.long), output_hidden_states=True)
    dense = model(x[None])
    assert packed.dtype == residual_dtype and all(h.dtype == residual_dtype for h in hidden)
    torch.testing.assert_close(dense[0], packed, rtol=0, atol=0)
    assert model.dtype == base_dtype
    assert all(p.dtype == (torch.float32 if '.lora_' in n else base_dtype)
               for n, p in model.named_parameters())
    with closing(model.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=256)) as session:
        session.defer(0, x[:2])
        actual, layers = model(x[None, 2:], cache=session, lengths=[4], output_hidden_states=True)
        assert session.residual_dtype == residual_dtype
        assert actual.dtype == residual_dtype and all(h.dtype == residual_dtype for h in layers)
        torch.testing.assert_close(actual[0], packed[2:], rtol=0, atol=0)
        allocator = session._cache.layer_to_allocator[0]
        key, value = allocator._kv_token_views[0]
        assert key.dtype == value.dtype == (autocast or base_dtype)


def test_residual_addition_keeps_small_updates_and_gradients_through_frozen_layers():
    from types import SimpleNamespace
    # Zero MLP isolates the attention residual addition.
    class ZeroMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        def forward(self, x):
            return x * 0
    projection = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    projection.weight.data.copy_(torch.eye(4, dtype=torch.bfloat16))
    projection.requires_grad_(False)
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=projection),
                            mlp=ZeroMLP(), post_attention_layernorm=nn.Identity())
    branch = torch.full((1, 1, 2, 4), .001, dtype=torch.bfloat16, requires_grad=True)
    low = _attention_output(layer, torch.ones(2, 4, dtype=torch.bfloat16), branch)
    high = _attention_output(layer, torch.ones(2, 4), branch)
    assert torch.equal(low, torch.ones_like(low))
    assert torch.all(high > 1)
    high.sum().backward()
    torch.testing.assert_close(branch.grad, torch.ones_like(branch))
    assert projection.weight.grad is None


@pytest.mark.parametrize("invalid", ["float32", torch.int32, 32])
def test_invalid_residual_precision_fails_at_construction(invalid):
    with pytest.raises(ValueError, match="residual_dtype"):
        make_backbone(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA")
def test_fp32_residuals_with_bf16_frozen_backbone_train_and_decode_on_cuda():
    model = make_backbone(torch.float32, lora=True).cuda()
    x = torch.randn(8, 32, device="cuda", requires_grad=True)
    out = model(x, sequence_ids=torch.zeros(8, dtype=torch.long, device="cuda"))
    assert out.dtype == torch.float32
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if '.lora_' in n)
    assert all(p.grad is None for n,p in model.named_parameters() if '.lora_' not in n)
    with torch.no_grad(), closing(model.decode_session(1, max_batch_tokens=16, cache_capacity_tokens=256)) as session:
        decoded = model(x.detach()[None], cache=session, lengths=[8])
        torch.testing.assert_close(decoded[0], out.detach(), rtol=.02, atol=.02)


@pytest.mark.parametrize("dtype,autocast,residual", [
    (torch.float32, None, torch.float32),
    (torch.bfloat16, None, torch.bfloat16),
    (torch.float32, torch.bfloat16, torch.float32),
    (torch.bfloat16, None, torch.float32),
])
@torch.no_grad()
def test_every_layer_uses_same_precision_and_weights_in_train_and_decode(dtype, autocast, residual):
    model = make_backbone(residual, dtype=dtype, autocast=autocast, lora=True)
    seen = {}
    def capture(name):
        def hook(module, args, output):
            if isinstance(output, torch.Tensor):
                seen[name] = (tuple(x.dtype for x in args if isinstance(x, torch.Tensor)),
                              output.dtype, tuple((id(p), p.dtype) for p in module.parameters()))
        return hook
    from mouse_core.models.lora import LoRALinear
    hooks = [module.register_forward_hook(capture(name)) for name, module in model.named_modules()
             if isinstance(module, LoRALinear) or not list(module.children())]
    before = [(p, p.clone(), p.data_ptr(), p.dtype) for p in model.parameters()]
    x = torch.randn(4, 32)
    try:
        model.train()
        expected = model(x, sequence_ids=torch.zeros(4, dtype=torch.long))
        train_seen = seen.copy()
        seen.clear()
        model.eval()
        with closing(model.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=256)) as cache:
            actual = model(x[None], cache=cache, lengths=[4])[0]
            assert cache.autocast_dtype == model.autocast_dtype == autocast
        assert seen == train_seen
        assert any(name.endswith('q_proj') for name in seen)
        assert any('layernorm' in name for name in seen)
        # Packed and cached attention can differ by FP32 reduction-order rounding.
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        for parameter, value, pointer, stored_dtype in before:
            assert parameter.dtype == stored_dtype and parameter.data_ptr() == pointer
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("obsolete", ["train_autocast_dtype", "decode_autocast_dtype"])
def test_separate_training_or_decode_precision_is_rejected(obsolete):
    with pytest.raises(TypeError, match="separate precision"):
        Backbone(architecture="qwen3", dtype=torch.float32, **{obsolete: torch.bfloat16})


@pytest.mark.parametrize("field,value", [("autocast_dtype", torch.bfloat16),
                                        ("residual_dtype", torch.bfloat16)])
def test_decode_session_rejects_changed_precision(field, value):
    model = make_backbone(torch.float32, dtype=torch.float32)
    with closing(model.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=256)) as cache:
        setattr(model, field, value)
        with pytest.raises(RuntimeError, match="Backbone precision changed"):
            cache.submit(0, torch.zeros(1, 32))
        with pytest.raises(ValueError, match="precision must match"):
            model(torch.zeros(1, 1, 32), cache=cache, lengths=[1])


def test_decode_cache_cannot_supply_another_models_weights():
    model = make_backbone(torch.float32, dtype=torch.float32)
    other = make_backbone(torch.float32, dtype=torch.float32)
    with closing(other.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=256)) as cache:
        with pytest.raises(ValueError, match="this backbone's weights"):
            model(torch.zeros(1, 1, 32), cache=cache, lengths=[1])


@pytest.mark.parametrize("path", ["packed", "dense", "cached", "deferred"])
@pytest.mark.parametrize("residual_dtype", [torch.float32, torch.float64])
@torch.no_grad()
def test_embedding_values_enter_residual_stream_without_bf16_round_trip(path, residual_dtype):
    model = make_backbone(residual_dtype, dtype=torch.bfloat16)
    # Zero the branches so the transformer must preserve the incoming residual
    # exactly. These embedding values collapse to 1.0 if rounded through BF16.
    for layer in model.model.layers:
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    table = nn.Embedding(4, 32, dtype=residual_dtype)
    table.weight.copy_(1 + torch.arange(1, 129, dtype=residual_dtype).reshape(4, 32) / 100000)
    embeds = table(torch.arange(4))
    assert not torch.equal(embeds, embeds.bfloat16().to(residual_dtype))
    if path == "packed":
        actual = model(embeds, sequence_ids=torch.zeros(4, dtype=torch.long))
    elif path == "dense":
        actual = model(embeds[None])[0]
    else:
        with closing(model.decode_session(1, max_batch_tokens=8, cache_capacity_tokens=256)) as cache:
            if path == "deferred":
                cache.defer(0, embeds[:2])
                actual = model(embeds[None, 2:], cache=cache, lengths=[2])[0]
                embeds = embeds[2:]
            else:
                prefix = model(embeds[None, :2], cache=cache, lengths=[2])[0]
                suffix = model(embeds[None, 2:], cache=cache, lengths=[2])[0]
                actual = torch.cat((prefix, suffix))
    assert actual.dtype == residual_dtype
    torch.testing.assert_close(actual, embeds, rtol=0, atol=0)
