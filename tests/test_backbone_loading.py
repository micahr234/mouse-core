from __future__ import annotations
import warnings
from typing import Any, cast
import pytest
import torch
from transformers import LlamaConfig, LlamaModel, Qwen3Config, Qwen3Model
from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone

def _save_tiny_llama(tmp_path) -> LlamaModel:
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    source = LlamaModel(config)
    source.save_pretrained(tmp_path)
    return source

def test_llama_backbone_loads_pretrained_checkpoint(tmp_path) -> None:
    source = _save_tiny_llama(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        backbone = LlamaBackbone(pretrained=tmp_path, num_layers=1)
    assert backbone.hidden_dim == 8
    assert len(backbone.model.layers) == 1
    loaded_layer = cast(Any, backbone.model.layers[0])
    source_layer = cast(Any, source.layers[0])
    assert torch.equal(loaded_layer.self_attn.q_proj.weight, source_layer.self_attn.q_proj.weight)

def test_llama_backbone_warns_on_unloaded_tensors(tmp_path) -> None:
    _save_tiny_llama(tmp_path)
    with pytest.warns(UserWarning, match='did not receive pretrained weights') as records:
        LlamaBackbone(pretrained=tmp_path, intermediate_size=32)
    message = str(records[0].message)
    assert 'layers.0.mlp.gate_proj.weight' in message
    assert 'layers.1.mlp.down_proj.weight' in message
    assert 'self_attn.q_proj.weight' not in message

def test_llama_backbone_warns_on_unconsumed_pretrained_tensors(tmp_path) -> None:
    """A checkpoint tensor with no slot in the backbone must not vanish silently."""
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32, attention_bias=True)
    LlamaModel(config).save_pretrained(tmp_path)
    with pytest.warns(UserWarning, match='no matching backbone tensor') as records:
        LlamaBackbone(pretrained=tmp_path, attention_bias=False)
    messages = ' '.join(str(r.message) for r in records)
    assert 'layers.0.self_attn.q_proj.bias' in messages


def test_num_layers_truncation_does_not_warn_about_dropped_layers(tmp_path) -> None:
    _save_tiny_llama(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        LlamaBackbone(pretrained=tmp_path, num_layers=1)


def test_qwen3_backbone_rejects_non_qwen3_checkpoint(tmp_path) -> None:
    from transformers import Qwen2Config, Qwen2Model

    config = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    Qwen2Model(config).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="model_type='qwen3'"):
        Qwen3Backbone(pretrained=tmp_path)


def test_qwen3_backbone_direct_constructor_exposes_hidden_dim() -> None:
    backbone = Qwen3Backbone(hidden_dim=8, num_layers=1, num_heads=2)
    assert backbone.hidden_dim == 8


def test_qwen3_backbone_copies_pretrained_rope_parameters(tmp_path) -> None:
    """RoPE is computed from config, so pretrained rope_theta must be copied."""
    config = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        rope_parameters={"rope_theta": 123456.0, "rope_type": "default"},
    )
    Qwen3Model(config).save_pretrained(tmp_path)
    backbone = Qwen3Backbone(pretrained=tmp_path)
    assert backbone.model.config.rope_parameters["rope_theta"] == 123456.0
    assert backbone._config_kwargs["rope_parameters"]["rope_theta"] == 123456.0
