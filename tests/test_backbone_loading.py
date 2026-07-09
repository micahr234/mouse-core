from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone


def _save_tiny_llama(tmp_path) -> LlamaModel:
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    source = LlamaModel(config)
    source.save_pretrained(tmp_path)
    return source


def test_llama_backbone_loads_pretrained_checkpoint(tmp_path) -> None:
    source = _save_tiny_llama(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a full load must not warn
        backbone = LlamaBackbone(pretrained=tmp_path, num_layers=1)

    assert backbone.hidden_dim == 8
    assert len(backbone.model.layers) == 1
    loaded_layer = cast(Any, backbone.model.layers[0])
    source_layer = cast(Any, source.layers[0])
    assert torch.equal(
        loaded_layer.self_attn.q_proj.weight,
        source_layer.self_attn.q_proj.weight,
    )


def test_llama_backbone_warns_on_unloaded_tensors(tmp_path) -> None:
    _save_tiny_llama(tmp_path)

    # Overriding intermediate_size makes every MLP weight shape-mismatched,
    # so those tensors keep their random init and must be reported.
    with pytest.warns(UserWarning, match="did not receive pretrained weights") as records:
        LlamaBackbone(pretrained=tmp_path, intermediate_size=32)

    message = str(records[0].message)
    assert "layers.0.mlp.gate_proj.weight" in message
    assert "layers.1.mlp.down_proj.weight" in message
    # Attention weights still match and must not be listed.
    assert "self_attn.q_proj.weight" not in message


def test_qwen3_backbone_direct_constructor_exposes_hidden_dim() -> None:
    backbone = Qwen3Backbone(hidden_dim=8, num_layers=1, num_heads=2)

    assert backbone.hidden_dim == 8
