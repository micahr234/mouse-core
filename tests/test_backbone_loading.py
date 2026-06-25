from __future__ import annotations

from typing import Any, cast

import torch
from transformers import LlamaConfig, LlamaModel

from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone


def test_llama_backbone_loads_pretrained_checkpoint(tmp_path) -> None:
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

    backbone = LlamaBackbone(pretrained=tmp_path, num_layers=1)

    assert backbone.hidden_dim == 8
    assert len(backbone.model.layers) == 1
    loaded_layer = cast(Any, backbone.model.layers[0])
    source_layer = cast(Any, source.layers[0])
    assert torch.equal(
        loaded_layer.self_attn.q_proj.weight,
        source_layer.self_attn.q_proj.weight,
    )


def test_qwen3_backbone_direct_constructor_exposes_hidden_dim() -> None:
    backbone = Qwen3Backbone(hidden_dim=8, num_layers=1, num_heads=2)

    assert backbone.hidden_dim == 8
