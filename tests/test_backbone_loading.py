from __future__ import annotations

from typing import Any, cast

import torch
from transformers import LlamaConfig, LlamaModel, ModernBertConfig, ModernBertModel

from mouse_core.models.backbone import LlamaBackbone, ModernBertBackbone, Qwen3Backbone


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


def test_modernbert_backbone_is_bidirectional_and_rejects_cache() -> None:
    backbone = ModernBertBackbone(hidden_dim=8, num_layers=2, num_heads=2)
    embeds = torch.randn(1, 3, 8)

    out_full, cache = backbone(embeds)
    assert cache is None
    assert out_full.shape == (1, 3, 8)

    embeds_long = torch.cat([embeds, torch.randn(1, 1, 8)], dim=1)
    out_long, _ = backbone(embeds_long)
    assert not torch.allclose(out_full[0, 0], out_long[0, 0])

    try:
        backbone(embeds, use_cache=True)
    except ValueError as exc:
        assert "KV caching" in str(exc)
    else:
        raise AssertionError("expected ValueError for use_cache=True")


def test_modernbert_backbone_loads_pretrained_checkpoint(tmp_path) -> None:
    config = ModernBertConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        cls_token_id=None,
        sep_token_id=None,
    )
    source = ModernBertModel(config)
    source.save_pretrained(tmp_path)

    backbone = ModernBertBackbone(pretrained=tmp_path)

    assert backbone.hidden_dim == 8
    assert len(backbone.model.layers) == 2
    loaded_layer = cast(Any, backbone.model.layers[0])
    source_layer = cast(Any, source.layers[0])
    assert torch.equal(
        loaded_layer.attn.Wqkv.weight,
        source_layer.attn.Wqkv.weight,
    )
