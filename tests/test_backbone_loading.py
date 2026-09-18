from __future__ import annotations
import warnings
from typing import Any, cast
import pytest
import torch
from transformers import LlamaConfig, LlamaModel, Qwen3Config, Qwen3Model
from mouse_core.models.backbone import TransformerBackbone

def _save_tiny_llama(tmp_path) -> LlamaModel:
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    source = LlamaModel(config)
    source.save_pretrained(tmp_path)
    return source

def test_llama_backbone_loads_pretrained_checkpoint(tmp_path) -> None:
    source = _save_tiny_llama(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        backbone = TransformerBackbone(train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path, num_layers=1)
    assert backbone.hidden_dim == 8
    assert len(backbone.model.layers) == 1
    loaded_layer = cast(Any, backbone.model.layers[0])
    source_layer = cast(Any, source.layers[0])
    assert torch.equal(loaded_layer.self_attn.q_proj.weight, source_layer.self_attn.q_proj.weight)

def test_backbones_keep_and_load_the_final_norm(tmp_path) -> None:
    source = _save_tiny_llama(tmp_path)
    with torch.no_grad():
        source.norm.weight.fill_(3.0)
    source.save_pretrained(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        backbone = TransformerBackbone(train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path)
    assert type(backbone.model.norm).__name__.endswith('RMSNorm')
    assert torch.equal(backbone.model.norm.weight, torch.full((8,), 3.0))
    # Output is the residual stream through that norm: per-token RMS equals the gain.
    with torch.no_grad():
        out = backbone(torch.randn(1, 5, 8) * 50.0)
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.full((1, 5), 3.0), atol=1e-3)
    qwen = TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, hidden_dim=8, num_layers=1, num_heads=2)
    assert type(qwen.model.norm).__name__.endswith('RMSNorm')


def test_backbones_can_drop_the_final_norm(tmp_path) -> None:
    source = _save_tiny_llama(tmp_path)
    with torch.no_grad():
        source.norm.weight.fill_(3.0)
    source.save_pretrained(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        backbone = TransformerBackbone(
            train_kernel="reference", decode_kernel="flex", dtype=torch.float32,
            use_norm=False, pretrained=tmp_path)
    assert isinstance(backbone.model.norm, torch.nn.Identity)
    assert "norm.weight" not in backbone.state_dict()
    assert backbone._config_kwargs["use_norm"] is False
    with torch.no_grad():
        out = backbone(torch.randn(1, 5, 8) * 50.0)
    assert not torch.allclose(out.pow(2).mean(-1).sqrt(), torch.full((1, 5), 3.0), atol=1e-3)
    qwen = TransformerBackbone(architecture="qwen3", 
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32,
        use_norm=False, hidden_dim=8, num_layers=1, num_heads=2,
    )
    assert isinstance(qwen.model.norm, torch.nn.Identity)
    assert qwen._config_kwargs["use_norm"] is False


def test_transformer_backbone_requires_use_norm() -> None:
    with pytest.raises(TypeError, match="use_norm"):
        TransformerBackbone(architecture="qwen3", 
            train_kernel="reference", decode_kernel="flex", dtype=torch.float32,
            hidden_dim=8, num_layers=1, num_heads=2,
        )


def test_llama_backbone_warns_on_unloaded_tensors(tmp_path) -> None:
    _save_tiny_llama(tmp_path)
    with pytest.warns(UserWarning, match='did not receive pretrained weights') as records:
        TransformerBackbone(train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path, intermediate_size=32)
    message = str(records[0].message)
    assert 'layers.0.mlp.gate_proj.weight' in message
    assert 'layers.1.mlp.down_proj.weight' in message
    assert 'self_attn.q_proj.weight' not in message

def test_llama_backbone_warns_on_unconsumed_pretrained_tensors(tmp_path) -> None:
    """A checkpoint tensor with no slot in the backbone must not vanish silently."""
    config = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32, attention_bias=True)
    LlamaModel(config).save_pretrained(tmp_path)
    with pytest.warns(UserWarning, match='no matching backbone tensor') as records:
        TransformerBackbone(train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path, attention_bias=False)
    messages = ' '.join(str(r.message) for r in records)
    assert 'layers.0.self_attn.q_proj.bias' in messages


def test_num_layers_truncation_does_not_warn_about_dropped_layers(tmp_path) -> None:
    _save_tiny_llama(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        TransformerBackbone(train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path, num_layers=1)


def test_qwen2_checkpoint_uses_packed_kernels(tmp_path) -> None:
    from transformers import Qwen2Config, Qwen2Model

    config = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    Qwen2Model(config).save_pretrained(tmp_path)
    backbone = TransformerBackbone(
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path)
    assert backbone.architecture == "hf"
    assert backbone.uses_packed is True
    assert backbone.hidden_dim == 8


def test_hybrid_linear_attention_is_rejected(tmp_path) -> None:
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "layer_types": ["linear_attention", "full_attention"],
            }
        )
    )
    with pytest.raises(ValueError, match="hybrid linear/full-attention"):
        TransformerBackbone(
            train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path,
        )


def test_generic_softmax_decoder_uses_hf_forward(tmp_path) -> None:
    from transformers import GPT2Config, GPT2Model

    config = GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=2, n_inner=16, n_positions=32)
    GPT2Model(config).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="train_kernel='reference'"):
        TransformerBackbone(
            train_kernel="flex", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path,
        )
    backbone = TransformerBackbone(
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path)
    assert backbone.uses_packed is False
    assert backbone.architecture == "hf"
    with torch.no_grad():
        out = backbone(torch.randn(1, 5, 8))
    assert out.shape == (1, 5, 8)
    with pytest.raises(NotImplementedError, match="cached decode"):
        backbone.decode_session(batch_size=1)


def test_architecture_is_required_without_pretrained() -> None:
    with pytest.raises(TypeError, match="pretrained= or architecture="):
        TransformerBackbone(
            train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True,
            hidden_dim=8, num_layers=1, num_heads=2,
        )


def test_qwen3_backbone_direct_constructor_exposes_hidden_dim() -> None:
    backbone = TransformerBackbone(architecture="qwen3", train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, hidden_dim=8, num_layers=1, num_heads=2)
    assert backbone.hidden_dim == 8
    assert backbone.uses_packed is True
    assert backbone.architecture == "qwen3"


def test_generic_hf_backbone_roundtrip(tmp_path) -> None:
    from transformers import GPT2Config, GPT2Model
    from mouse_core.models import Model, load_model, save_model
    from mouse_core.models.heads import RegressionHead

    config = GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=2, n_inner=16, n_positions=32)
    GPT2Model(config).save_pretrained(tmp_path / "gpt2")
    backbone = TransformerBackbone(
        train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True,
        pretrained=tmp_path / "gpt2",
    )
    model = Model(
        backbone=backbone,
        heads=(head := RegressionHead(in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True)),
        action_source=head,
        reasoner=None,
    )
    save_model(model=model, path=tmp_path / "ckpt")
    loaded = load_model(repo_id_or_path=tmp_path / "ckpt", train_kernel="reference", decode_kernel="flex", dtype=torch.float32)
    loaded_bb = cast(TransformerBackbone, loaded.backbone)
    assert loaded_bb.architecture == "hf"
    assert loaded_bb.uses_packed is False
    assert loaded_bb.hidden_dim == 8


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
    backbone = TransformerBackbone(train_kernel="reference", decode_kernel="flex", dtype=torch.float32, use_norm=True, pretrained=tmp_path)
    rope = backbone.model.config.rope_parameters
    assert rope is not None
    assert rope["rope_theta"] == 123456.0
    assert backbone._config_kwargs["rope_parameters"]["rope_theta"] == 123456.0
