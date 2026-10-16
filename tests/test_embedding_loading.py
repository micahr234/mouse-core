"""Standalone table loading, independent of decoder construction and training."""
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from transformers import (
    AutoModel, GPT2Config, GPT2Model, LlamaConfig, LlamaModel,
    Qwen3Config, Qwen3ForCausalLM, PreTrainedModel,
)

from mouse_core.models import Backbone, Embedding, LoRAConfig


def _source(kind) -> PreTrainedModel:
    if kind == "gpt2":
        return GPT2Model(GPT2Config(vocab_size=16, n_embd=8, n_layer=1,
                                   n_head=2, n_inner=16, n_positions=16))
    config: dict[str, Any] = dict(vocab_size=16, hidden_size=8, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=2, intermediate_size=16)
    if kind == "llama":
        return LlamaModel(LlamaConfig(**config))
    return Qwen3ForCausalLM(Qwen3Config(**config, head_dim=4))


@pytest.mark.parametrize("kind", ["llama", "gpt2", "qwen3_lm"])
@pytest.mark.parametrize("sharded", [False, True])
def test_embedding_loads_only_the_table_without_allocating_a_decoder(tmp_path, monkeypatch, kind, sharded):
    source = _source(kind)
    cast(torch.nn.Module, source).to(dtype=torch.bfloat16)
    source.save_pretrained(tmp_path, max_shard_size="1KB" if sharded else "1GB")
    table = source.get_input_embeddings()
    assert isinstance(table, torch.nn.Embedding)
    expected = table.weight.detach().float().clone()
    build = AutoModel.from_config
    inspected = []

    def meta_layout(*args, **kwargs):
        model = build(*args, **kwargs)
        assert all(p.device.type == "meta" for p in model.parameters())
        inspected.append(True)
        return model

    def no_full_load(*args, **kwargs):
        raise AssertionError("loading an embedding must not load a decoder")

    monkeypatch.setattr(AutoModel, "from_config", meta_layout)
    monkeypatch.setattr(AutoModel, "from_pretrained", no_full_load)
    from transformers import modeling_utils
    load_state_dict = modeling_utils.load_state_dict
    loaded_files = []

    def load_shard(path, *args, **kwargs):
        loaded_files.append(Path(path).name)
        return load_state_dict(path, *args, **kwargs)

    monkeypatch.setattr(modeling_utils, "load_state_dict", load_shard)
    embedding = Embedding(pretrained=tmp_path, num_frequencies=2,
                          hub_kwargs={"local_files_only": True})
    assert inspected == [True]
    assert len(loaded_files) == 1
    if sharded:
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
        table_key = {"gpt2": "wte.weight", "llama": "embed_tokens.weight",
                     "qwen3_lm": "model.embed_tokens.weight"}[kind]
        assert loaded_files == [index["weight_map"][table_key]]
    torch.testing.assert_close(embedding.embed_tokens.weight, expected, atol=0, rtol=0)
    assert embedding.hidden_dim == 8 and embedding.vocab_size == 16
    assert all(p.dtype == torch.float32 and p.requires_grad for p in embedding.parameters())
    assert set(embedding.state_dict()) == {"embed_tokens.weight", "fourier.proj.weight"}
    ids = torch.tensor([2, 5, 2])
    embedding.embed_tokens(ids).sum().backward()
    grad = embedding.embed_tokens.weight.grad
    assert grad is not None
    torch.testing.assert_close(grad[2], torch.full((8,), 2.))


def test_embedding_supports_local_pytorch_weights_and_checks_dimensions(tmp_path):
    source = _source("llama")
    source.config.save_pretrained(tmp_path)
    torch.save(source.state_dict(), tmp_path / "pytorch_model.bin")
    embedding = Embedding(pretrained=tmp_path)
    torch.testing.assert_close(embedding.embed_tokens.weight, source.get_input_embeddings().weight)
    with pytest.raises(ValueError, match="does not match"):
        Embedding(pretrained=tmp_path, hidden_dim=4)
    with pytest.raises(ValueError, match="does not match"):
        Embedding(pretrained=tmp_path, vocab_size=7)


def test_embedding_trainability_is_independent_of_backbone_lora():
    embedding = Embedding(hidden_dim=8, vocab_size=16, num_frequencies=2)
    backbone = Backbone(architecture="qwen3", hidden_dim=8, num_layers=1,
                        num_heads=2, train_kernel="reference", decode_kernel="flex",
                        dtype=torch.bfloat16, lora=LoRAConfig(rank=2))
    assert any(not p.requires_grad for p in backbone.parameters())
    assert all(p.requires_grad for p in embedding.parameters())
    embedding.embed_tokens.requires_grad_(False)
    assert embedding.fourier is not None and embedding.fourier.proj.weight.requires_grad
    assert any(p.requires_grad for p in backbone.parameters())
