"""Compiled decoder body matches the eager decoder (no Hub download, no GPU)."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import LlamaConfig, LlamaModel, Qwen3Config, Qwen3Model

from mouse_core.models.backbone import flex_train as flex_train_mod
from mouse_core.models.backbone.flex_train import install_compiled_decoder

os.environ.setdefault("PYTHON_GIL", "0")


def _dummy_flex(q, k, v, block_mask=None, scale=1.0, enable_gqa=True):
    """Stand-in for Flex so this test only checks the compiled decoder body."""
    return q * (1.0 if scale is None else scale)


@pytest.fixture(scope="module")
def tiny_qwen() -> Qwen3Model:
    torch.manual_seed(0)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
    )
    return Qwen3Model(config).eval()


@pytest.fixture(scope="module")
def tiny_llama() -> LlamaModel:
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    )
    return LlamaModel(config).eval()


def _decoder_out(hf, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(length)
    x = torch.randn(1, length, hf.config.hidden_size)
    pos = torch.arange(length).unsqueeze(0)
    cos, sin = hf.rotary_emb(x, pos)
    with torch.no_grad():
        eager = flex_train_mod._run_decoder_layers(hf, x, cos, sin, None, _dummy_flex)
    return x, cos, sin, eager


def test_install_compiled_decoder_idempotent() -> None:
    was = flex_train_mod._compiled_decoder
    try:
        flex_train_mod._compiled_decoder = None
        assert install_compiled_decoder() is True
        assert flex_train_mod._compiled_decoder is not None
        compiled = flex_train_mod._compiled_decoder
        assert install_compiled_decoder() is False
        assert flex_train_mod._compiled_decoder is compiled
    finally:
        flex_train_mod._compiled_decoder = was


@pytest.mark.parametrize("fixture_name", ["tiny_qwen", "tiny_llama"])
def test_compiled_decoder_matches_eager(request: pytest.FixtureRequest, fixture_name: str) -> None:
    hf = request.getfixturevalue(fixture_name)
    x, cos, sin, eager = _decoder_out(hf, 6)
    compiled = torch.compile(flex_train_mod._run_decoder_layers, dynamic=True)
    with torch.no_grad():
        compiled(hf, x, cos, sin, None, _dummy_flex)
        out = compiled(hf, x, cos, sin, None, _dummy_flex)
    torch.testing.assert_close(out, eager, rtol=1e-4, atol=1e-4)

    x2, cos2, sin2, eager2 = _decoder_out(hf, 9)
    with torch.no_grad():
        out2 = compiled(hf, x2, cos2, sin2, None, _dummy_flex)
    torch.testing.assert_close(out2, eager2, rtol=1e-4, atol=1e-4)
