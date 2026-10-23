"""Disabling RoPE preserves causal attention across all backbone forwards."""

from typing import Literal

import pytest
import torch
from transformers import GPT2Config

from mouse_core.models import Backbone, LoRAConfig
from mouse_core.models.backbone import TrainKernel


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def backbone(architecture: Literal["llama", "qwen3"], *, rope_enabled: bool,
             train_kernel: TrainKernel = "reference", lora: bool = False) -> Backbone:
    return Backbone(
        architecture=architecture, hidden_dim=16, num_layers=2, num_heads=2,
        num_key_value_heads=1, intermediate_size=32,
        train_kernel=train_kernel, decode_kernel="flex", dtype=torch.float32,
        rope_enabled=rope_enabled,
        lora=LoRAConfig(rank=2, alpha=2.0, dropout=0.0) if lora else None,
    )


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
@pytest.mark.parametrize("train_kernel", ["reference", "padded"])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_disabled_rope_matches_zero_position_reference_and_gradients(
    architecture, train_kernel, checkpoint,
):
    torch.manual_seed(16)
    disabled = backbone(architecture, rope_enabled=False, train_kernel=train_kernel).eval()
    disabled.gradient_checkpointing = checkpoint
    reference = backbone(architecture, rope_enabled=True).eval()
    reference.load_state_dict(disabled.state_dict())
    x = torch.randn(6, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    ids = torch.tensor([0, 1, 0, 1, 0, 1])
    # Zero positions make the original RoPE an identity, without replacing it.
    expected = torch.empty_like(x)
    for sequence in (0, 1):
        selected = ids == sequence
        expected[selected] = reference(
            reference_x[selected].unsqueeze(0), position_ids=torch.zeros(1, 3, dtype=torch.long),
        )[0]
    actual = disabled(x, sequence_ids=ids)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=2e-6, rtol=2e-5)
    for (_, parameter), (_, ref_parameter) in zip(disabled.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_disabled_rope_flex_forward_matches_reference(architecture):
    model = backbone(architecture, rope_enabled=False).eval()
    x = torch.randn(6, 16)
    ids = torch.tensor([0, 1, 0, 1, 0, 1])
    with torch.no_grad():
        expected = model(x, sequence_ids=ids)
        model.train_kernel = "flex"
        torch.testing.assert_close(model(x, sequence_ids=ids), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_position_ids_affect_only_enabled_rope(architecture):
    torch.manual_seed(25)
    disabled = backbone(architecture, rope_enabled=False).eval()
    enabled = backbone(architecture, rope_enabled=True).eval()
    enabled.load_state_dict(disabled.state_dict())
    x = torch.randn(1, 5, 16)
    normal = torch.arange(5).unsqueeze(0)
    irregular = torch.tensor([[0, 5, 19, 43, 103]])
    with torch.no_grad():
        torch.testing.assert_close(disabled(x, position_ids=normal), disabled(x, position_ids=irregular), rtol=0, atol=0)
        assert not torch.allclose(enabled(x, position_ids=normal), enabled(x, position_ids=irregular))


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
@pytest.mark.parametrize("lora", [False, True])
def test_disabled_rope_cached_ragged_chunks_and_reset_match_full_forward(architecture, lora):
    torch.manual_seed(8)
    model = backbone(architecture, rope_enabled=False, lora=lora).eval()
    rows = [torch.randn(5, 16), torch.randn(3, 16)]
    with torch.no_grad():
        expected = [model(row, sequence_ids=torch.zeros(len(row), dtype=torch.long)) for row in rows]
        session = model.decode_session(batch_size=2)
        try:
            # Unequal prefill lengths followed by incremental steps exercise KV positions.
            prefill = torch.zeros(2, 3, 16)
            prefill[0] = rows[0][:3]
            prefill[1, -1:] = rows[1][:1]
            first = model(prefill, cache=session, lengths=[3, 1])
            torch.testing.assert_close(first[0], expected[0][:3], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(first[1, -1:], expected[1][:1], atol=2e-6, rtol=2e-5)
            for i in range(2):
                chunk = torch.stack([rows[0][3 + i], rows[1][1 + i]])[:, None]
                actual = model(chunk, cache=session, lengths=[1, 1])
                torch.testing.assert_close(actual[0, 0], expected[0][3 + i], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual[1, 0], expected[1][1 + i], atol=2e-6, rtol=2e-5)
            session.reset_rows([0])
            actual = model(rows[0][:1].expand(2, -1, -1), cache=session, lengths=[1, 0])
            torch.testing.assert_close(actual[0, 0], expected[0][0], atol=2e-6, rtol=2e-5)
        finally:
            session.close()


@pytest.mark.parametrize("value", [None, 0, 1, "false"])
def test_rope_enabled_requires_boolean(value):
    with pytest.raises(TypeError, match="rope_enabled must be a boolean"):
        backbone("qwen3", rope_enabled=value)


def test_disabling_rope_rejects_non_rotary_decoder():
    config = GPT2Config(n_embd=8, n_layer=1, n_head=2, n_inner=16, vocab_size=16)
    with pytest.raises(ValueError, match="packed-compatible rotary decoder"):
        Backbone(architecture="hf", hf_config=config.to_dict(), rope_enabled=False,
                 train_kernel="reference", decode_kernel="flex", dtype=torch.float32)
