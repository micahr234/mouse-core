"""Qwen-compatible MLP structure, scaling, and gradient contract."""
import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from mouse_core.models import SequentialReasoner


def make_reasoner(**overrides):
    settings = dict(hidden_dim=8, input_mlp_intermediate_dim=24, input_mlp_output_scale=100.,
                    min_steps=1, max_steps=3, rms_norm_eps=1e-6,
                    residual_connections=True, wrap_latent="sequence", adaptive_depth=True)
    return SequentialReasoner(**(settings | overrides))


def test_input_matches_qwen_mlp_without_a_residual_and_preserves_gradients():
    torch.manual_seed(113)
    reasoner = make_reasoner()
    reference = Qwen3MLP(Qwen3Config(hidden_size=8, intermediate_size=24, hidden_act="silu"))
    reference.load_state_dict(reasoner.input_projection.state_dict(), strict=True)
    x = torch.randn(5, 8, requires_grad=True)
    expected = reference(reasoner.norm(x)) * reasoner.input_mlp_output_scale + reasoner.start_embedding.weight
    actual = reasoner.project_inputs(x)
    torch.testing.assert_close(actual, expected)
    expected_gradient, = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, expected_gradient)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in reasoner.input_projection.parameters())
    assert reasoner.norm.weight.grad.abs().sum() > 0
    with torch.no_grad():
        reasoner.input_projection.down_proj.weight.zero_()
    torch.testing.assert_close(reasoner.project_inputs(x), reasoner.start_embedding.weight.expand_as(x))


def test_scale_is_static_and_applied_in_generation_input_function():
    reasoner = make_reasoner(input_mlp_output_scale=1.)
    x = torch.randn(2, 8)
    before = reasoner.project_inputs(x)
    reasoner.input_mlp_output_scale = 17.
    torch.testing.assert_close(reasoner.project_inputs(x), (before - reasoner.start_embedding.weight) * 17. + reasoner.start_embedding.weight)


@pytest.mark.parametrize('scale', [None, True, 0., -1., float('inf'), float('nan')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match='input_mlp_output_scale'):
        make_reasoner(input_mlp_output_scale=scale)


@pytest.mark.parametrize('width', [None, True, 0, -1, 2.5])
def test_invalid_intermediate_width(width):
    with pytest.raises(ValueError, match='input_mlp_intermediate_dim'):
        make_reasoner(input_mlp_intermediate_dim=width)


def test_required_mlp_settings():
    with pytest.raises(TypeError, match='input_mlp_intermediate_dim.*input_mlp_output_scale'):
        SequentialReasoner(hidden_dim=8, min_steps=1, max_steps=1, rms_norm_eps=1e-6,
                           residual_connections=False, wrap_latent="none", adaptive_depth=True)


def test_zero_feedback_injects_exact_live_embedding_at_every_token():
    reasoner = make_reasoner()
    zeros = torch.zeros(4, 8, requires_grad=True)
    output = reasoner.project_inputs(zeros)
    torch.testing.assert_close(output, reasoner.start_embedding.weight.expand(4, -1))
    output.sum().backward()
    torch.testing.assert_close(reasoner.start_embedding.weight.grad, torch.full((1, 8), 4.))
    assert reasoner.norm.weight.grad.count_nonzero() == 0
    assert all(p.grad.count_nonzero() == 0 for p in reasoner.input_projection.parameters())
    assert zeros.grad.count_nonzero() == 0
