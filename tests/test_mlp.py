from __future__ import annotations
from tests._model_helpers import fresh_model, ExampleModel as Model
from tests._model_helpers import simple_model
import copy
import json
import pytest
import torch
from mouse_core.models import load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.mlp import MLP, ResidualBlock, SwiGLU
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

_TOK = token_tokenizer("action")

def test_mlp_preserves_batch_dimensions() -> None:
    head = MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0)
    out = head(torch.randn(2, 5, 8))
    assert out.shape == (2, 5, 4)

def test_output_name_is_explicit() -> None:
    head = MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.)
    model = simple_model(backbone=IdentityBackbone(hidden_dim=8), heads={"custom_action": head}, reasoner=None)
    assert tuple(model.heads) == ("custom_action",)

@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("normalize_input,normalize_output", [(True, False), (True, True), (False, True), (False, False)])
def test_mlp_prediction_save_load_roundtrip(tmp_path, num_layers, normalize_input, normalize_output) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    head = MLP(in_features=hidden_dim, out_features=4, hidden_dim=12, num_layers=num_layers,
               propagate_gradient=1.0, normalize_input=normalize_input, normalize_output=normalize_output)
    model = simple_model(backbone=IdentityBackbone(hidden_dim=hidden_dim), heads={'action': head}, reasoner=None).eval()
    batch = [[{'action': 0, 'reward': 0.0}, {'action': 1, 'reward': 1.0}]]
    expected = model(batch_to_token_batch(_TOK, batch))["predictions"]
    save_model(model=model, path=tmp_path)
    loaded = load_model(model=fresh_model(model), repo_id_or_path=tmp_path).eval()
    actual = loaded(batch_to_token_batch(_TOK, batch))["predictions"]
    assert torch.allclose(actual['action'], expected['action'])
    with (tmp_path / 'config.json').open() as fh:
        config = json.load(fh)
    assert config['class'].endswith('ExampleModel')


@pytest.mark.parametrize("num_layers", [1, 2, 4])
@pytest.mark.parametrize("hidden_dim", [5, 12])
def test_layer_count_includes_residual_blocks_and_final_linear(num_layers, hidden_dim):
    head = MLP(in_features=8, out_features=3, hidden_dim=hidden_dim, num_layers=num_layers)
    assert len(head.layers) == num_layers
    assert isinstance(head.input_norm, torch.nn.RMSNorm)
    assert head.input_norm.normalized_shape == (8,)
    assert head.input_norm.eps == 1e-5
    assert head.input_norm.elementwise_affine
    torch.testing.assert_close(head.input_norm.weight, torch.ones(8))
    blocks = list(head.layers[:-1])
    assert len(blocks) == num_layers - 1
    for block in blocks:
        assert isinstance(block, ResidualBlock)
        assert block.input_linear.in_features == 8
        assert block.input_linear.out_features == 2 * hidden_dim
        assert isinstance(block.activation, SwiGLU)
        assert block.output_linear.in_features == hidden_dim
        assert block.output_linear.out_features == 8
    assert isinstance(head.layers[-1], torch.nn.Linear)
    assert head.layers[-1].in_features == 8 and head.layers[-1].out_features == 3
    z = torch.randn(2, 5, 8, requires_grad=True)
    result = head(z)
    assert result.shape == (2, 5, 3)
    result.square().sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert all(p.grad is not None for p in head.parameters())


def test_swiglu_splits_the_fused_linear_output():
    gate = torch.tensor([[-2.0, 0.0, 2.0]])
    value = torch.tensor([[3.0, -4.0, 5.0]])
    actual = SwiGLU()(torch.cat((gate, value), dim=-1))
    torch.testing.assert_close(actual, torch.nn.functional.silu(gate) * value)


@pytest.mark.parametrize("hidden_dim", [3, 8, 16])
def test_zero_residual_branch_preserves_features_and_gradients(hidden_dim):
    block = ResidualBlock(in_features=8, hidden_dim=hidden_dim)
    with torch.no_grad():
        block.output_linear.weight.zero_()
        block.output_linear.bias.zero_()
    x = torch.randn(2, 4, 8, requires_grad=True)
    upstream = torch.randn_like(x)
    result = block(x)
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    result.backward(upstream)
    torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)


@pytest.mark.parametrize("gradient_scale", [0.0, 0.25, 1.0])
def test_gradient_scaling_includes_residual_skips_but_not_parameter_gradients(gradient_scale):
    head = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=3,
               propagate_gradient=gradient_scale)
    reference = copy.deepcopy(head)
    reference.propagate_gradient = 1.0
    x = torch.randn(4, 8, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    output, reference_output = head(x), reference(reference_x)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    output.square().sum().backward()
    reference_output.square().sum().backward()
    if gradient_scale == 0:
        assert x.grad is None
    else:
        torch.testing.assert_close(x.grad, reference_x.grad * gradient_scale)
    for actual, expected in zip(head.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad)


def test_initialization_scale_affects_only_the_final_linear():
    torch.manual_seed(91)
    unscaled = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=3)
    torch.manual_seed(91)
    scaled = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=3, scale=0.1)
    for name, parameter in scaled.named_parameters():
        expected = dict(unscaled.named_parameters())[name]
        if name.startswith("layers.2."):
            expected = expected * 0.1
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_layers", [1, 3])
def test_input_norm_precedes_residual_blocks_without_an_extra_final_norm(num_layers):
    head = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=num_layers)
    with torch.no_grad():
        for block in head.layers[:-1]:
            assert isinstance(block, ResidualBlock)
            block.output_linear.weight.zero_()
            block.output_linear.bias.zero_()
        if num_layers > 1:
            block = head.layers[-2]
            assert isinstance(block, ResidualBlock)
            block.output_linear.bias.fill_(2.0)
    x = torch.randn(2, 8) * torch.tensor([[0.01], [100.]])
    output = head.layers[-1]
    assert isinstance(output, torch.nn.Linear)
    assert isinstance(head.input_norm, torch.nn.RMSNorm)
    features = torch.nn.functional.rms_norm(x, (8,), head.input_norm.weight, eps=1e-5)
    if num_layers > 1:
        features = features + 2.0
    expected = torch.nn.functional.linear(features, output.weight, output.bias)
    torch.testing.assert_close(head(x), expected)
    assert sum(isinstance(module, torch.nn.RMSNorm) for module in head.modules()) == num_layers


@pytest.mark.parametrize("num_layers", [1, 3])
def test_input_norm_stabilizes_outputs_under_large_feature_rescaling(num_layers):
    torch.manual_seed(11)
    head = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=num_layers)
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(head(100 * x), head(x), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("num_layers", [1, 3])
def test_output_norm_normalizes_after_the_final_linear_with_learned_gain(num_layers):
    torch.manual_seed(11)
    head = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=num_layers,
               normalize_input=False, normalize_output=True)
    assert isinstance(head.input_norm, torch.nn.Identity)
    assert isinstance(head.output_norm, torch.nn.RMSNorm)
    assert head.output_norm.normalized_shape == (3,)
    assert head.output_norm.eps == 1e-5
    assert head.output_norm.elementwise_affine
    torch.testing.assert_close(head.output_norm.weight, torch.ones(3))
    x = (10 * torch.randn(2, 5, 8)).requires_grad_()
    result = head(x)
    torch.testing.assert_close(result.square().mean(-1), torch.ones(2, 5), atol=1e-4, rtol=0)
    result.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert head.output_norm.weight.grad is not None
    assert torch.isfinite(head.output_norm.weight.grad).all()
    assert head.output_norm.weight.grad.abs().sum() > 0
    with torch.no_grad():
        output = head.layers[-1]
        assert isinstance(output, torch.nn.Linear) and output.bias is not None
        output.weight.mul_(100)
        output.bias.mul_(100)
    torch.testing.assert_close(head(x), result, atol=1e-5, rtol=1e-4)
    gain = torch.tensor([0.5, 1.5, 2.0])
    with torch.no_grad():
        head.output_norm.weight.copy_(gain)
    torch.testing.assert_close(head(x), result * gain, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("field", ["normalize_input", "normalize_output"])
@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_normalization_options_require_booleans(field, value):
    with pytest.raises(ValueError, match=field):
        MLP(in_features=8, out_features=3, hidden_dim=12, **{field: value})


@pytest.mark.parametrize("num_layers", [1, 3])
def test_final_linear_has_no_output_activation(num_layers):
    head = MLP(in_features=8, out_features=3, hidden_dim=12, num_layers=num_layers)
    with torch.no_grad():
        output = head.layers[-1]
        assert isinstance(output, torch.nn.Linear) and output.bias is not None
        output.weight.zero_()
        output.bias.fill_(-3.0)
    result = head(torch.randn(2, 8, dtype=torch.bfloat16))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full((2, 3), -3.0), rtol=0, atol=0)
