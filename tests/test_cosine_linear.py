import pytest
import torch
from torch.nn import functional as F

from mouse_core.models import CosineLinear


@pytest.mark.parametrize("width", [4, 32])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 2.0])
@pytest.mark.parametrize("bias", [False, True])
def test_cosine_linear_matches_scaled_cosine_and_preserves_batch_dimensions(width, alpha, bias):
    head = CosineLinear(width, 3, bias=bias, dtype=torch.float64)
    torch.testing.assert_close(head.alpha, torch.ones(3, dtype=torch.float64))
    if bias:
        torch.testing.assert_close(head.beta, torch.zeros(3, dtype=torch.float64))
    scales = torch.tensor([alpha, alpha + 0.5, -alpha], dtype=torch.float64)
    with torch.no_grad():
        head.alpha.copy_(scales)
        if head.beta is not None:
            head.beta.copy_(torch.tensor([-0.5, 0.2, 1.0]))
    x = torch.randn(2, 5, width, dtype=torch.float64)
    expected = scales * F.linear(F.normalize(x, dim=-1), F.normalize(head.weight, dim=-1))
    if head.beta is not None:
        expected = expected + head.beta
    torch.testing.assert_close(head(x), expected, atol=1e-4, rtol=1e-4)
    assert head(x).shape == (2, 5, 3)
    assert set(dict(head.named_parameters())) == ({"weight", "alpha", "beta"} if bias else {"weight", "alpha"})


def test_cosine_linear_is_invariant_to_positive_rescaling_of_features_and_each_weight_row():
    head = CosineLinear(16, 3)
    x = torch.randn(4, 16)
    expected = head(x)
    with torch.no_grad():
        head.weight.mul_(torch.tensor([[10.], [30.], [100.]]))
    torch.testing.assert_close(head(x * 50), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("zero_input,zero_weight", [(True, False), (False, True), (True, True)])
def test_cosine_linear_handles_zeros_with_finite_gradients(zero_input, zero_weight):
    head = CosineLinear(8, 3)
    x = (torch.zeros(2, 8) if zero_input else torch.randn(2, 8)).requires_grad_()
    if zero_weight:
        with torch.no_grad():
            head.weight.zero_()
    values = head(x)
    torch.testing.assert_close(values, head.beta.expand(2, 3))
    values.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize("global_affine", [False, True])
def test_cosine_linear_differentiates_through_normalization_scale_and_offset(global_affine):
    head = CosineLinear(3, 2, dtype=torch.float64, global_affine=global_affine)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    def evaluate(x, weight, alpha, beta):
        return torch.func.functional_call(head, {"weight": weight, "alpha": alpha, "beta": beta}, (x,))
    assert torch.autograd.gradcheck(evaluate, (x, head.weight, head.alpha, head.beta))


def test_cosine_linear_can_learn_values_outside_the_initial_cosine_range():
    head = CosineLinear(4, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., 0., 0., 0.]]))
    x = torch.tensor([[1., 0., 0., 0.], [-1., 0., 0., 0.]])
    target = torch.tensor([[4.], [-2.]])  # alpha=3 and beta=1.
    optimizer = torch.optim.SGD(head.parameters(), lr=0.2)
    for _ in range(60):
        optimizer.zero_grad()
        (head(x) - target).square().mean().backward()
        optimizer.step()
    torch.testing.assert_close(head(x), target, atol=1e-4, rtol=0)
    assert head.alpha.item() > 2.9
    assert abs(head.beta.item() - 1) < 1e-4


@pytest.mark.parametrize("width", [4, 1024])
@pytest.mark.parametrize("global_affine", [False, True])
def test_reset_initializes_weight_directions_at_unit_rms_and_resets_affine_parameters(width, global_affine):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        head = CosineLinear(width, 64, global_affine=global_affine)
        with torch.no_grad():
            head.alpha.fill_(4)
            head.beta.fill_(-2)
        head.reset_parameters()
    torch.testing.assert_close(head.alpha, torch.ones(() if global_affine else (64,)))
    assert head.beta.count_nonzero() == 0
    assert abs(head.weight.square().mean().item() - 1) < 0.15


@pytest.mark.parametrize("field,value", [
    ("eps", 0), ("eps", -1), ("eps", float("nan")), ("eps", True),
    ("in_features", 0), ("out_features", True), ("bias", 1),
    ("global_affine", 1), ("global_affine", "true"), ("global_affine", None),
])
def test_cosine_linear_rejects_invalid_settings(field, value):
    args = dict(in_features=4, out_features=2)
    args[field] = value
    with pytest.raises(ValueError, match=field):
        CosineLinear(**args)


@pytest.mark.parametrize("bias", [False, True])
def test_global_affine_broadcasts_and_accumulates_gradients_from_selected_actions(bias):
    shared = CosineLinear(8, 3, bias=bias, global_affine=True, dtype=torch.float64)
    separate = CosineLinear(8, 3, bias=bias, global_affine=False, dtype=torch.float64)
    assert shared.alpha.shape == () and shared.alpha.item() == 1
    if bias:
        assert shared.beta.shape == () and shared.beta.item() == 0
    else:
        assert shared.beta is None
    with torch.no_grad():
        separate.weight.copy_(shared.weight)
        for head in (shared, separate):
            head.alpha.fill_(2)
            if bias:
                head.beta.fill_(-0.5)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual, expected = shared(x), separate(reference_x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actions = torch.tensor([[0], [2], [2], [1]])
    actual.gather(1, actions).square().sum().backward()
    expected.gather(1, actions).square().sum().backward()
    torch.testing.assert_close(shared.alpha.grad, separate.alpha.grad.sum())
    if bias:
        torch.testing.assert_close(shared.beta.grad, separate.beta.grad.sum())
    torch.testing.assert_close(shared.weight.grad, separate.weight.grad)
    torch.testing.assert_close(x.grad, reference_x.grad)
