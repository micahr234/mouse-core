"""``propagate_gradient`` scales backbone grads through MLP head inputs."""

from __future__ import annotations

import copy

import pytest
import torch

from mouse_core.models.heads import ClassificationHead, RegressionHead


def _filled_head(*, propagate_gradient: float) -> RegressionHead:
    head = RegressionHead(
        in_features=8,
        out_features=4,
        hidden_dim=8,
        num_layers=1,
        use_norm=False,
        scale=1.0,
        propagate_gradient=propagate_gradient,
    )
    with torch.no_grad():
        for param in head.parameters():
            param.fill_(0.1)
    return head


def _backbone_grad(head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor | None:
    probe = hidden.detach().clone().requires_grad_(True)
    head.zero_grad(set_to_none=True)
    head(probe).sum().backward()
    return probe.grad


def test_propagate_gradient_scales_backbone_grad() -> None:
    """``1`` full flow; ``0`` detaches; ``0.5`` halves the backbone grad."""
    torch.manual_seed(0)
    hidden = torch.randn(3, 8)
    full_head = _filled_head(propagate_gradient=1.0)
    half_head = _filled_head(propagate_gradient=0.5)
    none_head = _filled_head(propagate_gradient=0.0)
    half_head.load_state_dict(full_head.state_dict())
    none_head.load_state_dict(full_head.state_dict())
    full = _backbone_grad(full_head, hidden)
    half = _backbone_grad(half_head, hidden)
    assert full is not None and half is not None
    assert torch.allclose(half, full * 0.5)
    assert _backbone_grad(none_head, hidden) is None
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0.0
        for param in none_head.parameters()
    )
    copied = copy.deepcopy(none_head)
    assert _backbone_grad(copied, hidden) is None
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0.0
        for param in copied.parameters()
    )


@pytest.mark.parametrize("bad", [None, True, -0.1, 1.1, float("nan"), float("inf")])
def test_propagate_gradient_rejects_bad_values(bad: object) -> None:
    with pytest.raises(ValueError, match="propagate_gradient"):
        RegressionHead(
            in_features=8,
            out_features=4,
            hidden_dim=8,
            num_layers=1,
            use_norm=True,
            propagate_gradient=bad,  # type: ignore[arg-type]
        )


def test_classification_head_accepts_propagate_gradient() -> None:
    head = ClassificationHead(
        in_features=8,
        out_features=4,
        hidden_dim=8,
        num_layers=1,
        use_norm=True,
        propagate_gradient=0.0,
    )
    assert head.propagate_gradient == 0.0
    assert _backbone_grad(head, torch.randn(2, 8)) is None
