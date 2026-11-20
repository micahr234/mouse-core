"""LoRA membership is structural; frozen backbones preserve input gradients."""
import torch
from torch import nn
from mouse_core.models import backbone_parameter_view, named_backbone_parameters
from mouse_core.models.lora import LoRAConfig, LoRALinear
from mouse_core.models.backbone.flex_utils import module_device_dtype


def test_partition_is_disjoint_complete_and_preserves_names_and_identity():
    layer = LoRALinear(nn.Linear(3, 3), LoRAConfig(rank=2))
    model = nn.ModuleDict({"first": layer, "alias": layer, "norm": nn.RMSNorm(3)})
    keys = set(model.state_dict())
    base = backbone_parameter_view(model, adapters=False)
    adapters = backbone_parameter_view(model, adapters=True)
    left, right = set(map(id, base.parameters())), set(map(id, adapters.parameters()))
    assert not left & right
    assert left | right == set(map(id, model.parameters()))
    for view in (base, adapters):
        for name, parameter in view.named_parameters():
            assert model.get_parameter(name) is parameter
    adapters.requires_grad_(False)
    assert set(map(id, backbone_parameter_view(model, adapters=True).parameters())) == right
    assert set(model.state_dict()) == keys


def test_frozen_backbone_still_backpropagates_into_embeddings():
    layer = LoRALinear(nn.Linear(3, 3), LoRAConfig(rank=2))
    for adapters in (False, True):
        backbone_parameter_view(layer, adapters=adapters).requires_grad_(False)
    embedding = nn.Embedding(2, 3)
    layer(embedding(torch.tensor([0]))).square().sum().backward()
    assert embedding.weight.grad[0].abs().sum() > 0
    assert all(p.grad is None for p in layer.parameters())


def test_base_precision_does_not_use_adapter_names_as_a_heuristic():
    # A normal module named lora_A is not an adapter.
    model = nn.ModuleDict({"nested": nn.ModuleDict({"lora_A": nn.Linear(2, 2, dtype=torch.float64)})})
    assert len(list(named_backbone_parameters(model, adapters=True))) == 0
    assert module_device_dtype(model)[1] == torch.float64
    layer = LoRALinear(nn.Linear(2, 2, dtype=torch.bfloat16), LoRAConfig(rank=1))
    assert module_device_dtype(layer)[1] == torch.bfloat16


def test_identity_backbone_preserves_input_precision_and_gradient():
    from mouse_core.models import IdentityBackbone
    backbone = IdentityBackbone(hidden_dim=3)
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        x = torch.randn(2, 3, dtype=dtype, requires_grad=True)
        y = backbone(x, sequence_ids=torch.zeros(2, dtype=torch.int64))
        assert y is x
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
