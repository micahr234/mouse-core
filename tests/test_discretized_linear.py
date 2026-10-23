import pytest
import torch

from mouse_core.models import DiscretizedLinear, load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from tests._model_helpers import fresh_model, simple_model


def test_direct_linear_preserves_action_table_conversions_gradients_and_checkpoint(tmp_path):
    table = torch.tensor([[2., -1.], [-2., 1.], [0., 0.25]], requires_grad=True)
    original = table.detach().clone()
    head = DiscretizedLinear(in_features=8, action_vectors=table)
    assert isinstance(head, torch.nn.Linear)
    assert not list(head.children())
    with torch.no_grad():
        table.fill_(99)
    torch.testing.assert_close(head.action_vectors, original)
    assert not head.action_vectors.requires_grad
    ids = torch.tensor([[2, 0], [1, 2]])
    torch.testing.assert_close(head.discrete_to_continuous(ids), original[ids])
    torch.testing.assert_close(head.continuous_to_discrete(original[ids]), ids)
    with pytest.raises(ValueError, match="action_ids"):
        head.discrete_to_continuous(-1)
    x = torch.randn(2, 8, requires_grad=True)
    head(x).square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None for p in head.parameters())
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    optimizer.step()
    model = simple_model(backbone=IdentityBackbone(hidden_dim=8), heads={'action': head}, reasoner=None)
    save_model(model=model, path=tmp_path)
    restored = load_model(model=fresh_model(model), repo_id_or_path=tmp_path).heads['action']
    assert isinstance(restored, DiscretizedLinear)
    torch.testing.assert_close(restored(x), head(x), rtol=0, atol=0)
    torch.testing.assert_close(restored.discrete_to_continuous(ids), original[ids])
    torch.testing.assert_close(restored.continuous_to_discrete(original), torch.arange(3))


@pytest.mark.parametrize('table', [torch.empty(0, 2), torch.ones(2, 2), torch.tensor([[float('nan')]])])
def test_direct_linear_rejects_invalid_tables(table):
    with pytest.raises(ValueError, match='action_vectors'):
        DiscretizedLinear(in_features=8, action_vectors=table)
