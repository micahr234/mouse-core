"""Weighted SP reduction: masks, head normalization and invalid weights."""

import pytest
import torch
import torch.nn.functional as F
from mouse_core.objectives import SupervisedObjective


@pytest.mark.parametrize('batched', [False, True])
def test_core_weighted_reduction_and_mask(batched):
    logits = torch.tensor([[2., 0.], [0., 1.], [1., 1.]], requires_grad=True)
    targets = torch.tensor([[1., 0.], [0., 1.], [float('nan'), float('nan')]])
    weights = torch.tensor([1., 3., float('nan')])
    if batched:
        predictions = logits[:, None, :].expand(3, 2, 2)
        weights = torch.stack([weights, weights * 2], -1)
    else:
        predictions = logits
    losses, _ = SupervisedObjective(mask_key='skip', weight_key='weights', batched_heads=batched)(
        objective_data={'skip': torch.tensor([False, False, True]), 'weights': weights},
        predictions=predictions, targets=targets,
    )
    expected = (F.cross_entropy(logits[:2], torch.tensor([0, 1]), reduction='none')
                * torch.tensor([1., 3.])).sum() / 4
    torch.testing.assert_close(losses['supervised'], expected.expand(2) if batched else expected)


@pytest.mark.parametrize('weights', [[0., 0.], [1., -1.], [1., float('nan')],
                                    [1., float('inf')], [1., 2., 3.]])
def test_core_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match='weight'):
        SupervisedObjective(mask_key=None, weight_key='weights')(
            objective_data={'weights': torch.tensor(weights)},
            predictions=torch.zeros(2, 2), targets=torch.ones(2, 2),
        )
