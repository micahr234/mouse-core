import pytest
import torch
from mouse_core.objectives import ReplicateObjective


def test_replication_averages_all_actions_and_stops_online_gradient():
    delayed = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    online = torch.zeros_like(delayed, requires_grad=True)
    losses, _ = ReplicateObjective()(predictions=delayed, targets=online)
    loss = losses["replicate"]
    torch.testing.assert_close(loss, delayed.square().sum() / 6)
    loss.backward()
    torch.testing.assert_close(delayed.grad, 2 * delayed.detach() / 6)
    assert online.grad is None


@pytest.mark.parametrize("shared_targets", [False, True])
def test_replication_batched_heads_match_independent_losses_and_gradients(shared_targets):
    predictions = torch.tensor(
        [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], requires_grad=True,
    )
    targets = (
        torch.tensor([[0., 1.], [2., 3.]], requires_grad=True)
        if shared_targets else
        torch.tensor([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]], requires_grad=True)
    )
    losses, metrics = ReplicateObjective()(predictions=predictions, targets=targets)
    assert losses["replicate"].shape == (2,)
    assert isinstance(metrics["replicate"], torch.Tensor)
    assert not metrics["replicate"].requires_grad
    losses["replicate"].sum().backward()
    assert targets.grad is None
    assert predictions.grad is not None
    for head in range(2):
        single_predictions = predictions.detach()[:, head].requires_grad_()
        single_targets = targets if shared_targets else targets[:, head]
        scalar, _ = ReplicateObjective()(
            predictions=single_predictions, targets=single_targets,
        )
        scalar["replicate"].backward()
        torch.testing.assert_close(losses["replicate"][head], scalar["replicate"])
        torch.testing.assert_close(predictions.grad[:, head], single_predictions.grad)


def test_replication_rejects_misaligned_or_empty_head_batches():
    objective = ReplicateObjective()
    with pytest.raises(ValueError, match="targets"):
        objective(predictions=torch.zeros(2, 3, 4), targets=torch.zeros(2, 2, 4))
    with pytest.raises(ValueError, match="nonempty"):
        objective(predictions=torch.zeros(2, 0, 4), targets=torch.zeros(2, 0, 4))
