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
