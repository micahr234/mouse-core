"""CE imitates supplied actions, sharing supervised masks and head reductions."""

from typing import Literal

import pytest
import torch
import torch.nn.functional as F

from mouse_core.objectives import SupervisedObjective


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_ce_matches_action_nll_and_gradients(dtype):
    logits = torch.tensor([[3., -1., 0.], [-2., 4., 1.]], dtype=dtype, requires_grad=True)
    actions = torch.tensor([1, 0], dtype=torch.int32)
    losses, metrics = SupervisedObjective(loss_type="ce", oracle_temperature=None)(
        objective_data={}, predictions=logits, actions=actions,
    )
    expected = F.cross_entropy(logits.to(torch.promote_types(dtype, torch.float32)), actions.long())
    torch.testing.assert_close(losses["supervised"], expected)
    actual_grad = torch.autograd.grad(losses["supervised"], logits, retain_graph=True)[0]
    torch.testing.assert_close(actual_grad, torch.autograd.grad(expected, logits)[0])
    metric = metrics["supervised"]
    assert isinstance(metric, torch.Tensor) and not metric.requires_grad


def test_ce_skips_invalid_action_placeholders_and_padded_logits():
    logits = torch.tensor([[1., 2., -torch.inf], [torch.nan] * 3], requires_grad=True)
    losses, _ = SupervisedObjective(loss_type="ce", oracle_temperature=None, mask_key="skip")(
        objective_data={"skip": torch.tensor([0, 2])},
        predictions=logits, actions=torch.tensor([0, -99]),
    )
    torch.testing.assert_close(losses["supervised"], F.cross_entropy(logits[:1, :2], torch.tensor([0])))
    losses["supervised"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[1].count_nonzero() == 0
    assert logits.grad[0, 2] == 0


@pytest.mark.parametrize("actions,error,match", [
    (torch.tensor([0., 1.]), TypeError, "integer action IDs"),
    (torch.tensor([True, False]), TypeError, "integer action IDs"),
    (torch.tensor([[0.2, 0.8], [0.7, 0.3]]), TypeError, "integer action IDs"),
    (torch.tensor([[0, 1], [1, 0]]), ValueError, "actions must match"),
    (torch.tensor([-1, 0]), ValueError, "action IDs must be in"),
    (torch.tensor([0, 2]), ValueError, "action IDs must be in"),
])
def test_ce_rejects_non_action_targets(actions, error, match):
    with pytest.raises(error, match=match):
        SupervisedObjective(loss_type="ce", oracle_temperature=None)(
            objective_data={}, predictions=torch.zeros(2, 2), actions=actions,
        )


def test_ce_requires_actions_and_rejects_q_targets():
    objective = SupervisedObjective(loss_type="ce", oracle_temperature=None)
    with pytest.raises(TypeError, match="actions"):
        objective(objective_data={}, predictions=torch.zeros(2, 2))
    with pytest.raises(TypeError, match="unexpected keyword argument 'targets'"):
        objective(objective_data={}, predictions=torch.zeros(2, 2),
                  actions=torch.zeros(2, dtype=torch.long), targets=torch.ones(2, 2))


@pytest.mark.parametrize("shared_actions", [False, True])
@pytest.mark.parametrize("shared_weights", [False, True])
def test_ce_batched_heads_match_independent_calls(shared_actions, shared_weights):
    torch.manual_seed(23)
    logits = torch.randn(4, 3, 3, requires_grad=True)
    actions = (torch.tensor([2, 0, 1, 0]) if shared_actions else
               torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2]]))
    mask = torch.tensor([[False, True, False], [False, False, True],
                         [False, False, False], [True, False, False]])
    weights = torch.rand(4) if shared_weights else torch.rand(4, 3)
    weights.requires_grad_()
    losses, _ = SupervisedObjective(loss_type="ce", oracle_temperature=None, batched_heads=True, mask_key="skip", weight_key="w")(
        objective_data={"skip": mask, "w": weights}, predictions=logits, actions=actions,
    )
    expected = []
    for head in range(3):
        keep = ~mask[:, head]
        chosen = actions if shared_actions else actions[:, head]
        w = weights.detach() if shared_weights else weights.detach()[:, head]
        terms = F.cross_entropy(logits[keep, head], chosen[keep], reduction="none")
        expected.append((terms * w[keep]).sum() / w[keep].sum())
    expected_loss = torch.stack(expected)
    torch.testing.assert_close(losses["supervised"], expected_loss)
    actual = torch.autograd.grad(losses["supervised"].sum(), (logits, weights),
                                 retain_graph=True, allow_unused=True)
    torch.testing.assert_close(actual[0], torch.autograd.grad(expected_loss.sum(), logits)[0])
    assert actual[1] is None


@pytest.mark.parametrize("loss_type", ["mse", "mae"])
@pytest.mark.parametrize("batched", [False, True])
def test_regression_weights_average_finite_entries_per_row(
    loss_type: Literal["mse", "mae"], batched: bool,
):
    logits = torch.tensor([[1., 3.], [4., 6.], [torch.nan, torch.nan]], requires_grad=True)
    targets = torch.tensor([[0., 1.], [2., -torch.inf], [7., 8.]], requires_grad=True)
    predictions = logits[:, None, :].expand(3, 2, 2) if batched else logits
    weights = torch.tensor([1., 3., torch.nan], requires_grad=True)
    losses, _ = SupervisedObjective(loss_type=loss_type, oracle_temperature=None, batched_heads=batched,
                                    mask_key="skip", weight_key="w")(
        objective_data={"skip": torch.tensor([False, False, True]), "w": weights},
        predictions=predictions, targets=targets,
    )
    errors = torch.stack((logits[0, 0], logits[0, 1] - 1, logits[1, 0] - 2))
    terms = errors.square() if loss_type == "mse" else errors.abs()
    expected = (terms[:2].mean() + 3 * terms[2]) / 4
    torch.testing.assert_close(losses["supervised"], expected.expand(2) if batched else expected)
    actual = torch.autograd.grad(losses["supervised"].sum(), (logits, targets, weights),
                                 retain_graph=True, allow_unused=True)
    torch.testing.assert_close(actual[0], torch.autograd.grad(expected * (2 if batched else 1), logits)[0])
    assert actual[1:] == (None, None)
