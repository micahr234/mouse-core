from __future__ import annotations

from typing import Literal, cast

import pytest
import torch

from mouse_core.objectives import SupervisedObjective, best_action


def _ended(*shape: int, fill: int = 0) -> torch.Tensor:
    return torch.full(shape, fill, dtype=torch.int64)


def test_supervised_policy_objective_zero_temperature_returns_scalar_metrics() -> None:
    q = torch.tensor([[[0.0, 1.0, -torch.inf]]])
    objective_data = {"episode_ended": _ended(*q.shape[:-1])}
    predictions = torch.tensor([[[0.0, 1.0, 100.0]]])
    loss, metrics = SupervisedObjective(mask_key="episode_ended")(
        objective_data=objective_data,
        predictions=predictions,
        targets=q,
    )
    assert loss["supervised"].ndim == 0
    assert isinstance(metrics["supervised"], torch.Tensor)
    assert metrics["supervised"].ndim == 0
    assert metrics["supervised"] >= 0.0
    assert not metrics["supervised"].requires_grad


def test_supervised_policy_objective_zero_temperature_skips_nonzero_mask_rows() -> None:
    """Any nonzero mask drops the row."""
    q = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    logits = torch.tensor([[[0.0, 100.0], [100.0, 0.0], [0.0, 100.0]]])
    objective_data = {"episode_ended": torch.tensor([[1, 1, 0]], dtype=torch.int64)}
    loss, _ = SupervisedObjective(mask_key="episode_ended")(
        objective_data=objective_data,
        predictions=logits,
        targets=q,
    )
    assert loss["supervised"].item() < 1e-4


def test_supervised_policy_objective_zero_temperature_mask_key_none_keeps_terminals() -> None:
    predictions = torch.tensor([[[0.0, 100.0]]])
    loss, _ = SupervisedObjective(mask_key=None)(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[[1.0, 0.0]]]),
    )
    assert loss["supervised"].item() > 1.0


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_supervised_policy_objective_kl_softmaxes_q_targets_at_temperature(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    q = torch.tensor([[[0.0, 2.0]]])
    logits = torch.tensor([[[1.0, -0.5]]], requires_grad=True)
    temperature = 0.5
    loss, metrics = SupervisedObjective(
        loss_type=loss_type, oracle_temperature=temperature, mask_key=None
    )(objective_data={}, predictions=logits, targets=q)

    oracle = torch.softmax(q[0, 0] / temperature, dim=-1)
    student = torch.softmax(logits[0, 0], dim=-1)
    if loss_type == "forward_kl":
        expected = (oracle * (oracle.log() - student.log())).sum()
    else:
        expected = (student * (student.log() - oracle.log())).sum()
    torch.testing.assert_close(loss["supervised"], expected)
    torch.testing.assert_close(metrics["supervised"], expected.detach())
    loss["supervised"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_supervised_policy_objective_kl_directions_are_distinct() -> None:
    q = torch.tensor([[0.0, 2.0]])
    logits = torch.tensor([[3.0, -1.0]])
    forward, _ = SupervisedObjective(loss_type="forward_kl", oracle_temperature=1.0, mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    reverse, _ = SupervisedObjective(loss_type="reverse_kl", oracle_temperature=1.0, mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    assert not torch.allclose(forward["supervised"], reverse["supervised"])


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_supervised_policy_objective_kl_excludes_nonfinite_q_actions(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    q = torch.tensor([[0.0, 1.0, -torch.inf, torch.nan, torch.inf]])
    logits = torch.tensor([[0.2, -0.4, 100.0, 100.0, 100.0]], requires_grad=True)
    loss, _ = SupervisedObjective(loss_type=loss_type, oracle_temperature=1.0, mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    expected, _ = SupervisedObjective(loss_type=loss_type, oracle_temperature=1.0, mask_key=None)(
        objective_data={}, predictions=logits[:, :2], targets=q[:, :2]
    )
    torch.testing.assert_close(loss["supervised"], expected["supervised"])
    loss["supervised"].backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[0, 2:], torch.zeros(3))


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_supervised_policy_objective_kl_all_zero_q_is_uniform(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    q = torch.zeros(1, 3)
    logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    loss, _ = SupervisedObjective(loss_type=loss_type, oracle_temperature=1.0, mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    student = torch.softmax(logits[0], dim=-1)
    uniform = torch.full_like(student, 1 / 3)
    if loss_type == "forward_kl":
        expected = (uniform * (uniform.log() - student.log())).sum()
    else:
        expected = (student * (student.log() - uniform.log())).sum()
    torch.testing.assert_close(loss["supervised"], expected)
    loss["supervised"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_supervised_policy_objective_kl_skips_masked_rows_before_validating_q() -> None:
    q = torch.tensor([[[-torch.inf, torch.nan], [0.0, 1.0]]])
    logits = torch.tensor([[[100.0, -100.0], [0.0, 1.0]]])
    loss, _ = SupervisedObjective(loss_type="forward_kl", oracle_temperature=1.0, mask_key="episode_ended")(
        objective_data={"episode_ended": torch.tensor([[1, 0]])},
        predictions=logits,
        targets=q,
    )
    torch.testing.assert_close(loss["supervised"], torch.tensor(0.0), atol=1e-7, rtol=0)


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_supervised_policy_objective_kl_rejects_unmasked_row_without_finite_q(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    with pytest.raises(ValueError, match="finite Q value"):
        SupervisedObjective(loss_type=loss_type, oracle_temperature=1.0, mask_key=None)(
            objective_data={},
            predictions=torch.tensor([[0.0, 1.0]]),
            targets=torch.tensor([[-torch.inf, torch.nan]]),
        )


def test_supervised_policy_objective_kl_rejects_action_ids_and_wrong_shape() -> None:
    objective = SupervisedObjective(loss_type="forward_kl", oracle_temperature=1.0, mask_key=None)
    logits = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="same shape"):
        objective(objective_data={}, predictions=logits, targets=torch.tensor([1]))
    with pytest.raises(TypeError, match="floating-point Q"):
        objective(objective_data={}, predictions=logits, targets=torch.tensor([[0, 1]]))


@pytest.mark.parametrize("temperature", [-1.0, torch.nan, torch.inf])
def test_supervised_policy_objective_rejects_invalid_oracle_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="oracle_temperature"):
        SupervisedObjective(loss_type="forward_kl", oracle_temperature=temperature, mask_key=None)


@pytest.mark.parametrize("loss_type", ["ce-soft", "not_a_loss"])
def test_supervised_policy_objective_rejects_unknown_loss_type(loss_type: str) -> None:
    with pytest.raises(ValueError, match="loss_type"):
        SupervisedObjective(
            loss_type=cast(Literal["forward_kl", "reverse_kl"], loss_type),
            mask_key=None,
        )


def test_supervised_policy_default_three_dimensional_input_keeps_scalar_loss() -> None:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    loss, metrics = SupervisedObjective(mask_key=None)(
        objective_data={}, predictions=logits, targets=q,
    )
    assert loss["supervised"].ndim == 0
    assert isinstance(metrics["supervised"], torch.Tensor)
    assert metrics["supervised"].ndim == 0
    loss["supervised"].backward()
    assert logits.grad is not None


@pytest.mark.parametrize("per_head_targets", [False, True])
@pytest.mark.parametrize("per_head_mask", [False, True])
def test_supervised_policy_zero_temperature_batched_heads_match_independent_calls(
    per_head_targets: bool, per_head_mask: bool,
) -> None:
    torch.manual_seed(31)
    logits = torch.randn(4, 3, 3, requires_grad=True)
    shared_targets = torch.tensor([2, 1, 0, 2])
    targets = (
        torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]])
        if per_head_targets else shared_targets
    )
    targets = torch.nn.functional.one_hot(targets, num_classes=3).float()
    mask = (
        torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
        if per_head_mask else torch.tensor([0, 1, 0, 0])
    )
    batched_objective = SupervisedObjective(
        mask_key="skip", batched_heads=True,
    )
    single_objective = SupervisedObjective(mask_key="skip")
    losses, metrics = batched_objective(
        objective_data={"skip": mask}, predictions=logits, targets=targets,
    )
    assert losses["supervised"].shape == (3,)
    metric = metrics["supervised"]
    assert isinstance(metric, torch.Tensor)
    assert metric.shape == (3,)
    assert not metric.requires_grad
    losses["supervised"].sum().backward()
    batched_grad = logits.grad
    assert batched_grad is not None
    for head in range(3):
        single = logits.detach()[:, head].requires_grad_()
        single_targets = targets[:, head] if per_head_targets else targets
        single_mask = mask[:, head] if per_head_mask else mask
        expected, _ = single_objective(
            objective_data={"skip": single_mask},
            predictions=single, targets=single_targets,
        )
        expected["supervised"].backward()
        single_grad = single.grad
        assert single_grad is not None
        torch.testing.assert_close(losses["supervised"][head], expected["supervised"])
        torch.testing.assert_close(batched_grad[:, head], single_grad)


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
@pytest.mark.parametrize("per_head_targets", [False, True])
def test_supervised_policy_kl_batched_heads_match_independent_calls(
    loss_type: Literal["forward_kl", "reverse_kl"], per_head_targets: bool,
) -> None:
    torch.manual_seed(37)
    logits = torch.randn(4, 3, 4, requires_grad=True)
    shared_q = torch.tensor([
        [0.0, 1.0, -torch.inf, 0.5],
        [2.0, -1.0, 0.0, -torch.inf],
        [1.0, 0.0, -torch.inf, -2.0],
        [0.0, 0.0, 0.0, -torch.inf],
    ])
    q = shared_q[:, None, :].expand(-1, 3, -1).clone() if per_head_targets else shared_q
    mask = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
    if per_head_targets:
        q[:, 0, 0] += 0.4
        q[:, 1, 1] -= 0.8
        q[1, 0] = torch.tensor([-torch.inf, torch.nan, -torch.inf, -torch.inf])
    batched_objective = SupervisedObjective(
        loss_type=loss_type, oracle_temperature=0.65, mask_key="skip",
        batched_heads=True,
    )
    single_objective = SupervisedObjective(
        loss_type=loss_type, oracle_temperature=0.65, mask_key="skip",
    )
    losses, metrics = batched_objective(
        objective_data={"skip": mask}, predictions=logits, targets=q,
    )
    assert losses["supervised"].shape == (3,)
    torch.testing.assert_close(metrics["supervised"], losses["supervised"].detach())
    losses["supervised"].sum().backward()
    batched_grad = logits.grad
    assert batched_grad is not None
    for head in range(3):
        single = logits.detach()[:, head].requires_grad_()
        single_q = q[:, head] if per_head_targets else q
        expected, _ = single_objective(
            objective_data={"skip": mask[:, head]},
            predictions=single, targets=single_q,
        )
        expected["supervised"].backward()
        single_grad = single.grad
        assert single_grad is not None
        torch.testing.assert_close(losses["supervised"][head], expected["supervised"])
        torch.testing.assert_close(batched_grad[:, head], single_grad)


def test_supervised_policy_batched_heads_reject_misaligned_shared_inputs() -> None:
    objective = SupervisedObjective(mask_key="skip", batched_heads=True)
    logits = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="same shape"):
        objective(
            objective_data={"skip": torch.zeros(2)},
            predictions=logits, targets=torch.zeros(2, 5),
        )
    with pytest.raises(ValueError, match="skip mask must have shape"):
        objective(
            objective_data={"skip": torch.zeros(2, 4)},
            predictions=logits, targets=torch.zeros(2, dtype=torch.int64),
        )


def test_best_action_unique_max_is_deterministic() -> None:
    q = torch.tensor([[0.0, 2.0, 1.0, -torch.inf], [3.0, 1.0, 3.0 - 1e-6, -torch.inf]])
    for _ in range(20):
        assert best_action(q).tolist() == [1, 0]


def test_best_action_samples_uniformly_among_maxima() -> None:
    torch.manual_seed(0)
    q = torch.tensor([[1.0, 1.0, 0.5, -torch.inf]])
    seen = {best_action(q).item() for _ in range(80)}
    assert seen == {0, 1}


def test_best_action_never_selects_padding() -> None:
    torch.manual_seed(0)
    q = torch.tensor([[-torch.inf, 2.0, 2.0]])
    for _ in range(40):
        assert best_action(q).item() in (1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_zero_temperature_matches_nll_loss_and_gradients(dtype) -> None:
    q = torch.tensor([[0.93, 0.94, 0.0], [-2.0, -3.0, -1.0]], requires_grad=True)
    logits = torch.tensor([[0.2, -0.4, 1.1], [100.0, 0.0, -100.0]],
                          dtype=dtype, requires_grad=True)
    losses, _ = SupervisedObjective(oracle_temperature=0.0, mask_key=None)(
        objective_data={}, predictions=logits, targets=q,
    )
    expected = torch.nn.functional.cross_entropy(
        logits.to(torch.promote_types(dtype, torch.float32)), torch.tensor([1, 2]),
    )
    torch.testing.assert_close(losses["supervised"], expected)
    actual_grad = torch.autograd.grad(losses["supervised"], logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
    assert q.grad is None


@pytest.mark.parametrize("q", [[2.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -3.0]])
def test_zero_temperature_averages_ties_with_correct_entropy_constant(q) -> None:
    targets = torch.tensor([q], requires_grad=True)
    logits = torch.tensor([[0.2, -0.4, 1.1]], requires_grad=True)
    losses, _ = SupervisedObjective(mask_key=None)(
        objective_data={}, predictions=logits, targets=targets,
    )
    maxima = targets[0] == targets[0].max()
    expected_nll = -torch.log_softmax(logits[0], dim=-1)[maxima].mean()
    expected_kl = expected_nll - maxima.sum().float().log()
    torch.testing.assert_close(losses["supervised"], expected_kl)
    actual_grad = torch.autograd.grad(losses["supervised"], (logits, targets),
                                      retain_graph=True, allow_unused=True)
    torch.testing.assert_close(actual_grad[0], torch.autograd.grad(expected_nll, logits)[0])
    assert actual_grad[1] is None


def test_zero_temperature_masks_missing_rows_and_unavailable_actions() -> None:
    q = torch.tensor([[1.0, 2.0, -torch.inf, torch.nan, torch.inf],
                      [torch.nan] * 5])
    logits = torch.tensor([[0.2, -0.4, 100.0, 100.0, 100.0], [0.0] * 5],
                          requires_grad=True)
    losses, _ = SupervisedObjective(mask_key="skip")(
        objective_data={"skip": torch.tensor([False, True])},
        predictions=logits, targets=q,
    )
    expected = torch.nn.functional.cross_entropy(logits[:1, :2], torch.tensor([1]))
    torch.testing.assert_close(losses["supervised"], expected)
    losses["supervised"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2:].count_nonzero() == 0
    assert logits.grad[1].count_nonzero() == 0


def test_forward_kl_small_temperature_converges_to_zero_temperature() -> None:
    q = torch.tensor([[1.0, 1.0, 0.99], [0.99, 1.0, 0.0]])
    logits = torch.tensor([[0.2, -0.4, 1.1], [0.3, 0.6, -0.2]], requires_grad=True)
    losses = [SupervisedObjective(oracle_temperature=t, mask_key=None)(
        objective_data={}, predictions=logits, targets=q,
    )[0]["supervised"] for t in (0.0, 0.0001)]
    torch.testing.assert_close(losses[0], losses[1])
    grads = [torch.autograd.grad(loss, logits, retain_graph=True)[0] for loss in losses]
    torch.testing.assert_close(grads[0], grads[1])


def test_reverse_kl_rejects_zero_temperature_before_execution() -> None:
    with pytest.raises(ValueError, match="reverse_kl requires oracle_temperature > 0"):
        SupervisedObjective(loss_type="reverse_kl", oracle_temperature=0.0, mask_key=None)
