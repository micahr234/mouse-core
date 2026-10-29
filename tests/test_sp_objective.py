from __future__ import annotations

from typing import Literal, cast

import pytest
import torch

from mouse_core.objectives import SpObjective, best_action
from mouse_core.objectives.sp import sp_ce


def _ended(*shape: int, fill: int = 0) -> torch.Tensor:
    return torch.full(shape, fill, dtype=torch.int64)


def _ce_targets(q: torch.Tensor) -> torch.Tensor:
    """Hard-CE target action ids from Q* via ``best_action``."""
    return best_action(q.reshape(-1, q.shape[-1])).reshape(q.shape[:-1])


def test_sp_objective_ce_uses_best_action_ids() -> None:
    q = torch.tensor([[[0.0, 1.0, -torch.inf]]])
    objective_data = {"episode_ended": _ended(*q.shape[:-1])}
    predictions = torch.tensor([[[0.0, 1.0, 100.0]]])
    loss, metrics = SpObjective(mask_key="episode_ended")(
        objective_data=objective_data,
        predictions=predictions,
        targets=_ce_targets(q),
    )
    assert loss["action"].ndim == 0
    assert isinstance(metrics["action"], torch.Tensor)
    assert metrics["action"].ndim == 0
    assert metrics["action"] >= 0.0
    assert not metrics["action"].requires_grad


def test_sp_objective_ce_skips_nonzero_mask_rows() -> None:
    """Any nonzero mask drops the row."""
    q = torch.tensor([[[0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    logits = torch.tensor([[[0.0, 100.0], [100.0, 0.0], [0.0, 100.0]]])
    objective_data = {"episode_ended": torch.tensor([[1, 1, 0]], dtype=torch.int64)}
    loss, _ = SpObjective(mask_key="episode_ended")(
        objective_data=objective_data,
        predictions=logits,
        targets=_ce_targets(q),
    )
    assert loss["action"].item() < 1e-4


def test_sp_objective_ce_mask_key_none_keeps_terminals() -> None:
    predictions = torch.tensor([[[0.0, 100.0]]])
    loss, _ = SpObjective(mask_key=None)(
        objective_data={},
        predictions=predictions,
        targets=torch.tensor([[0]]),
    )
    assert loss["action"].item() > 1.0


def test_sp_objective_ce_rejects_out_of_range_actions() -> None:
    objective_data = {"episode_ended": _ended(1, 1)}
    predictions = torch.tensor([[[0.0, 1.0]]])
    try:
        SpObjective(mask_key="episode_ended")(
            objective_data=objective_data,
            predictions=predictions,
            targets=torch.tensor([[3]]),
        )
    except ValueError as exc:
        assert "action ids must be in" in str(exc)
    else:
        raise AssertionError("expected ValueError for out-of-range action")


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_sp_objective_kl_softmaxes_q_targets_at_temperature(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    q = torch.tensor([[[0.0, 2.0]]])
    logits = torch.tensor([[[1.0, -0.5]]], requires_grad=True)
    temperature = 0.5
    loss, metrics = SpObjective(
        loss_type=loss_type, oracle_temperature=temperature, mask_key=None
    )(objective_data={}, predictions=logits, targets=q)

    oracle = torch.softmax(q[0, 0] / temperature, dim=-1)
    student = torch.softmax(logits[0, 0], dim=-1)
    if loss_type == "forward_kl":
        expected = (oracle * (oracle.log() - student.log())).sum()
    else:
        expected = (student * (student.log() - oracle.log())).sum()
    torch.testing.assert_close(loss["action"], expected)
    torch.testing.assert_close(metrics["action"], expected.detach())
    loss["action"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sp_objective_kl_directions_are_distinct() -> None:
    q = torch.tensor([[0.0, 2.0]])
    logits = torch.tensor([[3.0, -1.0]])
    forward, _ = SpObjective(loss_type="forward_kl", mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    reverse, _ = SpObjective(loss_type="reverse_kl", mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    assert not torch.allclose(forward["action"], reverse["action"])


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_sp_objective_kl_excludes_nonfinite_q_actions(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    q = torch.tensor([[0.0, 1.0, -torch.inf, torch.nan, torch.inf]])
    logits = torch.tensor([[0.2, -0.4, 100.0, 100.0, 100.0]], requires_grad=True)
    loss, _ = SpObjective(loss_type=loss_type, mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    expected, _ = SpObjective(loss_type=loss_type, mask_key=None)(
        objective_data={}, predictions=logits[:, :2], targets=q[:, :2]
    )
    torch.testing.assert_close(loss["action"], expected["action"])
    loss["action"].backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[0, 2:], torch.zeros(3))


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_sp_objective_kl_all_zero_q_is_uniform(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    q = torch.zeros(1, 3)
    logits = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    loss, _ = SpObjective(loss_type=loss_type, mask_key=None)(
        objective_data={}, predictions=logits, targets=q
    )
    student = torch.softmax(logits[0], dim=-1)
    uniform = torch.full_like(student, 1 / 3)
    if loss_type == "forward_kl":
        expected = (uniform * (uniform.log() - student.log())).sum()
    else:
        expected = (student * (student.log() - uniform.log())).sum()
    torch.testing.assert_close(loss["action"], expected)
    loss["action"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sp_objective_kl_skips_masked_rows_before_validating_q() -> None:
    q = torch.tensor([[[-torch.inf, torch.nan], [0.0, 1.0]]])
    logits = torch.tensor([[[100.0, -100.0], [0.0, 1.0]]])
    loss, _ = SpObjective(loss_type="forward_kl", mask_key="episode_ended")(
        objective_data={"episode_ended": torch.tensor([[1, 0]])},
        predictions=logits,
        targets=q,
    )
    torch.testing.assert_close(loss["action"], torch.tensor(0.0), atol=1e-7, rtol=0)


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
def test_sp_objective_kl_rejects_unmasked_row_without_finite_q(
    loss_type: Literal["forward_kl", "reverse_kl"],
) -> None:
    with pytest.raises(ValueError, match="finite Q value"):
        SpObjective(loss_type=loss_type, mask_key=None)(
            objective_data={},
            predictions=torch.tensor([[0.0, 1.0]]),
            targets=torch.tensor([[-torch.inf, torch.nan]]),
        )


def test_sp_objective_kl_rejects_action_ids_and_wrong_shape() -> None:
    objective = SpObjective(loss_type="forward_kl", mask_key=None)
    logits = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="same shape"):
        objective(objective_data={}, predictions=logits, targets=torch.tensor([1]))
    with pytest.raises(TypeError, match="floating-point Q"):
        objective(objective_data={}, predictions=logits, targets=torch.tensor([[0, 1]]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, torch.nan, torch.inf])
def test_sp_objective_rejects_invalid_oracle_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="oracle_temperature"):
        SpObjective(loss_type="forward_kl", oracle_temperature=temperature, mask_key=None)


def test_sp_objective_kl_rejects_label_smoothing() -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        SpObjective(loss_type="reverse_kl", label_smoothing=0.1, mask_key=None)


def test_sp_objective_rejects_unknown_loss_type() -> None:
    with pytest.raises(ValueError, match="loss_type"):
        SpObjective(
            loss_type=cast(Literal["ce", "forward_kl", "reverse_kl"], "not_a_loss"),
            mask_key=None,
        )


def test_sp_default_three_dimensional_input_keeps_scalar_loss() -> None:
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    labels = torch.tensor([[0, 1]])
    loss, metrics = SpObjective(mask_key=None)(
        objective_data={}, predictions=logits, targets=labels,
    )
    assert loss["action"].ndim == 0
    assert isinstance(metrics["action"], torch.Tensor)
    assert metrics["action"].ndim == 0
    loss["action"].backward()
    assert logits.grad is not None


@pytest.mark.parametrize("per_head_targets", [False, True])
@pytest.mark.parametrize("per_head_mask", [False, True])
def test_sp_ce_batched_heads_match_independent_calls(
    per_head_targets: bool, per_head_mask: bool,
) -> None:
    torch.manual_seed(31)
    logits = torch.randn(4, 3, 3, requires_grad=True)
    shared_targets = torch.tensor([2, 1, 0, 2])
    targets = (
        torch.tensor([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]])
        if per_head_targets else shared_targets
    )
    mask = (
        torch.tensor([[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
        if per_head_mask else torch.tensor([0, 1, 0, 0])
    )
    batched_objective = SpObjective(
        mask_key="skip", label_smoothing=0.15, batched_heads=True,
    )
    single_objective = SpObjective(mask_key="skip", label_smoothing=0.15)
    losses, metrics = batched_objective(
        objective_data={"skip": mask}, predictions=logits, targets=targets,
    )
    assert losses["action"].shape == (3,)
    metric = metrics["action"]
    assert isinstance(metric, torch.Tensor)
    assert metric.shape == (3,)
    assert not metric.requires_grad
    losses["action"].sum().backward()
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
        expected["action"].backward()
        single_grad = single.grad
        assert single_grad is not None
        torch.testing.assert_close(losses["action"][head], expected["action"])
        torch.testing.assert_close(batched_grad[:, head], single_grad)


@pytest.mark.parametrize("loss_type", ["forward_kl", "reverse_kl"])
@pytest.mark.parametrize("per_head_targets", [False, True])
def test_sp_kl_batched_heads_match_independent_calls(
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
    batched_objective = SpObjective(
        loss_type=loss_type, oracle_temperature=0.65, mask_key="skip",
        batched_heads=True,
    )
    single_objective = SpObjective(
        loss_type=loss_type, oracle_temperature=0.65, mask_key="skip",
    )
    losses, metrics = batched_objective(
        objective_data={"skip": mask}, predictions=logits, targets=q,
    )
    assert losses["action"].shape == (3,)
    torch.testing.assert_close(metrics["action"], losses["action"].detach())
    losses["action"].sum().backward()
    batched_grad = logits.grad
    assert batched_grad is not None
    for head in range(3):
        single = logits.detach()[:, head].requires_grad_()
        single_q = q[:, head] if per_head_targets else q
        expected, _ = single_objective(
            objective_data={"skip": mask[:, head]},
            predictions=single, targets=single_q,
        )
        expected["action"].backward()
        single_grad = single.grad
        assert single_grad is not None
        torch.testing.assert_close(losses["action"][head], expected["action"])
        torch.testing.assert_close(batched_grad[:, head], single_grad)


def test_sp_batched_heads_reject_misaligned_shared_inputs() -> None:
    objective = SpObjective(mask_key="skip", batched_heads=True)
    logits = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="CE targets must have shape"):
        objective(
            objective_data={"skip": torch.zeros(2)},
            predictions=logits, targets=torch.zeros(2, 4, dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="skip mask must have shape"):
        objective(
            objective_data={"skip": torch.zeros(2, 4)},
            predictions=logits, targets=torch.zeros(2, dtype=torch.int64),
        )


@pytest.mark.parametrize("target", [0.9, 1.9, True])
def test_sp_objective_rejects_noninteger_action_labels(target) -> None:
    with pytest.raises(TypeError, match="integer"):
        SpObjective(mask_key=None)(
            objective_data={},
            predictions=torch.tensor([[0.0, 1.0]]),
            targets=torch.tensor([target]),
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


def test_sp_ce_matches_cross_entropy_on_unique_max() -> None:
    actions = torch.tensor([1])
    logits = torch.tensor([[0.5, -1.0, 2.0]])
    expected = torch.nn.functional.cross_entropy(logits, actions)
    assert torch.allclose(sp_ce(target_actions=actions, logits=logits), expected)
    smoothed = torch.nn.functional.cross_entropy(logits, actions, label_smoothing=0.2)
    assert torch.allclose(
        sp_ce(target_actions=actions, logits=logits, label_smoothing=0.2), smoothed
    )


def test_sp_ce_ignores_padded_student_logits() -> None:
    """A junk student logit at a padded slot must not affect the hard CE loss."""
    actions = torch.tensor([1])
    invalid = torch.tensor([[False, False, False, True]])
    logits = torch.tensor([[0.5, -1.0, 2.0, 100.0]], requires_grad=True)
    clean = sp_ce(target_actions=actions, logits=logits[:, :3])
    padded = sp_ce(target_actions=actions, logits=logits, invalid=invalid)
    assert torch.allclose(padded, clean)
    clean_smoothed = sp_ce(target_actions=actions, logits=logits[:, :3], label_smoothing=0.1)
    padded_smoothed = sp_ce(
        target_actions=actions, logits=logits, label_smoothing=0.1, invalid=invalid
    )
    assert torch.isfinite(padded_smoothed)
    assert torch.allclose(padded_smoothed, clean_smoothed)
    padded_smoothed.backward()
    assert logits.grad is not None
    assert logits.grad[0, -1].item() == 0.0


def test_sp_ce_excludes_padding_even_when_valid_logits_are_extremely_negative() -> None:
    logits = torch.full((1, 3), torch.finfo(torch.float32).min / 2)
    invalid = torch.tensor([[False, False, True]])
    expected = sp_ce(torch.tensor([0]), logits[:, :2], label_smoothing=0.1)
    actual = sp_ce(torch.tensor([0]), logits, invalid=invalid, label_smoothing=0.1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("invalid", [[[True, True]], [[False, True]]])
def test_sp_ce_rejects_targeting_an_invalid_action(invalid) -> None:
    with pytest.raises(ValueError, match="invalid action"):
        sp_ce(torch.tensor([1]), torch.zeros(1, 2), invalid=torch.tensor(invalid))


@pytest.mark.parametrize("smoothing", [-0.1, 1.1, float("nan")])
def test_sp_ce_rejects_invalid_label_smoothing(smoothing) -> None:
    with pytest.raises(ValueError, match="label_smoothing"):
        sp_ce(torch.tensor([0]), torch.zeros(1, 2), label_smoothing=smoothing)
