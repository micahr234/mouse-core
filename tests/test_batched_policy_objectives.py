"""Independent policy heads match separate scalar objective calls."""

import pytest
import torch

from mouse_core.objectives import (
    GrpoObjective,
    PpoObjective,
    affine_reward,
    affine_value,
    boundary_discount,
)


def _columns() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    columns = {
        "action": torch.tensor([-99, 1, 0, 2, -99, 1, 0]),
        "action_executed": torch.tensor([False, True, True, True, False, True, True]),
        "reward": torch.tensor([0.0, 1.0, 0.5, 0.2, 0.0, 1.5, 0.1]),
        "terminated": torch.tensor([False, False, False, True, False, False, True]),
        "truncated": torch.zeros(7, dtype=torch.bool),
        "head_output_count": torch.ones(7, dtype=torch.int64),
    }
    return columns, torch.tensor([0, 0, 0, 0, 1, 1, 1])


@pytest.mark.parametrize("normalize_advantage", [False, True])
@pytest.mark.parametrize("value_last_dim", [False, True])
def test_batched_ppo_matches_independent_heads_and_gradients(
    normalize_advantage: bool, value_last_dim: bool,
) -> None:
    torch.manual_seed(37)
    columns, groups = _columns()
    logits = torch.randn(7, 3, 3, requires_grad=True)
    values = torch.randn(7, 3, requires_grad=True)
    old = torch.randn(7, 3)
    old[0] = old[4] = torch.nan  # Reset placeholders never enter the ratio.
    columns["old_log_prob"] = old.requires_grad_()
    objective = PpoObjective(
        discount=boundary_discount(
            gamma_step=0.9, gamma_terminated=0.0, gamma_truncated=0.4,
        ),
        reward=affine_reward(scale=1.0, shift=0.0),
        value=affine_value(scale=1.3, shift=-0.2),
        cross_group_backups="bootstrap",
        gae_lambda=0.8,
        ent_coef=0.05,
        normalize_advantage=normalize_advantage,
    )
    value_predictions = values[..., None] if value_last_dim else values
    scalar_losses = []
    scalar_metrics = []
    for head in range(3):
        head_columns = {**columns, "old_log_prob": columns["old_log_prob"][:, head]}
        losses, metrics = objective(
            objective_data=head_columns,
            group_id=groups,
            predictions=logits[:, head],
            value_predictions=values[:, head],
        )
        scalar_losses.append(losses["ppo"])
        scalar_metrics.append(metrics)
    expected_loss = torch.stack(scalar_losses)
    expected_grads = torch.autograd.grad(
        expected_loss.mean(), (logits, values), retain_graph=True,
    )

    losses, metrics = objective(
        objective_data=columns,
        group_id=groups,
        predictions=logits,
        value_predictions=value_predictions,
    )
    torch.testing.assert_close(losses["ppo"], expected_loss)
    assert losses["ppo"].shape == (3,)
    for name in scalar_metrics[0]:
        expected = torch.tensor([row[name] for row in scalar_metrics])
        metric = metrics[name]
        assert isinstance(metric, torch.Tensor)
        assert metric.shape == (3,)
        torch.testing.assert_close(metric, expected)
        assert not metric.requires_grad
    actual_grads = torch.autograd.grad(
        losses["ppo"].mean(), (logits, values, columns["old_log_prob"]),
        allow_unused=True,
    )
    torch.testing.assert_close(actual_grads[0], expected_grads[0])
    torch.testing.assert_close(actual_grads[1], expected_grads[1])
    assert actual_grads[2] is None


@pytest.mark.parametrize("importance_sampling_level", ["action", "episode"])
@pytest.mark.parametrize("head_specific", [False, True])
def test_batched_grpo_matches_independent_heads_and_gradients(
    importance_sampling_level: str, head_specific: bool,
) -> None:
    torch.manual_seed(41)
    columns, groups = _columns()
    logits = torch.randn(7, 3, 3, requires_grad=True)
    width = 3 if head_specific else 1
    old = torch.randn(7, width)
    old[0] = old[4] = torch.nan
    advantages = torch.randn(7, width)
    weights = torch.rand(7, width)
    weights[0] = weights[4] = torch.nan
    columns["old_log_prob"] = old.requires_grad_() if head_specific else old[:, 0].requires_grad_()
    columns["advantage"] = advantages.requires_grad_() if head_specific else advantages[:, 0].requires_grad_()
    columns["sample_weight"] = weights.requires_grad_() if head_specific else weights[:, 0].requires_grad_()
    objective = GrpoObjective(
        importance_sampling_level=importance_sampling_level,
        sampling_weight_key="sample_weight",
        ent_coef=0.05,
    )
    scalar_losses = []
    scalar_metrics = []
    for head in range(3):
        head_columns = {
            **columns,
            **{
                key: columns[key][:, head]
                for key in ("old_log_prob", "advantage", "sample_weight")
                if head_specific
            },
        }
        losses, metrics = objective(
            objective_data=head_columns,
            group_id=groups,
            predictions=logits[:, head],
        )
        scalar_losses.append(losses["grpo"])
        scalar_metrics.append(metrics)
    expected_loss = torch.stack(scalar_losses)
    expected_grad = torch.autograd.grad(expected_loss.mean(), logits, retain_graph=True)[0]

    losses, metrics = objective(
        objective_data=columns, group_id=groups, predictions=logits,
    )
    torch.testing.assert_close(losses["grpo"], expected_loss)
    assert losses["grpo"].shape == (3,)
    for name in scalar_metrics[0]:
        expected = torch.tensor([row[name] for row in scalar_metrics])
        metric = metrics[name]
        assert isinstance(metric, torch.Tensor)
        assert metric.shape == (3,)
        torch.testing.assert_close(metric, expected)
        assert not metric.requires_grad
    actual_grads = torch.autograd.grad(
        losses["grpo"].mean(),
        (logits, columns["old_log_prob"], columns["advantage"], columns["sample_weight"]),
        allow_unused=True,
    )
    torch.testing.assert_close(actual_grads[0], expected_grad)
    assert actual_grads[1:] == (None, None, None)


@pytest.mark.parametrize("kind", ["ppo", "grpo"])
def test_batched_policy_objectives_reject_empty_or_misaligned_heads(kind: str) -> None:
    columns, groups = _columns()
    if kind == "ppo":
        objective = PpoObjective(
            discount=None, reward=None, value=None,
            cross_group_backups="bootstrap",
        )
        with pytest.raises(ValueError, match="batched value shape"):
            objective(
                objective_data=columns, group_id=groups,
                predictions=torch.zeros(7, 2, 3),
                value_predictions=torch.zeros(7, 3),
            )
        with pytest.raises(ValueError, match="at least one head"):
            objective(
                objective_data=columns, group_id=groups,
                predictions=torch.zeros(7, 0, 3),
                value_predictions=torch.zeros(7, 0),
            )
    else:
        objective = GrpoObjective()
        with pytest.raises(ValueError, match="at least one head"):
            objective(
                objective_data=columns, group_id=groups,
                predictions=torch.zeros(7, 0, 3),
            )
