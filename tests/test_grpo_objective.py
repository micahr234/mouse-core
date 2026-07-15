"""Tests for GRPO objective and group-relative advantages."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import GrpoObjective, group_relative_advantages


def test_group_relative_advantages_zscore() -> None:
    rewards = torch.tensor([1.0, 3.0, 5.0])
    adv = group_relative_advantages(rewards)
    assert adv.shape == (3,)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv[0].item() + adv[2].item()) < 1e-5  # symmetric around mean
    assert adv[2].item() > 0 and adv[0].item() < 0


def test_group_relative_advantages_identical_is_zero() -> None:
    adv = group_relative_advantages(torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(adv, torch.zeros(3))


def test_group_relative_advantages_singleton_is_zero() -> None:
    adv = group_relative_advantages(torch.tensor([7.0]))
    assert torch.allclose(adv, torch.zeros(1))


def test_grpo_objective_runs() -> None:
    b, s, a = 2, 4, 3
    objective_data = TensorDict(
        {
            "action": torch.randint(0, a, (b, s)),
            "old_log_prob": torch.randn(b, s),
            "advantage": torch.randn(b, s),
        },
        batch_size=(b, s),
    )
    predictions = TensorDict(
        {"action": torch.randn(b, s, a)},
        batch_size=(b, s),
    )
    loss, metrics = GrpoObjective()(objective_data, predictions)
    assert loss.ndim == 0
    assert "grpo" in metrics
    assert "policy_loss" in metrics
    assert "entropy" in metrics


def test_grpo_objective_requires_advantage() -> None:
    objective_data = TensorDict(
        {
            "action": torch.zeros(1, 2, dtype=torch.long),
            "old_log_prob": torch.zeros(1, 2),
        },
        batch_size=(1, 2),
    )
    predictions = TensorDict({"action": torch.zeros(1, 2, 2)}, batch_size=(1, 2))
    try:
        GrpoObjective()(objective_data, predictions)
    except KeyError as e:
        assert "advantage" in str(e)
    else:
        raise AssertionError("expected KeyError when advantage is missing")


def test_grpo_objective_closed_form() -> None:
    """ratio≈1, advantage=3 → policy_loss = -3; ent_coef=0 → loss = -3."""
    objective_data = TensorDict(
        {
            "action": torch.tensor([[0, 0]]),
            "old_log_prob": torch.tensor([[0.0, 0.0]]),
            "advantage": torch.tensor([[0.0, 3.0]]),
        },
        batch_size=(1, 2),
    )
    predictions = TensorDict(
        {"action": torch.tensor([[[20.0, -20.0], [20.0, -20.0]]])},
        batch_size=(1, 2),
    )
    loss, metrics = GrpoObjective(ent_coef=0.0)(objective_data, predictions)
    assert abs(loss.item() - (-3.0)) < 1e-3
    assert abs(metrics["policy_loss"] - (-3.0)) < 1e-3


def test_grpo_objective_skips_pack_seams() -> None:
    # Only the second pair is same-segment; advantage there is 2 → loss = -2.
    objective_data = TensorDict(
        {
            "action": torch.tensor([[0, 0, 0]]),
            "old_log_prob": torch.zeros(1, 3),
            "advantage": torch.tensor([[9.0, 9.0, 2.0]]),
            "segment_id": torch.tensor([[0, 1, 1]]),
        },
        batch_size=(1, 3),
    )
    predictions = TensorDict(
        {
            "action": torch.tensor(
                [[[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]]
            )
        },
        batch_size=(1, 3),
    )
    loss, _ = GrpoObjective(ent_coef=0.0)(objective_data, predictions)
    assert abs(loss.item() - (-2.0)) < 1e-3
