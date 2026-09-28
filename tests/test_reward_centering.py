"""Tests for RewardCentering and DQN reward-centering loss."""

from __future__ import annotations

import math

import pytest
import torch

from mouse_core.models import RewardCentering
from mouse_core.objectives import DqnObjective, affine_reward, affine_value, boundary_discount


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    kwargs.update(overrides)
    return boundary_discount(**kwargs)


def _rew(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_reward(**kwargs)  # type: ignore[arg-type]


def _val(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_value(**kwargs)  # type: ignore[arg-type]


def test_reward_centering_rejects_bad_init_and_scale() -> None:
    with pytest.raises(ValueError, match="init"):
        RewardCentering(init=math.nan, loss_scale=1.0)
    with pytest.raises(ValueError, match="loss_scale"):
        RewardCentering(init=0.0, loss_scale=-1.0)
    with pytest.raises(ValueError, match="loss_scale"):
        RewardCentering(init=0.0, loss_scale=math.inf)


def test_reward_centering_mse_trains_only_the_constant() -> None:
    center = RewardCentering(init=0.0, loss_scale=1.0)
    td = torch.tensor([2.0, 4.0], requires_grad=True)
    weight = torch.ones(2)
    loss = center.centering_mse(td_error=td, weight=weight)
    loss.backward()
    assert center.center.grad is not None
    assert td.grad is None
    # mean target is 3; grad of (c - 3)^2 w.r.t. c at 0 is -6
    assert center.center.grad.item() == pytest.approx(-6.0)


def test_reward_centering_loss_scale_multiplies_mse() -> None:
    center = RewardCentering(init=1.0, loss_scale=0.5)
    td = torch.tensor([1.0, 1.0])
    weight = torch.ones(2)
    # (1-1)^2 = 0
    assert center.centering_mse(td_error=td, weight=weight).item() == pytest.approx(0.0)
    center2 = RewardCentering(init=0.0, loss_scale=2.0)
    td2 = torch.tensor([1.0, 1.0])
    # mean (0-1)^2 = 1, * 2 = 2
    assert center2.centering_mse(td_error=td2, weight=weight).item() == pytest.approx(2.0)


def test_dqn_requires_reward_centering_argument() -> None:
    with pytest.raises(TypeError, match="reward_centering"):
        DqnObjective(  # type: ignore[call-arg]
            reward=_rew(),
            value=_val(),
            discount=_disc(),
            grouping_field=None,
            temperature=0.0,
            double=False,
            gate=None,
            bootstrap_cutoff=True,
        )


def test_dqn_rejects_non_reward_centering() -> None:
    with pytest.raises(TypeError, match="RewardCentering"):
        DqnObjective(
            reward=_rew(),
            value=_val(),
            discount=_disc(),
            grouping_field=None,
            temperature=0.0,
            double=False,
            gate=None,
            bootstrap_cutoff=True,
            reward_centering=object(),  # type: ignore[arg-type]
        )


def test_dqn_reward_centering_centers_td_and_trains_constant() -> None:
    """gamma=0 so targets are rewards; constant learns mean δ with its own grad."""
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    # Taken Q at next-action rows: actions [1,0] → Q 2 and 3; targets 1 and 5.
    # δ = [1-2, 5-3] = [-1, 2]; mean δ = 0.5
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]], requires_grad=True)
    delayed = torch.zeros(3, 2)
    center = RewardCentering(init=0.0, loss_scale=1.0)
    objective = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.0),
        grouping_field=None,
        temperature=0.0,
        double=False,
        gate=None,
        bootstrap_cutoff=True,
        reward_centering=center,
    )
    loss, metrics = objective(
        objective_data=step_stream,
        predictions=online,
        delayed_predictions=delayed,
    )
    # Centered residuals: (-1 - 0)^2 and (2 - 0)^2 → mean 2.5
    # Centering MSE: (0 - (-1))^2 and (0 - 2)^2 → mean 2.5
    # Total 5.0
    assert loss.item() == pytest.approx(5.0)
    assert metrics["action_value"].item() == pytest.approx(2.5)
    assert metrics["reward_center"].item() == pytest.approx(0.0)
    assert metrics["reward_center_loss"].item() == pytest.approx(2.5)

    loss.backward()
    assert center.center.grad is not None
    # d/dc mean((c - δ_det)^2) at 0 with δ=[-1,2]: mean(2(c-δ)) = mean(-2δ) = -1
    assert center.center.grad.item() == pytest.approx(-1.0)
    assert online.grad is not None


def test_dqn_without_centering_omits_center_metrics() -> None:
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    delayed = torch.zeros(3, 2)
    loss, metrics = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.0),
        grouping_field=None,
        temperature=0.0,
        double=False,
        gate=None,
        bootstrap_cutoff=True,
        reward_centering=None,
    )(
        objective_data=step_stream,
        predictions=online,
        delayed_predictions=delayed,
    )
    assert loss.item() == pytest.approx(2.5)
    assert "reward_center" not in metrics
    assert "reward_center_loss" not in metrics
