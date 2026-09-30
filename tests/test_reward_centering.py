"""Tests for RewardCentering and DQN reward-centering loss."""

from __future__ import annotations

import pytest
import torch

from mouse_core.models import RewardCentering, reward_centering_polyak
from mouse_core.objectives import (
    DqnObjective,
    affine_reward,
    affine_value,
    boundary_discount,
    lambda_gate,
    nstep_gate,
)


def _group_id(data: dict[str, torch.Tensor]) -> torch.Tensor:
    if "group_id" in data:
        return data["group_id"]
    n = int(next(iter(data.values())).shape[0])
    return torch.zeros(n, dtype=torch.int64)


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


def test_reward_centering_starts_at_zero() -> None:
    center = RewardCentering()
    assert center.center.dtype is torch.float32
    assert center.center.item() == 0.0
    assert not center.center.requires_grad
    assert list(center.parameters()) == []


def _objective() -> DqnObjective:
    return DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.0),
        temperature=0.0,
        double=False,
        gate=None,
        cross_group_backups="bootstrap",
    )


def test_dqn_requires_reward_center_argument() -> None:
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    with pytest.raises(TypeError, match="reward_center"):
        _objective()(  # type: ignore[call-arg]
            objective_data=step_stream, group_id=_group_id(step_stream),
            predictions=torch.zeros(3, 2),
            delayed_predictions=torch.zeros(3, 2),
        )


def test_dqn_rejects_non_scalar_reward_center() -> None:
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    objective = _objective()
    online = torch.zeros(3, 2)
    delayed = torch.zeros(3, 2)
    with pytest.raises(TypeError, match="reward_center"):
        objective(
            objective_data=step_stream, group_id=_group_id(step_stream),
            predictions=online,
            delayed_predictions=delayed,
            reward_center=RewardCentering(),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="reward_center"):
        objective(
            objective_data=step_stream, group_id=_group_id(step_stream),
            predictions=online,
            delayed_predictions=delayed,
            reward_center=torch.zeros(1),
        )


def test_dqn_reward_centering_centers_td_without_training_constant() -> None:
    """gamma=0 so targets are rewards; the loss does not train the constant."""
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
    center = RewardCentering()
    loss, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=center.center,
    )
    # c = 0, so (δ - c)² is the plain residual: δ = [-1, 2], mean 2.5.
    # c is detached, so only Q receives a gradient.
    assert loss.item() == pytest.approx(2.5)
    assert isinstance(metrics["action_value"], torch.Tensor)
    assert metrics["action_value"].item() == pytest.approx(2.5)
    assert isinstance(metrics["reward_center"], torch.Tensor)
    assert metrics["reward_center"].item() == pytest.approx(0.0)
    assert "reward_center_loss" not in metrics

    loss.backward()
    assert center.center.grad is None
    assert online.grad is not None


def test_dqn_centering_subtracts_same_offset_from_every_horizon() -> None:
    """Episodic nstep(2), γ=1 in-episode and 0 at the terminal: K = [1, 1].

    Constant-potential shaping subtracts (1 - γ)·c per reward slot, so
    the two-reward return and the one-reward return each lose exactly
    one ``c`` — the same offset for every horizon, which is what keeps
    the policy ordering intact (a per-reward count K = [2, 1] would
    penalize the longer backup). G_0 = r_1 + r_2 = 6, G_1 = r_2 = 5;
    δ = [6-2, 5-3] = [4, 2]. With c = 0.5 the residuals are
    [3.5, 1.5] → loss (12.25 + 2.25) / 2 = 7.25. c is detached, so it
    receives no gradient.
    """
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 0, 1]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]], requires_grad=True)
    delayed = torch.zeros(3, 2)
    center = torch.tensor(0.5, requires_grad=True)
    objective = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=1.0),
        temperature=0.0,
        double=False,
        gate=nstep_gate(n=2),
        cross_group_backups="bootstrap",
    )
    loss, metrics = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=center,
    )
    assert loss.item() == pytest.approx(7.25)
    # backup stays the uncentered G.
    assert isinstance(metrics["backup"], torch.Tensor)
    assert metrics["backup"].tolist() == pytest.approx([6.0, 5.0, 0.0])
    assert isinstance(metrics["in_run_backup"], torch.Tensor)
    assert metrics["in_run_backup"].tolist() == pytest.approx([6.0, 5.0])

    loss.backward()
    assert center.grad is None
    assert online.grad is not None


def test_dqn_centering_subtracts_nothing_on_undiscounted_bootstrapped_steps() -> None:
    """γ=1 with no terminal in the sample: K = 0, centering is a no-op.

    With every slot at γ = 1 the shaped reward r + (γ - 1)·c is r, and
    any constant offset in Q is self-consistent — there is nothing for a
    constant to remove, and subtracting one per reward would skew long
    backups against short ones.
    """
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 0, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    delayed = torch.zeros(3, 2)
    objective = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=1.0),
        temperature=0.0,
        double=False,
        gate=nstep_gate(n=2),
        cross_group_backups="bootstrap",
    )
    with_center, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=torch.tensor(3.0),
    )
    without, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=None,
    )
    assert with_center.item() == pytest.approx(without.item())


def test_dqn_centering_matches_classic_reward_centering_when_continuing() -> None:
    """Constant γ < 1, one-step: each slot subtracts (1 - γ)·c.

    With c tracking the mean value r̄/(1-γ), the per-slot subtraction is
    the average reward r̄ — classic reward centering. Here γ = 0.9 and
    c = 2: δ = [1-2, 5-3] = [-1, 2] (delayed Q is 0), K = 0.1, so the
    residuals are [-1.2, 1.8] → loss (1.44 + 3.24) / 2 = 2.34.
    """
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 0, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    delayed = torch.zeros(3, 2)
    objective = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.9),
        temperature=0.0,
        double=False,
        gate=None,
        cross_group_backups="bootstrap",
    )
    loss, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=torch.tensor(2.0),
    )
    assert loss.item() == pytest.approx(2.34)


def test_dqn_centering_is_a_pure_value_shift() -> None:
    """Shifting all Q by -k and setting c = k reproduces the exact loss.

    The shaping telescopes to the same offset on every action value, so
    the centered system is a reparameterization of the plain one: greedy
    actions, soft policies, and the policy ordering never change — for
    any gate, any done pattern, and any c.
    """
    torch.manual_seed(0)
    step_stream = {
        "action": torch.tensor([0, 1, 0, 2, 1, 0]),
        "reward": torch.randn(6),
        "episode_done": torch.tensor([0, 0, 1, 0, 0, 2]),
        "task_done": torch.tensor([0, 0, 0, 0, 0, 0]),
    }
    online = torch.randn(6, 3)
    delayed = torch.randn(6, 3)
    objective = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(
            gamma_step=0.9,
            gamma_episode_terminal=0.0,
            gamma_episode_truncated=1.0,
        ),
        temperature=0.7,
        double=True,
        gate=lambda_gate(td_lambda=0.8),
        cross_group_backups="bootstrap",
    )
    k = 2.5
    base, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=torch.tensor(0.0),
    )
    shifted, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online - k,
        delayed_predictions=delayed - k,
        reward_center=torch.tensor(k),
    )
    assert shifted.item() == pytest.approx(base.item(), rel=1e-5)


def test_reward_centering_update_is_polyak_average_of_values() -> None:
    """center ← τ·mean(values) + (1−τ)·center. τ = 0 keeps it; τ = 1 replaces it."""
    center = RewardCentering()
    values = torch.tensor([0.0, 1.0, 5.0])
    reward_centering_polyak(center=center, tau=0.0, values=values)
    assert center.center.item() == pytest.approx(0.0)
    reward_centering_polyak(center=center, tau=1.0, values=values)
    assert center.center.item() == pytest.approx(2.0)
    reward_centering_polyak(center=center, tau=0.5, values=torch.tensor([0.0, 0.0]))
    assert center.center.item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="tau"):
        reward_centering_polyak(center=center, tau=1.5, values=values)
    with pytest.raises(TypeError, match="RewardCentering"):
        reward_centering_polyak(
            center=torch.zeros(()),  # type: ignore[arg-type]
            tau=0.0,
            values=torch.zeros(1),
        )


def test_dqn_reward_center_metric_is_a_snapshot() -> None:
    """The logged center must not change when ``update`` writes the buffer."""
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    center = RewardCentering()
    _, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=torch.zeros(3, 2),
        delayed_predictions=torch.zeros(3, 2),
        reward_center=center.center,
    )
    with torch.no_grad():
        center.center.add_(1.0)
    assert isinstance(metrics["reward_center"], torch.Tensor)
    assert metrics["reward_center"].item() == pytest.approx(0.0)


def test_dqn_without_centering_omits_center_metrics() -> None:
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    delayed = torch.zeros(3, 2)
    loss, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=None,
    )
    assert loss.item() == pytest.approx(2.5)
    assert "reward_center" not in metrics
