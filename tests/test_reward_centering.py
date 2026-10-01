"""Tests for ConstantHead and DQN reward-centering loss."""

from __future__ import annotations

import pytest
import torch

from mouse_core.models import ConstantHead
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


def test_constant_head_starts_at_zero_and_ignores_backbone() -> None:
    center = ConstantHead(scale=2.0)
    assert center.scale == 2.0
    assert center.value.dtype is torch.float32
    assert center.value.item() == 0.0
    assert center.value.requires_grad
    h = torch.randn(3, 4, requires_grad=True)
    out = center(h)
    assert out.shape == ()
    assert out.item() == pytest.approx(0.0)
    with torch.no_grad():
        center.value.fill_(3.0)
    scaled = center(h)
    assert scaled.item() == pytest.approx(6.0)
    scaled.backward()
    assert h.grad is None
    assert center.value.grad is not None
    assert center.value.grad.item() == pytest.approx(2.0)


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
            reward_center=ConstantHead(scale=1.0),  # type: ignore[arg-type]
            delayed_reward_center=torch.zeros(()),
        )
    with pytest.raises(TypeError, match="reward_center"):
        objective(
            objective_data=step_stream, group_id=_group_id(step_stream),
            predictions=online,
            delayed_predictions=delayed,
            reward_center=torch.zeros(1),
            delayed_reward_center=torch.zeros(()),
        )


def test_dqn_loss_does_not_train_the_center() -> None:
    """gamma=0 so the target is r - c'. action_value trains Q. reward_center trains c."""
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
    center = ConstantHead(scale=1.0)
    losses, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=center.value,
        delayed_reward_center=center.value,
    )
    # c' = 0, so the shaped target is the plain residual: δ = [-1, 2], loss 2.5.
    assert losses["action_value"].item() == pytest.approx(2.5)
    assert metrics["action_value"].item() == pytest.approx(2.5)
    assert metrics["reward_center"].item() == pytest.approx(0.0)
    assert "reward_center_loss" not in metrics
    in_run_delta = metrics["in_run_delta"]
    assert isinstance(in_run_delta, torch.Tensor)
    assert in_run_delta.tolist() == pytest.approx([-1.0, 2.0])
    assert not in_run_delta.requires_grad

    losses["action_value"].backward()
    assert center.value.grad is None
    assert online.grad is not None
    # δ = [-1, 2], mean 0.5. The center loss is -c · mean(δ), so ∂L/∂c = -0.5.
    online.grad = None
    losses["reward_center"].backward()
    assert center.value.grad is not None
    assert center.value.grad.item() == pytest.approx(-0.5)


def test_dqn_centering_subtracts_same_offset_from_every_horizon() -> None:
    """Episodic nstep(2), γ=1 in-episode and 0 at the terminal: K = [1, 1].

    Constant-potential shaping subtracts (1 - γ)·c per reward slot, so
    the two-reward return and the one-reward return each lose exactly
    one ``c`` — the same offset for every horizon, which is what keeps
    the policy ordering intact (a per-reward count K = [2, 1] would
    penalize the longer backup). G_0 = r_1 + r_2 = 6, G_1 = r_2 = 5;
    δ = [6-2, 5-3] = [4, 2]. With c = 0.5 the residuals are
    [3.5, 1.5] → loss (12.25 + 2.25) / 2 = 7.25. The loss trains Q.
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
    losses, metrics = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=center,
        delayed_reward_center=center,
    )
    assert losses["action_value"].item() == pytest.approx(7.25)
    # Shaped backup is G − c. K = 1 on both in-run rows, so
    # [6, 5, 0] − 0.5 = [5.5, 4.5, 0].
    assert metrics["backup"].tolist() == pytest.approx([5.5, 4.5, 0.0])
    assert metrics["in_run_backup"].tolist() == pytest.approx([5.5, 4.5])
    # δ = shaped G − Q = [5.5 − 2, 4.5 − 3] = [3.5, 1.5].
    in_run_delta = metrics["in_run_delta"]
    assert isinstance(in_run_delta, torch.Tensor)
    assert in_run_delta.tolist() == pytest.approx([3.5, 1.5])
    assert not in_run_delta.requires_grad

    losses["action_value"].backward()
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
        delayed_reward_center=torch.tensor(3.0),
    )
    without, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=None, delayed_reward_center=None,
    )
    assert with_center["action_value"].item() == pytest.approx(without["action_value"].item())


def test_dqn_requires_both_centers_or_neither() -> None:
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="both be set or both be None"):
        _objective()(
            objective_data=step_stream, group_id=_group_id(step_stream),
            predictions=online,
            delayed_predictions=online,
            reward_center=torch.zeros(()),
            delayed_reward_center=None,
        )


def test_dqn_backup_uses_delayed_center() -> None:
    """G = r - (1 - γ) c' + γ V. The loss does not train c.

    γ = 0.9, c' = 2, V = 0. G = [1 - 0.2, 5 - 0.2] = [0.8, 4.8],
    δ = [-1.2, 1.8], loss (1.44 + 3.24) / 2 = 2.34.
    """
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 0, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]], requires_grad=True)
    delayed = torch.zeros(3, 2)
    center = torch.tensor(0.0, requires_grad=True)
    delayed_center = torch.tensor(2.0, requires_grad=True)
    objective = DqnObjective(
        reward=_rew(),
        value=_val(),
        discount=_disc(gamma_step=0.9),
        temperature=0.0,
        double=False,
        gate=None,
        cross_group_backups="bootstrap",
    )
    losses, metrics = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=center,
        delayed_reward_center=delayed_center,
    )
    assert losses["action_value"].item() == pytest.approx(2.34)
    assert metrics["backup"].tolist() == pytest.approx([0.8, 4.8, 0.0])
    losses["action_value"].backward()
    assert center.grad is None
    assert delayed_center.grad is None


def test_dqn_terminal_target_uses_delayed_center() -> None:
    """γ = 0 drops the bootstrap, so G = r - c'. The online center is unused."""
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    delayed = torch.zeros(3, 2)
    losses, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=torch.tensor(0.5),
        delayed_reward_center=torch.tensor(2.0),
    )
    # Unshaped G = [1, 5]. Delayed c' = 2 → [-1, 3]. Online c = 0.5 is unused.
    assert metrics["in_run_backup"].tolist() == pytest.approx([-1.0, 3.0])
    assert losses["action_value"].item() == pytest.approx(((-1.0 - 2.0) ** 2 + (3.0 - 3.0) ** 2) / 2)


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
    losses, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=torch.tensor(2.0),
        delayed_reward_center=torch.tensor(2.0),
    )
    assert losses["action_value"].item() == pytest.approx(2.34)


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
    base_losses, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=torch.tensor(0.0),
        delayed_reward_center=torch.tensor(0.0),
    )
    shifted_losses, _ = objective(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online - k,
        delayed_predictions=delayed - k,
        reward_center=torch.tensor(k),
        delayed_reward_center=torch.tensor(k),
    )
    assert shifted_losses["action_value"].item() == pytest.approx(
        base_losses["action_value"].item(), rel=1e-5
    )


def test_dqn_reward_center_metric_is_a_snapshot() -> None:
    """The logged center must not change when the parameter is written."""
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    center = ConstantHead(scale=1.0)
    _, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=torch.zeros(3, 2),
        delayed_predictions=torch.zeros(3, 2),
        reward_center=center.value,
        delayed_reward_center=center.value,
    )
    with torch.no_grad():
        center.value.add_(1.0)
    assert metrics["reward_center"].item() == pytest.approx(0.0)
    assert metrics["delayed_reward_center"].item() == pytest.approx(0.0)
    in_run_delta = metrics["in_run_delta"]
    assert isinstance(in_run_delta, torch.Tensor)
    assert in_run_delta.tolist() == pytest.approx([1.0, 5.0])


def test_dqn_without_centering_omits_center_metrics() -> None:
    step_stream = {
        "action": torch.tensor([0, 1, 0]),
        "reward": torch.tensor([0.0, 1.0, 5.0]),
        "episode_done": torch.tensor([0, 1, 0]),
        "task_done": torch.tensor([0, 0, 0]),
    }
    online = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    delayed = torch.zeros(3, 2)
    losses, metrics = _objective()(
        objective_data=step_stream, group_id=_group_id(step_stream),
        predictions=online,
        delayed_predictions=delayed,
        reward_center=None, delayed_reward_center=None,
    )
    assert losses["action_value"].item() == pytest.approx(2.5)
    assert "reward_center" not in metrics
    assert "delayed_reward_center" not in metrics
    assert "in_run_delta" not in metrics
