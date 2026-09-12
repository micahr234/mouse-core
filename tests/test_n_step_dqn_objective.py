"""Tests for the n-step return DQN objective on synthetic tensors."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import DqnObjective, NStepDqnObjective
import pytest


def _q(online: torch.Tensor, delayed: torch.Tensor) -> tuple[TensorDict, TensorDict]:
    n = online.shape[0]
    return (
        TensorDict({"action_value": online}, batch_size=[n]),
        TensorDict({"action_value": delayed}, batch_size=[n]),
    )


def _objective(*, n: int, **kwargs) -> NStepDqnObjective:
    defaults = dict(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    defaults.update(kwargs)
    return NStepDqnObjective(n=n, **defaults)


def _return_fixture() -> tuple[TensorDict, TensorDict, TensorDict]:
    """Four in-run steps so n=1 / n=2 / n=3 targets differ at s0.

    Rewards out of s0 / s1 / s2 are 1, 10, 100. Delayed max-Q is 3 at s1,
    7 at s2, and 1000 at s3. Online Q(s0, a=0) = 5; later taken Q is 0.
    """
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 10.0, 100.0]),
            "episode_done": torch.zeros(4, dtype=torch.int64),
            "task_done": torch.zeros(4, dtype=torch.int64),
            "sequence_id": torch.zeros(4, dtype=torch.int64),
        },
        batch_size=[4],
    )
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 7.0], [1000.0, 0.0]])
    return step_stream, *_q(online, delayed)


def _mse(online_taken: list[float], targets: list[float]) -> float:
    errs = [(o - t) ** 2 for o, t in zip(online_taken, targets, strict=True)]
    return sum(errs) / len(errs)


# Taken Q: 5, 0, 0. Targets below; last pair has no next step (weight 0).
_ONE_STEP = _mse([5.0, 0.0, 0.0], [4.0, 17.0, 1100.0])  # 1 + 3, 10 + 7, 100 + 1000
_TWO_STEP = _mse([5.0, 0.0, 0.0], [18.0, 1110.0, 1100.0])  # 1+17, 10+1100, 100+1000
_THREE_STEP = _mse([5.0, 0.0, 0.0], [1111.0, 1110.0, 1100.0])  # 1+1110, 10+1100, 100+1000


def test_n_step_objective_runs() -> None:
    n, a = (8, 3)
    step_stream = TensorDict(
        {
            "action": torch.randint(0, a, (n,)),
            "reward": torch.randn(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
            "sequence_id": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        },
        batch_size=[n],
    )
    predictions, delayed = _q(torch.randn(n, a), torch.randn(n, a))
    loss, metrics = _objective(n=3, gamma_step=0.99)(step_stream, predictions, delayed)
    assert loss.ndim == 0
    assert "action_value" in metrics
    assert metrics["action_value"] >= 0.0
    assert "watkins_greedy_frac" not in metrics


def test_n_step_one_is_the_one_step_target() -> None:
    step_stream, predictions, delayed = _return_fixture()
    loss, metrics = _objective(n=1)(step_stream, predictions, delayed)
    assert abs(loss.item() - _ONE_STEP) < 1e-03
    assert "watkins_greedy_frac" not in metrics


def test_n_step_one_matches_dqn_objective() -> None:
    step_stream, predictions, delayed = _return_fixture()
    nstep, _ = _objective(n=1)(step_stream, predictions, delayed)
    dqn, _ = DqnObjective(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )(step_stream, predictions, delayed)
    assert abs(nstep.item() - dqn.item()) < 1e-05


def test_n_step_two_uses_two_rewards_then_bootstraps() -> None:
    step_stream, predictions, delayed = _return_fixture()
    loss, _ = _objective(n=2)(step_stream, predictions, delayed)
    assert loss.item() == pytest.approx(_TWO_STEP, rel=1e-05)


def test_n_step_three_uses_three_rewards_then_bootstraps() -> None:
    step_stream, predictions, delayed = _return_fixture()
    loss, _ = _objective(n=3)(step_stream, predictions, delayed)
    assert loss.item() == pytest.approx(_THREE_STEP, rel=1e-05)


def test_n_step_past_horizon_matches_available_return() -> None:
    """n larger than the run is the full in-run return, same as n=3 here."""
    step_stream, predictions, delayed = _return_fixture()
    loss, _ = _objective(n=8)(step_stream, predictions, delayed)
    assert loss.item() == pytest.approx(_THREE_STEP, rel=1e-05)


def test_n_step_does_not_cut_on_non_greedy_actions() -> None:
    """Taken action at s1 is not online-greedy; the 3-step return still continues."""
    step_stream, predictions, delayed = _return_fixture()
    predictions = predictions.clone()
    q = predictions["action_value"].clone()
    q[1] = torch.tensor([10.0, 0.0])
    predictions["action_value"] = q
    loss, _ = _objective(n=3)(step_stream, predictions, delayed)
    assert loss.item() == pytest.approx(_THREE_STEP, rel=1e-05)


def test_n_step_terminal_gamma_zero_ends_the_return() -> None:
    """s0 → s1 terminates the episode (code stored at s1): G_0 = r only."""
    step_stream, predictions, delayed = _return_fixture()
    step_stream = step_stream.clone()
    step_stream["episode_done"] = torch.tensor([0, 1, 0, 0])
    loss, _ = _objective(n=3)(step_stream, predictions, delayed)
    # s0: (5 - 1)^2 = 16; s1/s2 still use the remaining in-run 2-step / 1-step.
    expected = _mse([5.0, 0.0, 0.0], [1.0, 1110.0, 1100.0])
    assert loss.item() == pytest.approx(expected, rel=1e-05)


def test_n_step_truncation_gamma_carries_the_return_discounted() -> None:
    """A non-zero truncation gamma bootstraps through the reset, so n-step does too."""
    step_stream, predictions, delayed = _return_fixture()
    step_stream = step_stream.clone()
    step_stream["episode_done"] = torch.tensor([0, 2, 0, 0])
    loss, _ = _objective(n=2, gamma_episode_truncated=0.5)(step_stream, predictions, delayed)
    # s0 n=2: 1 + 0.5 * (10 + 7) = 9.5; s1 n=2: 10 + 1 * 1100 = 1110; s2: 1100.
    expected = _mse([5.0, 0.0, 0.0], [9.5, 1110.0, 1100.0])
    assert loss.item() == pytest.approx(expected, rel=1e-05)


def test_n_step_does_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _return_fixture()
    step_stream = step_stream.clone()
    step_stream["sequence_id"] = torch.tensor([0, 0, 1, 1])
    loss, _ = _objective(n=3)(step_stream, predictions, delayed)
    # Pair (0,1) is in-run but (1,2) is not, so G_0 bootstraps V(s1)=3.
    # Pair (2,3) is its own run: G_2 = 100 + 1000 = 1100.
    expected = _mse([5.0, 0.0], [4.0, 1100.0])
    assert abs(loss.item() - expected) < 1e-04


def test_n_step_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's n-step return; bootstrap reads the last row."""
    step_stream, predictions, delayed = _return_fixture()
    step_stream = step_stream.clone()
    step_stream["head_output_count"] = torch.tensor([2, 1, 1, 1])
    online = torch.tensor([[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed_q = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [0.0, 7.0], [1000.0, 0.0]])
    predictions, delayed = _q(online, delayed_q)
    loss, _ = _objective(n=2)(step_stream, predictions, delayed)
    # s0 rows both use G=18: (5-18)^2=169, (7-18)^2=121; s1: (0-1110)^2; s2: (0-1100)^2.
    expected = (169.0 + 121.0 + (1110.0 ** 2) + (1100.0 ** 2)) / 4
    assert abs(loss.item() - expected) < 1e-02


def test_n_step_does_not_backprop_through_delayed_q() -> None:
    n, a = 4, 2
    step_stream = TensorDict(
        {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
        },
        batch_size=[n],
    )
    online = torch.randn(n, a, requires_grad=True)
    delayed = torch.randn(n, a, requires_grad=True)
    predictions, delayed_td = _q(online, delayed)
    loss, _ = _objective(n=2)(step_stream, predictions, delayed_td)
    loss.backward()
    assert online.grad is not None
    assert delayed.grad is None


def test_n_step_reward_scale_and_shift() -> None:
    """n=1 and gamma=0 so targets are the affine rewards stored at i+1."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0]),
            "task_done": torch.tensor([0, 0, 0]),
        },
        batch_size=[3],
    )
    predictions, delayed = _q(torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]), torch.zeros(3, 2))
    scaled, _ = _objective(n=1, gamma_step=0.0, reward_scale=2.0)(step_stream, predictions, delayed)
    assert abs(scaled.item() - 24.5) < 1e-05
    shifted, _ = _objective(n=1, gamma_step=0.0, reward_shift=1.0)(step_stream, predictions, delayed)
    assert abs(shifted.item() - 4.5) < 1e-05
    assert torch.equal(step_stream["reward"], torch.tensor([0.0, 1.0, 5.0]))


def test_n_step_rejects_n_less_than_one() -> None:
    with pytest.raises(ValueError, match="integer >= 1"):
        _objective(n=0)


def test_n_step_rejects_non_integer_n() -> None:
    with pytest.raises(ValueError, match="integer >= 1"):
        NStepDqnObjective(
            n=1.5,  # type: ignore[arg-type]
            gamma_step=1.0,
            gamma_episode_terminal=0.0,
            gamma_episode_truncated=0.0,
            gamma_task_terminal=0.0,
            gamma_task_truncated=0.0,
        )


def test_n_step_requires_delayed_predictions() -> None:
    step_stream, predictions, _ = _return_fixture()
    with pytest.raises(ValueError, match="delayed_predictions"):
        _objective(n=2)(step_stream, predictions)


def test_n_step_rejects_non_fp32_q() -> None:
    step_stream, predictions, delayed = _return_fixture()
    predictions = predictions.clone()
    predictions["action_value"] = predictions["action_value"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        _objective(n=2)(step_stream, predictions, delayed)
