"""Tests for the n-step DQN objective on synthetic tensors."""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from mouse_core.objectives import DqnObjective, NStepDqnObjective
from mouse_core.objectives.nstep import _n_step_targets


_GAMMAS = dict(
    gamma_step=1.0,
    gamma_episode_terminal=0.0,
    gamma_episode_truncated=0.0,
    gamma_task_terminal=0.0,
    gamma_task_truncated=0.0,
)


def _q(
    online: torch.Tensor,
    delayed: torch.Tensor,
    *,
    key: str = "action_value",
) -> tuple[TensorDict, TensorDict]:
    n = online.shape[0]
    return (
        TensorDict({key: online}, batch_size=[n]),
        TensorDict({key: delayed}, batch_size=[n]),
    )


def _nstep(**overrides: object) -> NStepDqnObjective:
    kwargs: dict[str, object] = dict(
        n=1, prediction_key="action_value", grouping_field=None, **_GAMMAS
    )
    kwargs.update(overrides)
    return NStepDqnObjective(**kwargs)  # type: ignore[arg-type]


def _lambda_fixture() -> tuple[TensorDict, TensorDict, TensorDict]:
    """Three in-run steps. Action from s0 is 0; from s1 is 1.

    Delayed max-Q is 3 at s1 and 100 at s2. Rewards out of s0 / s1 are 1
    and 10. One-step target at s0 is 4; the two-step target at s0 is 111.
    """
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.zeros(3, dtype=torch.int64),
        },
        batch_size=[3],
    )
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    return step_stream, *_q(online, delayed)


# One-step MSEs: (5-4)^2 = 1 and (0-110)^2 = 12100 → mean 6050.5.
_ONE_STEP = 6050.5
# Two-step target at s0 is 1 + 10 + 100 = 111 → (5-111)^2 = 11236; s1 stays 12100 → mean 11668.
_TWO_STEP = 11668.0


def test_nstep_requires_n_prediction_key_and_grouping_field() -> None:
    with pytest.raises(TypeError, match="n"):
        NStepDqnObjective(  # type: ignore[call-arg]
            prediction_key="action_value", grouping_field=None, **_GAMMAS
        )
    with pytest.raises(TypeError, match="prediction_key"):
        NStepDqnObjective(n=1, grouping_field=None, **_GAMMAS)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="grouping_field"):
        NStepDqnObjective(n=1, prediction_key="action_value", **_GAMMAS)  # type: ignore[call-arg]


def test_nstep_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="int >= 1"):
        _nstep(n=0)
    with pytest.raises(ValueError, match="int >= 1"):
        _nstep(n=1.5)
    with pytest.raises(ValueError, match="int >= 1"):
        _nstep(n=True)


def test_nstep_rejects_empty_prediction_key() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _nstep(prediction_key="")


def test_nstep_requires_delayed_predictions() -> None:
    step_stream, predictions, _ = _lambda_fixture()
    with pytest.raises(ValueError, match="delayed_predictions"):
        _nstep()(step_stream, predictions)


def test_nstep_requires_prediction_key_on_both_sides() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    with pytest.raises(KeyError, match="action_value_3"):
        _nstep(prediction_key="action_value_3")(step_stream, predictions, delayed)
    delayed_only = TensorDict(
        {"action_value_3": delayed["action_value"]}, batch_size=delayed.batch_size
    )
    with pytest.raises(KeyError, match="delayed_predictions"):
        _nstep()(step_stream, predictions, delayed_only)


def test_nstep_one_matches_dqn() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    nstep, metrics = _nstep(n=1)(step_stream, predictions, delayed)
    dqn, _ = DqnObjective(grouping_field=None, **_GAMMAS)(
        step_stream, predictions, delayed
    )
    assert abs(nstep.item() - dqn.item()) < 1e-05
    assert abs(nstep.item() - _ONE_STEP) < 1e-03
    assert "n_step" in metrics
    assert "watkins_greedy_frac" not in metrics


def test_nstep_two_is_the_two_step_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = _nstep(n=2)(step_stream, predictions, delayed)
    assert abs(loss.item() - _TWO_STEP) < 1e-03


def test_nstep_past_batch_end_truncates_to_available_steps() -> None:
    """n larger than the remaining run is the same as the longest available."""
    step_stream, predictions, delayed = _lambda_fixture()
    two, _ = _nstep(n=2)(step_stream, predictions, delayed)
    ten, _ = _nstep(n=10)(step_stream, predictions, delayed)
    assert abs(two.item() - ten.item()) < 1e-05


def test_n_step_targets_window() -> None:
    reward = torch.tensor([0.0, 1.0, 10.0, 100.0])
    discount = torch.ones(4)
    v = torch.tensor([0.0, 3.0, 7.0, 11.0])
    pair_weight = torch.ones(3)
    got = _n_step_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight, n=2
    )
    # G0 = 1 + 10 + 7 = 18; G1 = 10 + 100 + 11 = 121; G2 = 100 + 11 = 111.
    assert torch.allclose(got, torch.tensor([18.0, 121.0, 111.0]))


def test_nstep_terminal_gamma_zero_ends_the_sum() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["episode_done"] = torch.tensor([0, 1, 0])
    loss, _ = _nstep(n=2)(step_stream, predictions, delayed)
    # s0: (5 - 1)^2 = 16 — neither V(s1) nor the next episode's return; s1: 12100.
    assert abs(loss.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_nstep_truncation_gamma_carries_the_sum_discounted() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["episode_done"] = torch.tensor([0, 2, 0])
    loss, _ = _nstep(
        n=2, gamma_episode_truncated=0.5, gamma_episode_terminal=0.0
    )(step_stream, predictions, delayed)
    # s0: 1 + 0.5 * 10 + 0.5 * 100 = 56 → (5 - 56)^2 = 2601; s1: 12100.
    assert abs(loss.item() - (2601.0 + 12100.0) / 2) < 1e-03


def test_nstep_does_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["sequence_id"] = torch.tensor([0, 0, 1])
    two, _ = _nstep(n=2)(step_stream, predictions, delayed)
    one, _ = _nstep(n=1)(step_stream, predictions, delayed)
    assert abs(two.item() - one.item()) < 1e-04


def test_nstep_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.tensor([0, 1, 2]),
        },
        batch_size=[3],
    )
    predictions, delayed = _q(torch.ones(3, 2), torch.ones(3, 2))
    loss, metrics = _nstep(n=2)(step_stream, predictions, delayed)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics["q_values_mean"]) < 1e-05


def test_nstep_trains_multiple_heads_independently() -> None:
    """Two keys, two n's: each loss reads only its own Q and they add."""
    step_stream, pred_1, delayed_1 = _lambda_fixture()
    q1 = pred_1["action_value"].detach().clone().requires_grad_(True)
    q3 = pred_1["action_value"].detach().clone().requires_grad_(True)
    delayed_q = delayed_1["action_value"]
    predictions = TensorDict(
        {"action_value_1": q1, "action_value_3": q3}, batch_size=[3]
    )
    delayed = TensorDict(
        {"action_value_1": delayed_q, "action_value_3": delayed_q},
        batch_size=[3],
    )
    loss_1, m1 = _nstep(n=1, prediction_key="action_value_1")(
        step_stream, predictions, delayed
    )
    loss_3, m3 = _nstep(n=3, prediction_key="action_value_3")(
        step_stream, predictions, delayed
    )
    assert abs(loss_1.item() - _ONE_STEP) < 1e-03
    assert abs(loss_3.item() - _TWO_STEP) < 1e-03
    assert m1["n_step"] != m3["n_step"]
    total = loss_1 + loss_3
    total.backward()
    assert q1.grad is not None
    assert q3.grad is not None
    assert not torch.equal(q1.grad, torch.zeros_like(q1.grad))
    assert not torch.equal(q3.grad, torch.zeros_like(q3.grad))


def test_nstep_does_not_backprop_through_delayed_q() -> None:
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
    delayed_q = torch.randn(n, a, requires_grad=True)
    predictions, delayed = _q(online, delayed_q)
    loss, _ = _nstep(n=2)(step_stream, predictions, delayed)
    loss.backward()
    assert online.grad is not None
    assert delayed_q.grad is None


def test_nstep_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's n-step return."""
    step_stream, _, _ = _lambda_fixture()
    step_stream = step_stream.clone()
    step_stream["head_output_count"] = torch.tensor([2, 1, 2])
    online = torch.tensor(
        [[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]]
    )
    delayed_q = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]]
    )
    predictions, delayed = _q(online, delayed_q)
    loss, _ = _nstep(n=2)(step_stream, predictions, delayed)
    # s0 rows: (5-111)^2 = 11236, (7-111)^2 = 10816; s1 row: (0-110)^2 = 12100.
    assert abs(loss.item() - (11236.0 + 10816.0 + 12100.0) / 3) < 1e-02


def test_nstep_q_affine_applies_to_online_and_delayed() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = _nstep(n=1, q_scale=2.0)(step_stream, predictions, delayed)
    expected = (9.0 + 44100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_nstep_requires_min_sequence() -> None:
    step_stream = TensorDict(
        {
            "action": torch.zeros(1, dtype=torch.long),
            "reward": torch.zeros(1),
            "episode_done": torch.zeros(1, dtype=torch.long),
            "task_done": torch.zeros(1, dtype=torch.long),
        },
        batch_size=[1],
    )
    predictions, delayed = _q(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Not enough"):
        _nstep()(step_stream, predictions, delayed)


def test_nstep_rejects_non_fp32_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.clone()
    predictions["action_value"] = predictions["action_value"].to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        _nstep()(step_stream, predictions, delayed)
