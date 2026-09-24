"""Tests for fixed-horizon DQN backups on synthetic tensors."""
from __future__ import annotations

import pytest
import torch

from mouse_core.objectives import (
    DqnObjective,
    affine_reward,
    affine_value,
    boundary_discount,
    lambda_gate,
    nstep_gate,
)
from mouse_core.objectives.dqn import _continuation_targets


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


def _q(
    online: torch.Tensor,
    delayed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return online, delayed


def _nstep(**overrides: object) -> DqnObjective:
    n = overrides.pop("n", 1)
    kwargs: dict[str, object] = dict(
        gate=nstep_gate(n=n),  # type: ignore[arg-type]
        grouping_field=None, temperature=0.0, double=False,         discount=_disc(), reward=_rew(), value=_val(),
    )
    kwargs.update(overrides)
    return DqnObjective(**kwargs)  # type: ignore[arg-type]


def _lambda_fixture() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Three in-run steps. Action from s0 is 0; from s1 is 1.

    Delayed max-Q is 3 at s1 and 100 at s2. Rewards out of s0 / s1 are 1
    and 10. One-step target at s0 is 4; the two-step target at s0 is 111.
    """
    step_stream = {
            "action": torch.tensor([0, 0, 1]),
            "reward": torch.tensor([0.0, 1.0, 10.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.zeros(3, dtype=torch.int64),
        }
    online = torch.tensor([[5.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 100.0]])
    return step_stream, *_q(online, delayed)


# One-step MSEs: (5-4)^2 = 1 and (0-110)^2 = 12100 → mean 6050.5.
_ONE_STEP = 6050.5
# Two-step target at s0 is 1 + 10 + 100 = 111 → (5-111)^2 = 11236; s1 stays 12100 → mean 11668.
_TWO_STEP = 11668.0


def test_nstep_requires_gate_and_double() -> None:
    with pytest.raises(TypeError, match="gate"):
        DqnObjective(  # type: ignore[call-arg]
            grouping_field=None, temperature=0.0, double=False,
            discount=_disc(), reward=_rew(), value=_val(),
        )
    with pytest.raises(TypeError, match="double"):
        DqnObjective(  # type: ignore[call-arg]
            gate=nstep_gate(n=1),
            grouping_field=None, temperature=0.0,
            discount=_disc(), reward=_rew(), value=_val(),
        )


def test_nstep_gate_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="int >= 1"):
        nstep_gate(n=0)
    with pytest.raises(ValueError, match="int >= 1"):
        nstep_gate(n=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="int >= 1"):
        nstep_gate(n=True)  # type: ignore[arg-type]


def test_nstep_requires_delayed_predictions() -> None:
    step_stream, predictions, _ = _lambda_fixture()
    with pytest.raises(TypeError, match="delayed_predictions"):
        _nstep()(objective_data=step_stream, predictions=predictions)  # type: ignore[call-arg]


def test_nstep_one_matches_dqn() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    nstep, metrics = _nstep(n=1)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    dqn, _ = DqnObjective( grouping_field=None, temperature=0.0, double=False, gate=None, discount=_disc(), reward=_rew(), value=_val())(
        objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed
    )
    assert abs(nstep.item() - dqn.item()) < 1e-05
    assert abs(nstep.item() - _ONE_STEP) < 1e-03
    assert "action_value" in metrics
    assert "watkins_greedy_frac" not in metrics


def test_nstep_two_is_the_two_step_return() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    assert abs(loss.item() - _TWO_STEP) < 1e-03


def test_nstep_past_batch_end_truncates_to_available_steps() -> None:
    """n larger than the remaining run is the same as the longest available."""
    step_stream, predictions, delayed = _lambda_fixture()
    two, _ = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    ten, _ = _nstep(n=10)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    assert abs(two.item() - ten.item()) < 1e-05


def _blank_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(n, 1), torch.zeros(n, dtype=torch.long)


def test_n_step_targets_window() -> None:
    reward = torch.tensor([0.0, 1.0, 10.0, 100.0])
    discount = torch.ones(4)
    v = torch.tensor([0.0, 3.0, 7.0, 11.0])
    pair_weight = torch.ones(3)
    q, action = _blank_batch(4)
    got = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=nstep_gate(n=2)(q=q, action=action),
    )
    # G0 = 1 + 10 + 7 = 18; G1 = 10 + 100 + 11 = 121; G2 = 100 + 11 = 111.
    assert torch.allclose(got, torch.tensor([18.0, 121.0, 111.0]))
    full = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=lambda_gate(td_lambda=1.0)(q=q, action=action),
    )
    # λ = 1 out to the run break keeps going: G0 = 1 + 10 + 100 + 11 = 122.
    assert torch.allclose(full, torch.tensor([122.0, 121.0, 111.0]))


def test_nstep_gate_returns_a_square_matrix() -> None:
    """``nstep_gate`` is ``[N, N]``: 1 on ``t < s < t + n``, else 0."""
    reward = torch.tensor([0.0, 1.0, 10.0, 100.0])
    discount = torch.ones(4)
    v = torch.tensor([0.0, 3.0, 7.0, 11.0])
    pair_weight = torch.ones(3)
    N = 4
    n = 2
    q, action = _blank_batch(N)
    got = nstep_gate(n=n)(q=q, action=action)
    matrix = torch.zeros(N, N)
    for t in range(N):
        for s in range(t + 1, min(N, t + n)):
            matrix[t, s] = 1.0
    assert torch.equal(got, matrix)
    targets = _continuation_targets(
        reward=reward, discount_all=discount, v_step=v, pair_weight=pair_weight,
        continuation=got,
    )
    assert torch.allclose(targets, torch.tensor([18.0, 121.0, 111.0]))


def test_nstep_terminal_gamma_zero_ends_the_sum() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 1, 0])
    loss, _ = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    # s0: (5 - 1)^2 = 16 — neither V(s1) nor the next episode's return; s1: 12100.
    assert abs(loss.item() - (16.0 + 12100.0) / 2) < 1e-03


def test_nstep_truncation_gamma_carries_the_sum_discounted() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["episode_done"] = torch.tensor([0, 2, 0])
    loss, _ = _nstep(
        n=2, discount=_disc(gamma_episode_truncated=0.5)
    )(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    # s0: 1 + 0.5 * 10 + 0.5 * 100 = 56 → (5 - 56)^2 = 2601; s1: 12100.
    assert abs(loss.item() - (2601.0 + 12100.0) / 2) < 1e-03


def test_nstep_does_not_cross_sequence_boundary() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["sequence_id"] = torch.tensor([0, 0, 1])
    two, _ = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    one, _ = _nstep(n=1)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    assert abs(two.item() - one.item()) < 1e-04


def test_nstep_all_out_of_run_pairs_yield_zero_loss() -> None:
    step_stream = {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.tensor([0, 1, 2]),
        }
    predictions, delayed = _q(torch.ones(3, 2), torch.ones(3, 2))
    loss, metrics = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics["q_values_mean"]) < 1e-05


def test_nstep_trains_multiple_heads_independently() -> None:
    """Two tensors, two n's: each loss reads only its own Q and they add."""
    step_stream, pred_1, delayed_1 = _lambda_fixture()
    q1 = pred_1.detach().clone().requires_grad_(True)
    q3 = pred_1.detach().clone().requires_grad_(True)
    delayed_q = delayed_1
    loss_1, m1 = _nstep(n=1)(
        objective_data=step_stream, predictions=q1, w=None, rho=1.0, delayed_predictions=delayed_q
    )
    loss_3, m3 = _nstep(n=3)(
        objective_data=step_stream, predictions=q3, w=None, rho=1.0, delayed_predictions=delayed_q
    )
    assert abs(loss_1.item() - _ONE_STEP) < 1e-03
    assert abs(loss_3.item() - _TWO_STEP) < 1e-03
    assert m1["action_value"] != m3["action_value"]
    total = loss_1 + loss_3
    total.backward()
    assert q1.grad is not None
    assert q3.grad is not None
    assert not torch.equal(q1.grad, torch.zeros_like(q1.grad))
    assert not torch.equal(q3.grad, torch.zeros_like(q3.grad))


def test_nstep_does_not_backprop_through_delayed_q() -> None:
    n, a = 4, 2
    step_stream = {
            "action": torch.zeros(n, dtype=torch.long),
            "reward": torch.ones(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
        }
    online = torch.randn(n, a, requires_grad=True)
    delayed_q = torch.randn(n, a, requires_grad=True)
    predictions, delayed = _q(online, delayed_q)
    loss, _ = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    loss.backward()
    assert online.grad is not None
    assert delayed_q.grad is None


def test_nstep_with_multiple_head_output_rows_per_step() -> None:
    """Every row of a step trains toward that step's n-step return."""
    step_stream, _, _ = _lambda_fixture()
    step_stream = {key: value.clone() for key, value in step_stream.items()}
    step_stream["head_output_count"] = torch.tensor([2, 1, 2])
    online = torch.tensor(
        [[5.0, 0.0], [7.0, 0.0], [0.0, 0.0], [0.0, -9.0], [0.0, 0.0]]
    )
    delayed_q = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [-9.0, -9.0], [0.0, 100.0]]
    )
    predictions, delayed = _q(online, delayed_q)
    loss, _ = _nstep(n=2)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    # s0 rows: (5-111)^2 = 11236, (7-111)^2 = 10816; s1 row: (0-110)^2 = 12100.
    assert abs(loss.item() - (11236.0 + 10816.0 + 12100.0) / 3) < 1e-02


def test_nstep_q_affine_applies_to_online_and_delayed() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    loss, _ = _nstep(n=1, value=_val(scale=2.0))(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    expected = (9.0 + 44100.0) / 2
    assert abs(loss.item() - expected) < 1e-03


def test_nstep_requires_min_sequence() -> None:
    step_stream = {
            "action": torch.zeros(1, dtype=torch.long),
            "reward": torch.zeros(1),
            "episode_done": torch.zeros(1, dtype=torch.long),
            "task_done": torch.zeros(1, dtype=torch.long),
        }
    predictions, delayed = _q(torch.zeros(1, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="Not enough"):
        _nstep()(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)


def test_nstep_rejects_non_fp32_q() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    predictions = predictions.to(torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        _nstep()(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)


def test_nstep_temperature_matches_dqn_one_step() -> None:
    """n=1 with the same α is DqnObjective's soft backup."""
    step_stream = {
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 0.0]),
            "episode_done": torch.zeros(2, dtype=torch.int64),
            "task_done": torch.zeros(2, dtype=torch.int64),
        }
    predictions, delayed = _q(torch.zeros(2, 2), torch.zeros(2, 2))
    nstep_loss, nstep_m = _nstep(temperature=1.0)(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    dqn_loss, dqn_m = DqnObjective( temperature=1.0, double=False, gate=None, grouping_field=None, discount=_disc(), reward=_rew(), value=_val())(
        objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed
    )
    assert abs(nstep_loss.item() - dqn_loss.item()) < 1e-06
    assert abs(nstep_m["entropy"] - dqn_m["entropy"]) < 1e-06


def test_nstep_temperature_zero_omits_metric() -> None:
    step_stream, predictions, delayed = _lambda_fixture()
    _, metrics = _nstep()(objective_data=step_stream, predictions=predictions, w=None, rho=1.0, delayed_predictions=delayed)
    assert "entropy" not in metrics


def test_nstep_temperature_rejects_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _nstep(temperature=-0.1)
