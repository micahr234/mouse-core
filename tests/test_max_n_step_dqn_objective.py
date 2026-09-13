"""Tests for the max-over-n-step DQN objective."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import MaxNStepDqnObjective
from mouse_core.objectives.n_step_dqn import (
    _n_step_horizon_targets,
    _n_step_return_targets,
)
import pytest


def _heads(
    selector: torch.Tensor,
    max_return: torch.Tensor,
    delayed_max: torch.Tensor,
) -> tuple[TensorDict, TensorDict]:
    n = selector.shape[0]
    return (
        TensorDict(
            {"selector": selector, "max_return": max_return},
            batch_size=[n],
        ),
        TensorDict({"max_return": delayed_max}, batch_size=[n]),
    )


def _objective(**kwargs) -> MaxNStepDqnObjective:
    defaults = dict(
        horizons=(1, 3),
        max_return_weight=1.0,
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    defaults.update(kwargs)
    return MaxNStepDqnObjective(**defaults)


def test_pdf_numerical_example() -> None:
    """N={1,3}, (r,γ)=(1,2,3)/(0.9,0.8,0.5), b_{t+1}=4, b_{t+3}=10."""
    reward = torch.tensor([0.0, 1.0, 2.0, 3.0])
    discount = torch.tensor([0.0, 0.9, 0.8, 0.5])
    bootstrap = torch.tensor([0.0, 4.0, 0.0, 10.0])
    pair_weight = torch.ones(3)
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=pair_weight,
        horizons=(1, 3),
    )
    assert bool(valid[0, 0]) and bool(valid[0, 1])
    assert abs(candidates[0, 0].item() - 4.6) < 1e-05
    assert abs(candidates[0, 1].item() - 8.56) < 1e-05
    assert abs(candidates[0].max().item() - 8.56) < 1e-05


def test_pdf_numerical_example_through_objective() -> None:
    """Done-code gammas reconstruct the PDF discounts; losses hit 4.6 and 8.56."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 0, 0]),
            "reward": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            "episode_done": torch.tensor([0, 0, 1, 2]),
            "task_done": torch.zeros(4, dtype=torch.int64),
            "sequence_id": torch.zeros(4, dtype=torch.int64),
        },
        batch_size=[4],
    )
    # Taken action from s0 is 0. Selector at s1 prefers 1; at s3 prefers 0.
    selector = torch.tensor(
        [[0.0, -1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, -1.0]]
    )
    max_return = torch.zeros(4, 2)
    delayed = torch.tensor(
        [[0.0, 0.0], [99.0, 4.0], [0.0, 0.0], [10.0, 99.0]]
    )
    preds, delayed_td = _heads(selector, max_return, delayed)
    loss, metrics = _objective(
        gamma_step=0.9,
        gamma_episode_terminal=0.8,
        gamma_episode_truncated=0.5,
    )(step_stream, preds, delayed_td)
    # Pair 0: selector (0-4.6)^2, max-return (0-8.56)^2. Later pairs also train.
    reward = step_stream["reward"]
    discount = torch.tensor([0.0, 0.9, 0.8, 0.5])
    bootstrap = torch.tensor([0.0, 4.0, 0.0, 10.0])
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=torch.ones(3),
        horizons=(1, 3),
    )
    y1 = torch.where(valid[:, 0], candidates[:, 0], torch.zeros(3))
    ym = torch.where(valid.any(-1), candidates.max(-1).values, torch.zeros(3))
    taken = selector[:3, 0]
    expected_sel = ((taken - y1) ** 2).mean()
    expected_max = ((torch.zeros(3) - ym) ** 2).mean()
    assert metrics["selector"] == pytest.approx(expected_sel.item(), rel=1e-05)
    assert metrics["max_return"] == pytest.approx(expected_max.item(), rel=1e-05)
    assert loss.item() == pytest.approx(
        expected_sel.item() + expected_max.item(), rel=1e-05
    )
    assert metrics["horizon_1_selected_frac"] == pytest.approx(2.0 / 3.0, abs=1e-06)
    assert metrics["horizon_3_selected_frac"] == pytest.approx(1.0 / 3.0, abs=1e-06)


def test_bootstrap_uses_selector_action_not_max_return_argmax() -> None:
    """Delayed max-return prefers the other action; bootstrap still uses selector a*."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 0]),
            "reward": torch.tensor([0.0, 1.0, 0.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.tensor([0, 0, 1]),
        },
        batch_size=[3],
    )
    # Endpoint s1: selector prefers action 1 (value 4). Max-return delayed prefers 0 (100).
    selector = torch.tensor([[0.0, -1.0], [-1.0, 2.0], [0.0, 0.0]])
    delayed = torch.tensor([[0.0, 0.0], [100.0, 4.0], [0.0, 0.0]])
    preds, delayed_td = _heads(selector, torch.zeros(3, 2), delayed)
    _, metrics = _objective(horizons=(1,), gamma_step=1.0)(
        step_stream, preds, delayed_td
    )
    # Y1 = 1 + 1*4 = 5, not 1+100. Selector taken Q(s0,a=0)=0 → loss 25.
    assert metrics["selector"] == pytest.approx(25.0, rel=1e-05)


def test_zero_discount_fills_longer_horizons_without_later_q() -> None:
    reward = torch.tensor([0.0, 1.0, 2.0, 3.0])
    discount = torch.tensor([0.0, 0.0, 0.8, 0.5])
    bootstrap = torch.tensor([0.0, 4.0, 0.0, 10.0])
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=torch.ones(3),
        horizons=(1, 3),
    )
    assert bool(valid[0].all())
    assert abs(candidates[0, 0].item() - 1.0) < 1e-06
    assert abs(candidates[0, 1].item() - 1.0) < 1e-06


def test_truncation_masks_longer_horizon_keeps_shorter() -> None:
    reward = torch.tensor([0.0, 1.0, 2.0, 3.0])
    discount = torch.tensor([0.0, 0.9, 0.8, 0.5])
    bootstrap = torch.tensor([0.0, 4.0, 0.0, 10.0])
    pair_weight = torch.tensor([1.0, 0.0, 1.0])
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=pair_weight,
        horizons=(1, 3),
    )
    assert bool(valid[0, 0]) and not bool(valid[0, 1])
    assert abs(candidates[0, 0].item() - 4.6) < 1e-05
    assert not bool(valid[1].any())


def test_all_invalid_pairs_yield_zero_loss() -> None:
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
    preds, delayed = _heads(torch.ones(3, 2), torch.ones(3, 2), torch.ones(3, 2))
    loss, metrics = _objective()(step_stream, preds, delayed)
    assert abs(loss.item()) < 1e-06
    assert abs(metrics["selector"]) < 1e-06
    assert abs(metrics["max_return"]) < 1e-06


def test_negative_returns_are_selected_when_they_are_the_max() -> None:
    reward = torch.tensor([0.0, -5.0, -1.0, 0.0])
    discount = torch.ones(4)
    bootstrap = torch.zeros(4)
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=torch.ones(3),
        horizons=(1, 3),
    )
    assert bool(valid[0].all())
    assert candidates[0, 0].item() == pytest.approx(-5.0)
    assert candidates[0, 1].item() == pytest.approx(-6.0)
    assert candidates[0].max().item() == pytest.approx(-5.0)


def test_targets_are_detached() -> None:
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
    selector = torch.zeros(n, a, requires_grad=True)
    max_return = torch.zeros(n, a, requires_grad=True)
    delayed = torch.randn(n, a, requires_grad=True)
    preds, delayed_td = _heads(selector, max_return, delayed)
    loss, _ = _objective(horizons=(1, 3), gamma_step=0.9)(
        step_stream, preds, delayed_td
    )
    loss.backward()
    assert selector.grad is not None
    assert max_return.grad is not None
    assert delayed.grad is None


def test_horizons_one_makes_both_targets_coincide() -> None:
    reward = torch.tensor([0.0, 1.0, 2.0])
    discount = torch.tensor([0.0, 0.5, 0.5])
    bootstrap = torch.tensor([0.0, 4.0, 8.0])
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=torch.ones(2),
        horizons=(1,),
    )
    assert bool(valid[:, 0].all())
    assert torch.equal(candidates[:, 0], candidates.max(dim=-1).values)


def test_rejects_horizons_missing_one() -> None:
    with pytest.raises(ValueError, match="contain 1"):
        _objective(horizons=(3, 5))


def test_rejects_empty_horizons() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _objective(horizons=())


def test_rejects_duplicate_horizons() -> None:
    with pytest.raises(ValueError, match="unique"):
        _objective(horizons=(1, 3, 3))


def test_rejects_non_positive_horizons() -> None:
    with pytest.raises(ValueError, match="integers >= 1"):
        _objective(horizons=(1, 0))


def test_requires_delayed_predictions() -> None:
    step_stream = TensorDict(
        {
            "action": torch.zeros(3, dtype=torch.long),
            "reward": torch.ones(3),
            "episode_done": torch.zeros(3, dtype=torch.long),
            "task_done": torch.zeros(3, dtype=torch.long),
        },
        batch_size=[3],
    )
    preds, _ = _heads(torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(3, 2))
    with pytest.raises(ValueError, match="delayed_predictions"):
        _objective()(step_stream, preds)


def test_rejects_non_fp32_q() -> None:
    step_stream = TensorDict(
        {
            "action": torch.zeros(3, dtype=torch.long),
            "reward": torch.ones(3),
            "episode_done": torch.zeros(3, dtype=torch.long),
            "task_done": torch.zeros(3, dtype=torch.long),
        },
        batch_size=[3],
    )
    selector = torch.zeros(3, 2, dtype=torch.bfloat16)
    preds, delayed = _heads(selector, torch.zeros(3, 2), torch.zeros(3, 2))
    with pytest.raises(TypeError, match="float32"):
        _objective()(step_stream, preds, delayed)


def test_multi_token_rows_share_the_step_target() -> None:
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 0]),
            "reward": torch.tensor([0.0, 1.0, 0.0]),
            "episode_done": torch.zeros(3, dtype=torch.int64),
            "task_done": torch.zeros(3, dtype=torch.int64),
            "sequence_id": torch.zeros(3, dtype=torch.int64),
            "head_output_count": torch.tensor([2, 1, 1]),
        },
        batch_size=[3],
    )
    selector = torch.tensor(
        [[3.0, 0.0], [7.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )
    delayed = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 4.0], [0.0, 0.0]]
    )
    preds, delayed_td = _heads(selector, torch.zeros(4, 2), delayed)
    _, metrics = _objective(horizons=(1,), gamma_step=1.0)(
        step_stream, preds, delayed_td
    )
    # Y1 = 1 + 4 = 5. Rows of s0: (3-5)^2=4, (7-5)^2=4; s1 weight 0 for... 
    # pair (0,1) valid, pair (1,2) valid with Y=0+0=0. s1 taken action 0, Q=0.
    # s0 two rows + s1 one row: (4+4+0)/3 = 8/3
    assert metrics["selector"] == pytest.approx(8.0 / 3.0, rel=1e-05)


def test_max_horizons_match_n_step_returns() -> None:
    """Each max-n-step candidate is the same complete n-step return."""
    reward = torch.tensor([0.0, 1.0, 10.0, 100.0])
    discount = torch.ones(4)
    bootstrap = torch.tensor([0.0, 3.0, 7.0, 1000.0])
    pair_weight = torch.ones(3)
    horizons = (1, 2, 3)
    candidates, valid = _n_step_horizon_targets(
        reward=reward,
        discount_all=discount,
        bootstrap=bootstrap,
        pair_weight=pair_weight,
        horizons=horizons,
    )
    for h, n in enumerate(horizons):
        target, ok = _n_step_return_targets(
            reward=reward,
            discount_all=discount,
            bootstrap=bootstrap,
            pair_weight=pair_weight,
            n=n,
        )
        assert torch.equal(ok, valid[:, h])
        assert torch.equal(
            target, torch.where(ok, candidates[:, h], torch.zeros_like(target))
        )
