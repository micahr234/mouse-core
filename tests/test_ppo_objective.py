"""Tests for PPO objective on synthetic tensors."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import PpoObjective, batch_field, sample_discrete_action


def _ppo_batch(
    b: int = 2,
    s: int = 4,
    a: int = 3,
    *,
    with_old_log_prob: bool = True,
    segment_id: list[int] | None = None,
) -> tuple[TensorDict, TensorDict]:
    action = torch.randint(0, a, (b, s))
    reward = torch.randn(b, s)
    done = torch.zeros(b, s, dtype=torch.long)
    data: dict[str, torch.Tensor] = {
        "action": action,
        "reward": reward,
        "done": done,
    }
    if with_old_log_prob:
        data["old_log_prob"] = torch.randn(b, s)
    if segment_id is not None:
        data["segment_id"] = torch.tensor([segment_id] * b)
    objective_data = TensorDict(data, batch_size=(b, s))
    predictions = TensorDict(
        {
            "action": torch.randn(b, s, a),
            "value": torch.randn(b, s, 1),
        },
        batch_size=(b, s),
    )
    return objective_data, predictions


def test_ppo_objective_runs() -> None:
    objective_data, predictions = _ppo_batch()
    loss, metrics = PpoObjective(gamma_step=0.99)(objective_data, predictions)
    assert loss.ndim == 0
    assert "ppo" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_ppo_objective_runs_without_old_log_prob() -> None:
    objective_data, predictions = _ppo_batch(with_old_log_prob=False)
    loss, metrics = PpoObjective()(objective_data, predictions)
    assert loss.ndim == 0
    # Without behavior log-probs, ratio is 1 so clipfrac should be ~0.
    assert metrics["clipfrac"] == 0.0


def test_ppo_objective_accepts_squeezed_value() -> None:
    objective_data, predictions = _ppo_batch()
    predictions["value"] = predictions["value"].squeeze(-1)
    loss, _ = PpoObjective()(objective_data, predictions)
    assert loss.ndim == 0


def test_ppo_objective_rejects_wrong_action_shape() -> None:
    objective_data, predictions = _ppo_batch()
    objective_data["action"] = torch.randint(0, 3, (2, 4, 1))
    try:
        PpoObjective()(objective_data, predictions)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for action shape [B, S, 1]")


def test_ppo_objective_requires_min_sequence() -> None:
    objective_data = TensorDict(
        {
            "action": torch.zeros(1, 1, dtype=torch.long),
            "reward": torch.zeros(1, 1),
            "done": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1, 1),
    )
    predictions = TensorDict(
        {"action": torch.zeros(1, 1, 2), "value": torch.zeros(1, 1, 1)},
        batch_size=(1, 1),
    )
    try:
        PpoObjective()(objective_data, predictions)
    except ValueError as e:
        assert "S >= 2" in str(e)
    else:
        raise AssertionError("expected ValueError for S < 2")


def test_ppo_objective_closed_form_single_transition() -> None:
    """One valid transition: known policy and value losses with λ=1, γ=0."""
    # Sequence length 2 → one transition. γ=0 → return = reward, advantage = r - V.
    # Logits force action 0 with probability 1; old_log_prob = 0 → ratio = 1.
    objective_data = TensorDict(
        {
            "action": torch.tensor([[0, 0]]),
            "reward": torch.tensor([[0.0, 4.0]]),
            "done": torch.tensor([[0, 0]]),
            "old_log_prob": torch.tensor([[0.0, 0.0]]),
        },
        batch_size=(1, 2),
    )
    predictions = TensorDict(
        {
            # Huge logit on action 0 → log_prob ≈ 0
            "action": torch.tensor([[[20.0, -20.0], [20.0, -20.0]]]),
            "value": torch.tensor([[[1.0], [0.0]]]),
        },
        batch_size=(1, 2),
    )
    objective = PpoObjective(
        gamma_step=0.0,
        gae_lambda=1.0,
        clip_eps=0.2,
        vf_coef=1.0,
        ent_coef=0.0,
        normalize_advantage=False,
    )
    loss, metrics = objective(objective_data, predictions)

    # advantage = 4 - 1 = 3; ratio ≈ 1 → policy_loss = -3
    # value_loss = (1 - 4)^2 = 9
    # total = -3 + 9 = 6
    assert abs(loss.item() - 6.0) < 1e-3
    assert abs(metrics["policy_loss"] - (-3.0)) < 1e-3
    assert abs(metrics["value_loss"] - 9.0) < 1e-3


def test_ppo_objective_skips_transitions_across_pack_segments() -> None:
    # Only the (1, 2) pair is same-segment; γ=0, λ=1, no entropy, no normalize.
    objective_data = TensorDict(
        {
            "action": torch.tensor([[0, 0, 0]]),
            "reward": torch.tensor([[0.0, 1.0, 5.0]]),
            "done": torch.tensor([[0, 0, 0]]),
            "old_log_prob": torch.tensor([[0.0, 0.0, 0.0]]),
            "segment_id": torch.tensor([[0, 1, 1]]),
        },
        batch_size=(1, 3),
    )
    predictions = TensorDict(
        {
            "action": torch.tensor(
                [[[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]]
            ),
            "value": torch.tensor([[[0.0], [2.0], [0.0]]]),
        },
        batch_size=(1, 3),
    )
    loss, _ = PpoObjective(
        gamma_step=0.0,
        gae_lambda=1.0,
        vf_coef=1.0,
        ent_coef=0.0,
        normalize_advantage=False,
    )(objective_data, predictions)

    # Only t=1: adv = 5 - 2 = 3 → policy = -3; value = (2-5)^2 = 9 → loss = 6
    assert abs(loss.item() - 6.0) < 1e-3


def test_ppo_objective_raises_when_all_pairs_cross_segments() -> None:
    objective_data = TensorDict(
        {
            "action": torch.tensor([[0, 1, 0]]),
            "reward": torch.tensor([[0.0, 1.0, 5.0]]),
            "done": torch.tensor([[0, 0, 0]]),
            "segment_id": torch.tensor([[0, 1, 2]]),
        },
        batch_size=(1, 3),
    )
    predictions = TensorDict(
        {
            "action": torch.zeros(1, 3, 2),
            "value": torch.zeros(1, 3, 1),
        },
        batch_size=(1, 3),
    )
    try:
        PpoObjective()(objective_data, predictions)
    except ValueError as e:
        assert "packed-segment" in str(e)
    else:
        raise AssertionError("expected ValueError when every pair crosses a segment boundary")


def test_sample_discrete_action_shapes() -> None:
    logits = torch.randn(4, 5)
    actions, log_probs = sample_discrete_action(logits, num_actions=3)
    assert actions.shape == (4,)
    assert log_probs.shape == (4,)
    assert int(actions.max()) < 3


def test_batch_field_extracts_old_log_prob() -> None:
    batch = [
        [{"old_log_prob": 0.1}, {"old_log_prob": 0.2}],
        [{"old_log_prob": -0.5}, {"old_log_prob": 1.0}],
    ]
    t = batch_field(batch, "old_log_prob")
    assert t.shape == (2, 2)
    assert abs(t[0, 0].item() - 0.1) < 1e-6
    assert abs(t[1, 1].item() - 1.0) < 1e-6
