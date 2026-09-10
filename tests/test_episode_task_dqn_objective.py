"""Tests for episode/task Q-heads and EpisodeTaskDqnObjective."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.models import Model, load_model, save_model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import NumericEmbedder
from mouse_core.models.heads import DiscreteActionValueHead
from mouse_core.objectives import EpisodeTaskDqnObjective
from tests._token_batch_helpers import batch_to_token_batch, tok_from_encoder


def _q_pair(
    online_e: torch.Tensor,
    online_t: torch.Tensor,
    delayed_e: torch.Tensor,
    delayed_t: torch.Tensor,
) -> tuple[TensorDict, TensorDict]:
    n = online_e.shape[0]
    return (
        TensorDict(
            {"action_value_episode": online_e, "action_value_task": online_t},
            batch_size=[n],
        ),
        TensorDict(
            {"action_value_episode": delayed_e, "action_value_task": delayed_t},
            batch_size=[n],
        ),
    )


def _head(hidden_dim: int = 8, out_features: int = 3) -> DiscreteActionValueHead:
    return DiscreteActionValueHead(
        in_features=hidden_dim,
        out_features=out_features,
        hidden_dim=hidden_dim,
        num_layers=1,
    )


def _model(heads: dict[str, DiscreteActionValueHead], hidden_dim: int = 8) -> Model:
    encoder = NumericEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4, "std": 0.02, "positions": 1}
        ],
    )
    return Model(encoder=encoder, backbone=IdentityBackbone(hidden_dim=hidden_dim), heads=heads)


def test_episode_task_heads_must_appear_together() -> None:
    try:
        _model({"action_value_episode": _head()})
    except ValueError as e:
        assert "together" in str(e)
    else:
        raise AssertionError("expected ValueError for episode head without task head")
    try:
        _model({"action_value_task": _head()})
    except ValueError as e:
        assert "together" in str(e)
    else:
        raise AssertionError("expected ValueError for task head without episode head")


def test_episode_task_heads_cannot_combine_with_action_value() -> None:
    try:
        _model(
            {
                "action_value": _head(),
                "action_value_episode": _head(),
                "action_value_task": _head(),
            }
        )
    except ValueError as e:
        assert "cannot be combined" in str(e)
    else:
        raise AssertionError("expected ValueError for mixed Q-head names")


def test_get_action_sums_episode_and_task_q() -> None:
    model = _model(
        {
            "action_value_episode": _head(out_features=3),
            "action_value_task": _head(out_features=3),
        }
    )
    # Episode prefers 0, task prefers 2, sum prefers 1.
    preds = TensorDict(
        {
            "action_value_episode": torch.tensor([[3.0, 1.0, 0.0]]),
            "action_value_task": torch.tensor([[0.0, 3.0, 2.0]]),
        }
    )
    action = model.get_action(preds, temperature=0.0)
    assert action.shape == (1,)
    assert int(action.item()) == 1


def test_get_action_sums_batched_decode_scores() -> None:
    model = _model(
        {
            "action_value_episode": _head(out_features=2),
            "action_value_task": _head(out_features=2),
        }
    )
    preds = TensorDict(
        {
            "action_value_episode": torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [4.0, 0.0]]]
            ),
            "action_value_task": torch.tensor(
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 5.0], [0.0, 5.0]]]
            ),
        }
    )
    action = model.get_action(preds, temperature=0.0)
    assert action.tolist() == [1, 1]


def test_episode_task_save_load_roundtrip(tmp_path) -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    encoder = NumericEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {
                "type": "discrete",
                "field": "action",
                "vocab_size": 4,
                "std": 0.02,
                "positions": 1,
            },
            {
                "type": "fourier",
                "field": "reward",
                "std": 0.02,
                "positions": 1,
                "fourier_min": 0.01,
                "fourier_max": 10.0,
            },
        ],
    )
    model = Model(
        encoder=encoder,
        backbone=IdentityBackbone(hidden_dim=hidden_dim),
        heads={
            "action_value_episode": _head(hidden_dim, 4),
            "action_value_task": _head(hidden_dim, 4),
        },
    ).eval()
    batch = [[{"action": 0, "reward": 0.0}, {"action": 1, "reward": 1.0}]]
    expected = model(batch_to_token_batch(tok_from_encoder(model.encoder), batch)).predictions
    save_model(model, tmp_path)
    loaded = load_model(tmp_path, train_kernel="varlen", decode_kernel="flex", dtype=torch.float32).eval()
    actual = loaded(batch_to_token_batch(tok_from_encoder(loaded.encoder), batch)).predictions
    assert torch.allclose(actual["action_value_episode"], expected["action_value_episode"])
    assert torch.allclose(actual["action_value_task"], expected["action_value_task"])
    assert loaded.action_head == "action_value_episode"
    assert set(model.state_dict()) == set(loaded.state_dict())


def test_shared_astar_not_per_head_argmax() -> None:
    """Episode target uses Q_e[a*] where a* is argmax of the delayed sum."""
    # Delayed step 1: Q_e max is action 0 (5), Q_t max is action 1 (3),
    # sum is [5, 7] so a* = 1 and Q_e[a*] = 4.
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1]),
            "reward": torch.zeros(2),
            "episode_done": torch.zeros(2, dtype=torch.long),
            "task_done": torch.zeros(2, dtype=torch.long),
        },
        batch_size=[2],
    )
    online_e = torch.tensor([[0.0, 4.0], [0.0, 0.0]])
    online_t = torch.tensor([[0.0, 3.0], [0.0, 0.0]])
    delayed_e = torch.tensor([[0.0, 0.0], [5.0, 4.0]])
    delayed_t = torch.tensor([[0.0, 0.0], [0.0, 3.0]])
    preds, delayed = _q_pair(online_e, online_t, delayed_e, delayed_t)
    loss, metrics = EpisodeTaskDqnObjective(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        episode_td_lambda=0.0,
        task_td_lambda=0.0,
        task_gamma_step=1.0,
    )(step_stream, preds, delayed)
    # Episode: gathered Q=4, target = 0 + 1 * Q_e[a*]=4 → 0.
    # Task: gathered Q=3, target = 0 + 1 * Q_t[a*]=3 → 0.
    # Own-max episode target would be 5 and loss_e would be 1.
    assert abs(metrics["action_value_episode"] - 0.0) < 1e-5
    assert abs(metrics["action_value_task"] - 0.0) < 1e-5
    assert abs(loss.item() - 0.0) < 1e-5


def test_episode_head_ignores_next_episode_reward() -> None:
    """gamma_episode=0 so the episode head target is just the step reward."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 1, 0], dtype=torch.long),
            "task_done": torch.tensor([0, 0, 0], dtype=torch.long),
        },
        batch_size=[3],
    )
    # Pair 0 is the episode end (done codes live at i+1). Taken actions: 1 then 0.
    online_e = torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]])
    online_t = torch.zeros(3, 2)
    delayed_e = torch.tensor([[0.0, 0.0], [9.0, 9.0], [0.0, 0.0]])
    delayed_t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    preds, delayed = _q_pair(online_e, online_t, delayed_e, delayed_t)
    _, metrics = EpisodeTaskDqnObjective(
        gamma_step=0.0,
        gamma_episode_terminal=0.0,
        episode_td_lambda=0.0,
        task_td_lambda=0.0,
    )(step_stream, preds, delayed)
    # Both pairs have γ=0, so targets are rewards 1 and 5: (2-1)^2, (3-5)^2 → 2.5.
    # Bootstrapping the next-episode start into pair 0 would use Q_e[a*]=9.
    assert abs(metrics["action_value_episode"] - 2.5) < 1e-5


def test_task_lambda_one_bootstraps_sum_at_next_episode() -> None:
    """λ=1 skips intra-episode steps to (Q_e + Q_t)[a*] at the next start."""
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 0, 0]),
            "reward": torch.tensor([0.0, 1.0, 0.0]),
            "episode_done": torch.tensor([0, 0, 1], dtype=torch.long),
            "task_done": torch.tensor([0, 0, 0], dtype=torch.long),
        },
        batch_size=[3],
    )
    online_e = torch.zeros(3, 2)
    online_t = torch.zeros(3, 2)
    delayed_e = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 2.0]])
    delayed_t = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
    # Step 2 sum [4, 2], a*=0, V_sum = 4. Env reward is ignored by the task head.
    preds, delayed = _q_pair(online_e, online_t, delayed_e, delayed_t)
    _, metrics = EpisodeTaskDqnObjective(
        gamma_step=0.0,
        gamma_episode_terminal=0.0,
        episode_td_lambda=0.0,
        task_td_lambda=1.0,
        task_gamma_step=1.0,
        task_gamma_episode_terminal=1.0,
    )(step_stream, preds, delayed)
    # Both in-run pairs target 4; online task Q gathered is 0 → MSE 16.
    assert abs(metrics["action_value_task"] - 16.0) < 1e-5


def test_task_head_zero_at_task_end() -> None:
    step_stream = TensorDict(
        {
            "action": torch.tensor([0, 1]),
            "reward": torch.tensor([0.0, 1.0]),
            "episode_done": torch.tensor([0, 1], dtype=torch.long),
            "task_done": torch.tensor([0, 2], dtype=torch.long),
        },
        batch_size=[2],
    )
    online_e = torch.zeros(2, 2)
    online_t = torch.zeros(2, 2)
    delayed_e = torch.tensor([[0.0, 0.0], [8.0, 0.0]])
    delayed_t = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    preds, delayed = _q_pair(online_e, online_t, delayed_e, delayed_t)
    _, metrics = EpisodeTaskDqnObjective(
        gamma_step=0.0,
        gamma_episode_terminal=0.0,
        gamma_task_truncated=0.0,
        episode_td_lambda=0.0,
        task_td_lambda=1.0,
        task_gamma_episode_terminal=1.0,
    )(step_stream, preds, delayed)
    assert abs(metrics["action_value_task"] - 0.0) < 1e-5


def test_episode_task_objective_metrics() -> None:
    n, a = 6, 3
    step_stream = TensorDict(
        {
            "action": torch.randint(0, a, (n,)),
            "reward": torch.randn(n),
            "episode_done": torch.zeros(n, dtype=torch.long),
            "task_done": torch.zeros(n, dtype=torch.long),
            "sequence_id": torch.tensor([0, 0, 0, 1, 1, 1]),
        },
        batch_size=[n],
    )
    preds, delayed = _q_pair(
        torch.randn(n, a),
        torch.randn(n, a),
        torch.randn(n, a),
        torch.randn(n, a),
    )
    loss, metrics = EpisodeTaskDqnObjective()(step_stream, preds, delayed)
    assert loss.ndim == 0
    assert metrics["action_value_episode"] >= 0.0
    assert metrics["action_value_task"] >= 0.0
    assert "q_episode_mean" in metrics
    assert "q_task_mean" in metrics
    assert "watkins_greedy_frac" not in metrics
