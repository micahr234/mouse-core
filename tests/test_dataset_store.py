"""Tests for DatasetStore append and encoding (mouse-env rollout contract)."""

from __future__ import annotations

import pytest

from mouse_core.data import DatasetStore


def test_append_and_len() -> None:
    store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
    store.append({
        "observation": {"discrete": 0},
        "action": {"discrete": 1},
        "reward": 0.5,
        "done": 0,
        "time": 0,
    })
    store.append({
        "observation": {"discrete": 1},
        "action": {"discrete": 2},
        "reward": 1.0,
        "done": 0,
        "time": 1,
    })
    assert len(store) == 2


def test_encode_single_row() -> None:
    store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
    store.append({
        "observation": {"discrete": 3},
        "action": {"discrete": 2},
        "reward": -1.0,
        "done": 1,
        "time": 4,
    })
    td = store[0]
    assert td["action"].item() == 2
    assert td["reward"].item() == -1.0
    assert td["done"].item() == 1
    assert td["obs_discrete"].item() == 3
    assert td["time"].item() == 4


def test_encode_optional_fields() -> None:
    store = DatasetStore(max_action_dim=4, max_obs_continuous_dim=2)
    store.append({
        "observation": {"continuous": [0.1, 0.2]},
        "action": {"discrete": 1},
        "reward": 0.5,
        "reward_episodic": 0.25,
        "done": 0,
        "time": 0,
        "q_star": [1.0, 2.0, 3.0, 4.0],
    })
    td = store[0]
    assert td["xformed_reward"].item() == 0.25
    assert td["obs_continuous"].shape[-1] == 2
    assert td["q_star"].shape[-1] == 4


def test_dataset_roundtrip() -> None:
    """Append rows, export to a HuggingFace Dataset, reload, and re-encode."""
    store = DatasetStore(max_action_dim=4, max_obs_continuous_dim=2)
    for t in range(3):
        store.append({
            "observation": {"continuous": [float(t), float(t) + 0.5]},
            "action": {"discrete": t % 2},
            "reward": float(t),
            "reward_episodic": float(t) / 10.0,
            "done": 0,
            "time": t,
            "group_id": "CartPole-v1#0",
            "episode_index": 0,
        })

    ds = store.to_dataset()
    assert len(ds) == 3

    reloaded = DatasetStore(max_action_dim=4, max_obs_continuous_dim=2)
    reloaded.from_dataset(ds)
    td = reloaded[[0, 1, 2]]
    assert td["action"].tolist() == [0, 1, 0]
    assert td["obs_continuous"].shape == (3, 2)
    assert abs(td["xformed_reward"][2].item() - 0.2) < 1e-6


def test_encode_continuous_action() -> None:
    store = DatasetStore(max_action_dim=4, max_action_continuous_dim=2, max_obs_continuous_dim=3)
    store.append({
        "observation": {"continuous": [0.1, 0.2, 0.3]},
        "action": {"continuous": [0.5, -0.5]},
        "reward": 1.0,
        "done": 0,
        "time": 0,
    })
    td = store[0]
    assert td["action_continuous"].shape[-1] == 2
    assert td["action_continuous"][0].tolist() == pytest.approx([0.5, -0.5])
    # Discrete action index is still emitted (0 placeholder for continuous-only rows).
    assert td["action"].item() == 0


def test_encode_mixed_modalities() -> None:
    """One store holding discrete, continuous, and image steps zero-fills absent modalities."""
    store = DatasetStore(
        max_action_dim=6,
        max_action_continuous_dim=1,
        max_obs_continuous_dim=4,
        max_obs_discrete_dim=1,
        max_obs_image_pixels=4,
    )
    store.append({  # discrete obs + discrete action
        "observation": {"discrete": 2},
        "action": {"discrete": 3},
        "reward": 0.0, "done": 0, "time": 0,
    })
    store.append({  # continuous obs + continuous action
        "observation": {"continuous": [0.1, 0.2, 0.3, 0.4]},
        "action": {"continuous": [0.9]},
        "reward": 1.0, "done": 0, "time": 1,
    })
    store.append({  # image obs + discrete action
        "observation": {"image": [10, 20, 30, 40]},
        "action": {"discrete": 5},
        "reward": 0.0, "done": 1, "time": 2,
    })

    td = store[[0, 1, 2]]
    assert td["action"].tolist() == [3, 0, 5]
    assert td["action_continuous"].shape == (3, 1)
    assert td["action_continuous"][:, 0].tolist() == pytest.approx([0.0, 0.9, 0.0])
    # Discrete obs: real index on row 0, 0 placeholder elsewhere.
    assert td["obs_discrete"].tolist() == [2, 0, 0]
    # Continuous obs: populated on row 1, zero-filled elsewhere.
    assert td["obs_continuous"].shape == (3, 4)
    assert td["obs_continuous"][0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert td["obs_continuous"][1].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    # Image obs: populated on row 2, zero-filled elsewhere.
    assert td["obs_image"].shape == (3, 4)
    assert td["obs_image"][2].tolist() == [10, 20, 30, 40]
    assert td["obs_image"][0].tolist() == [0, 0, 0, 0]
