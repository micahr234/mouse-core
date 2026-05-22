"""Tests for DatasetStore append and encoding."""

from __future__ import annotations

from mouse.data import DatasetStore


def test_append_and_len() -> None:
    store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
    store.append({
        "observation_discrete": [0],
        "action": 1,
        "reward": 0.5,
        "done": 0,
        "episode_step": 0,
    })
    store.append({
        "observation_discrete": [1],
        "action": 2,
        "reward": 1.0,
        "done": 0,
        "episode_step": 1,
    })
    assert len(store) == 2


def test_encode_single_row() -> None:
    store = DatasetStore(max_action_dim=4, max_obs_discrete_dim=1)
    store.append({
        "observation_discrete": [3],
        "action": 2,
        "reward": -1.0,
        "done": 1,
        "episode_step": 4,
    })
    td = store[0]
    assert td["action"].item() == 2
    assert td["reward"].item() == -1.0
    assert td["done"].item() == 1
    assert td["obs_discrete"].item() == 3
