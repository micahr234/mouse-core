"""Tests for ``frozenlake_group_prefix``."""

from __future__ import annotations

import pytest

from mouse_core.data import frozenlake_group_prefix


def test_frozenlake_group_prefix_matches_episode_budget() -> None:
    text = frozenlake_group_prefix(max_task_episodes=20)
    assert text == (
        "Your job is to predict the future sum of rewards in FrozenLake. "
        "Navigate a grid; reach the goal for reward; a hole ends the "
        "episode with none. You have 20 episodes to solve the task. The "
        "grid is permuted, so squares are not in order; action ids may be "
        "remapped.\n"
        "Strategy: explore; keep a mental map of what has and has not been "
        "explored; avoid holes you have already fallen in; once you have a "
        "path to the goal, repeat it.\n"
        "Predict when you see a new line. Step format: "
        "action,observation[,r=reward][,d=done][,e=episode].\n"
    )
    assert "reuse" not in text
    assert "action,observation,r=reward,d=done" not in text


def test_frozenlake_group_prefix_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="max_task_episodes"):
        frozenlake_group_prefix(max_task_episodes=0)
