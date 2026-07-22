"""Tests for held-out task evaluation helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("mouse_gym")
pytest.importorskip("procedural_frozenlake")

from mouse_core.task_eval import (
    DEFAULT_EVAL_SEED_OFFSET,
    make_procedural_frozenlake_group,
)


def test_make_eval_group_uses_offset_seeds() -> None:
    train = make_procedural_frozenlake_group(
        num_envs=2,
        episodes_per_task=2,
        seed_offset=0,
        name_prefix="train",
    )
    eval_env = make_procedural_frozenlake_group(
        num_envs=2,
        episodes_per_task=2,
        seed_offset=DEFAULT_EVAL_SEED_OFFSET,
        name_prefix="eval",
    )
    try:
        assert train.names[0].startswith("train_")
        assert eval_env.names[0].startswith("eval_")
        # Distinct seed streams (map_seed / reset_seed encoded in names only —
        # stepping once should not raise).
        train.step(train.sample_random_input())
        eval_env.step(eval_env.sample_random_input())
    finally:
        train.close()
        eval_env.close()
