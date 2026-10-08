"""Pipeline stages: step-independence / prefix consistency for train↔eval parity."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from mouse_core.data import (
    Tokenizer,
    pack_token_batch,
)
from tests._token_batch_helpers import IntIdTokenizer

def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0



def _io(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    for src, dst in pairs:
        spec = {"input_field": src}
        if dst != src:
            spec["output_field"] = dst
        fields.append(spec)
    return fields


def _tok_in(
    *names: str, head_output: str | None = None
) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = [
        {
            "type": "text",
            "input_field": name,
            "format": "{field}",
            **({"head_output": True} if name == head_output else {}),
        }
        for name in names
    ]
    if not any(
        field.get("input_field") == "episode_index"
        for field in fields
    ):
        fields.append(
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            }
        )
    return fields


def _rows() -> list[dict]:
    return [
        {"action": 0, "observation": 1, "reward": 0.0, "episode_done": 0, "task_done": 0, "task_index": 0, "noise": 9},
        {"action": 1, "observation": 2, "reward": 0.5, "episode_done": 0, "task_done": 0, "task_index": 0, "noise": 8},
        {"action": 2, "observation": 3, "reward": 1.0, "episode_done": 1, "task_done": 2, "task_index": 0, "noise": 7},
        {"action": 0, "observation": 4, "reward": 0.0, "episode_done": 0, "task_done": 0, "task_index": 1, "noise": 6},
        {"action": 1, "observation": 5, "reward": 0.25, "episode_done": 2, "task_done": 2, "task_index": 1, "noise": 5},
        {"action": 3, "observation": 6, "reward": 0.0, "episode_done": 0, "task_done": 0, "task_index": 2, "noise": 4},
    ]


def test_missing_objective_fields_key_raises() -> None:
    tokenizer = Tokenizer(
        input_fields=_tok_in("action", head_output="action"),
        objective_fields=_io(("action", "action"), ("old_log_prob", "old_log_prob")),
        tokenizer=IntIdTokenizer(),
    )
    with pytest.raises(KeyError, match="old_log_prob"):
        tokenizer({"action": 1, "task_index": 0})


def test_tokenizer_output_defaults_to_input() -> None:
    tokenizer = Tokenizer(
        input_fields=_tok_in("action", head_output="action"),
        objective_fields=_io(("reward", "reward")),
        tokenizer=IntIdTokenizer(),
    )
    tokens = tokenizer({"action": 2, "reward": 0.5, "task_index": 0})
    assert tokens.modality_names == ("__text__",)
    assert tokens.objective_fields["reward"] == pytest.approx(0.5)


def test_tokenizer_renames_objective_fields() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "act",
                "format": "{field}",
                "head_output": True,
            },
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_io(("q", "info_q_star")),
        tokenizer=IntIdTokenizer(),
    )
    tokens = tokenizer({"act": 3, "q": 1.5, "task_index": 0})
    assert tokens.modality_names == ("__text__",)
    assert tokens.objective_fields["info_q_star"] == pytest.approx(1.5)


def test_tokenizer_full_matches_per_step_concat() -> None:
    """Full-window pack == head/tail step lists packed together."""
    tokenizer = Tokenizer(
        input_fields=[
            *_tok_in("action", "observation", head_output="observation"),
        ],
        objective_fields=_io(
            ("action", "action"),
            ("observation", "observation"),
            ("reward", "reward"),
            ("episode_done", "episode_done"),
            ("task_done", "task_done"),
        ),
        tokenizer=IntIdTokenizer(),
    )
    rows = _rows()
    full, full_obj, _sid = pack_token_batch(steps=[tokenizer(step) for step in rows], continuing=None)
    head = [tokenizer(s) for s in rows[:3]]
    tail = [tokenizer(s) for s in rows[3:]]
    cat, cat_obj, _sid = pack_token_batch(steps=head + tail, continuing=None)
    assert np.array_equal(full.modality_ids, cat.modality_ids)
    assert np.array_equal(full.ids, cat.ids)
    assert np.allclose(full.values, cat.values)
    assert np.array_equal(full.group_ids, cat.group_ids)
    assert np.array_equal(full.head_output_indices, cat.head_output_indices)
    for key in ("action", "observation", "reward", "episode_done", "task_done"):
        assert torch.equal(full_obj[key], cat_obj[key])
