"""Pipeline stages: step-independence / prefix consistency for train↔eval parity."""

from __future__ import annotations
from tests._event_fixtures import encode_measurement

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
        field.get("input_field") == "repeat_episode_index"
        for field in fields
    ):
        fields.append(
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            }
        )
    return fields


def _rows() -> list[dict]:
    return [
        {"action": 0, "observation": 1, "reward": 0.0, "repeat_episode_done": 0, "episode_done": 0, "episode_index": 0, "noise": 9},
        {"action": 1, "observation": 2, "reward": 0.5, "repeat_episode_done": 0, "episode_done": 0, "episode_index": 0, "noise": 8},
        {"action": 2, "observation": 3, "reward": 1.0, "repeat_episode_done": 1, "episode_done": 2, "episode_index": 0, "noise": 7},
        {"action": 0, "observation": 4, "reward": 0.0, "repeat_episode_done": 0, "episode_done": 0, "episode_index": 1, "noise": 6},
        {"action": 1, "observation": 5, "reward": 0.25, "repeat_episode_done": 2, "episode_done": 2, "episode_index": 1, "noise": 5},
        {"action": 3, "observation": 6, "reward": 0.0, "repeat_episode_done": 0, "episode_done": 0, "episode_index": 2, "noise": 4},
    ]


def test_missing_objective_fields_key_raises() -> None:
    tokenizer = Tokenizer(
        input_fields=_tok_in("action", head_output="action"),
        objective_fields=_io(("action", "action"), ("old_log_prob", "old_log_prob")),
        tokenizer=IntIdTokenizer(),
    )
    with pytest.raises(KeyError, match="old_log_prob"):
        encode_measurement(tokenizer, {"action": 1, "episode_index": 0})


def test_tokenizer_output_defaults_to_input() -> None:
    tokenizer = Tokenizer(
        input_fields=_tok_in("action", head_output="action"),
        objective_fields=_io(("reward", "reward")),
        tokenizer=IntIdTokenizer(),
    )
    tokens = encode_measurement(tokenizer, {"action": 2, "reward": 0.5, "episode_index": 0})
    assert not tokens.numeric_mask.any()
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
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_io(("q", "q_star")),
        tokenizer=IntIdTokenizer(),
    )
    tokens = encode_measurement(tokenizer, {"act": 3, "q": 1.5, "episode_index": 0})
    assert not tokens.numeric_mask.any()
    assert tokens.objective_fields["q_star"] == pytest.approx(1.5)


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
            ("repeat_episode_done", "repeat_episode_done"),
            ("episode_done", "episode_done"),
        ),
        tokenizer=IntIdTokenizer(),
    )
    rows = _rows()
    full, full_obj, _sid = pack_token_batch(steps=[encode_measurement(tokenizer, step) for step in rows], continuing=None)
    head = [encode_measurement(tokenizer, s) for s in rows[:3]]
    tail = [encode_measurement(tokenizer, s) for s in rows[3:]]
    cat, cat_obj, _sid = pack_token_batch(steps=head + tail, continuing=None)
    assert np.array_equal(full.numeric_mask, cat.numeric_mask)
    assert np.array_equal(full.token_ids, cat.token_ids)
    assert np.allclose(full.numeric_values, cat.numeric_values)
    assert np.array_equal(full.sequence_ids, cat.sequence_ids)
    assert np.array_equal(full.head_token_ids, cat.head_token_ids)
    for key in ("action", "observation", "reward", "repeat_episode_done", "episode_done"):
        assert torch.equal(full_obj[key], cat_obj[key])
