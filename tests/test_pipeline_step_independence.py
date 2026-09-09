"""Pipeline stages: step-independence / prefix consistency for train↔eval parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.data import (
    NumericTokenizer,
    pack_token_batch,
)


def _io(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    fields: list[dict[str, str]] = []
    for src, dst in pairs:
        spec = {"input_field": src}
        if dst != src:
            spec["output_field"] = dst
        fields.append(spec)
    return fields


def _tok_in(
    *names: str, type: str = "discrete", head_output: str | None = None
) -> list[dict]:
    return [
        {
            "type": type,
            "input_field": name,
            **({"head_output": True} if name == head_output else {}),
        }
        for name in names
    ]


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
    tokenizer = NumericTokenizer(
        input_fields=_tok_in("action", head_output="action"),
        objective_fields=_io(("action", "action"), ("old_log_prob", "old_log_prob")),
        grouping_field="task_index",
    )
    with pytest.raises(KeyError, match="old_log_prob"):
        tokenizer({"action": 1, "task_index": 0})


def test_tokenizer_rejects_legacy_field_key() -> None:
    with pytest.raises(TypeError, match="input_field=/output_field="):
        NumericTokenizer(
            input_fields=[{"type": "discrete", "field": "action"}],
            grouping_field="task_index",
        )


def test_tokenizer_output_defaults_to_input() -> None:
    tokenizer = NumericTokenizer(
        input_fields=_tok_in("action", head_output="action"),
        objective_fields=_io(("reward", "reward")),
        grouping_field="task_index",
    )
    tokens = tokenizer({"action": 2, "reward": 0.5, "task_index": 0})
    assert tokens.modality_names == ("action",)
    assert tokens.objective_fields["reward"] == pytest.approx(0.5)


def test_tokenizer_renames_input_and_objective_fields() -> None:
    tokenizer = NumericTokenizer(
        input_fields=[
            {
                "type": "discrete",
                "input_field": "act",
                "output_field": "action",
                "head_output": True,
            },
        ],
        objective_fields=_io(("q", "info_q_star")),
        grouping_field="task_index",
    )
    tokens = tokenizer({"act": 3, "q": 1.5, "task_index": 0})
    assert tokens.modality_names == ("action",)
    assert tokens.objective_fields["info_q_star"] == pytest.approx(1.5)


def test_tokenizer_full_matches_per_step_concat() -> None:
    """Full-window pack == head/tail step lists packed together."""
    tokenizer = NumericTokenizer(
        input_fields=[
            *_tok_in("action", "observation"),
            *_tok_in("reward", type="fourier"),
            *_tok_in("episode_done", head_output="episode_done"),
        ],
        objective_fields=_io(
            ("action", "action"),
            ("observation", "observation"),
            ("reward", "reward"),
            ("episode_done", "episode_done"),
            ("task_done", "task_done"),
        ),
        grouping_field="task_index",
    )
    rows = _rows()
    full, full_obj = pack_token_batch([tokenizer(step) for step in rows])
    head = [tokenizer(s) for s in rows[:3]]
    tail = [tokenizer(s) for s in rows[3:]]
    cat, cat_obj = pack_token_batch(head + tail)
    assert np.array_equal(full.modality_ids, cat.modality_ids)
    assert np.array_equal(full.ids, cat.ids)
    assert np.allclose(full.values, cat.values)
    assert np.array_equal(full.grouping_ids, cat.grouping_ids)
    assert np.array_equal(full.head_output_indices, cat.head_output_indices)
    for key in ("action", "observation", "reward", "episode_done", "task_done", "task_index"):
        assert torch.equal(full_obj[key], cat_obj[key])
