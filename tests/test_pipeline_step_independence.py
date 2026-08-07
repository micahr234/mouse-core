"""Pipeline stages: step-independence / prefix consistency for train↔eval parity."""

from __future__ import annotations

import numpy as np
import pytest

from mouse_core.data import (
    Grouper,
    NumericTokenizer,
    Selector,
    compose,
    pack_token_batch,
)


def _rows() -> list[dict]:
    return [
        {"action": 0, "observation": 1, "reward": 0.0, "done": 0, "task_index": 0, "noise": 9},
        {"action": 1, "observation": 2, "reward": 0.5, "done": 0, "task_index": 0, "noise": 8},
        {"action": 2, "observation": 3, "reward": 1.0, "done": 3, "task_index": 0, "noise": 7},
        {"action": 0, "observation": 4, "reward": 0.0, "done": 0, "task_index": 1, "noise": 6},
        {"action": 1, "observation": 5, "reward": 0.25, "done": 4, "task_index": 1, "noise": 5},
        {"action": 3, "observation": 6, "reward": 0.0, "done": 0, "task_index": 2, "noise": 4},
    ]


def test_selector_concat_matches_full() -> None:
    selector = Selector(
        fields={
            "action": "action",
            "observation": "observation",
            "reward": "reward",
            "done": "done",
            "task_index": "task_index",
        }
    )
    rows = _rows()
    full = [selector(r) for r in rows]
    head = [selector(r) for r in rows[:-1]]
    tail = [selector(r) for r in rows[-1:]]
    assert full == head + tail


def test_selector_renames_before_tokenizer() -> None:
    selector = Selector(fields={"act": "action", "obs": "observation", "task_index": "task_index"})
    grouper = Grouper(input_field="task_index", output_field="grouping_id")
    tokenizer = NumericTokenizer(
        modalities=[
            {"type": "discrete", "field": "action"},
            {"type": "discrete", "field": "observation"},
        ],
        step_fields=["action", "observation"],
        grouping_field="grouping_id",
    )
    transform = compose(selector, grouper, tokenizer)
    step = {"act": 1, "obs": 2, "task_index": 0, "noise": 9}
    tokens = transform(step)
    assert tokens.step_fields["action"] == 1
    assert tokens.step_fields["observation"] == 2


def test_grouper_field_ids() -> None:
    grouper = Grouper(input_field="task_index", output_field="grouping_id")
    rows = _rows()
    full = [grouper(r) for r in rows]
    prefix = [grouper(r) for r in rows[:-1]]
    assert [r["grouping_id"] for r in full[:-1]] == [r["grouping_id"] for r in prefix]
    assert [r["grouping_id"] for r in full] == [0, 0, 0, 1, 1, 2]


def test_grouper_copies_grouping_field() -> None:
    grouper = Grouper(input_field="grouping_id", output_field="grouping_id")
    assert grouper({"x": 1, "grouping_id": 0})["grouping_id"] == 0


def test_grouper_rename_preserves_non_int_value() -> None:
    grouper = Grouper(input_field="task_index", output_field="grouping_id")
    out = grouper({"task_index": "episode-a", "action": 1})
    assert out["grouping_id"] == "episode-a"
    assert out["task_index"] == "episode-a"


def test_grouper_missing_input_raises() -> None:
    grouper = Grouper(input_field="task_index", output_field="grouping_id")
    try:
        grouper({"action": 1})
    except KeyError as e:
        assert "task_index" in str(e)
    else:
        raise AssertionError("expected KeyError for missing input_field")


def test_missing_step_fields_key_raises() -> None:
    tokenizer = NumericTokenizer(
        modalities=[{"type": "discrete", "field": "action"}],
        step_fields=["action", "old_log_prob"],
        grouping_field="grouping_id",
    )
    with pytest.raises(KeyError, match="old_log_prob"):
        tokenizer({"action": 1, "grouping_id": 0})


def test_tokenizer_rejects_legacy_io_keys() -> None:
    try:
        NumericTokenizer(
            modalities=[
                {
                    "type": "discrete",
                    "input_field": "action",
                    "output_field": "action",
                }
            ],
            grouping_field="grouping_id",
        )
    except TypeError as e:
        assert "field=" in str(e)
    else:
        raise AssertionError("expected TypeError for input_field=/output_field=")


def test_tokenizer_full_matches_per_step_concat() -> None:
    """Full-window pack == head/tail step lists packed together."""
    grouper = Grouper(input_field="task_index", output_field="grouping_id")
    tokenizer = NumericTokenizer(
        modalities=[
            {"type": "discrete", "field": "action"},
            {"type": "discrete", "field": "observation"},
            {"type": "fourier", "field": "reward"},
            {"type": "discrete", "field": "done"},
        ],
        step_fields=["action", "observation", "reward", "done"],
        grouping_field="grouping_id",
    )
    transform = compose(grouper, tokenizer)
    rows = _rows()
    full = pack_token_batch([transform(step) for step in rows])
    head = [transform(s) for s in rows[:3]]
    tail = [transform(s) for s in rows[3:]]
    cat = pack_token_batch(head + tail)
    assert np.array_equal(full.modality_ids, cat.modality_ids)
    assert np.array_equal(full.ids, cat.ids)
    assert np.allclose(full.values, cat.values)
    assert np.array_equal(full.grouping_ids, cat.grouping_ids)
    assert np.array_equal(full.prediction_indices, cat.prediction_indices)
    for key in ("action", "observation", "reward", "done", "grouping_id"):
        assert np.array_equal(full.step_fields[key], cat.step_fields[key])


def test_pipeline_without_augmenter_full_matches_head_plus_tail_tokens() -> None:
    """selector→grouper→tokenizer on full window matches head/tail pack."""
    selector = Selector(
        fields={
            "action": "action",
            "observation": "observation",
            "reward": "reward",
            "done": "done",
            "task_index": "task_index",
            "grouping_id": "grouping_id",
        }
    )
    grouper = Grouper(input_field="task_index", output_field="grouping_id")
    tokenizer = NumericTokenizer(
        modalities=[
            {"type": "discrete", "field": "action"},
            {"type": "discrete", "field": "observation"},
            {"type": "fourier", "field": "reward"},
            {"type": "discrete", "field": "done"},
        ],
        step_fields=["action", "observation", "reward", "done"],
        grouping_field="grouping_id",
    )
    transform = compose(grouper, selector, tokenizer)
    rows = _rows()
    full = pack_token_batch([transform(s) for s in rows])
    head = [transform(s) for s in rows[:-1]]
    tail = transform(rows[-1])
    cat = pack_token_batch(head + [tail])
    assert np.array_equal(full.modality_ids, cat.modality_ids)
    assert np.array_equal(full.ids, cat.ids)
    assert np.array_equal(full.grouping_ids, cat.grouping_ids)
    assert np.array_equal(full.prediction_indices, cat.prediction_indices)
