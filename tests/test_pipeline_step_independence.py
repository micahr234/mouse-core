"""Pipeline stages: step-independence / prefix consistency for train↔eval parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.data import (
    NumericTokenizer,
    Selector,
    compose,
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


def _tok_in(*names: str, type: str = "discrete") -> list[dict[str, str]]:
    return [{"type": type, "input_field": name} for name in names]


def _rows() -> list[dict]:
    return [
        {"action": 0, "observation": 1, "reward": 0.0, "episode_done": 0, "task_done": 0, "task_index": 0, "noise": 9},
        {"action": 1, "observation": 2, "reward": 0.5, "episode_done": 0, "task_done": 0, "task_index": 0, "noise": 8},
        {"action": 2, "observation": 3, "reward": 1.0, "episode_done": 1, "task_done": 2, "task_index": 0, "noise": 7},
        {"action": 0, "observation": 4, "reward": 0.0, "episode_done": 0, "task_done": 0, "task_index": 1, "noise": 6},
        {"action": 1, "observation": 5, "reward": 0.25, "episode_done": 2, "task_done": 2, "task_index": 1, "noise": 5},
        {"action": 3, "observation": 6, "reward": 0.0, "episode_done": 0, "task_done": 0, "task_index": 2, "noise": 4},
    ]


def test_selector_concat_matches_full() -> None:
    selector = Selector(
        fields=_io(
            ("action", "action"),
            ("observation", "observation"),
            ("reward", "reward"),
            ("episode_done", "episode_done"),
            ("task_done", "task_done"),
            ("task_index", "task_index"),
        )
    )
    rows = _rows()
    full = [selector(r) for r in rows]
    head = [selector(r) for r in rows[:-1]]
    tail = [selector(r) for r in rows[-1:]]
    assert full == head + tail


def test_selector_renames_before_tokenizer() -> None:
    selector = Selector(fields=_io(("act", "action"), ("obs", "observation"), ("task_index", "task_index")))
    tokenizer = NumericTokenizer(
        input_fields=_tok_in("action", "observation"),
        objective_fields=_io(("action", "action"), ("observation", "observation")),
        grouping_field="task_index",
    )
    transform = compose(selector, tokenizer)
    step = {"act": 1, "obs": 2, "task_index": 0, "noise": 9}
    tokens = transform(step)
    assert tokens.objective_fields["action"] == 1
    assert tokens.objective_fields["observation"] == 2
    assert tokens.grouping_id == 0


def test_missing_objective_fields_key_raises() -> None:
    tokenizer = NumericTokenizer(
        input_fields=_tok_in("action"),
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
        input_fields=_tok_in("action"),
        objective_fields=_io(("reward", "reward")),
        grouping_field="task_index",
    )
    tokens = tokenizer({"action": 2, "reward": 0.5, "task_index": 0})
    assert tokens.modality_names == ("action",)
    assert tokens.objective_fields["reward"] == pytest.approx(0.5)


def test_tokenizer_renames_input_and_objective_fields() -> None:
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "act", "output_field": "action"},
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
            *_tok_in("episode_done"),
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
    assert np.array_equal(full.prediction_indices, cat.prediction_indices)
    for key in ("action", "observation", "reward", "episode_done", "task_done", "task_index"):
        assert torch.equal(full_obj[key], cat_obj[key])


def test_pipeline_without_augmenter_full_matches_head_plus_tail_tokens() -> None:
    """selector→tokenizer on full window matches head/tail pack."""
    selector = Selector(
        fields=_io(
            ("action", "action"),
            ("observation", "observation"),
            ("reward", "reward"),
            ("episode_done", "episode_done"),
            ("task_done", "task_done"),
            ("task_index", "task_index"),
        )
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            *_tok_in("action", "observation"),
            *_tok_in("reward", type="fourier"),
            *_tok_in("episode_done"),
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
    transform = compose(selector, tokenizer)
    rows = _rows()
    full, _ = pack_token_batch([transform(s) for s in rows])
    head = [transform(s) for s in rows[:-1]]
    tail = transform(rows[-1])
    cat, _ = pack_token_batch(head + [tail])
    assert np.array_equal(full.modality_ids, cat.modality_ids)
    assert np.array_equal(full.ids, cat.ids)
    assert np.array_equal(full.grouping_ids, cat.grouping_ids)
    assert np.array_equal(full.prediction_indices, cat.prediction_indices)
