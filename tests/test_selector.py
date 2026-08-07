from __future__ import annotations

import pytest

from mouse_core.data import Augmenter, Selector


def test_selector_keeps_listed_keys() -> None:
    step = {"obs": 0, "action": 1, "reward": 2.0, "done": False}
    result = Selector(fields={"obs": "obs", "action": "action"})(step)
    assert result == {"obs": 0, "action": 1}


def test_selector_renames_fields() -> None:
    step = {"obs": 0, "act": 1}
    result = Selector(fields={"obs": "observation", "act": "action"})(step)
    assert result == {"observation": 0, "action": 1}


def test_selector_does_not_mutate_original() -> None:
    step = {"obs": 0, "action": 1, "reward": 2.0}
    Selector(fields={"obs": "obs"})(step)
    assert "reward" in step


def test_selector_missing_input_raises() -> None:
    with pytest.raises(KeyError, match="Selector input field 'obs'"):
        Selector(fields={"obs": "observation"})({"action": 1})


def test_selector_duplicate_outputs_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate output"):
        Selector(fields={"a": "x", "b": "x"})


def test_selector_empty_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Selector(fields={})


def test_selector_after_augmentation() -> None:
    step = {"obs": 0, "action": 1, "reward": 1.0, "task_index": 0}
    augment = Augmenter(
        seed_field="task_index",
        modalities=[
            {
                "type": "linear",
                "input_field": "reward",
                "output_field": "reward",
                "scale_in_low": 0.0,
                "scale_out_low": 0.0,
                "scale_in_high": 1.0,
                "scale_out_high": 2.0,
            }
        ],
    )
    result = Selector(fields={"obs": "obs", "action": "action"})(augment(step))
    assert list(result.keys()) == ["obs", "action"]
