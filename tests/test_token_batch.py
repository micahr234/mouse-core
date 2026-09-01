"""Tests for StepTokens / TokenBatch packing invariants."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.data import NumericTokenizer, pack_token_batch


def _tok(**kwargs) -> NumericTokenizer:
    return NumericTokenizer(
        input_fields=[{"type": "discrete", "input_field": "action"}],
        objective_fields=[{"input_field": "reward"}, {"input_field": "action"}],
        grouping_field="task_index",
        **kwargs,
    )


def test_objective_column_dtype_promotes_to_float_when_any_step_is_float() -> None:
    tok = _tok()
    steps = [
        tok({"action": 0, "reward": 1, "task_index": 0}),     # int-typed reward first
        tok({"action": 1, "reward": 0.75, "task_index": 0}),
        tok({"action": 2, "reward": 0, "task_index": 0}),
    ]
    _, objective = pack_token_batch(steps, sequence_ids=[0, 0, 0], batch_size=1)
    assert objective["reward"].dtype == torch.float32
    assert objective["reward"].tolist() == [1.0, 0.75, 0.0]
    assert objective["action"].dtype == torch.int64


def test_objective_column_stays_int_when_all_steps_are_int() -> None:
    tok = _tok()
    steps = [tok({"action": a, "reward": a, "task_index": 0}) for a in range(3)]
    _, objective = pack_token_batch(steps, sequence_ids=[0, 0, 0], batch_size=1)
    assert objective["reward"].dtype == torch.int64


def test_objective_vector_column_promotes_dtype() -> None:
    tok = NumericTokenizer(
        input_fields=[{"type": "discrete", "input_field": "action"}],
        objective_fields=[{"input_field": "q"}],
        grouping_field="task_index",
    )
    steps = [
        tok({"action": 0, "q": np.array([1, 2]), "task_index": 0}),
        tok({"action": 0, "q": np.array([0.5, 0.25]), "task_index": 0}),
    ]
    _, objective = pack_token_batch(steps, sequence_ids=[0, 0], batch_size=1)
    assert objective["q"].dtype == torch.float32
    assert objective["q"].tolist() == [[1.0, 2.0], [0.5, 0.25]]


def test_objective_mixed_rank_raises() -> None:
    tok = NumericTokenizer(
        input_fields=[{"type": "discrete", "input_field": "action"}],
        objective_fields=[{"input_field": "q"}],
        grouping_field="task_index",
    )
    steps = [
        tok({"action": 0, "q": np.array([1.0, 2.0]), "task_index": 0}),
        tok({"action": 0, "q": 0.5, "task_index": 0}),
    ]
    with pytest.raises(ValueError, match="mixes array ranks"):
        pack_token_batch(steps, sequence_ids=[0, 0], batch_size=1)


def test_continuous_dim_mismatch_raises() -> None:
    tok = NumericTokenizer(
        input_fields=[{"type": "continuous", "input_field": "obs", "dim": 3}],
        objective_fields=[],
        grouping_field="task_index",
    )
    with pytest.raises(ValueError, match="dim=3 but the step value has 2"):
        tok({"obs": np.array([1.0, 2.0]), "task_index": 0})
    with pytest.raises(ValueError, match="dim=3 but the step value has 4"):
        tok({"obs": np.array([1.0, 2.0, 3.0, 4.0]), "task_index": 0})
    st = tok({"obs": np.array([1.0, 2.0, 3.0]), "task_index": 0})
    assert st.T == 3


def test_skip_on_vector_modality_compares_elementwise() -> None:
    tok = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {"type": "continuous", "input_field": "obs", "dim": 2, "skip": 0.0},
        ],
        objective_fields=[],
        grouping_field="task_index",
    )
    assert tok({"action": 0, "obs": np.array([0.0, 0.0]), "task_index": 0}).T == 1
    assert tok({"action": 0, "obs": np.array([0.0, 1.0]), "task_index": 0}).T == 3
    tok_vec = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {"type": "continuous", "input_field": "obs", "dim": 2, "skip": [1.0, 2.0]},
        ],
        objective_fields=[],
        grouping_field="task_index",
    )
    assert tok_vec({"action": 0, "obs": [1.0, 2.0], "task_index": 0}).T == 1
    assert tok_vec({"action": 0, "obs": [1.0, 3.0], "task_index": 0}).T == 3


def test_interleaved_sequence_ids_rejected() -> None:
    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0}) for _ in range(4)]
    with pytest.raises(ValueError, match="non-decreasing"):
        pack_token_batch(steps, sequence_ids=[0, 1, 0, 1], batch_size=2)


def test_sequence_ids_out_of_range_rejected() -> None:
    from mouse_core.data.token_batch import TokenBatch

    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0}) for _ in range(2)]
    inputs, _ = pack_token_batch(steps, sequence_ids=[0, 1], batch_size=2)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        TokenBatch(
            modality_ids=inputs.modality_ids,
            ids=inputs.ids,
            values=inputs.values,
            modality_names=inputs.modality_names,
            modality_map=inputs.modality_map,
            sequence_ids=inputs.sequence_ids,
            grouping_ids=inputs.grouping_ids,
            prediction_indices=inputs.prediction_indices,
            grouping_field=inputs.grouping_field,
            B=1,
        )


def test_contiguous_sequence_ids_with_gaps_accepted() -> None:
    """Rows with zero steps (decode) leave gaps in the id set; that is fine."""
    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0}) for _ in range(3)]
    inputs, _ = pack_token_batch(steps, sequence_ids=[0, 2, 2], batch_size=3)
    assert inputs.step_counts().tolist() == [1, 0, 2]
