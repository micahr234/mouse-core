"""Tests for StepTokens / TokenBatch packing invariants."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mouse_core.data import Tokenizer, pack_token_batch, to_device
from tests._token_batch_helpers import IntIdTokenizer

def when_episode_done_nonzero(ctx):
    return "episode_done" in ctx and ctx["episode_done"] != 0


def when_group_start(ctx):
    return bool(ctx["group_start"])


def when_reward_nonzero(ctx):
    return "reward" in ctx and ctx["reward"] != 0.0


def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0


def when_step_index_zero_or_group_start(ctx):
    return (ctx.get("step_index") == 0) | bool(ctx["group_start"])



def _tok(**kwargs) -> Tokenizer:
    return Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "reward"}, {"input_field": "action"}],
        tokenizer=kwargs.pop("tokenizer", IntIdTokenizer()),
        **kwargs,
    )


def test_tokenizer_pack_rows_packs_ragged_rows() -> None:
    tok = _tok()
    rows = [
        [
            {"action": 0, "reward": 0.0, "task_index": 0},
            {"action": 1, "reward": 1.0, "task_index": 0},
        ],
        [],
        [{"action": 2, "reward": 0.5, "task_index": 3}],
    ]
    inputs = tok.pack_rows(rows=rows, continuing=None)
    assert inputs.B == 3
    assert inputs.step_counts().tolist() == [2, 0, 1]
    manual, _, _sid = pack_token_batch(
        steps=[tok(step) for row in rows for step in row],
        group_ids=[0, 0, 2],
        batch_size=3,continuing=None)
    assert inputs.ids.tolist() == manual.ids.tolist()
    assert inputs.group_ids.tolist() == manual.group_ids.tolist()
    assert inputs.head_output_indices.tolist() == manual.head_output_indices.tolist()


def test_tokenizer_pack_rows_all_empty_keeps_batch_slots() -> None:
    tok = _tok()
    inputs = tok.pack_rows(rows=[[], [], []], continuing=None)
    assert inputs.B == 3
    assert inputs.L == 0
    assert inputs.step_counts().tolist() == [0, 0, 0]


def test_objective_column_dtype_promotes_to_float_when_any_step_is_float() -> None:
    tok = _tok()
    steps = [
        tok({"action": 0, "reward": 1, "task_index": 0}),     # int-typed reward first
        tok({"action": 1, "reward": 0.75, "task_index": 0}),
        tok({"action": 2, "reward": 0, "task_index": 0}),
    ]
    _, objective, _sid = pack_token_batch(steps=steps, group_ids=[0, 0, 0], batch_size=1, continuing=None)
    assert objective["reward"].dtype == torch.float32
    assert objective["reward"].tolist() == [1.0, 0.75, 0.0]
    assert objective["action"].dtype == torch.int64


def test_to_device_moves_every_tensor() -> None:
    data = {
        "reward": torch.zeros(2, dtype=torch.float32),
        "action": torch.zeros(2, dtype=torch.int64),
    }
    moved = to_device(data=data, device="cpu")
    assert set(moved) == {"reward", "action"}
    assert all(value.device.type == "cpu" for value in moved.values())


def test_objective_column_stays_int_when_all_steps_are_int() -> None:
    tok = _tok()
    steps = [tok({"action": a, "reward": a, "task_index": 0}) for a in range(3)]
    _, objective, group_id = pack_token_batch(steps=steps, group_ids=[0, 0, 0], batch_size=1, continuing=None)
    assert "group_id" not in objective
    assert group_id.tolist() == [0, 0, 0]
    assert objective["reward"].dtype == torch.int64


def test_objective_vector_column_promotes_dtype() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        tok({"action": 0, "q": np.array([1, 2]), "task_index": 0}),
        tok({"action": 0, "q": np.array([0.5, 0.25]), "task_index": 0}),
    ]
    _, objective, _sid = pack_token_batch(steps=steps, group_ids=[0, 0], batch_size=1, continuing=None)
    assert objective["q"].dtype == torch.float32
    assert objective["q"].tolist() == [[1.0, 2.0], [0.5, 0.25]]


def test_objective_ragged_float_vectors_pad_with_neg_inf() -> None:
    """Shorter rows are padded with -inf (the nonexistent-action sentinel), not 0."""
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        tok({"action": 0, "q": np.array([1.0, 2.0, 3.0]), "task_index": 0}),
        tok({"action": 0, "q": np.array([0.5, 0.25]), "task_index": 0}),
    ]
    _, objective, _sid = pack_token_batch(steps=steps, group_ids=[0, 0], batch_size=1, continuing=None)
    q = objective["q"]
    assert q[0].tolist() == [1.0, 2.0, 3.0]
    assert q[1, :2].tolist() == [0.5, 0.25]
    assert q[1, 2].item() == -torch.inf


def test_objective_ragged_int_vectors_raise() -> None:
    """Integer columns have no padding sentinel, so ragged shapes are an error."""
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        tok({"action": 0, "q": np.array([1, 2, 3]), "task_index": 0}),
        tok({"action": 0, "q": np.array([4, 5]), "task_index": 0}),
    ]
    with pytest.raises(ValueError, match="ragged"):
        pack_token_batch(steps=steps, group_ids=[0, 0], batch_size=1, continuing=None)


def test_objective_mixed_rank_raises() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        tok({"action": 0, "q": np.array([1.0, 2.0]), "task_index": 0}),
        tok({"action": 0, "q": 0.5, "task_index": 0}),
    ]
    with pytest.raises(ValueError, match="mixes array ranks"):
        pack_token_batch(steps=steps, group_ids=[0, 0], batch_size=1, continuing=None)




class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [ord(c) % 20 + 1 for c in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def test_positions_index_tokens_within_modality_per_step() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {"type": "text", "format": "ab", "head_output": True},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
        objective_fields=[],
    )
    st = tok({"action": 1, "task_index": 0})
    # Shared __text__ stream: token field then two-id text const.
    assert st.positions.tolist() == [0, 1, 2]
    inputs, _, _sid = pack_token_batch(steps=[st, st], group_ids=[0, 0], batch_size=1, continuing=None)
    # Positions restart every step; they never accumulate across the sequence.
    assert inputs.positions.tolist() == [0, 1, 2] * 2
    assert inputs.to_tensors()["positions"].dtype == torch.int64


def test_negative_positions_rejected() -> None:
    from mouse_core.data.token_batch import TokenBatch

    tok = _tok()
    inputs, _, _sid = pack_token_batch(steps=[tok({"action": 0, "reward": 0.0, "task_index": 0})], continuing=None)
    with pytest.raises(ValueError, match="positions must be >= 0"):
        TokenBatch(
            modality_ids=inputs.modality_ids,
            ids=inputs.ids,
            values=inputs.values,
            positions=np.array([-1]),
            modality_names=inputs.modality_names,
            modality_map=inputs.modality_map,
            group_ids=inputs.group_ids,
            head_output_indices=inputs.head_output_indices,
            head_output_steps=inputs.head_output_steps,
            B=inputs.B,
        )


def test_interleaved_group_ids_rejected() -> None:
    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0}) for _ in range(4)]
    with pytest.raises(ValueError, match="non-decreasing"):
        pack_token_batch(steps=steps, group_ids=[0, 1, 0, 1], batch_size=2, continuing=None)


def test_group_ids_out_of_range_rejected() -> None:
    from mouse_core.data.token_batch import TokenBatch

    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0}) for _ in range(2)]
    inputs, _, _sid = pack_token_batch(steps=steps, group_ids=[0, 1], batch_size=2, continuing=None)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        TokenBatch(
            modality_ids=inputs.modality_ids,
            ids=inputs.ids,
            values=inputs.values,
            positions=inputs.positions,
            modality_names=inputs.modality_names,
            modality_map=inputs.modality_map,
            group_ids=inputs.group_ids,
            head_output_indices=inputs.head_output_indices,
            head_output_steps=inputs.head_output_steps,
            B=1,
        )


def test_contiguous_group_ids_with_gaps_accepted() -> None:
    """Rows with zero steps (decode) leave gaps in the id set; that is fine."""
    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0}) for _ in range(3)]
    inputs, _, _sid = pack_token_batch(steps=steps, group_ids=[0, 2, 2], batch_size=3, continuing=None)
    assert inputs.step_counts().tolist() == [1, 0, 2]


def test_pack_rejects_negative_group_ids() -> None:
    tok = _tok()
    steps = [tok({"action": 0, "reward": 0.0, "task_index": 0})]
    with pytest.raises(ValueError, match="group_ids must be >= 0"):
        pack_token_batch(steps=steps, group_ids=[-1], batch_size=2, continuing=None)


def test_pack_rejects_objective_key_missing_on_some_steps() -> None:
    tok = _tok()
    steps = [
        tok({"action": 0, "reward": 1.0, "task_index": 0}),
        tok({"action": 1, "reward": 0.0, "task_index": 0}),
    ]
    steps[1].objective_fields.pop("reward")
    with pytest.raises(KeyError, match="reward"):
        pack_token_batch(steps=steps, group_ids=[0, 0], batch_size=1, continuing=None)
