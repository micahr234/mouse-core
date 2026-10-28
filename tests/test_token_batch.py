"""Tests for StepTokens / TokenBatch packing invariants."""
from __future__ import annotations
from tests._event_fixtures import encode_measurement, measurement_events

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from mouse_core.data import (EventLog, Tokenizer, latest_environment_decision,
                             pack_token_batch, reuse_token_tensors, to_device)
from tests._token_batch_helpers import IntIdTokenizer


def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0


def _tok(**kwargs) -> Tokenizer:
    return Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
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
            {"action": 0, "reward": 0.0, "episode_index": 0},
            {"action": 1, "reward": 1.0, "episode_index": 0},
        ],
        [],
        [{"action": 2, "reward": 0.5, "episode_index": 3}],
    ]
    inputs = tok.pack_rows(rows=[measurement_events(payloads) for payloads in rows], continuing=None)
    assert inputs.B == 3
    assert inputs.step_counts().tolist() == [2, 0, 1]
    manual, _, _sid = pack_token_batch(
        steps=[encode_measurement(tok, step) for row in rows for step in row],
        sequence_ids=[0, 0, 2],
        batch_size=3,continuing=None)
    assert inputs.token_ids.tolist() == manual.token_ids.tolist()
    assert inputs.sequence_ids.tolist() == manual.sequence_ids.tolist()
    assert inputs.head_token_ids.tolist() == manual.head_token_ids.tolist()


def test_tokenizer_pack_rows_all_empty_keeps_batch_slots() -> None:
    tok = _tok()
    inputs = tok.pack_rows(rows=[measurement_events(payloads) for payloads in [[], [], []]], continuing=None)
    assert inputs.B == 3
    assert inputs.L == 0
    assert inputs.step_counts().tolist() == [0, 0, 0]


def test_sparse_inference_inputs_and_strict_training_labels():
    tok = Tokenizer(input_fields=[
        dict(type='text', provider='environment', input_field='observation', format='{field}'),
        dict(type='token', provider='environment', format='4', head_output=True),
    ], objective_fields=[dict(input_field='advantage')], tokenizer=IntIdTokenizer())
    log = EventLog()
    log.event(provider='coordinator', episode_index=0, step_index=0, payload={
        'episode_start': True,
    })
    log.event(provider='environment', episode_index=0, step_index=0, payload={
        'observation': 2,
        'terminated': False,
        'truncated': False,
    })
    row = latest_environment_decision(log)
    inputs = tok.pack_rows(rows=[log.events, []], continuing=None)
    assert inputs.token_ids.tolist() == [2, 4]
    assert inputs.step_counts().tolist() == [1, 0]
    # Training stays strict; inference must not remove configured labels or invent them.
    with pytest.raises(KeyError, match='advantage'):
        tok.encode(log)
    events = log.events
    events[-1]['payload']['advantage'] = 1.5
    training = tok.encode(events)[0]
    np.testing.assert_array_equal(inputs.token_ids, training.token_ids)
    assert training.objective_fields['advantage'] == 1.5
    assert 'advantage' not in row
    missing_log = EventLog()
    missing_log.event(provider='coordinator', episode_index=0, step_index=0, payload={
        'episode_start': True,
    })
    missing_log.event(provider='environment', episode_index=0, step_index=0, payload={
        'terminated': False,
        'truncated': False,
    })
    sparse = tok.pack_rows(rows=[missing_log.events], continuing=None)
    assert sparse.token_ids.tolist() == [4]
    assert sparse.step_counts().tolist() == [1]


@pytest.mark.parametrize('provider', ['environment', 'agent'])
def test_event_tokenizer_groups_repeated_providers_by_marker(provider):
    tok = Tokenizer(input_fields=[dict(type='text', provider=provider, input_field='value', format='{field}'),
                                          dict(type='token', provider=provider, format='4', head_output=True)], tokenizer=IntIdTokenizer())
    log = EventLog()
    for step, value in [(0, 1), (0, 2), (1, 3)]:
        log.event(provider=provider, episode_index=0, step_index=step, payload={'value': value})
    steps = tok.encode(log)
    assert [step.token_ids.tolist() for step in steps] == [[1, 4, 2, 4], [3, 4]]


def test_objective_column_dtype_promotes_to_float_when_any_step_is_float() -> None:
    tok = _tok()
    steps = [
        encode_measurement(tok, {"action": 0, "reward": 1, "episode_index": 0}),     # int-typed reward first
        encode_measurement(tok, {"action": 1, "reward": 0.75, "episode_index": 0}),
        encode_measurement(tok, {"action": 2, "reward": 0, "episode_index": 0}),
    ]
    _, objective, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 0, 0], batch_size=1, continuing=None)
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


def test_tensor_reuse_is_scoped_by_batch_and_restores_after_errors() -> None:
    tok = _tok()
    batch = tok.pack_rows(rows=[measurement_events(payloads) for payloads in [[{"action": 0, "reward": 0.0}]]], continuing=None)
    other = tok.pack_rows(rows=[measurement_events(payloads) for payloads in [[{"action": 1, "reward": 1.0}]]], continuing=None)
    with reuse_token_tensors():
        first = batch.to_tensors()
        with torch.no_grad():
            assert batch.to_tensors("cpu") is first
        assert other.to_tensors() is not first
        with pytest.raises(RuntimeError, match="interrupted"):
            with reuse_token_tensors():
                assert batch.to_tensors() is not first
                raise RuntimeError("interrupted")
        assert batch.to_tensors() is first
    assert batch.to_tensors() is not first
    batch.token_ids[:] = 7
    with reuse_token_tensors():
        fresh = batch.to_tensors()
        assert fresh is not first
        assert (fresh["token_ids"] == 7).all()


def test_tensor_reuse_separates_inference_tensors_from_autograd_inputs() -> None:
    batch = _tok().pack_rows(rows=[measurement_events(payloads) for payloads in [[{"action": 1, "reward": 0.0}]]], continuing=None)
    with reuse_token_tensors():
        with torch.inference_mode():
            inference = batch.to_tensors()
        normal = batch.to_tensors()
        assert normal is not inference
        assert not normal["token_ids"].is_inference()
        # Embedding backward needs to save the indices, which inference-mode
        # tensors cannot do. Inference-first reuse must still allow training.
        embedding = torch.nn.Embedding(32, 4)
        embedding(normal["token_ids"]).sum().backward()
        assert embedding.weight.grad is not None


def test_objective_column_stays_int_when_all_steps_are_int() -> None:
    tok = _tok()
    steps = [encode_measurement(tok, {"action": a, "reward": a, "episode_index": 0}) for a in range(3)]
    _, objective, group_id = pack_token_batch(steps=steps, sequence_ids=[0, 0, 0], batch_size=1, continuing=None)
    assert "group_id" not in objective
    assert group_id.tolist() == [0, 0, 0]
    assert objective["reward"].dtype == torch.int64


def test_objective_vector_column_promotes_dtype() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1, 2]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": np.array([0.5, 0.25]), "episode_index": 0}),
    ]
    _, objective, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1, continuing=None)
    assert objective["q"].dtype == torch.float32
    assert objective["q"].tolist() == [[1.0, 2.0], [0.5, 0.25]]


def test_objective_ragged_float_vectors_pad_with_neg_inf() -> None:
    """Shorter rows are padded with -inf (the nonexistent-action sentinel), not 0."""
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1.0, 2.0, 3.0]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": np.array([0.5, 0.25]), "episode_index": 0}),
    ]
    _, objective, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1, continuing=None)
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
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1, 2, 3]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": np.array([4, 5]), "episode_index": 0}),
    ]
    with pytest.raises(ValueError, match="ragged"):
        pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1, continuing=None)


def test_objective_mixed_rank_raises() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1.0, 2.0]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": 0.5, "episode_index": 0}),
    ]
    with pytest.raises(ValueError, match="mixes array ranks"):
        pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1, continuing=None)


def test_interleaved_sequence_ids_rejected() -> None:
    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 0.0, "episode_index": 0}) for _ in range(4)]
    with pytest.raises(ValueError, match="non-decreasing"):
        pack_token_batch(steps=steps, sequence_ids=[0, 1, 0, 1], batch_size=2, continuing=None)


def test_sequence_ids_out_of_range_rejected() -> None:
    from mouse_core.data.token_batch import TokenBatch

    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 0.0, "episode_index": 0}) for _ in range(2)]
    inputs, _, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 1], batch_size=2, continuing=None)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        TokenBatch(
            numeric_mask=inputs.numeric_mask,
            token_ids=inputs.token_ids,
            numeric_values=inputs.numeric_values,
            sequence_ids=inputs.sequence_ids,
            head_token_ids=inputs.head_token_ids,
            head_step_ids=inputs.head_step_ids,
            B=1,
        )


def test_contiguous_sequence_ids_with_gaps_accepted() -> None:
    """Rows with zero steps (decode) leave gaps in the id set; that is fine."""
    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 0.0, "episode_index": 0}) for _ in range(3)]
    inputs, _, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 2, 2], batch_size=3, continuing=None)
    assert inputs.step_counts().tolist() == [1, 0, 2]


@pytest.mark.parametrize("ids", [[0.1, 0.9], [False, True], ["0", "1"]])
def test_sequence_ids_are_not_silently_coerced(ids):
    step = encode_measurement(_tok(), {"action": 0, "reward": 0.0})
    with pytest.raises(ValueError, match="integer"):
        pack_token_batch(steps=[step, step], sequence_ids=ids, continuing=None)
    valid, _, _ = pack_token_batch(steps=[step, step], sequence_ids=[0, 1], continuing=None)
    with pytest.raises(ValueError, match="integer"):
        replace(valid, sequence_ids=np.asarray(ids))


def test_head_tokens_for_one_step_cannot_cross_sequence_boundaries():
    step = encode_measurement(_tok(), {"action": 0, "reward": 0.0})
    valid, _, _ = pack_token_batch(steps=[step, step], sequence_ids=[0, 1], continuing=None)
    with pytest.raises(ValueError, match="one sequence"):
        replace(valid, head_step_ids=np.array([0, 0]))


def test_pack_rejects_negative_sequence_ids() -> None:
    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 0.0, "episode_index": 0})]
    with pytest.raises(ValueError, match="sequence_ids must be >= 0"):
        pack_token_batch(steps=steps, sequence_ids=[-1], batch_size=2, continuing=None)


def test_pack_rejects_objective_key_missing_on_some_steps() -> None:
    tok = _tok()
    steps = [
        encode_measurement(tok, {"action": 0, "reward": 1.0, "episode_index": 0}),
        encode_measurement(tok, {"action": 1, "reward": 0.0, "episode_index": 0}),
    ]
    steps[1].objective_fields.pop("reward")
    with pytest.raises(KeyError, match="reward"):
        pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1, continuing=None)
