"""Tests for StepTokens / TokenBatch packing invariants."""
from __future__ import annotations
from _token_batch_fixture import token_batch_fixture
from tests._event_fixtures import encode_measurement, measurement_events

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from mouse_core.data import EventLog, Tokenizer, latest_environment_decision, pack_token_batch
from tests._token_batch_helpers import IntIdTokenizer


def _tok(**kwargs) -> Tokenizer:
    return Tokenizer(
        input_fields=[
            {"provider": "sensor", "type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {"provider": "sensor",
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": lambda ctx: f"{ctx['field']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        ],
        objective_fields=[{"input_field": "reward"}, {"input_field": "action"}],
        tokenizers={"default": kwargs.pop("tokenizer", IntIdTokenizer())},
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
    inputs, inputs_objective_data = tok.pack_rows(rows=[measurement_events(payloads) for payloads in rows])
    assert inputs["num_sequences"] == 3
    assert torch.bincount(inputs_objective_data["step_sequence_ids"], minlength=inputs["num_sequences"]).tolist() == [2, 0, 1]
    manual, manual_objective_data = pack_token_batch(
        steps=[encode_measurement(tok, step) for row in rows for step in row],
        sequence_ids=[0, 0, 2],
        batch_size=3)
    assert inputs['token_ids'].tolist() == manual['token_ids'].tolist()
    assert inputs['sequence_ids'].tolist() == manual['sequence_ids'].tolist()
    assert inputs["head_token_mask"].nonzero(as_tuple=True)[0].tolist() == manual["head_token_mask"].nonzero(as_tuple=True)[0].tolist()


def test_tokenizer_pack_rows_all_empty_keeps_batch_slots() -> None:
    tok = _tok()
    inputs, inputs_objective_data = tok.pack_rows(rows=[measurement_events(payloads) for payloads in [[], [], []]])
    assert inputs["num_sequences"] == 3
    assert inputs["token_ids"].shape[0] == 0
    assert torch.bincount(inputs_objective_data["step_sequence_ids"], minlength=inputs["num_sequences"]).tolist() == [0, 0, 0]


def test_sparse_inference_inputs_and_strict_training_labels():
    tok = Tokenizer(input_fields=[
        dict(type='text', provider='environment', input_field='observation', format='{field}'),
        dict(type='token', provider='environment', format='4', head_output=True),
    ], objective_fields=[dict(input_field='advantage')], tokenizers={"default": IntIdTokenizer()})
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
    inputs, inputs_objective_data = tok.pack_rows(rows=[log.events, []])
    assert inputs['token_ids'].tolist() == [2, 4]
    assert torch.bincount(inputs_objective_data["step_sequence_ids"], minlength=inputs["num_sequences"]).tolist() == [1, 0]
    # Training stays strict; inference must not remove configured labels or invent them.
    with pytest.raises(KeyError, match='advantage'):
        tok.encode(log)
    events = log.events
    events[-1]['payload']['advantage'] = 1.5
    training = tok.encode(events)[0]
    np.testing.assert_array_equal(inputs['token_ids'], training.token_ids)
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
    sparse, sparse_objective_data = tok.pack_rows(rows=[missing_log.events])
    assert sparse['token_ids'].tolist() == [4]
    assert torch.bincount(sparse_objective_data["step_sequence_ids"], minlength=sparse["num_sequences"]).tolist() == [1]


@pytest.mark.parametrize('provider', ['environment', 'agent'])
def test_event_tokenizer_groups_repeated_providers_by_marker(provider):
    tok = Tokenizer(input_fields=[dict(type='text', provider=provider, input_field='value', format='{field}'),
                                          dict(type='token', provider=provider, format='4', head_output=True)], tokenizers={"default": IntIdTokenizer()})
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
    inputs, objective = pack_token_batch(steps=steps, sequence_ids=[0, 0, 0], batch_size=1)
    assert objective["reward"].dtype == torch.float32
    assert objective["reward"].tolist() == [1.0, 0.75, 0.0]
    assert objective["action"].dtype == torch.float64


def test_objective_column_stays_int_when_all_steps_are_int() -> None:
    tok = _tok()
    steps = [encode_measurement(tok, {"action": a, "reward": a, "episode_index": 0}) for a in range(3)]
    inputs, objective = pack_token_batch(steps=steps, sequence_ids=[0, 0, 0], batch_size=1)
    group_id = objective["step_sequence_ids"]
    assert "group_id" not in objective
    assert group_id.tolist() == [0, 0, 0]
    assert objective["reward"].dtype == torch.int64


def test_objective_vector_column_promotes_dtype() -> None:
    tok = Tokenizer(
        input_fields=[
            {"provider": "sensor", "type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {"provider": "sensor",
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": lambda ctx: f"{ctx['field']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizers={"default": IntIdTokenizer()},
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1, 2]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": np.array([0.5, 0.25]), "episode_index": 0}),
    ]
    inputs, objective = pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1)
    assert objective["q"].dtype == torch.float32
    assert objective["q"].tolist() == [[1.0, 2.0], [0.5, 0.25]]


def test_objective_ragged_float_vectors_pad_with_neg_inf() -> None:
    """Shorter rows are padded with -inf (the nonexistent-action sentinel), not 0."""
    tok = Tokenizer(
        input_fields=[
            {"provider": "sensor", "type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {"provider": "sensor",
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": lambda ctx: f"{ctx['field']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizers={"default": IntIdTokenizer()},
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1.0, 2.0, 3.0]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": np.array([0.5, 0.25]), "episode_index": 0}),
    ]
    inputs, objective = pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1)
    q = objective["q"]
    assert q[0].tolist() == [1.0, 2.0, 3.0]
    assert q[1, :2].tolist() == [0.5, 0.25]
    assert q[1, 2].item() == -torch.inf


def test_objective_ragged_int_vectors_raise() -> None:
    """Integer columns have no padding sentinel, so ragged shapes are an error."""
    tok = Tokenizer(
        input_fields=[
            {"provider": "sensor", "type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {"provider": "sensor",
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": lambda ctx: f"{ctx['field']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizers={"default": IntIdTokenizer()},
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1, 2, 3]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": np.array([4, 5]), "episode_index": 0}),
    ]
    with pytest.raises(ValueError, match="ragged"):
        pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1)


def test_objective_mixed_rank_raises() -> None:
    tok = Tokenizer(
        input_fields=[
            {"provider": "sensor", "type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {"provider": "sensor",
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": lambda ctx: f"{ctx['field']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        ],
        objective_fields=[{"input_field": "q"}],
        tokenizers={"default": IntIdTokenizer()},
    )
    steps = [
        encode_measurement(tok, {"action": 0, "q": np.array([1.0, 2.0]), "episode_index": 0}),
        encode_measurement(tok, {"action": 0, "q": 0.5, "episode_index": 0}),
    ]
    with pytest.raises(ValueError, match="mixes array ranks"):
        pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1)


def test_contiguous_sequence_ids_with_gaps_accepted() -> None:
    """Rows with zero steps (decode) leave gaps in the id set; that is fine."""
    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 0.0, "episode_index": 0}) for _ in range(3)]
    inputs, inputs_objective_data = pack_token_batch(steps=steps, sequence_ids=[0, 2, 2], batch_size=3)
    assert torch.bincount(inputs_objective_data["step_sequence_ids"], minlength=inputs["num_sequences"]).tolist() == [1, 0, 2]


@pytest.mark.parametrize("ids", [[0.1, 0.9], [False, True], ["0", "1"]])
def test_sequence_ids_are_not_silently_coerced(ids):
    step = encode_measurement(_tok(), {"action": 0, "reward": 0.0})
    with pytest.raises(ValueError, match="integer"):
        pack_token_batch(steps=[step, step], sequence_ids=ids)
    valid, valid_objective_data = pack_token_batch(steps=[step, step], sequence_ids=[0, 1])


def test_pack_rejects_negative_sequence_ids() -> None:
    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 0.0, "episode_index": 0})]
    with pytest.raises(ValueError, match="sequence_ids must be >= 0"):
        pack_token_batch(steps=steps, sequence_ids=[-1], batch_size=2)


def test_pack_rejects_objective_key_missing_on_some_steps() -> None:
    tok = _tok()
    steps = [
        encode_measurement(tok, {"action": 0, "reward": 1.0, "episode_index": 0}),
        encode_measurement(tok, {"action": 1, "reward": 0.0, "episode_index": 0}),
    ]
    steps[1].objective_fields.pop("reward")
    with pytest.raises(KeyError, match="reward"):
        pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1)


def test_tensor_inputs_and_objectives_have_independent_batch_dimensions():
    from tensordict import TensorDict
    tok = _tok()
    steps = [encode_measurement(tok, {"action": 0, "reward": 1.0}) for _ in range(2)]
    inputs, objective = pack_token_batch(steps=steps, sequence_ids=[0, 1], batch_size=3)
    assert isinstance(inputs, TensorDict) and isinstance(objective, TensorDict)
    assert inputs.batch_size == torch.Size([])
    assert objective.batch_size == torch.Size([])
    assert inputs["num_sequences"] == 3
    assert "head_token_ids" not in inputs
    assert "head_step_ids" not in inputs and "step_sequence_ids" not in inputs
    assert inputs["head_token_mask"].dtype == torch.bool
    assert objective["head_step_ids"].tolist() == [0, 1]
    assert objective["step_sequence_ids"].tolist() == [0, 1]
    assert inputs.to("cpu") is inputs
    assert objective.to("cpu") is objective
    # Tensor storage survives inference use and can then be saved for backward.
    with torch.inference_mode():
        inputs.to("cpu")
    embedding = torch.nn.Embedding(32, 4)
    embedding(inputs["token_ids"]).sum().backward()
    assert embedding.weight.grad is not None


def test_tensor_batch_construction_does_not_validate_indices():
    from mouse_core.data import make_token_batch
    inputs = make_token_batch(token_ids=[1], numeric_values=[0.0], numeric_mask=[False],
                             sequence_ids=[7], head_token_mask=[True], num_sequences=1)
    assert inputs["sequence_ids"].tolist() == [7]
    assert inputs["head_token_mask"].tolist() == [True]
