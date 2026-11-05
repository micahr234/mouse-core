"""Packed metadata preserves sparse steps, prefixes, and objective columns."""

import numpy as np
import pytest
import torch

from mouse_core.data.token_batch import StepTokens, pack_token_batch


def _step(ids, heads=(), prefix=None, **fields):
    return StepTokens(
        numeric_field_names=("sensor.value",),
        numeric_field_ids=np.where(np.asarray(ids) % 2 == 0, 0, -1),
        group_start_numeric_field_ids=None if prefix is None else np.where(np.asarray(prefix) % 2 == 0, 0, -1),
        token_ids=np.asarray(ids),
        numeric_values=np.asarray(ids, dtype=np.float32) / 10,
        numeric_mask=np.asarray(ids) % 2 == 0,
        head_token_mask=np.isin(np.arange(len(ids)), heads),
        tokenizer_ids=np.asarray(ids, dtype=np.int64) % 2,
        tokenizer_names=("text", "numeric"),
        group_start_token_ids=prefix,
        group_start_numeric_values=None if prefix is None else np.asarray(prefix) / 10,
        group_start_numeric_mask=None if prefix is None else np.asarray(prefix) % 2 == 0,
        group_start_tokenizer_ids=None if prefix is None else np.asarray(prefix) % 2,
        objective_fields=fields,
    )


@pytest.mark.parametrize("continuing", [None, [True, False, False, False, True, False]])
def test_pack_prefixes_and_readouts_around_zero_token_steps(continuing):
    steps = [
        _step([], prefix=[90, 91], action=None, reward=1),
        _step([10, 11, 12], heads=[0, 2], prefix=[99], action=2, reward=0.5),
        _step([], action=None, reward=0),
        # The previous empty step already started sequence 2: omit this prefix.
        _step([20, 21], prefix=[99], action=None, reward=0),
        _step([], action=None, reward=0),
        _step([30], heads=[0], prefix=[92], action=1, reward=-0.25),
    ]
    batch, fields = pack_token_batch(
        steps=steps, sequence_ids=[0, 0, 2, 2, 2, 4],
        batch_size=6, continuing=continuing,
    )
    group_id = fields["step_sequence_ids"]
    if continuing is None:
        expected_ids = [90, 91, 10, 11, 12, 20, 21, 92, 30]
        assert batch['sequence_ids'].tolist() == [0, 0, 0, 0, 0, 2, 2, 4, 4]
        assert batch["head_token_mask"].nonzero(as_tuple=True)[0].tolist() == [2, 4, 8]
    else:
        expected_ids = [10, 11, 12, 20, 21, 30]
        assert batch['sequence_ids'].tolist() == [0, 0, 0, 2, 2, 4]
        assert batch["head_token_mask"].nonzero(as_tuple=True)[0].tolist() == [0, 2, 5]
    assert batch['token_ids'].tolist() == expected_ids
    np.testing.assert_allclose(batch['numeric_values'], np.asarray(expected_ids) / 10)
    np.testing.assert_array_equal(batch['numeric_mask'], np.asarray(expected_ids) % 2 == 0)
    np.testing.assert_array_equal(batch['tokenizer_ids'], np.asarray(expected_ids) % 2)
    assert batch['tokenizer_names'] == ("text", "numeric")
    assert fields["head_step_ids"].tolist() == [1, 1, 5]
    assert torch.bincount(fields["step_sequence_ids"], minlength=batch["num_sequences"]).tolist() == [2, 0, 3, 0, 1, 0]
    assert group_id.tolist() == [0, 0, 2, 2, 2, 4]
    assert "head_output_count" not in fields
    assert (~torch.isnan(fields["action"])).tolist() == [0, 1, 0, 0, 0, 1]
    torch.testing.assert_close(fields["action"], torch.tensor([float("nan"), 2, float("nan"), float("nan"), float("nan"), 1], dtype=torch.float64), equal_nan=True)
    assert fields["reward"].dtype == torch.float32
    assert fields["reward"].tolist() == [1.0, 0.5, 0.0, 0.0, 0.0, -0.25]
    assert steps[0].objective_fields["action"] is None


def test_pack_all_zero_token_steps_preserves_steps_and_empty_readouts():
    batch, fields = pack_token_batch(
        steps=[_step([]), _step([]), _step([])],
        sequence_ids=[0, 2, 2], batch_size=4, continuing=None,
    )
    group_id = fields["step_sequence_ids"]
    assert batch["token_ids"].shape[0] == batch["head_token_mask"].nonzero(as_tuple=True)[0].shape[0] == 0
    assert fields["step_sequence_ids"].shape[0] == 3
    assert torch.bincount(fields["step_sequence_ids"], minlength=batch["num_sequences"]).tolist() == [1, 0, 2, 0]
    assert group_id.tolist() == [0, 2, 2]
    assert "head_output_count" not in fields
    assert fields["head_step_ids"].numel() == 0


@pytest.mark.parametrize("shape", [(), (0,), (2,), (2, 3)])
def test_uniform_objective_columns_keep_shapes_and_promote_across_steps(shape):
    batch, fields = pack_token_batch(
        steps=[
            _step([], target=np.full(shape, 2, dtype=np.int32)),
            _step([], target=torch.full(shape, 0.75, dtype=torch.float64)),
        ], continuing=None,
    )
    assert fields["step_sequence_ids"].shape[0] == 2
    assert fields["target"].shape == (2, *shape)
    assert fields["target"].dtype == torch.float32
    torch.testing.assert_close(fields["target"][0], torch.full(shape, 2.0))
    torch.testing.assert_close(fields["target"][1], torch.full(shape, 0.75))
