"""Packed metadata preserves sparse steps, and objective columns."""

import numpy as np
import pytest
import torch

from mouse_core.data.token_batch import StepTokens, pack_token_batch


def _step(ids, heads=(), **fields):
    return StepTokens(
        numeric_field_names=("sensor.value",),
        numeric_field_ids=np.where(np.asarray(ids) % 2 == 0, 0, -1),
        token_ids=np.asarray(ids),
        numeric_values=np.asarray(ids, dtype=np.float32) / 10,
        numeric_mask=np.asarray(ids) % 2 == 0,
        head_mask=np.isin(np.arange(len(ids)), heads),
        tokenizer_ids=np.asarray(ids, dtype=np.int64) % 2,
        tokenizer_names=("text", "numeric"),
        objective_fields=fields,
    )


def test_pack_readouts_around_zero_token_steps():
    steps = [
        _step([], action=None, reward=1),
        _step([10, 11, 12], heads=[0, 2], action=2, reward=0.5),
        _step([], action=None, reward=0),
        _step([20, 21], action=None, reward=0),
        _step([], action=None, reward=0),
        _step([30], heads=[0], action=1, reward=-0.25),
    ]
    batch, fields = pack_token_batch(
        steps=steps, sequence_ids=[0, 0, 2, 2, 2, 4],
        batch_size=6,
    )
    sequence_ids = fields["sequence_ids"]
    expected_ids = [10, 11, 12, 20, 21, 30]
    assert batch['sequence_ids'].tolist() == [0, 0, 0, 2, 2, 4]
    assert batch["head_mask"].nonzero(as_tuple=True)[0].tolist() == [0, 2, 5]
    assert batch['token_ids'].tolist() == expected_ids
    np.testing.assert_allclose(batch['numeric_values'], np.asarray(expected_ids) / 10)
    np.testing.assert_array_equal(batch['numeric_mask'], np.asarray(expected_ids) % 2 == 0)
    np.testing.assert_array_equal(batch['tokenizer_ids'], np.asarray(expected_ids) % 2)
    assert batch['tokenizer_names'] == ("text", "numeric")
    assert fields["step_ids"].tolist() == [1, 1, 5]
    assert torch.bincount(fields["sequence_ids"], minlength=batch["num_sequences"]).tolist() == [2, 0, 3, 0, 1, 0]
    assert sequence_ids.tolist() == [0, 0, 2, 2, 2, 4]
    assert "head_output_count" not in fields
    assert (~torch.isnan(fields["action"])).tolist() == [0, 1, 0, 0, 0, 1]
    torch.testing.assert_close(fields["action"], torch.tensor([float("nan"), 2, float("nan"), float("nan"), float("nan"), 1], dtype=torch.float64), equal_nan=True)
    assert fields["reward"].dtype == torch.float32
    assert fields["reward"].tolist() == [1.0, 0.5, 0.0, 0.0, 0.0, -0.25]
    assert steps[0].objective_fields["action"] is None


def test_pack_all_zero_token_steps_preserves_steps_and_empty_readouts():
    batch, fields = pack_token_batch(
        steps=[_step([]), _step([]), _step([])],
        sequence_ids=[0, 2, 2], batch_size=4,
    )
    sequence_ids = fields["sequence_ids"]
    assert batch["token_ids"].shape[0] == batch["head_mask"].nonzero(as_tuple=True)[0].shape[0] == 0
    assert fields["sequence_ids"].shape[0] == 3
    assert torch.bincount(fields["sequence_ids"], minlength=batch["num_sequences"]).tolist() == [1, 0, 2, 0]
    assert sequence_ids.tolist() == [0, 2, 2]
    assert "head_output_count" not in fields
    assert fields["step_ids"].numel() == 0


@pytest.mark.parametrize("shape", [(), (0,), (2,), (2, 3)])
def test_uniform_objective_columns_keep_shapes_and_promote_across_steps(shape):
    batch, fields = pack_token_batch(
        steps=[
            _step([], target=np.full(shape, 2, dtype=np.int32)),
            _step([], target=torch.full(shape, 0.75, dtype=torch.float64)),
        ],
    )
    assert fields["sequence_ids"].shape[0] == 2
    assert fields["target"].shape == (2, *shape)
    assert fields["target"].dtype == torch.float32
    torch.testing.assert_close(fields["target"][0], torch.full(shape, 2.0))
    torch.testing.assert_close(fields["target"][1], torch.full(shape, 0.75))
