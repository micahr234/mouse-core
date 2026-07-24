from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import VecDqnObjective

def _make_batch(*, N: int=4, A: int=3, D: int=2, sequence_id: list[int] | None=None):
    torch.manual_seed(0)
    done = torch.zeros(N, dtype=torch.int64)
    done[1] = 1
    sid = sequence_id if sequence_id is not None else [0] * N
    objective_data = TensorDict({'action': torch.zeros(N, dtype=torch.int64), 'reward': torch.zeros(N), 'done': done, 'sequence_id': torch.tensor(sid)}, batch_size=[N])
    predictions = TensorDict({'action_vector': torch.randn(N, A, D), 'action_vector_target': torch.randn(N, A, D)}, batch_size=[N])
    return (objective_data, predictions)

def test_vec_dqn_objective_does_not_mutate_predictions() -> None:
    objective_data, predictions = _make_batch()
    target_before = predictions['action_vector_target'].clone()
    VecDqnObjective()(objective_data, predictions)
    assert torch.equal(predictions['action_vector_target'], target_before)

def test_vec_dqn_objective_is_idempotent() -> None:
    objective_data, predictions = _make_batch()
    objective = VecDqnObjective()
    loss_first, _ = objective(objective_data, predictions)
    loss_second, _ = objective(objective_data, predictions)
    assert torch.equal(loss_first, loss_second)

def test_vec_dqn_objective_ignores_transitions_across_sequences() -> None:
    """Loss must not depend on data at a pair that straddles a sequence boundary."""
    objective_data, predictions = _make_batch(N=6, sequence_id=[0, 0, 0, 1, 1, 1])
    objective_data['done'].zero_()
    loss_before, _ = VecDqnObjective()(objective_data, predictions)
    corrupted_data = objective_data.clone()
    corrupted_preds = predictions.clone()
    corrupted_data['reward'][3] = 1000000.0
    corrupted_preds['action_vector_target'][3] += 100.0
    loss_after, _ = VecDqnObjective()(corrupted_data, corrupted_preds)
    assert torch.allclose(loss_before, loss_after)

def test_vec_dqn_objective_skips_substitution_from_other_sequence() -> None:
    """A boundary transition whose reset-substitute row is a different sequence is dropped."""
    objective_data, predictions = _make_batch(N=4, sequence_id=[0, 0, 1, 1])
    loss_before, _ = VecDqnObjective()(objective_data, predictions)
    corrupted_preds = predictions.clone()
    corrupted_preds['action_vector_target'][2] += 100.0
    loss_after, _ = VecDqnObjective()(objective_data, corrupted_preds)
    assert torch.allclose(loss_before, loss_after)
