from __future__ import annotations

import torch
from tensordict import TensorDict

from mouse_core.objectives import VecDqnObjective


def _make_batch(B: int = 1, S: int = 4, A: int = 3, D: int = 2):
    torch.manual_seed(0)
    done = torch.zeros(B, S, dtype=torch.int64)
    done[0, 1] = 1  # boundary mid-sequence triggers the target substitution path
    objective_data = TensorDict(
        {
            "action": torch.zeros(B, S, dtype=torch.int64),
            "reward": torch.zeros(B, S),
            "done": done,
        },
        batch_size=(B, S),
    )
    predictions = TensorDict(
        {
            "action_vector": torch.randn(B, S, A, D),
            "action_vector_target": torch.randn(B, S, A, D),
        },
        batch_size=(B, S),
    )
    return objective_data, predictions


def test_vec_dqn_objective_does_not_mutate_predictions() -> None:
    objective_data, predictions = _make_batch()
    target_before = predictions["action_vector_target"].clone()

    VecDqnObjective()(objective_data, predictions)

    assert torch.equal(predictions["action_vector_target"], target_before)


def test_vec_dqn_objective_is_idempotent() -> None:
    objective_data, predictions = _make_batch()
    objective = VecDqnObjective()

    loss_first, _ = objective(objective_data, predictions)
    loss_second, _ = objective(objective_data, predictions)

    assert torch.equal(loss_first, loss_second)


def test_vec_dqn_objective_ignores_transitions_across_pack_seams() -> None:
    """Loss must not depend on data at a pair that straddles a packed seam."""
    objective_data, predictions = _make_batch(S=6)
    objective_data["done"].zero_()
    objective_data["is_seam"] = torch.tensor([[0, 0, 0, 1, 0, 0]])

    loss_before, _ = VecDqnObjective()(objective_data, predictions)

    # Corrupt the seam pair: reward entering the seam row and the target
    # vectors at the seam row itself. A masked objective must not notice.
    corrupted_data = objective_data.clone()
    corrupted_preds = predictions.clone()
    corrupted_data["reward"][0, 3] = 1.0e6
    corrupted_preds["action_vector_target"][0, 3] += 100.0

    loss_after, _ = VecDqnObjective()(corrupted_data, corrupted_preds)

    assert torch.allclose(loss_before, loss_after)


def test_vec_dqn_objective_skips_substitution_from_seam_row() -> None:
    """A boundary transition whose reset-substitute row starts a new segment is dropped."""
    # done boundary at t+1=1 makes transition t=0 look for its target at t+2=2,
    # but row 2 begins a new packed segment — so transition 0 must be invalid
    # and the target vectors at row 2 must not influence the loss.
    objective_data, predictions = _make_batch(S=4)
    objective_data["is_seam"] = torch.tensor([[0, 0, 1, 0]])

    loss_before, _ = VecDqnObjective()(objective_data, predictions)

    corrupted_preds = predictions.clone()
    corrupted_preds["action_vector_target"][0, 2] += 100.0
    loss_after, _ = VecDqnObjective()(objective_data, corrupted_preds)

    assert torch.allclose(loss_before, loss_after)
