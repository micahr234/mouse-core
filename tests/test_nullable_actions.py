"""A nullable stored action becomes a validity mask only at the tensor boundary."""

import copy
from typing import Any

import numpy as np
import pytest
import torch
from datasets import load_from_disk

from mouse_core.data import (
    Augmenter,
    DataLoader,
    Datastore,
    Tokenizer,
    pack_token_batch,
)
from mouse_core.objectives import (
    DqnObjective,
    GrpoObjective,
    PpoObjective,
    boundary_discount,
)
from tests._token_batch_helpers import IntIdTokenizer


def tokenizer(*fields):
    return Tokenizer(
        tokenizer=IntIdTokenizer(),
        input_fields=[{"type": "text", "format": "1", "head_output": True}],
        objective_fields=[{"input_field": key} for key in fields],
    )


@pytest.mark.parametrize("actions", [[None, 0, None, 2], [None, None], [0, 2]])
def test_only_packing_introduces_action_mask(actions):
    rows = [{"action": action, "reward": 0.0} for action in actions]
    steps = [tokenizer("action", "reward")(row) for row in rows]
    _, data, _ = pack_token_batch(steps=steps, continuing=None)
    assert data["action"].dtype == torch.int64
    assert data["action"].tolist() == [0 if a is None else a for a in actions]
    assert data["action_executed"].tolist() == [a is not None for a in actions]
    assert [row["action"] for row in rows] == actions
    assert [step.objective_fields["action"] for step in steps] == actions
    assert all("action_executed" not in step.objective_fields for step in steps)


@pytest.mark.parametrize("row", [{}, {"action": None, "reward": None}])
def test_missing_action_and_null_reward_are_still_errors(row):
    with pytest.raises(KeyError):
        tokenizer("action", "reward")(row)


def test_missing_packed_action_is_not_treated_as_explicit_none():
    tok = tokenizer("action")
    steps = [tok({"action": None}), tok({"action": 0})]
    del steps[1].objective_fields["action"]
    with pytest.raises(KeyError, match="action"):
        pack_token_batch(steps=steps, continuing=None)


def test_stored_mask_is_rejected_in_favor_of_nullable_action():
    step = tokenizer("action", "action_executed")(
        {"action": 0, "action_executed": False}
    )
    with pytest.raises(ValueError, match="derived during packing"):
        pack_token_batch(steps=[step], continuing=None)


def test_augmentation_preserves_none_and_permutates_reset_value_targets():
    augmenter = Augmenter(
        seed=12,
        seed_field="episode_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "input_vector_field": "q",
                "vocab_size": 4,
                "permute": True,
            }
        ],
    )
    row = {"action": None, "q": [1.0, 3.0, 5.0, 7.0], "episode_index": 0}
    before = copy.deepcopy(row)
    reset = augmenter(row)
    assert reset["action"] is None
    for action, value in enumerate(row["q"]):
        actual = augmenter({**row, "action": action})
        assert reset["q"][actual["action"]] == value
        np.testing.assert_array_equal(reset["q"], actual["q"])
    assert row == before


def test_null_actions_survive_disk_roundtrip_and_appending_real_actions(tmp_path):
    store = Datastore()
    store.append(data={"action": None, "episode_index": 0})
    # Exercise a source whose inferred action type is initially null.
    reloaded = Datastore()
    reloaded.from_dataset(ds=store.to_dataset())
    reloaded.append(data={"action": 0, "episode_index": 0})
    reloaded.append(data={"action": None, "episode_index": 1})
    reloaded.to_dataset().save_to_disk(str(tmp_path / "replay"))
    saved = load_from_disk(str(tmp_path / "replay"))
    assert list(saved["action"]) == [None, 0, None]
    assert "action_executed" not in saved.column_names


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_derives_masks_without_mutating_replay(workers):
    store = Datastore()
    for episode in range(2):
        for action in (None, 0, None, 1):
            store.append(data={"action": action, "episode_index": episode})
    before = store[:]
    loader = DataLoader(
        stores=store,
        sample_field="episode_index",
        samples_budget=4,
        batch_size=1,
        transform=tokenizer("action"),
        num_workers=workers,
        seed=4,
    )
    try:
        _, data, _ = loader.next_batch()
        assert data["action"].tolist() == [0, 0, 0, 1]
        assert data["action_executed"].tolist() == [False, True, False, True]
        assert store[:] == before
    finally:
        loader.close()


@pytest.mark.parametrize("algorithm", ["dqn", "grpo", "ppo"])
def test_nullable_rows_through_objective_ignore_automatic_decisions(algorithm):
    rows = [
        dict(
            action=action,
            reward=reward,
            terminated=i == 4,
            truncated=False,
            old_log_prob=-0.7 if action is not None else float("nan"),
            advantage=1.0 if action is not None else float("nan"),
        )
        for i, (action, reward) in enumerate(
            zip([None, 0, None, 1, 0], [0.0, 2.0, 7.0, 3.0, 4.0])
        )
    ]
    tok = tokenizer(*rows[0])
    _, data, groups = pack_token_batch(
        steps=[tok(row) for row in rows], continuing=None
    )
    predictions = torch.tensor(
        [[1.0, 2.0], [float("nan")] * 2, [3.0, 4.0], [5.0, 6.0], [9.0, 10.0]],
        requires_grad=True,
    )
    discount = boundary_discount(
        gamma_step=0.9, gamma_terminated=0.0, gamma_truncated=0.0
    )
    kwargs: dict[str, Any] = dict(
        objective_data=data, group_id=groups, predictions=predictions
    )
    if algorithm == "dqn":
        obj = DqnObjective(
            reward=None,
            value=None,
            discount=discount,
            temperature=0.0,
            gate=None,
            cross_group_backups="fault",
        )
        target = predictions.detach().clone()
        target[2] = torch.tensor([4.0, 5.0])
        kwargs.update(backup_action_predictions=target, backup_value_predictions=target)
    elif algorithm == "grpo":
        obj = GrpoObjective()
    else:
        obj = PpoObjective(
            discount=discount, reward=None, value=None, cross_group_backups="fault"
        )
        kwargs["value_predictions"] = torch.arange(
            5, dtype=torch.float32, requires_grad=True
        )
    losses, metrics = obj(**kwargs)
    loss = torch.stack(list(losses.values())).sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad[1].count_nonzero() == 0
    if algorithm == "dqn":
        assert isinstance(metrics["backup"], torch.Tensor)
        torch.testing.assert_close(
            metrics["backup"][[0, 2, 3]], torch.tensor([12.35, 8.4, 4.0])
        )
