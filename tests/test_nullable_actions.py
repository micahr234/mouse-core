"""A nullable stored action becomes a validity mask only at the tensor boundary."""

from functools import partial
from tests._event_fixtures import encode_measurement

import copy
import json
from typing import Any

import numpy as np
import pytest
import torch
from datasets import load_from_disk

from mouse_core.data import (
    Augmenter,
    DataLoader,
    Datastore,
    EventLog,
    EventTransform,
    environment_episode_ranges,
    Tokenizer,
    environment_decisions,
    pack_token_batch,
)
from mouse_core.objectives import (
    DqnObjective,
    GrpoObjective,
    PpoObjective,
    boundary_discount,
)
from tests._token_batch_helpers import IntIdTokenizer
from mouse_core.data.dataloader import _require_free_threading


def tokenizer(*fields):
    return Tokenizer(
        tokenizer=IntIdTokenizer(),
        input_fields=[{"type": "text", "format": "1", "head_output": True}],
        objective_fields=[{"input_field": key} for key in fields],
    )


@pytest.mark.parametrize("actions", [[None, 0, None, 2], [None, None], [0, 2]])
def test_only_packing_introduces_action_mask(actions):
    rows = [{"action": action, "reward": 0.0} for action in actions]
    steps = [encode_measurement(tokenizer("action", "reward"), row) for row in rows]
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
        encode_measurement(tokenizer("action", "reward"), row)


def test_missing_packed_action_is_not_treated_as_explicit_none():
    tok = tokenizer("action")
    steps = [encode_measurement(tok, {"action": None}), encode_measurement(tok, {"action": 0})]
    del steps[1].objective_fields["action"]
    with pytest.raises(KeyError, match="action"):
        pack_token_batch(steps=steps, continuing=None)


def test_stored_mask_is_rejected_in_favor_of_nullable_action():
    step = encode_measurement(tokenizer("action", "action_executed"),
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


def reset_stream():
    log = EventLog()
    for episode in range(2):
        log.event(provider='coordinator', episode_index=episode, payload={
            'episode_start': True,
        })
        log.event(provider='environment', episode_index=episode, payload={
            'terminated': False,
            'truncated': False,
            "action_requested": True,
        })
        log.event(provider='coordinator', episode_index=episode, payload={
        })
        log.event(provider='agent', episode_index=episode, payload={
            'action': 0,
        })
        log.event(provider='environment', episode_index=episode, payload={
            'terminated': False,
            'truncated': False,
        })
        log.event(provider='coordinator', episode_index=episode, payload={
        })
        log.event(provider='environment', episode_index=episode, payload={
            'terminated': False,
            'truncated': False,
            "action_requested": True,
        })
        log.event(provider='coordinator', episode_index=episode, payload={
        })
        log.event(provider='agent', episode_index=episode, payload={
            'action': 1,
        })
        log.event(provider='environment', episode_index=episode, payload={
            'terminated': True,
            'truncated': False,
        })
    return log.to_store()


def test_resets_persist_without_null_agent_events(tmp_path):
    store = reset_stream()
    store.to_dataset().save_to_disk(str(tmp_path / "replay"))
    reloaded = Datastore()
    reloaded.from_dataset(ds=load_from_disk(str(tmp_path / "replay")))
    assert [r.get('action') for r in list(environment_decisions(reloaded[:]))] == [None, 0, None, 1] * 2
    actions = [json.loads(e['payload'])['action'] for e in reloaded[:] if e['provider'] == 'agent']
    assert actions == [0,1]*2
    assert 'action' not in reloaded.to_dataset().column_names


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_derives_masks_without_mutating_replay(workers):
    if workers:
        try:
            _require_free_threading()
        except RuntimeError as error:
            pytest.skip(str(error))
    store = reset_stream()
    before = store[:]
    loader = DataLoader(
        stores=store, sample_ranges=environment_episode_ranges, events_budget=10,
        batch_size=1, transform=EventTransform(encode=partial(encode_measurement, Tokenizer(
            tokenizer=IntIdTokenizer(), input_fields=[{"type": "text", "format": "1", "head_output": True}],
            objective_fields=[{"input_field": "action", "missing_value": None}]))), num_workers=workers, seed=4,
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
        steps=[encode_measurement(tok, row) for row in rows], continuing=None
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
