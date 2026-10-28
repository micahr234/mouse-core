"""Loader shortcuts retain window order, generations, and budget semantics."""

from typing import Any

import numpy as np
import pytest
import torch
from datasets import Dataset

from mouse_core.data import DataLoader, Datastore, StepTokens
from mouse_core.data.dataloader import _ColumnView


class _EncodeOneTokenPerEvent:
    def __init__(self):
        self.generations = []
        self.encoded_generations = []

    def reseed(self, *, generation):
        self.generations.append(generation)

    def __call__(self, events):
        generation = self.generations[-1]
        self.encoded_generations.append(generation)
        return [StepTokens(
            token_ids=np.array([generation + event["source_offset"]]),
            numeric_values=np.zeros(1),
            numeric_mask=np.zeros(1, dtype=bool),
            head_token_mask=np.ones(1, dtype=bool),
            objective_fields={"source_offset": event["source_offset"]},
        ) for event in events]


def _store(lengths):
    store = Datastore()
    for episode, count in enumerate(lengths):
        for _ in range(count):
            store.append(data={"episode_index": episode, "provider": "sensor", "payload": {}})
    return store


@pytest.mark.parametrize("budget", [3, 5, 8])
def test_event_budget_shortcut_preserves_sampling_and_generations(budget):
    # One token per source event makes both budget paths produce identical
    # batches. Token budgeting must still encode the rejected window to count
    # its cost, so it supplies an independent reference for the shortcut.
    event_transform = _EncodeOneTokenPerEvent()
    token_transform = _EncodeOneTokenPerEvent()
    options: dict[str, Any] = dict(stores=_store([1, 2, 3]), batch_size=3,
                                   sample_field="episode_index", index_field="source_offset",
                                   num_workers=0, seed=21)
    with DataLoader(**options, events_budget=budget, transform=event_transform) as events, \
            DataLoader(**options, token_budget=budget, transform=token_transform) as tokens:
        for _ in range(4):
            event_batch, event_data, event_groups = events.next_batch()
            token_batch, token_data, token_groups = tokens.next_batch()
            for field in ("token_ids", "sequence_ids", "head_token_ids", "numeric_values", "numeric_mask"):
                np.testing.assert_array_equal(getattr(event_batch, field), getattr(token_batch, field))
            assert torch.equal(event_groups, token_groups)
            assert torch.equal(event_data["source_offset"], token_data["source_offset"])
        assert event_transform.generations == token_transform.generations
        assert len(token_transform.encoded_generations) - len(event_transform.encoded_generations) == 12


def test_event_budget_rejects_oversized_first_window_before_transform():
    transform = _EncodeOneTokenPerEvent()
    with DataLoader(stores=_store([3]), events_budget=2, batch_size=1,
                    sample_field="episode_index", index_field="source_offset",
                    transform=transform, num_workers=0, seed=0) as loader:
        with pytest.raises(ValueError, match="3 source events.*events_budget=2"):
            loader.next_batch()
    assert transform.generations == [0]
    assert transform.encoded_generations == []


@pytest.mark.parametrize("format_type", [None, "numpy"])
def test_column_view_preserves_slice_order_and_normalization(format_type):
    # An indexed dataset also exercises noncontiguous Arrow row mappings.
    dataset = Dataset.from_dict({"episode_index": [10, 20, 30, 40], "payload": ["a", "b", "c", "d"]})
    dataset = dataset.select([3, 0, 2]).with_format(format_type)
    columns = _ColumnView(ds=dataset, offset=1, n=2)
    np.testing.assert_array_equal(columns["episode_index"], [10, 30])
    np.testing.assert_array_equal(columns["payload"], ["a", "c"])
    assert columns["episode_index"] is columns["episode_index"]
    with pytest.raises(KeyError, match="requires column 'missing'"):
        columns["missing"]
