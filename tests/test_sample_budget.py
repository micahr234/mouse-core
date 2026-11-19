"""Sample budgets retain whole episodes independently of tokenization length."""
from typing import Any
from functools import partial

import numpy as np
import pytest
import torch

from mouse_core.data import DataLoader, EpochDataLoader, EventLog, StepTokens


def store():
    log = EventLog()
    for episode, length in enumerate([2, 5, 3, 4]):
        for step in range(length):
            log.event(provider='environment', episode_index=episode, step_index=step,
                      payload={'step': step, 'complete': episode != 3 and step == length - 1})
    return log.to_store()


def eligible(events):
    return events[-1]['payload']['complete']


def transform(events, *, width=1, context=None):
    width = width if context is None else context['width']
    return [StepTokens(
        token_ids=np.full(width, event['episode_index'] + 1),
        numeric_values=np.zeros(width), numeric_mask=np.zeros(width, dtype=bool),
        head_mask=np.arange(width) == width - 1,
        objective_fields={'episode': event['episode_index'], 'step': event['payload']['step']},
    ) for event in events]


def arguments() -> dict[str, Any]:
    return dict(stores=[store()], sample_field='episode_index', sample_eligible=eligible,
                transform=transform, num_workers=0, seed=12)


def assert_complete(data, count):
    groups = data['sequence_ids'].unique_consecutive()
    assert groups.tolist() == list(range(count))
    for group in groups:
        mask = data['sequence_ids'] == group
        episodes = data['episode'][mask].unique()
        assert len(episodes) == 1
        episode = int(episodes[0])
        assert episode != 3  # incomplete tail is not a sample
        assert data['step'][mask].tolist() == list(range([2, 5, 3][episode]))


@pytest.mark.parametrize('workers', [0, 2])
def test_sample_budget_invariant_to_token_length_and_refresh(workers):
    kwargs = arguments()
    with DataLoader(**kwargs, sample_budget=4) as reference, DataLoader(
        **{**kwargs, 'num_workers': workers}, sample_budget=4,
        context_for_batch=lambda k: {'width': [1, 9, 2][k % 3]},
    ) as expanded:
        for batch in range(6):
            if batch == 3:
                reference.refresh()
                expanded.refresh()
            left, expected = reference.next_batch()
            right, actual = expanded.next_batch()
            assert_complete(actual, 4)
            for field in ('episode', 'step', 'sequence_ids', 'step_ids'):
                torch.testing.assert_close(actual[field], expected[field])
            width = [1, 9, 2][batch % 3]
            assert right['token_ids'].numel() == left['token_ids'].numel() * width
            assert right['head_mask'].sum() == left['head_mask'].sum()
        assert 'sample_budget=4' in repr(expanded)


@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('width', [1, 11])
def test_epoch_sample_budget_retains_partial_batches(workers, width):
    loader = EpochDataLoader(**{**arguments(), 'num_workers': workers,
        'transform': partial(transform, width=width)}, sample_budget=2, epochs=3)
    try:
        assert [len(batch) for batch in loader.batches] == [2, 1] * 3
        assert loader.epoch_ends == [2, 4, 6]
        seen = []
        while not loader.exhausted:
            inputs, data = loader.next_batch()
            count = 2 if loader.position % 2 else 1
            assert_complete(data, count)
            assert inputs['token_ids'].numel() == len(data['step']) * width
            seen.extend(int(data['episode'][data['sequence_ids'] == group][0]) for group in range(count))
        for start in range(0, len(seen), 3):
            assert sorted(seen[start:start + 3]) == [0, 1, 2]
    finally:
        loader.close()


@pytest.mark.parametrize('loader_class', [DataLoader, EpochDataLoader])
@pytest.mark.parametrize('budget', [0, -1, 1.5, True, '2'])
def test_sample_budget_rejects_invalid_count(loader_class, budget):
    with pytest.raises(ValueError, match='sample_budget must be a positive integer'):
        loader_class(**arguments(), sample_budget=budget)


@pytest.mark.parametrize('loader_class', [DataLoader, EpochDataLoader])
@pytest.mark.parametrize('budgets', [{}, {'token_budget': 10, 'sample_budget': 2}])
def test_exactly_one_budget_required(loader_class, budgets):
    with pytest.raises(ValueError, match='exactly one'):
        loader_class(**arguments(), **budgets)


def test_sample_and_event_budget_conflict():
    with pytest.raises(ValueError, match='exactly one'):
        DataLoader(**arguments(), sample_budget=2, events_budget=10)


def test_no_eligible_samples_fails_instead_of_short_batch():
    with DataLoader(**{**arguments(), 'sample_eligible': lambda events: False}, sample_budget=2) as loader:
        with pytest.raises(ValueError, match='no eligible samples'):
            loader.next_batch()


def test_sample_budget_stops_without_tokenizing_an_extra_sample():
    calls = []
    def counted(events):
        calls.append(events)
        return transform(events)
    with DataLoader(**{**arguments(), 'transform': counted}, sample_budget=1) as loader:
        _, data = loader.next_batch()
    assert len(calls) == 1
    assert_complete(data, 1)
