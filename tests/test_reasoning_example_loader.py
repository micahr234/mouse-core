"""Reasoning examples use ordinary episode sampling and loader-owned packing."""
import importlib
import json

import pytest
import torch

from mouse_core.data import Datastore
from tests._reasoning_replay_fixture import recorded_stores, reasoning_loader


@pytest.fixture(params=['autoregressive', 'residual'])
def example(request):
    return importlib.import_module(f'tests.test_{request.param}_reasoning').example_module()


def test_loader_skips_partial_prefixes_and_unfinished_tails(example):
    complete = recorded_stores(episodes=1)[0]
    events = [dict(row, payload=json.loads(row['payload'])) for row in complete.to_dataset().to_list()]
    partial, unfinished = Datastore(name='partial'), Datastore(name='unfinished')
    for event in events[2:]:
        partial.append(data=event)
    for event in events[:-2]:
        unfinished.append(data=event)
    with reasoning_loader(example, stores=[complete, partial, unfinished]) as loader:
        tokens, data = loader.next_batch()
    # Two complete three-observation samples, with no prediction on terminal rows.
    assert data['sequence_ids'].tolist() == [0, 0, 0, 1, 1, 1]
    assert data['step_ids'].tolist() == [0, 1, 3, 4]
    assert data['truncated'].tolist() == [False, False, True] * 2
    assert tokens['sequence_ids'][tokens['head_mask']].tolist() == [0, 0, 1, 1]
    assert tokens['autoregressive_mask'].equal(tokens['head_mask'])
    with reasoning_loader(example, stores=[partial, unfinished]) as loader:
        with pytest.raises(ValueError, match='no eligible samples'):
            loader.next_batch()


def test_loader_workers_preserve_batches_and_source_events(example):
    stores = recorded_stores(episodes=3)
    before = stores[0].to_dataset().to_dict()
    with reasoning_loader(example, stores=stores, num_workers=0) as synchronous:
        expected = [synchronous.next_batch() for _ in range(2)]
    with reasoning_loader(example, stores=stores, num_workers=1) as worker:
        actual = [worker.next_batch() for _ in range(2)]
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        for left, right in zip(expected_batch, actual_batch, strict=True):
            assert left.keys() == right.keys()
            for key in left.keys():
                if isinstance(left[key], torch.Tensor):
                    torch.testing.assert_close(left[key], right[key], equal_nan=True)
                else:
                    assert left[key] == right[key]
    assert stores[0].to_dataset().to_dict() == before
