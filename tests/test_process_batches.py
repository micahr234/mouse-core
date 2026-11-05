"""Process transport ownership, failure, and notebook callback contracts."""
import os
import threading
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
import torch

from mouse_core.data import StepTokens, pack_token_batch
from mouse_core.data.process_batches import ProcessBatches
from tests.test_dataloader import _loader


def _batch(index, length=3):
    step = StepTokens(
        token_ids=np.full(length, index, dtype=np.int64),
        numeric_values=np.zeros(length), numeric_mask=np.zeros(length, dtype=bool),
        head_token_mask=np.arange(length) == length - 1,
        objective_fields={"vector": np.arange(length, dtype=np.float32)},
    )
    return pack_token_batch(steps=[step], continuing=None)


def test_shared_slots_grow_and_returned_batches_survive_reuse_and_close():
    # A notebook-style closure, and a result larger than the initial slot.
    sizes = [3, 100_000, 5, 200_000, 7, 9]
    def build(index):
        return _batch(index, sizes[index % len(sizes)])

    with_names = []
    pool = ProcessBatches(build, workers=2, prefetch=2)
    try:
        first_inputs, first_data = pool.next_batch()
        for index in range(1, 6):
            inputs, data = pool.next_batch()
            assert inputs["token_ids"].shape[0] == sizes[index]
            assert bool((inputs['token_ids'] == index).all())
            torch.testing.assert_close(data["vector"][0], torch.arange(sizes[index], dtype=torch.float32))
            assert len(pool._active) + len(pool._ready) <= 2
        with_names = [slot.name for slot in pool._slots if slot is not None]
        assert pool._payload is not None
        with_names.append(pool._payload.name)
    finally:
        pool.close()
        pool.close()
    assert first_inputs['token_ids'].tolist() == [0, 0, 0]
    assert first_data["vector"].tolist() == [[0.0, 1.0, 2.0]]
    for name in with_names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_transform_traceback_reaches_consumer_and_cleans_up():
    def build(index):
        if index == 1:
            raise ValueError("test transform exploded")
        return _batch(index)
    pool = ProcessBatches(build, workers=1, prefetch=2)
    try:
        with pytest.raises(RuntimeError, match="test transform exploded"):
            for _ in range(3):
                pool.next_batch()
        assert pool._closed
    finally:
        pool.close()


def test_abrupt_worker_death_does_not_hang():
    def die(index):
        os._exit(7)
    pool = ProcessBatches(die, workers=1, prefetch=1)
    try:
        with pytest.raises(RuntimeError, match="worker"):
            pool.next_batch()
    finally:
        pool.close()


def test_unserializable_transform_fails_before_starting_workers():
    lock = threading.Lock()
    def build(index):
        with lock:
            return _batch(index)
    with pytest.raises(TypeError, match="serialize"):
        ProcessBatches(build, workers=1, prefetch=1)


def test_loader_close_is_terminal_and_old_batch_survives_refresh():
    loader = _loader(events_budget=3, num_workers=4, prefetch=4, seed=10)
    try:
        inputs, data = loader.next_batch()
        expected = inputs['token_ids'].clone(), data["action"].clone()
        loader.refresh()
        loader.next_batch()
    finally:
        loader.close()
    np.testing.assert_array_equal(inputs['token_ids'], expected[0])
    torch.testing.assert_close(data["action"], expected[1])
    with pytest.raises(RuntimeError, match="closed"):
        loader.next_batch()
    with pytest.raises(RuntimeError, match="closed"):
        loader.refresh()


def test_removed_fill_count_is_rejected():
    with pytest.raises(TypeError, match="batch_size"):
        _loader(events_budget=3, batch_size=1)
