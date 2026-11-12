"""Process transport ownership, failure, and notebook callback contracts."""
import os
import threading
import time
import weakref
from multiprocessing.shared_memory import SharedMemory

import cloudpickle
import numpy as np
import pytest
import torch

from mouse_core.data import StepTokens, pack_token_batch
from mouse_core.data import process_batches
from mouse_core.data.process_batches import ProcessBatches
from tests.test_dataloader import _loader


def _batch(index, length=3):
    step = StepTokens(
        token_ids=np.full(length, index, dtype=np.int64),
        numeric_values=np.zeros(length), numeric_mask=np.zeros(length, dtype=bool),
        head_mask=np.arange(length) == length - 1,
        objective_fields={"vector": np.arange(length, dtype=np.float32)},
    )
    return pack_token_batch(steps=[step])


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


def test_worker_diagnostics_include_idle_workers_and_exclude_consumer_delay():
    # An idle worker still needs to report readiness when prefetch < workers.
    started = time.perf_counter()
    pool = ProcessBatches(_batch, workers=2, prefetch=1, start=7)
    try:
        assert pool.worker_payload_bytes == len(cloudpickle.dumps(_batch))
        assert pool.worker_startup_seconds is None
        # Observe notifications waiting in the pipes without consuming them.
        for connection in pool._connections:
            assert connection.poll(30), "worker did not report startup"
        time.sleep(0.2)
        inputs, _ = pool.next_batch()
        assert inputs["token_ids"].tolist() == [7, 7, 7]
        assert pool.worker_startup_seconds is not None
        assert 0 < pool.worker_startup_seconds < time.perf_counter() - started - 0.15
        assert not pool._starting
        diagnostics = pool.worker_payload_bytes, pool.worker_startup_seconds
    finally:
        pool.close()
    assert (pool.worker_payload_bytes, pool.worker_startup_seconds) == diagnostics


@pytest.mark.parametrize("fail", [False, True])
def test_worker_mapping_survives_deserialized_views_and_closes_after_callable_cycles(monkeypatch, fail):
    # Exercise a deserializer retaining views rather than assume pickle always
    # copies its input. The callable deliberately owns an unreachable cycle.
    payload = SharedMemory(create=True, size=16)
    output = SharedMemory(create=True, size=65536)
    payload.buf[:] = bytes(range(16))
    references = []

    class Connection:
        def __init__(self):
            self.jobs = iter([(3, output.name, output.size, None), None])
            self.messages = []
            self.closed = False

        def recv(self):
            return next(self.jobs)

        def send(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    def loads(buffer):
        assert isinstance(buffer, memoryview)
        assert buffer.readonly

        class Build:
            def __init__(self):
                self.values = np.frombuffer(buffer, dtype=np.uint8)
                self.cycle = self

            def __call__(self, index):
                if fail:
                    raise ValueError(f"retained view still readable: {self.values[7]}")
                return _batch(index + int(self.values[7]))

        build = Build()
        references.append(weakref.ref(build))
        return build

    connection = Connection()
    monkeypatch.setattr(process_batches.cloudpickle, "loads", loads)
    monkeypatch.setattr(process_batches.torch, "set_num_threads", lambda _: None)
    try:
        process_batches._worker(connection, payload.name)
        assert connection.closed
        assert [message[0] for message in connection.messages] == ["started", "error" if fail else "ready"]
        assert references[0]() is None
        if fail:
            assert "retained view still readable: 7" in connection.messages[1][1]
        else:
            descriptors = connection.messages[1][2]
            _, _, shape, dtype, offset = next(item for item in descriptors if item[1] == "token_ids")
            actual = np.ndarray(shape, dtype=dtype, buffer=output.buf, offset=offset).copy()
            assert actual.tolist() == [10, 10, 10]
        # The worker closes its mapping but never unlinks the parent's payload.
        assert bytes(payload.buf) == bytes(range(16))
    finally:
        output.close()
        output.unlink()
        payload.close()
        payload.unlink()


def _fail_during_deserialization():
    raise ValueError("test startup deserialization exploded")


class _StartupFailure:
    def __reduce__(self):
        return _fail_during_deserialization, ()


def test_startup_deserialization_failure_reports_traceback_and_releases_payload():
    pool = ProcessBatches(_StartupFailure(), workers=2, prefetch=1)
    assert pool._payload is not None
    payload_name = pool._payload.name
    try:
        with pytest.raises(RuntimeError, match="test startup deserialization exploded"):
            pool.next_batch()
        assert pool._closed
        assert pool.worker_startup_seconds is None
        assert pool.worker_payload_bytes > 0
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=payload_name)
    finally:
        pool.close()
