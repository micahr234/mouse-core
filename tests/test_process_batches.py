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
            assert len(pool._pending) == 2
        with_names = list(pool._names)
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
    assert payload.buf is not None
    payload.buf[:] = bytes(range(16))
    references = []

    class Connection:
        def __init__(self):

            self.messages = []
            self.closed = False

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

    class Jobs:
        def __init__(self): self.jobs = iter([(3, 0, None), None])
        def get(self): return next(self.jobs)

    class Signal:
        def __init__(self): self.value = False
        def set(self): self.value = True
        def release(self): self.value = True

    ready, completed = Signal(), Signal()
    connection = Connection()
    monkeypatch.setattr(process_batches.cloudpickle, "loads", loads)
    monkeypatch.setattr(process_batches.torch, "set_num_threads", lambda _: None)
    try:
        process_batches._worker(connection, payload.name, Jobs(), [output.name], [ready], completed)
        assert connection.closed
        assert [message[0] for message in connection.messages] == (["started", "error"] if fail else ["started"])
        assert references[0]() is None
        if fail:
            assert "retained view still readable: 7" in connection.messages[1][1]
        else:
            assert ready.value and completed.value
            inputs, _ = process_batches._read_slot(output.name)
            assert inputs["token_ids"].tolist() == [10, 10, 10]
        # The worker closes its mapping but never unlinks the parent's payload.
        assert payload.buf is not None
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


@pytest.mark.parametrize("prefetch", [1, 16])
def test_background_prefetch_fills_and_refills_while_consumer_is_idle(prefetch):
    def build(index):
        # Different completion order must not change the consumer's batch order.
        if index % 2 == 0:
            time.sleep(.02)
        return _batch(index)

    threads = {thread.ident for thread in threading.enumerate()}
    pool = ProcessBatches(build, workers=2, prefetch=prefetch)
    names = []
    try:
        for first_index in (0, 3):
            for event in pool._ready:
                assert event.wait(30), "worker did not fill its slot while consumer was idle"
            assert pool._next_submit == first_index + prefetch
            assert {thread.ident for thread in threading.enumerate()} == threads
            for index in range(first_index, first_index + 3):
                inputs, _ = pool.next_batch()
                assert inputs['token_ids'].tolist() == [index] * 3
        names = list(pool._names)
    finally:
        pool.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


@pytest.mark.parametrize("in_order", [True, False])
def test_order_option_handles_a_slow_first_batch_without_duplicates(tmp_path, in_order):
    gate = tmp_path / "release-first"
    def build(index):
        if index == 0:
            deadline = time.monotonic() + 30
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("first batch was not released")
                time.sleep(.005)
        return _batch(index)

    pool = ProcessBatches(build, workers=2, prefetch=3, in_order=in_order)
    names = list(pool._names)
    try:
        assert pool._ready[1].wait(30)
        assert pool._ready[2].wait(30)
        assert not pool._ready[0].is_set()
        if in_order:
            gate.touch()
            received = [int(pool.next_batch()[0]['token_ids'][0]) for _ in range(3)]
            assert received == [0, 1, 2]
        else:
            assert int(pool.next_batch()[0]['token_ids'][0]) == 1
            gate.touch()
            assert pool._ready[0].wait(30)
            received = [1] + [int(pool.next_batch()[0]['token_ids'][0]) for _ in range(2)]
            assert sorted(received) == [0, 1, 2]
    finally:
        gate.touch()
        pool.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


def test_close_discards_unread_grown_slots_without_leaking():
    pool = ProcessBatches(lambda index: _batch(index, 100_000), workers=2, prefetch=4)
    names = list(pool._names)
    try:
        for event in pool._ready:
            assert event.wait(30)
    finally:
        pool.close()
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)


@pytest.mark.parametrize("in_order", [True, False])
def test_recycled_slots_keep_batch_data_and_context_aligned(in_order):
    def build(index, *, context):
        assert context["index"] == index
        if index % 7 == 0:
            time.sleep(.005)
        return _batch(index, length=3 + index % 5)

    pool = ProcessBatches(build, workers=3, prefetch=8, in_order=in_order,
                          context_for_batch=lambda index: {"index": index})
    seen = []
    try:
        for _ in range(100):
            inputs, objective = pool.next_batch()
            index = int(inputs['token_ids'][0])
            assert inputs['token_ids'].tolist() == [index] * (3 + index % 5)
            assert objective['vector'].tolist() == [list(range(3 + index % 5))]
            seen.append(index)
        assert len(set(seen)) == 100
        if in_order:
            assert seen == list(range(100))
    finally:
        pool.close()
