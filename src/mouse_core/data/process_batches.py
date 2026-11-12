"""Bounded, ordered process workers with reusable shared-memory result slots.

Only task indices, small per-job contexts, array descriptors and errors cross
the pipes. The parent owns all shared-memory allocations and copies results before recycling a slot, so a
returned batch remains valid after the next read, refresh or close.
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import traceback
from collections import deque
from multiprocessing.connection import Connection, wait
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter
from types import MappingProxyType
from typing import cast

import cloudpickle
import numpy as np
import torch
from tensordict import TensorDict


def _arrays(batch):
    inputs, objective = batch
    metadata, arrays = {}, []
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            arrays.append(("inputs", key, value.detach().numpy()))
        else:
            metadata[key] = value
    for key, value in objective.items():
        if value.device.type != "cpu":
            raise ValueError("Data preparation must return CPU objective tensors")
        arrays.append(("objective", key, value.detach().numpy()))
    descriptors, offset = [], 0
    for section, key, array in arrays:
        if array.dtype.hasobject:
            raise TypeError("Shared-memory batches require numeric arrays")
        offset = (offset + 63) // 64 * 64
        descriptors.append((section, key, array.shape, array.dtype.str, offset))
        offset += array.nbytes
    return metadata, arrays, descriptors, max(1, offset)


def _prepare_batches(connection, build):
    while True:
        job = connection.recv()
        if job is None:
            return
        index, name, capacity, context_payload = job
        if context_payload is None:
            batch = build(index)
        else:
            context = MappingProxyType(cloudpickle.loads(context_payload))
            batch = build(index, context=context)
        metadata, arrays, descriptors, required = _arrays(batch)
        if required > capacity:
            connection.send(("resize", required))
            name = connection.recv()
        memory = SharedMemory(name=name, track=False)
        try:
            for (_, _, array), (_, _, shape, dtype, offset) in zip(arrays, descriptors):
                view = np.ndarray(shape, dtype=dtype, buffer=memory.buf, offset=offset)
                np.copyto(view, array)
                del view
        finally:
            memory.close()
        connection.send(("ready", metadata, descriptors))


def _worker(connection, build_name):
    # Spawn never inherits the training process's CUDA context. This worker
    # executes CPU data preparation only; do not initialize CUDA here.
    payload = payload_view = build = None
    try:
        torch.set_num_threads(1)
        payload = SharedMemory(name=build_name, track=False)
        assert payload.buf is not None
        # Unpickling accepts a buffer directly. Avoid a payload-sized bytes
        # allocation in every worker, and keep the mapping alive in case a
        # deserializer retains views into it.
        payload_view = payload.buf.toreadonly()
        build = cloudpickle.loads(payload_view)
        connection.send(("started", perf_counter()))
        _prepare_batches(connection, build)
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        # The batch-loop frame is gone before releasing the startup mapping.
        # Collect callable cycles as well, since they can own buffer views.
        build = None
        gc.collect()
        try:
            if payload_view is not None:
                payload_view.release()
            if payload is not None:
                payload.close()
        finally:
            connection.close()


class ProcessBatches:
    """Persistent spawned workers; at most ``prefetch`` unconsumed batches."""

    def __init__(self, build, *, workers, prefetch, start=0, context_for_batch=None):
        self._startup_started = perf_counter()
        self._startup_ready_at = self._startup_started
        self._worker_startup_seconds: float | None = None
        self._worker_payload_bytes = 0
        self._context_for_batch = context_for_batch
        self._processes = []
        self._connections: list[Connection] = []
        self._starting: set[Connection] = set()
        self._active: dict[Connection, tuple[int, int]] = {}
        self._ready = {}
        self._slots: list[SharedMemory | None] = [None] * prefetch
        self._free = deque(range(prefetch))
        self._next_submit = start
        self._next_read = start
        self._closed = False
        self._payload: SharedMemory | None = None
        # cloudpickle supports notebook-local functions and format callbacks.
        # The transform is serialized once at startup. Only small scheduled
        # context snapshots are serialized per batch.
        try:
            payload = cloudpickle.dumps(build)
        except Exception as exc:
            raise TypeError("Cannot serialize the data transform for process workers") from exc
        self._worker_payload_bytes = len(payload)
        context = mp.get_context("spawn")
        try:
            self._payload = SharedMemory(create=True, size=len(payload))
            assert self._payload.buf is not None
            self._payload.buf[:] = payload
            del payload
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, self._payload.name), daemon=True)
                try:
                    process.start()
                except BaseException:
                    child.close()
                    parent.close()
                    raise
                child.close()
                self._processes.append(process)
                self._connections.append(parent)
                self._starting.add(parent)
            self._schedule()
        except BaseException:
            self.close()
            raise

    @property
    def worker_payload_bytes(self) -> int:
        """Serialized transform bytes shared with each worker in this generation.

        This is transport size, not a bound on deserialized worker memory.
        The diagnostic remains available after close.
        """
        return self._worker_payload_bytes

    @property
    def worker_startup_seconds(self) -> float | None:
        """Construction to the last worker finishing deserialization, in seconds.

        Includes serialization and process startup. None until next_batch has
        observed every startup notification; excludes delayed consumption and
        batch preparation. The last observed value remains available after close.
        """
        return self._worker_startup_seconds

    def _schedule(self):
        for connection in self._connections:
            if connection in self._active or not self._free:
                continue
            slot = self._free.popleft()
            memory = self._slots[slot]
            index = self._next_submit
            self._next_submit += 1
            self._active[connection] = (index, slot)
            context_payload = None if self._context_for_batch is None else cloudpickle.dumps(
                dict(self._context_for_batch(index)))
            connection.send((index, memory.name if memory else None,
                             memory.size if memory else 0, context_payload))

    def next_batch(self):
        if self._closed:
            raise RuntimeError("DataLoader is closed")
        try:
            while self._next_read not in self._ready:
                for ready_connection in wait(list(self._active.keys() | self._starting), timeout=0.05):
                    connection = cast(Connection, ready_connection)
                    try:
                        message = connection.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError("Data preparation worker disconnected unexpectedly") from exc
                    if message[0] == "started":
                        self._starting.remove(connection)
                        self._startup_ready_at = max(self._startup_ready_at, message[1])
                        if not self._starting:
                            self._worker_startup_seconds = self._startup_ready_at - self._startup_started
                        continue
                    if message[0] == "error":
                        raise RuntimeError("A data preparation worker raised an exception:\n" + message[1])
                    index, slot = self._active[connection]
                    if message[0] == "resize":
                        memory = self._slots[slot]
                        if memory is not None:
                            memory.close()
                            memory.unlink()
                            self._slots[slot] = None
                        # Grow geometrically; variable-size batches reuse slots.
                        capacity = max(65536, 1 << (message[1] - 1).bit_length())
                        memory = SharedMemory(create=True, size=capacity)
                        self._slots[slot] = memory
                        connection.send(memory.name)
                    else:
                        self._ready[index] = (slot, message[1], message[2])
                        del self._active[connection]
                for process in self._processes:
                    if not process.is_alive():
                        raise RuntimeError(f"Data preparation worker exited unexpectedly ({process.exitcode})")
                self._schedule()
            slot, metadata, descriptors = self._ready.pop(self._next_read)
            memory = self._slots[slot]
            assert memory is not None
            objective = {}
            for section, key, shape, dtype, offset in descriptors:
                array = np.ndarray(shape, dtype=dtype, buffer=memory.buf, offset=offset).copy()
                if section == "inputs":
                    metadata[key] = torch.from_numpy(array)
                else:
                    objective[key] = torch.from_numpy(array)
            inputs = TensorDict(metadata, batch_size=[], device="cpu")
            batch = inputs, TensorDict(objective, batch_size=[], device="cpu")
            self._next_read += 1
            self._free.append(slot)
            self._schedule()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        # A worker can be preparing, blocked on a resize, or waiting for work.
        # Terminate all before joining, then release parent-owned allocations.
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        for connection in self._connections:
            connection.close()
        for memory in self._slots:
            if memory is not None:
                memory.close()
                memory.unlink()
        if self._payload is not None:
            self._payload.close()
            self._payload.unlink()
            self._payload = None
        self._active.clear()
        self._ready.clear()
        self._starting.clear()

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()
