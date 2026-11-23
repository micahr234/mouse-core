"""Bounded process workers with reusable shared-memory result slots.

Task indices and small contexts cross the work queue; errors use worker pipes.
Workers fill shared-memory tensors and metadata independently of the consumer.
The parent owns slot lifetimes and copies results before recycling a slot, so a
returned batch remains valid after the next read, refresh or close.
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import pickle
import traceback
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter
from types import MappingProxyType

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


def _prepare_batches(build, jobs, names, ready, completed):
    while True:
        job = jobs.get()
        if job is None:
            return
        index, slot, context_payload = job
        if context_payload is None:
            batch = build(index)
        else:
            context = MappingProxyType(cloudpickle.loads(context_payload))
            batch = build(index, context=context)
        metadata, arrays, descriptors, required = _arrays(batch)
        header = pickle.dumps((metadata, descriptors), protocol=pickle.HIGHEST_PROTOCOL)
        base = (8 + len(header) + 63) // 64 * 64
        memory = SharedMemory(name=names[slot], track=False)
        try:
            if base + required > memory.size:
                # Only this worker owns the slot until ready is set. The parent
                # has no open mapping while work is scheduled for this slot.
                memory.unlink()
                memory.close()
                capacity = max(65536, 1 << (base + required - 1).bit_length())
                memory = SharedMemory(name=names[slot], create=True, size=capacity, track=False)
            assert memory.buf is not None
            memory.buf[:8] = len(header).to_bytes(8, "little")
            memory.buf[8:8 + len(header)] = header
            for (_, _, array), (_, _, shape, dtype, offset) in zip(arrays, descriptors):
                view = np.ndarray(shape, dtype=dtype, buffer=memory.buf, offset=base + offset)
                np.copyto(view, array)
                del view
        finally:
            memory.close()
        ready[slot].set()
        completed.release()


def _read_slot(name):
    memory = SharedMemory(name=name, track=False)
    try:
        assert memory.buf is not None
        size = int.from_bytes(memory.buf[:8], "little")
        metadata, descriptors = pickle.loads(bytes(memory.buf[8:8 + size]))
        base = (8 + size + 63) // 64 * 64
        objective = {}
        for section, key, shape, dtype, offset in descriptors:
            array = np.ndarray(shape, dtype=dtype, buffer=memory.buf, offset=base + offset).copy()
            if section == "inputs":
                metadata[key] = torch.from_numpy(array)
            else:
                objective[key] = torch.from_numpy(array)
        return (TensorDict(metadata, batch_size=[], device="cpu"),
                TensorDict(objective, batch_size=[], device="cpu"))
    finally:
        memory.close()


def _worker(connection, build_name, jobs, names, ready, completed):
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
        _prepare_batches(build, jobs, names, ready, completed)
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
    """Spawned workers fill bounded slots, with optional ordered delivery."""

    def __init__(self, build, *, workers, prefetch, start=0, context_for_batch=None, in_order=True):
        self._closed = False
        self._payload: SharedMemory | None = None
        self._jobs = None
        self._processes = []
        self._connections: list[Connection] = []
        self._starting: set[Connection] = set()
        self._names: list[str] = []
        self._ready = []
        self._pending = {}
        self._in_order = in_order
        self._startup_started = perf_counter()
        self._startup_ready_at = self._startup_started
        self._worker_startup_seconds: float | None = None
        self._worker_payload_bytes = 0
        self._context_for_batch = context_for_batch
        self._next_submit = start
        if not isinstance(in_order, bool):
            raise TypeError("in_order must be a bool")
        if workers < 1 or prefetch < 1:
            raise ValueError("workers and prefetch must be positive")
        try:
            payload = cloudpickle.dumps(build)
        except Exception as exc:
            raise TypeError("Cannot serialize the data transform for process workers") from exc
        self._worker_payload_bytes = len(payload)
        context = mp.get_context("spawn")
        try:
            # SimpleQueue uses pipes and process locks, without a feeder thread.
            self._jobs = context.SimpleQueue()
            self._completed = context.Semaphore(0)
            self._payload = SharedMemory(create=True, size=len(payload))
            assert self._payload.buf is not None
            self._payload.buf[:] = payload
            del payload
            for _ in range(prefetch):
                memory = SharedMemory(create=True, size=65536)
                self._names.append(memory.name)
                memory.close()
                self._ready.append(context.Event())
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(
                    child, self._payload.name, self._jobs, self._names, self._ready, self._completed), daemon=True)
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
            for slot in range(prefetch):
                self._schedule(slot)
        except BaseException:
            self.close()
            raise

    @property
    def worker_payload_bytes(self) -> int:
        """Serialized transform bytes shared with every worker; retained after close."""
        return self._worker_payload_bytes

    @property
    def worker_startup_seconds(self) -> float | None:
        """Construction to last worker startup, excluding delayed consumption.

        None until next_batch observes every worker's timestamp. Retained after close.
        """
        return self._worker_startup_seconds

    @property
    def next_submit_index(self):
        """First unused batch ID, including jobs not yet consumed."""
        return self._next_submit

    def _schedule(self, slot):
        index = self._next_submit
        self._ready[slot].clear()
        context_payload = None if self._context_for_batch is None else cloudpickle.dumps(
            dict(self._context_for_batch(index)))
        assert self._jobs is not None
        self._jobs.put((index, slot, context_payload))
        self._pending[slot] = index
        self._next_submit += 1

    def _check_workers(self):
        for connection in self._connections:
            while connection.poll():
                try:
                    message = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError("Data preparation worker disconnected unexpectedly") from exc
                if message[0] == "error":
                    raise RuntimeError("A data preparation worker raised an exception:\n" + message[1])
                self._starting.remove(connection)
                self._startup_ready_at = max(self._startup_ready_at, message[1])
                if not self._starting:
                    self._worker_startup_seconds = self._startup_ready_at - self._startup_started
        for process in self._processes:
            if not process.is_alive():
                raise RuntimeError(f"Data preparation worker exited unexpectedly ({process.exitcode})")

    def next_batch(self):
        if self._closed:
            raise RuntimeError("DataLoader is closed")
        try:
            while not self._completed.acquire(timeout=0.05):
                self._check_workers()
            if self._in_order:
                slot = min(self._pending, key=self._pending.__getitem__)
                while not self._ready[slot].wait(timeout=0.05):
                    self._check_workers()
            else:
                # Events are set before completion is signaled. Prefer the oldest
                # ready batch without waiting on any unfinished earlier batch.
                slot = min((slot for slot in self._pending if self._ready[slot].is_set()),
                           key=self._pending.__getitem__)
            self._check_workers()
            batch = _read_slot(self._names[slot])
            self._schedule(slot)
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Stop every writer before releasing any slot, including unread results.
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
        if self._jobs is not None:
            self._jobs.close()
        for name in self._names:
            try:
                memory = SharedMemory(name=name)
            except FileNotFoundError:
                # A worker may have been stopped between unlink and grow.
                memory = SharedMemory(name=name, create=True, size=1)
            memory.close()
            memory.unlink()
        if self._payload is not None:
            self._payload.close()
            self._payload.unlink()
            self._payload = None
        self._starting.clear()

    def __del__(self):
        if hasattr(self, "_closed"):
            self.close()
