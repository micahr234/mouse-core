"""Reproducible, CPU-only spawned replay-memory benchmark.

Run from mouse-core with its environment, for example:
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python \
    bench/bench_replay_memory.py --output /tmp/mouse-core-memory.json

The reference bypasses automatic DatasetSnapshot storage while retaining the
current ProcessBatches transport. It therefore isolates the snapshot benefit;
it does not reproduce the removed per-worker bytes(payload.buf) allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import gc
import json
import os
from pathlib import Path
import platform
import statistics
from tempfile import gettempdir
import time

import datasets
from datasets import Dataset
import numpy as np

from mouse_core.data import DataLoader, Datastore, StepTokens
from mouse_core.data.dataloader import _fetch_one_batch
from mouse_core.data.process_batches import ProcessBatches


MIB = 1024 * 1024


def transform(events):
    return [StepTokens(
        token_ids=np.array([int(event["payload"]["id"])], dtype=np.int64),
        numeric_values=np.zeros(1), numeric_mask=np.zeros(1, dtype=bool),
        head_mask=np.ones(1, dtype=bool),
        objective_fields={"label": np.array([int(event["payload"]["id"])], dtype=np.int64)},
    ) for event in events]


def source_dataset(start, count, blob_size):
    blob = "x" * blob_size
    return Dataset.from_dict({
        "episode_index": list(range(start, start + count)),
        "step_index": [0] * count,
        "provider": ["sensor"] * count,
        "payload": [json.dumps({"id": i, "blob": blob}, separators=(",", ":"))
                    for i in range(start, start + count)],
    })


def memory(pid):
    fields = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]:
        key, raw = line.split(":", 1)
        fields[key] = int(raw.strip().split()[0]) * 1024
    return {
        "pid": pid,
        "rss_bytes": fields["Rss"],
        "pss_bytes": fields["Pss"],
        "uss_bytes": fields.get("Private_Clean", 0) + fields.get("Private_Dirty", 0),
        "anonymous_bytes": fields["Anonymous"],
        "swap_bytes": fields.get("Swap", 0),
    }


def temporary_storage():
    path = Path(gettempdir()).resolve()
    mounts = []
    for line in Path("/proc/mounts").read_text().splitlines():
        _, mount, kind, *_ = line.split()
        if path.is_relative_to(mount):
            mounts.append((len(mount), mount, kind))
    _, mount, kind = max(mounts)
    return {"directory": str(path), "mount": mount, "filesystem": kind}


def summarize_memory(pool):
    samples = {process.pid: [] for process in pool._processes}
    for sample in range(3):
        for pid in samples:
            samples[pid].append(memory(pid))
        if sample != 2:
            time.sleep(0.05)
    workers = [{key: int(statistics.median(item[key] for item in values))
                for key in values[0]} for values in samples.values()]
    return {"workers": workers, "total": {
        key: sum(worker[key] for worker in workers) for key in workers[0] if key != "pid"}}


def owned_resources(loader=None, pool=None):
    pool = loader._pool if loader is not None else pool
    assert pool is not None
    return {
        "worker_pids": [process.pid for process in pool._processes],
        "shared_memory": [pool._payload.name, *[slot.name for slot in pool._slots if slot is not None]],
        "snapshot_directories": ([] if loader is None else
                                 [str(item.path) for item in loader._snapshots if item.path is not None]),
    }


def assert_closed(resources, *, retained_paths=()):
    status = {
        "workers_gone": all(not Path(f"/proc/{pid}").exists() for pid in resources["worker_pids"]),
        "shared_memory_unlinked": all(not Path("/dev/shm", name).exists()
                                      for name in resources["shared_memory"]),
        "obsolete_snapshot_directories_removed": all(
            not Path(path).exists() for path in resources["snapshot_directories"]
            if path not in retained_paths),
    }
    assert all(status.values()), status
    return status


def warm_and_measure(loader, *, stage, source_bytes, elapsed):
    first = None
    for _ in range(32):
        inputs, _ = loader.next_batch()
        if first is None:
            first = inputs["token_ids"].clone()
    pool = loader._pool
    assert pool is not None
    assert pool.worker_startup_seconds is not None
    return {
        "mode": "automatic_snapshot", "stage": stage,
        "source_arrow_bytes": source_bytes,
        "construct_or_refresh_seconds": elapsed,
        "diagnostics": loader.diagnostics,
        "parent_memory": memory(os.getpid()),
        **summarize_memory(pool),
    }, first


def run(output):
    datasets.disable_progress_bars()
    results = {
        "platform": platform.platform(), "python": platform.python_version(),
        "temporary_storage": temporary_storage(),
        "cpu_only": True, "workers": 2, "prefetch": 2, "warmup_batches": 32,
        "events_per_batch": 8, "blob_bytes_per_event": 2048, "seed": 17,
        "stages": [], "cleanup": [],
        "limitations": [
            "Synthetic JSON events, fixed small transform and batch size; not a training or GPU benchmark.",
            "USS/PSS are Linux smaps_rollup snapshots after warmup, not transient peak memory.",
            "Both modes use the new direct-buffer worker transport; the reference isolates automatic mmap snapshot protection.",
            "Per-sample indices and eligibility metadata legitimately grow with row count and remain in the worker payload.",
            "The parent retains the in-memory source; mmap snapshots prevent worker copies, not parent replay growth.",
            "On tmpfs, snapshot files consume RAM once; select a disk-backed TMPDIR to move snapshot storage off RAM.",
            "No claim that arbitrary user transform closures or objects are bounded by snapshot protection.",
        ],
    }
    store = Datastore(name="memory-benchmark")
    store.from_dataset(ds=source_dataset(0, 4096, 2048))
    start = time.perf_counter()
    loader = DataLoader(stores=store, transform=transform, events_budget=8,
                        sample_field="episode_index", prefetch=2, num_workers=2, seed=17)
    references = []
    retained_batch = None
    try:
        record, retained_batch = warm_and_measure(loader, stage="8_MiB", source_bytes=store.to_dataset().data.nbytes,
                                                  elapsed=time.perf_counter() - start)
        results["stages"].append(record)
        references.append(("8_MiB", replace(loader._snapshot_config(), datasets=(store.to_dataset(),))))
        prior = owned_resources(loader=loader)
        store.from_dataset(ds=source_dataset(4096, 28672, 2048))
        start = time.perf_counter()
        loader.refresh()
        elapsed = time.perf_counter() - start
        current = owned_resources(loader=loader)
        results["cleanup"].append({"after": "growth_refresh", **assert_closed(
            prior, retained_paths=current["snapshot_directories"])})
        record, _ = warm_and_measure(loader, stage="64_MiB", source_bytes=store.to_dataset().data.nbytes,
                                    elapsed=elapsed)
        results["stages"].append(record)
        references.append(("64_MiB", replace(loader._snapshot_config(), datasets=(store.to_dataset(),))))
        for refresh_index in range(1, 4):
            prior = owned_resources(loader=loader)
            start = time.perf_counter()
            loader.refresh()
            elapsed = time.perf_counter() - start
            current = owned_resources(loader=loader)
            results["cleanup"].append({"after": f"repeat_refresh_{refresh_index}", **assert_closed(
                prior, retained_paths=current["snapshot_directories"])})
            assert prior["snapshot_directories"] == current["snapshot_directories"]
            record, _ = warm_and_measure(loader, stage=f"64_MiB_refresh_{refresh_index}",
                                        source_bytes=store.to_dataset().data.nbytes, elapsed=elapsed)
            results["stages"].append(record)
        prior = owned_resources(loader=loader)
    finally:
        loader.close()
    results["cleanup"].append({"after": "automatic_loader_close", **assert_closed(prior)})
    assert retained_batch is not None and retained_batch.numel() == 8

    for label, cfg in references:
        gc.collect()
        start = time.perf_counter()
        pool = ProcessBatches(partial(_fetch_one_batch, cfg, 17, transform=transform), workers=2, prefetch=2)
        elapsed = time.perf_counter() - start
        try:
            for _ in range(32):
                pool.next_batch()
            assert pool.worker_startup_seconds is not None
            results["stages"].append({
                "mode": "in_memory_transport_reference", "stage": label,
                "source_arrow_bytes": cfg.datasets[0].data.nbytes,
                "construct_or_refresh_seconds": elapsed,
                "diagnostics": {"worker_payload_bytes": pool.worker_payload_bytes,
                                "worker_startup_seconds": pool.worker_startup_seconds},
                "parent_memory": memory(os.getpid()), **summarize_memory(pool),
            })
            prior = owned_resources(pool=pool)
        finally:
            pool.close()
        results["cleanup"].append({"after": f"reference_{label}_close", **assert_closed(prior)})
    output.write_text(json.dumps(results, indent=2) + "\n")
    for item in results["stages"]:
        print(json.dumps({"mode": item["mode"], "stage": item["stage"],
                          "payload_MiB": item["diagnostics"]["worker_payload_bytes"] / MIB,
                          "worker_total_USS_MiB": item["total"]["uss_bytes"] / MIB,
                          "worker_total_PSS_MiB": item["total"]["pss_bytes"] / MIB,
                          "startup_seconds": item["diagnostics"]["worker_startup_seconds"]}), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
