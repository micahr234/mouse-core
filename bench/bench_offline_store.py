"""Time persisted-event import/snapshot and compare content and seeded batches.

Run the same command before and after a Datastore change with separate --output
paths. Import and snapshot timings exclude file reads, hashing, and loader setup.
Loader setup uses locally defined header ranges and episode eligibility. The
loader owns structural validation and caches unchanged indexes. Local parquet files only; no model or GPU is used.

PYTHONPATH=src .venv/bin/python bench/bench_offline_store.py SHARD.parquet \
    --repeats 3 --workers 0 4 --output bench/results/offline_store.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import torch
from datasets import Dataset
from mouse_core.data import Datastore, DataLoader, Tokenizer


def digest_dataset(dataset):
    digest = hashlib.sha256()
    for batch in dataset.iter(batch_size=8192):
        digest.update(json.dumps(batch, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


def digest_batch(batch):
    digest = hashlib.sha256()
    for section, values in zip(("inputs", "objective"), batch):
        for key in sorted(values.keys()):
            value = values[key]
            digest.update(f"{section}/{key}".encode())
            if torch.is_tensor(value):
                value = value.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.numpy().tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.batches < 1 or any(n < 0 for n in args.workers):
        parser.error("repeats/batches must be positive; workers must be nonnegative")
    torch.set_num_threads(2)
    started = time.perf_counter()
    dataset = Dataset.from_parquet([str(p) for p in args.parquet], cache_dir="/tmp/mouse-core-offline-bench-cache")
    report: dict[str, Any] = dict(python=platform.python_version(), rows=len(dataset),
                  parquet_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.parquet},
                  source_sha256=hashlib.sha256(Path(inspect.getfile(Datastore)).read_bytes()).hexdigest(),
                  read_seconds=time.perf_counter()-started, trials=[], loaders=[])
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    store: Datastore | None = None
    snapshot: Dataset | None = None
    for repeat in range(args.repeats):
        gc.collect()
        started = time.perf_counter()
        store = Datastore(name="offline_benchmark")
        store.from_dataset(ds=dataset)
        imported = time.perf_counter()
        snapshot = store.to_dataset()
        finished = time.perf_counter()
        trial = dict(import_seconds=imported-started, snapshot_seconds=finished-imported,
                     combined_seconds=finished-started)
        report["trials"].append(trial)
        print(json.dumps(dict(repeat=repeat, **trial)), flush=True)
        if repeat + 1 < args.repeats:
            store = None
            snapshot = None
        save()
    assert store is not None and snapshot is not None
    report["median_seconds"] = {key: statistics.median(t[key] for t in report["trials"])
                                for key in report["trials"][0]}
    report["dataset_sha256"] = digest_dataset(snapshot)
    positions = [0, 1, len(store)//2, -2, -1, 0]
    report["selected_events_sha256"] = hashlib.sha256(json.dumps(
        store[positions], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    tokenizer = Tokenizer(tokenizers={"text": "Qwen/Qwen3-0.6B"}, hub_kwargs={"local_files_only": True},
        input_fields=[
            dict(type="text", provider="environment", input_field="repeat_episode_index", format="episode {repeat_episode_index} x"),
            dict(type="text", provider="agent", input_field="action", format="{action}"),
            dict(type="text", provider="environment", input_field="observation", format=",{observation}"),
            dict(type="text", provider="environment", input_field="reward", format=",r={reward}"),
            dict(type="token", provider="environment", input_field="action_requested", format="\n", head_output=True),
        ], objective_fields=[
            dict(input_field="action", missing_value=None),
            dict(input_field="reward", missing_value=0.),
            dict(input_field="_terminated", output_field="terminated", missing_value=False),
            dict(input_field="_truncated", output_field="truncated", missing_value=False),
        ])
    def sample_ranges(dataset):
        """Keep all events belonging to one outer episode together."""
        start, previous = 0, None
        for index, episode in enumerate(dataset["episode_index"][:]):
            if episode is not None and episode != previous:
                if index:
                    yield start, index
                start, previous = index, episode
        if len(dataset):
            yield start, len(dataset)


    def sample_eligible(events):
        """Train only on episodes with a recorded final environment result."""
        last = events[-1]
        return last["provider"] == "environment" and (
            last["payload"].get('_terminated') is True
            or last["payload"].get('_truncated') is True
        )


    for workers in args.workers:
        started = time.perf_counter()
        loader = DataLoader(stores=[store], token_budget=10000, prefetch=4, num_workers=workers,
            sample_ranges=sample_ranges, sample_eligible=sample_eligible,
            transform=tokenizer.encode, seed=147)
        try:
            setup = time.perf_counter()-started
            ranges = loader._runs[0]
            started = time.perf_counter()
            batches = [digest_batch(loader.next_batch()) for _ in range(args.batches)]
            result = dict(workers=workers, setup_seconds=setup, batches_seconds=time.perf_counter()-started,
                          episodes=len(ranges), ranges_sha256=hashlib.sha256(ranges.tobytes()).hexdigest(),
                          batch_sha256=batches)
            report["loaders"].append(result)
            print(json.dumps(result), flush=True)
            save()
        finally:
            loader.close()
    save()


if __name__ == "__main__":
    main()
