"""CPU-only benchmark of the production shared-memory DataLoader.

Use --workers 0 4 to compare serial preparation with four persistent processes.
The historical standard.json/free_threaded.json results predate the single-budget
API; they used the prototype transport described by their recorded settings.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from datasets import load_from_disk

from bench_dataloader import _stores, _train_transform
from mouse_core.data import DataLoader, Datastore


class DiskStore(Datastore):
    def __init__(self, path):
        super().__init__(name=str(path))
        self.dataset = load_from_disk(str(path))

    def to_dataset(self):
        return self.dataset


def fingerprint(batch):
    inputs, data = batch
    digest = hashlib.sha256()
    for field in fields(inputs):
        value = getattr(inputs, field.name)
        digest.update(field.name.encode())
        digest.update(value.tobytes() if isinstance(value, np.ndarray) else repr(value).encode())
    for name, value in sorted(data.items()):
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--token-budgets", type=int, nargs="+", default=[10000, 52000])
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for this CPU-only benchmark")
    if min(args.iters, args.repeats, *args.token_budgets) < 1 or min(args.workers) < 0:
        parser.error("Budgets, iterations and repeats must be positive; workers nonnegative")
    torch.set_num_threads(1)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if not (args.data_dir / "store0").exists():
        for i, store in enumerate(_stores(n_stores=2, steps=2560)):
            store.to_dataset().save_to_disk(str(args.data_dir / f"store{i}"))
    stores = [DiskStore(args.data_dir / f"store{i}") for i in range(2)]
    transform = _train_transform(local_files_only=True)
    results = {"python": sys.version, "torch": torch.__version__, "results": []}
    for budget in args.token_budgets:
        options = dict(stores=stores, transform=transform, token_budget=budget,
                       sample_field="episode_index", prefetch=4, seed=0)
        with DataLoader(**options, num_workers=0) as reference:
            expected = [fingerprint(reference.next_batch()) for _ in range(4)]
        for repeat in range(args.repeats):
            for workers in args.workers:
                started = time.perf_counter()
                with DataLoader(**options, num_workers=workers) as loader:
                    for index in range(4):
                        batch = loader.next_batch()
                        if index == 0:
                            first_ms = (time.perf_counter() - started) * 1000
                        assert fingerprint(batch) == expected[index]
                    started, tokens = time.perf_counter(), 0
                    waits = []
                    for _ in range(args.iters):
                        before = time.perf_counter()
                        inputs, _ = loader.next_batch()
                        waits.append((time.perf_counter() - before) * 1000)
                        tokens += inputs["token_ids"].shape[0]
                    elapsed = time.perf_counter() - started
                result = dict(token_budget=budget, workers=workers, repeat=repeat,
                              first_ms=first_ms, ms_per_batch=1000 * elapsed / args.iters,
                              tokens_per_second=tokens / elapsed,
                              mean_tokens=tokens / args.iters, verified_batches=4,
                              median_wait_ms=statistics.median(waits))
                results["results"].append(result)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
