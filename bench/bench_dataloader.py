"""Benchmark ``DataLoader.next_batch`` on a synthetic FrozenLake-shaped store.

Builds an in-memory ``Datastore`` (no Hub) with the same augmenter +
``NumericTokenizer`` pipeline as ``examples/02_train_offline_dqn.ipynb``, then
times ``next_batch()`` for every ``--workloads`` × ``--workers`` pair. Reports
average and max consumer wait (how long ``next_batch`` blocks), first-call
(queue fill) time, steps/second, and tokens/second. ``--profile`` adds a
``torch.profiler`` CPU breakdown after warmup. ``num_workers>0`` needs
free-threaded CPython with the GIL off.

    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workers 0 1 4 8
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workloads notebook long
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workloads notebook --workers 0 --profile
"""

from __future__ import annotations

import argparse
import statistics
import sys
import sysconfig
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from mouse_core.data import Augmenter, DataLoader, Datastore, NumericTokenizer, compose
from mouse_core.data.token_batch import TokenBatch

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from bench_profile import add_profile_args, profile_call, wants_profile


_WORKLOADS: dict[str, tuple[int, int]] = {
    "short": (64, 4),
    "notebook": (512, 4),
    "wide": (512, 16),
    "long": (2048, 4),
}

_MAX_ACTIONS = 4
_MAX_OBS = 64
_STEPS_PER_EPISODE = 30
_EPISODES_PER_TASK = 20


def _free_threading_ok() -> bool:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return not (callable(is_gil_enabled) and is_gil_enabled())


def _synthetic_rows(n: int, *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    task = 0
    episode = 0
    ep_step = 0
    for _ in range(n):
        episode_done = 2 if ep_step + 1 >= _STEPS_PER_EPISODE else 0
        task_done = 2 if episode_done and episode + 1 >= _EPISODES_PER_TASK else 0
        rows.append(
            {
                "action": int(rng.integers(0, _MAX_ACTIONS)),
                "observation": int(rng.integers(0, _MAX_OBS)),
                "reward": float(rng.random()),
                "episode_done": episode_done,
                "task_done": task_done,
                "task_index": task,
                "info_q_star": rng.random(_MAX_ACTIONS, dtype=np.float32).tolist(),
            }
        )
        ep_step += 1
        if episode_done:
            ep_step = 0
            episode += 1
        if task_done:
            episode = 0
            task += 1
    return rows


def _stores(*, n_stores: int, steps: int) -> list[Datastore]:
    stores: list[Datastore] = []
    for i in range(n_stores):
        store = Datastore(name=f"synth_{i}")
        store.from_dataset(Dataset.from_list(_synthetic_rows(steps, seed=i)))
        stores.append(store)
    return stores


def _train_transform() -> Any:
    augmenter = Augmenter(
        seed_field="task_index",
        seed=0,
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "input_vector_field": "info_q_star",
                "vocab_size": _MAX_ACTIONS,
                "permute": True,
            },
            {
                "type": "discrete",
                "input_field": "observation",
                "vocab_size": _MAX_OBS,
                "permute": True,
            },
        ],
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {"type": "discrete", "input_field": "observation"},
            {"type": "fourier", "input_field": "reward"},
            {"type": "discrete", "input_field": "episode_done"},
            {"type": "learnable", "output_field": "value", "tokens": 1, "head_output": True},
        ],
        objective_fields=[
            {"input_field": "action"},
            {"input_field": "reward"},
            {"input_field": "episode_done"},
            {"input_field": "task_done"},
        ],
        grouping_field="task_index",
    )
    return compose(augmenter, tokenizer)


def _timed(fn: Callable[[], Any], iters: int) -> tuple[float, float, float]:
    """(first-call ms, mean wait ms, max wait ms)."""
    t0 = time.perf_counter()
    fn()
    first = (time.perf_counter() - t0) * 1e3
    fn()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return first, statistics.mean(times), max(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", nargs="+", default=["short", "notebook", "wide", "long"])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 4])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--stores", type=int, default=4)
    parser.add_argument("--store-steps", type=int, default=8192)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    add_profile_args(parser)
    args = parser.parse_args()

    unknown = [w for w in args.workloads if w not in _WORKLOADS]
    if unknown:
        raise SystemExit(f"unknown workload(s) {unknown}; choose from {sorted(_WORKLOADS)}")
    if any(w < 0 for w in args.workers):
        raise SystemExit("num_workers must be >= 0")
    if any(w > 0 for w in args.workers) and not _free_threading_ok():
        raise SystemExit(
            "num_workers>0 needs free-threaded CPython with the GIL off "
            "(PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py)"
        )

    stores = _stores(n_stores=args.stores, steps=args.store_steps)
    transform = _train_transform()
    print(
        f"stores={args.stores}×{args.store_steps} | augmenter+NumericTokenizer | "
        f"prefetch={args.prefetch} seed={args.seed} | gil_off={_free_threading_ok()}"
    )

    for wname in args.workloads:
        seq_len, batch_size = _WORKLOADS[wname]
        print(f"\n### workload: {wname} | S={seq_len} B={batch_size}")
        for n_workers in args.workers:
            loader = DataLoader(
                stores=stores,
                sequence_length=seq_len,
                batch_size=batch_size,
                transform=transform,
                prefetch=args.prefetch,
                num_workers=n_workers,
                seed=args.seed,
            )
            sample: TokenBatch | None = None

            def next_batch() -> None:
                nonlocal sample
                sample, _ = loader.next_batch()

            first, avg_wait, max_wait = _timed(next_batch, args.iters)
            assert sample is not None
            steps = sample.N
            tokens = sample.L
            print(
                f"  workers={n_workers:<2d} | wait avg {avg_wait:7.2f} ms  max {max_wait:7.2f} ms | "
                f"{steps / avg_wait * 1e3:>8,.0f} step/s | {tokens / avg_wait * 1e3:>8,.0f} tok/s | "
                f"N={steps} L={tokens} | first {first:.0f} ms"
            )
            if wants_profile(args):
                profile_call(
                    next_batch,
                    label=f"{wname} workers={n_workers} next_batch",
                    steps=args.iters,
                    cuda=False,
                    trace_dir=args.profile_trace,
                )
            loader.close()


if __name__ == "__main__":
    main()
