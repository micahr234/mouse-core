"""Benchmark ``DataLoader.next_batch`` on a synthetic FrozenLake-shaped store.

Builds an in-memory ``Datastore`` (no Hub) with the same augmenter +
``Tokenizer`` pipeline as ``examples/02_train_offline_dqn.ipynb``, then
times ``next_batch()`` for every ``--workloads`` × ``--workers`` pair. Reports
average and max consumer wait (how long ``next_batch`` blocks), loader setup,
first-call (queue fill) time, steps/second, and tokens/second. ``--profile`` adds a
``torch.profiler`` CPU breakdown after warmup. ``num_workers>0`` needs
free-threaded CPython with the GIL off. ``--profile-stages`` profiles the
synchronous loading, augmentation, tokenization, and packing stages without
``torch.profiler``. Use ``--local-files-only`` for a cached tokenizer or
``--tokenizer /path/to/tokenizer`` for a local backend.

    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workers 0 1 4 8
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workloads notebook long
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workloads notebook --workers 0 --profile
    PYTHON_GIL=0 .venv/bin/python bench/bench_dataloader.py --workloads notebook --workers 0 --local-files-only --profile-stages
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import statistics
import sys
import sysconfig
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


from mouse_core.data import Augmenter, DataLoader, Datastore, EventLog, Tokenizer, compose
from mouse_core.data.token_batch import TokenBatch


def on_group_start(ctx):
    return bool(ctx["group_start"])

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
_STEPS_PER_EPISODE = 8
_EPISODES_PER_TASK = 10  # 80 observations per task; fits the shortest event budget
_group_prefix = (
    "Your job is to predict the future sum of rewards in FrozenLake. "
    "Navigate a grid; reach the goal for reward; a hole ends the "
    f"episode with none. Each episode lasts at most {_STEPS_PER_EPISODE - 1} steps. "
    f"You have {_EPISODES_PER_TASK} episodes to solve the task. The "
    "grid is permuted, so squares are not in order; action ids may be "
    "remapped.\n"
    "Strategy: explore; keep a mental map of what has and has not been "
    "explored; avoid holes you have already fallen in; once you have a "
    "path to the goal, repeat it.\n"
    "At each action-request newline, predict the value of all actions. Format:\n"
    "action,observation[,r=reward][,terminated][,truncated].\n"
)


def _free_threading_ok() -> bool:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return not (callable(is_gil_enabled) and is_gil_enabled())


def _stores(*, n_stores: int, steps: int) -> list[Datastore]:
    stores = []
    for stream in range(n_stores):
        rng = np.random.default_rng(stream)
        log = EventLog(name=f"synth_{stream}")
        for index in range(steps):
            task, within_task = divmod(index, _STEPS_PER_EPISODE * _EPISODES_PER_TASK)
            repeat, repeat_step = divmod(within_task, _STEPS_PER_EPISODE)
            repeat_ended = repeat_step == _STEPS_PER_EPISODE - 1
            task_ended = repeat_ended and repeat == _EPISODES_PER_TASK - 1
            if within_task == 0:
                log.event(provider="coordinator", episode_index=task, step_index=within_task, payload={
                    'episode_start': True,
                })
            if repeat_step > 0:
                log.event(provider='agent', episode_index=task, step_index=within_task, payload={
                    'action': int(rng.integers(0, _MAX_ACTIONS)),
                })
            reward = 0.0 if repeat_step == 0 else float(rng.random())
            log.event(provider='environment', episode_index=task, step_index=within_task, payload={
                **({'action_requested': True} if not repeat_ended else {}),
                'observation': int(rng.integers(0, _MAX_OBS)),
                'task_index': task,
                **({'inner_truncated': True} if repeat_ended else {}),
                **({'q_star': rng.random(_MAX_ACTIONS, dtype=np.float32).tolist()} if not repeat_ended else {}),
                **({'reward': reward} if reward != 0 else {}),
                **({'truncated': True} if task_ended else {}),
            })

        stores.append(log.to_store())
    return stores


def _train_transform(*, tokenizer_source: str = "Qwen/Qwen3-0.6B", local_files_only: bool = False) -> Any:
    augmenter = Augmenter(
        seed_field="episode_index",
        seed=0,
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "input_vector_field": "q_star",
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
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "format": _group_prefix,
                "when": on_group_start,
            },
            {"type": "text", "input_field": "action", "lookup": ["a", "b", "c", "d"], "format": "{field},"},
            {"type": "text", "input_field": "observation", "format": "{field},"},
            {
                "type": "text",
                "input_field": "reward",
                "format": "r={field:g},",
            },
            {
                "type": "text",
                "format": ",terminated",
                "input_field": "inner_terminated",
            },
            {
                "type": "text",
                "format": ",truncated",
                "input_field": "inner_truncated",
            },
            {
                "type": "token",
                "input_field": "action_requested",
                "format": "\n",
                "head_output": True,
            },
        ],
        objective_fields=[
            {"input_field": "action", "missing_value": None},
            {"input_field": "reward", "missing_value": 0.0},
            {"input_field": "terminated", "missing_value": False},
            {"input_field": "truncated", "missing_value": False},
        ],
        tokenizers={"text": tokenizer_source},
        hub_kwargs={"local_files_only": local_files_only},
    )
    return compose(stages=(augmenter.events, tokenizer.encode))


def _timed(fn: Callable[[], TokenBatch], iters: int) -> tuple[float, float, float, float, float]:
    """First/mean/max wait in ms, and mean steps/tokens per measured batch."""
    t0 = time.perf_counter()
    fn()
    first = (time.perf_counter() - t0) * 1e3
    fn()
    times: list[float] = []
    steps = tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        sample = fn()
        times.append((time.perf_counter() - t0) * 1e3)
        steps += sample.N
        tokens += sample.L
    return first, statistics.mean(times), max(times), steps / iters, tokens / iters


def _profile_stages(fn: Callable[[], Any], iters: int) -> None:
    """Profile synchronous stages separately; backend time is inside tokenize."""
    profile = cProfile.Profile()
    started = time.perf_counter()
    with profile:
        for _ in range(iters):
            fn()
    elapsed = time.perf_counter() - started
    stats = pstats.Stats(profile).stats
    stages = (
        ("choose sample", "dataloader.py", "_fetch_sequence"),
        ("load rows", "dataloader.py", "_rows_from_slice"),
        ("augment", "augmenter.py", "events"),
        ("tokenize", "tokenizer.py", "encode"),
        ("pack", "token_batch.py", "pack_token_batch"),
        ("text backend (inside tokenize)", "tokenizer.py", "_tokenize_ids"),
    )
    print(f"  cProfile stages | {iters} batches | {elapsed / iters * 1e3:.2f} ms/batch (includes profiler overhead)")
    for label, filename, name in stages:
        matches = [value for (path, _, function), value in stats.items()
                   if Path(path).name == filename and function == name]
        calls = sum(value[1] for value in matches)
        cumulative = sum(value[3] for value in matches)
        print(f"    {label:<31s} {cumulative / iters * 1e3:9.2f} ms/batch  {calls / iters:8.1f} calls/batch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", nargs="+", default=["short", "notebook", "wide", "long"])
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 4])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--stores", type=int, default=4)
    parser.add_argument("--store-steps", type=int, default=8192)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B", help="pretrained tokenizer name or local directory")
    parser.add_argument("--local-files-only", action="store_true", help="load the tokenizer without Hub requests")
    parser.add_argument("--profile-stages", action="store_true", help="profile synchronous pipeline stages (workers=0)")
    add_profile_args(parser)
    args = parser.parse_args()

    for name in ("iters", "stores", "store_steps", "prefetch"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")

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
    transform = _train_transform(tokenizer_source=args.tokenizer, local_files_only=args.local_files_only)
    print(
        f"stores={args.stores}×{args.store_steps} | augmenter+Tokenizer | "
        f"prefetch={args.prefetch} seed={args.seed} | tokenizer={args.tokenizer} | gil_off={_free_threading_ok()}"
    )

    for wname in args.workloads:
        seq_len, batch_size = _WORKLOADS[wname]
        print(f"\n### workload: {wname} | S={seq_len} B={batch_size}")
        for n_workers in args.workers:
            started = time.perf_counter()
            loader = DataLoader(
                stores=stores,
                events_budget=3 * seq_len,
                batch_size=batch_size,
                sample_field="episode_index",
                transform=transform,
                prefetch=args.prefetch,
                num_workers=n_workers,
                seed=args.seed,
            )
            setup_ms = (time.perf_counter() - started) * 1e3

            def next_batch() -> TokenBatch:
                sample, _, _sid = loader.next_batch()
                return sample

            first, avg_wait, max_wait, steps, tokens = _timed(next_batch, args.iters)
            print(
                f"  workers={n_workers:<2d} | wait avg {avg_wait:7.2f} ms  max {max_wait:7.2f} ms | "
                f"{steps / avg_wait * 1e3:>8,.0f} step/s | {tokens / avg_wait * 1e3:>8,.0f} tok/s | "
                f"mean N={steps:g} L={tokens:g} | setup {setup_ms:.0f} ms first {first:.0f} ms"
            )
            if args.profile_stages:
                if n_workers == 0:
                    _profile_stages(next_batch, args.iters)
                else:
                    print("  cProfile stages skipped: worker threads are not captured; use --workers 0")
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
