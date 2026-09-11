"""Shared ``torch.profiler`` helper for ``bench_*.py``.

``--profile`` prints CPU vs device time, launch-heavy op counts, the top
aten/CUDA ops, and a ``cProfile`` host stack. ``--profile-trace DIR`` also
writes a Chrome trace per call (``chrome://tracing``). The helper warms
the callable once so the sample is the steady-state path (compiled /
CUDA-graphed), not first-call compile.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.autograd.profiler import EventList
from torch.profiler import ProfilerActivity, profile


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run torch.profiler on each measured callable after warmup",
    )
    parser.add_argument(
        "--profile-trace",
        type=Path,
        default=None,
        metavar="DIR",
        help="write Chrome traces under DIR (implies --profile)",
    )


def wants_profile(args: argparse.Namespace) -> bool:
    return bool(args.profile or args.profile_trace is not None)


def profile_call(
    fn: Callable[[], Any],
    *,
    label: str,
    steps: int,
    cuda: bool,
    trace_dir: Path | None,
) -> None:
    """Profile ``fn`` for ``steps`` already-warmed calls and print a summary."""
    if steps < 1:
        raise ValueError("profile steps must be >= 1")
    activities = [ProfilerActivity.CPU]
    if cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA profiling needs a CUDA device")
        activities.append(ProfilerActivity.CUDA)
    # One unprofiled call so compile / graph capture is not in the sample.
    fn()
    if cuda:
        torch.cuda.synchronize()
    cpu_prof = cProfile.Profile()
    cpu_prof.enable()
    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(steps):
            fn()
        # Sync once at the end so per-step cudaDeviceSynchronize is not the
        # top CPU op. Steady decode does not sync after every call.
        if cuda:
            torch.cuda.synchronize()
    cpu_prof.disable()
    _print_summary(prof, label=label, steps=steps, cuda=cuda)
    print("    cProfile (cumtime, top 12):")
    pstats.Stats(cpu_prof).sort_stats("cumtime").print_stats(12)
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{_safe_label(label)}.json"
        prof.export_chrome_trace(str(path))
        print(f"    chrome trace {path}")


def _safe_label(label: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("_")
    return text or "profile"


def _print_summary(prof: profile, *, label: str, steps: int, cuda: bool) -> None:
    averages: EventList = prof.key_averages()
    cpu_us = sum(e.self_cpu_time_total for e in averages)
    device_us = sum(e.self_device_time_total for e in averages) if cuda else 0.0
    cpu_ms = cpu_us / 1e3
    device_ms = device_us / 1e3
    ratio = (cpu_ms / device_ms) if device_ms > 0 else float("inf")
    gpu_pct = (100.0 * device_ms / cpu_ms) if cpu_ms > 0 and cuda else 0.0
    print(
        f"  profile {label} | {steps} steps | "
        f"CPU {cpu_ms:8.1f} ms ({cpu_ms / steps:7.2f} ms/step) | "
        + (
            f"CUDA {device_ms:8.1f} ms ({device_ms / steps:7.2f} ms/step) | "
            f"CPU/CUDA {ratio:5.2f} | GPU~{gpu_pct:4.0f}%"
            if cuda
            else "CPU only"
        )
    )
    counts = _op_counts(averages, steps)
    if counts:
        print("    " + "  ".join(f"{name} {n:.1f}/step" for name, n in counts))
    if cuda or cpu_ms >= 1:
        print(averages.table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=48))
        if cuda:
            print(averages.table(sort_by="self_device_time_total", row_limit=8, max_name_column_width=48))
    else:
        print("    (no aten ops; host time is in cProfile below)")


def _op_counts(averages: EventList, steps: int) -> list[tuple[str, float]]:
    """Launch-heavy ops, averaged per profiled step."""
    groups: list[tuple[str, tuple[str, ...]]] = [
        ("aten::mm", ("aten::mm",)),
        ("aten::addmm", ("aten::addmm",)),
        ("aten::bmm", ("aten::bmm",)),
        ("flex_attention", ("flex_attention", "flex_attention_forward")),
        ("sdpa", ("aten::scaled_dot_product_attention", "aten::_scaled_dot_product")),
        ("cudaGraphLaunch", ("cudaGraphLaunch", "CudaGraph", "cuda_graph")),
    ]
    out: list[tuple[str, float]] = []
    for label, needles in groups:
        n = 0
        for event in averages:
            key = event.key
            if any(needle in key for needle in needles):
                n += event.count
        if n:
            out.append((label, n / steps))
    return out
