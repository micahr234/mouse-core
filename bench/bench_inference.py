"""Benchmark cached FlexAttention decode (``ContinuousDecodeSession``) on CUDA.

Measures, per workload: prefill of a context into an empty paged KV pool,
then a decode step of ``--step`` new tokens per row against that history.
Reports synchronized median wall time (including host scheduling), transient
peak allocation above the warmed baseline, resident session allocation,
tokens/second, and first-call time. The HF paged pool is fixed at creation;
this path does not use CUDA graphs. ``--profile`` adds a ``torch.profiler`` CPU/CUDA
breakdown after warmup. Decode always uses FlexAttention.
``torch.set_float32_matmul_precision("high")`` is set so fp32 matches the
README TF32 path.

    .venv/bin/python bench/bench_inference.py --layers 8 --step 1
    .venv/bin/python bench/bench_inference.py --layers 28 --workloads mid long --step 1
    .venv/bin/python bench/bench_inference.py --no-lora --step 1
    .venv/bin/python bench/bench_inference.py --layers 8 --step 1 --profile --workloads short

Default shape is Qwen3-0.6B (hidden 1024, 16 q / 8 kv heads, head_dim 128,
FFN 3072) with fp32 LoRA rank 16 on a frozen bf16 base; ``--layers`` trims the
depth so long contexts fit. Pass ``--no-lora`` for a fully trainable fp32
backbone (decode still runs under ``no_grad``).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from transformers.generation import ContinuousBatchingConfig

from mouse_core.models.backbone import Backbone
from mouse_core.models.lora import LoRAConfig

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from bench_profile import add_profile_args, profile_call, wants_profile


def _workload(name: str) -> list[int]:
    """Per-row context lengths. One row is one sequence."""
    if name == "short":
        return [512] * 4
    if name == "mid":
        return [4096] * 8
    if name == "mid_recurring":
        return [4096] * 4
    if name == "long":
        return [16384] * 4
    if name == "long_recurring":
        return [16384] * 4
    if name == "long_manysmall":
        return [16384] * 8
    if name == "high_variance":
        return [17, 3000, 5, 900, 4000, 61, 2400, 1]
    raise ValueError(f"unknown workload {name!r}")


def _left_pad(
    lengths: list[int],
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    """Random left-padded embeds ``[B, S, D]``."""
    B = len(lengths)
    S = max(lengths)
    g = torch.Generator(device=device).manual_seed(0)
    embeds = torch.zeros(B, S, hidden, device=device, dtype=dtype)
    for b, n in enumerate(lengths):
        if n == 0:
            continue
        embeds[b, S - n :] = torch.randn(n, hidden, device=device, dtype=dtype, generator=g)
    return embeds, lengths


def _timed(fn: Callable[[], Any], iters: int) -> tuple[float, float, float, float, float]:
    """Synchronized wall ms (including host work), then peak MB above baseline."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return first, statistics.median(times), min(times), max(times), peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn", type=int, default=3072)
    parser.add_argument("--no-lora", action="store_true", help="fp32 backbone (no LoRA)")
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="new tokens every row adds on the decode step (any constant S)",
    )
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cache-capacity-tokens", type=int,
                        help="shared pool; defaults to page-rounded histories plus all timed appends")
    parser.add_argument("--max-batch-tokens", type=int, default=1024)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short", "mid", "mid_recurring", "high_variance", "long", "long_recurring", "long_manysmall"],
    )
    add_profile_args(parser)
    args = parser.parse_args()
    if args.iters < 1 or args.step < 1 or args.max_batch_tokens < 1:
        parser.error("iters, step and max-batch-tokens must be positive")
    page_size = ContinuousBatchingConfig().page_size
    capacities = {name: sum(((n + (args.iters + 2) * args.step + page_size - 1) // page_size) * page_size
                            for n in _workload(name)) for name in args.workloads}
    if args.cache_capacity_tokens is not None:
        if args.cache_capacity_tokens % page_size or args.cache_capacity_tokens < max(capacities.values()):
            parser.error(f"cache capacity must be a multiple of {page_size} and at least {max(capacities.values())}")

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(0)
    lora = None if args.no_lora else LoRAConfig(rank=16, alpha=32.0)
    dtype = torch.float32 if lora is None else torch.bfloat16
    backbone = Backbone(
        architecture="qwen3",
        dtype=dtype,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        num_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        intermediate_size=args.ffn,
        lora=lora,
    )
    backbone.to(device)
    backbone.eval()
    for name, p in backbone.named_parameters():
        if ".lora_" in name:
            p.data = p.data.float()
            if ".lora_B." in name:
                torch.nn.init.normal_(p, std=0.02)
    props = torch.cuda.get_device_properties(device)
    print(
        f"{props.name} | torch {torch.__version__} | layers={args.layers} base dtype={dtype} "
        f"lora={'off' if lora is None else f'r{lora.rank}'} params={sum(p.numel() for p in backbone.parameters())/1e6:.1f}M "
        f"| FlexAttention TF32=high step={args.step} "
        f"query_budget={args.max_batch_tokens} wall timing"
    )

    step_s = args.step
    for wname in args.workloads:
        lengths = _workload(wname)
        B = len(lengths)
        prefill_embeds, prefill_lens = _left_pad(lengths, args.hidden, device, dtype)
        tokens = sum(prefill_lens)
        S = prefill_embeds.shape[1]
        step_embeds = torch.randn(B, step_s, args.hidden, device=device, dtype=dtype)
        step_lens = [step_s] * B
        before_session = torch.cuda.memory_allocated(device)
        session = backbone.decode_session(batch_size=B, max_batch_tokens=args.max_batch_tokens,
                                          cache_capacity_tokens=args.cache_capacity_tokens or capacities[wname])
        session_mb = (torch.cuda.memory_allocated(device) - before_session) / 2**20

        def prefill() -> None:
            session.reset_rows()
            session.forward(embeds=prefill_embeds, lengths=prefill_lens)

        def decode_step() -> None:
            session.forward(embeds=step_embeds, lengths=step_lens)

        iters = args.iters
        first_p, med_p, lo_p, hi_p, peak_p = _timed(prefill, iters)
        # Restore the requested context before measuring incremental decoding.
        # Positions advance across iterations; preflight reserves all measured appends.
        session.reset_rows()
        session.forward(embeds=prefill_embeds, lengths=prefill_lens)
        first_d, med_d, lo_d, hi_d, peak_d = _timed(decode_step, iters)
        print(
            f"  {wname:15s} session={session_mb:.0f} MB pool={session.cache_capacity_tokens} tokens B={B} S={S:5d} tokens={tokens:6d} step={step_s} | "
            f"prefill {med_p:8.2f} ms [{lo_p:.1f},{hi_p:.1f}] (+{peak_p:6.0f} MB) | "
            f"{tokens / med_p * 1e3:>9,.0f} tok/s | "
            f"decode {med_d:8.2f} ms [{lo_d:.1f},{hi_d:.1f}] (+{peak_d:6.0f} MB) | "
            f"{B * step_s / med_d * 1e3:>8,.0f} tok/s | "
            f"first prefill {first_p:.0f} ms, first decode {first_d:.0f} ms"
        )
        if wants_profile(args):
            profile_call(
                prefill, label=f"{wname} prefill", steps=iters,
                cuda=True, trace_dir=args.profile_trace,
            )
            session.reset_rows()
            session.forward(embeds=prefill_embeds, lengths=prefill_lens)
            profile_call(
                decode_step, label=f"{wname} decode S={step_s}", steps=iters,
                cuda=True, trace_dir=args.profile_trace,
            )
        session.close()
        del session, prefill_embeds, step_embeds
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
