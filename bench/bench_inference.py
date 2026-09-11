"""Benchmark cached FlexAttention decode (``FlexDecodeSession``) on CUDA.

Measures, per workload: prefill of a context into an empty paged KV pool,
then a one-token-per-row decode step against that history. Reports median
wall time, peak allocated memory above the parameter baseline, tokens/second,
and first-call (compile / page-grow / S=1 CUDA-graph capture) time. Decode is ``decode_kernel="flex"``
only. ``torch.set_float32_matmul_precision("high")`` is set so fp32 matches
the README TF32 path.

    PYTHON_GIL=0 .venv/bin/python bench/bench_inference.py --layers 8
    PYTHON_GIL=0 .venv/bin/python bench/bench_inference.py --layers 28 --workloads mid long
    PYTHON_GIL=0 .venv/bin/python bench/bench_inference.py --no-lora

Default shape is Qwen3-0.6B (hidden 1024, 16 q / 8 kv heads, head_dim 128,
FFN 3072) with fp32 LoRA rank 16 on a frozen bf16 base; ``--layers`` trims the
depth so long contexts fit. Pass ``--no-lora`` for a fully trainable fp32
backbone (decode still runs under ``no_grad``).
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from mouse_core.models.backbone import Qwen3Backbone
from mouse_core.models.lora import LoRAConfig


def _workload(name: str) -> tuple[int, list[int], list[int] | None]:
    """Return ``(B, lengths, grouping_pattern)``.

    ``grouping_pattern`` is ``None`` (one group) or a per-token pattern
    broadcast across rows: ``"recurring"`` or ``"manysmall"``.
    """
    if name == "short":
        return 4, [512] * 4, None
    if name == "mid":
        return 8, [4096] * 8, None
    if name == "mid_recurring":
        return 4, [4096] * 4, "recurring"
    if name == "long":
        return 4, [16384] * 4, None
    if name == "long_recurring":
        return 4, [16384] * 4, "recurring"
    if name == "long_manysmall":
        return 8, [16384] * 8, "manysmall"
    if name == "high_variance":
        return 8, [17, 3000, 5, 900, 4000, 61, 2400, 1], None
    raise ValueError(f"unknown workload {name!r}")


def _left_pad(
    lengths: list[int],
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
    grouping: str | None,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Random left-padded embeds ``[B, S, D]`` and matching grouping ids."""
    B = len(lengths)
    S = max(lengths)
    g = torch.Generator(device=device).manual_seed(0)
    embeds = torch.zeros(B, S, hidden, device=device, dtype=dtype)
    grouping_ids = torch.zeros(B, S, dtype=torch.long, device=device)
    for b, n in enumerate(lengths):
        if n == 0:
            continue
        embeds[b, S - n :] = torch.randn(n, hidden, device=device, dtype=dtype, generator=g)
        if grouping == "recurring":
            grouping_ids[b, S - n :] = torch.randint(0, 3, (n,), device=device, generator=g)
        elif grouping == "manysmall":
            grouping_ids[b, S - n :] = (torch.arange(n, device=device) // 32) % 64
    return embeds, lengths, grouping_ids


def _timed(fn: Callable[[], Any], iters: int) -> tuple[float, float, float, float, float]:
    """(first-call ms, median ms, min ms, max ms, peak MB above baseline)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
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
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short", "mid", "mid_recurring", "high_variance", "long", "long_recurring", "long_manysmall"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.manual_seed(0)
    lora = None if args.no_lora else LoRAConfig(rank=16, alpha=32.0)
    dtype = torch.float32 if lora is None else torch.bfloat16
    backbone = Qwen3Backbone(
        train_kernel="varlen",
        decode_kernel="flex",
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
        f"| decode_kernel=flex TF32=high"
    )

    for wname in args.workloads:
        B, lengths, grouping = _workload(wname)
        prefill_embeds, prefill_lens, prefill_grp = _left_pad(
            lengths, args.hidden, device, dtype, grouping
        )
        tokens = sum(prefill_lens)
        S = prefill_embeds.shape[1]
        step_embeds = torch.randn(B, 1, args.hidden, device=device, dtype=dtype)
        step_grp = torch.zeros(B, 1, dtype=torch.long, device=device)
        step_lens = [1] * B
        session = backbone.decode_session(batch_size=B)

        def prefill() -> None:
            session.reset_rows()
            session.forward(embeds=prefill_embeds, lengths=prefill_lens, grouping_ids=prefill_grp)

        def decode_step() -> None:
            session.forward(embeds=step_embeds, lengths=step_lens, grouping_ids=step_grp)

        iters = args.iters if tokens <= 4096 * 4 else max(3, args.iters // 2)
        first_p, med_p, lo_p, hi_p, peak_p = _timed(prefill, iters)
        # Prefill left the cache empty (last timed call resets then fills).
        # Grow once so decode-step timings are against a full context, not an empty pool.
        session.reset_rows()
        session.forward(embeds=prefill_embeds, lengths=prefill_lens, grouping_ids=prefill_grp)
        first_d, med_d, lo_d, hi_d, peak_d = _timed(decode_step, iters)
        print(
            f"  {wname:15s} B={B} S={S:5d} tokens={tokens:6d} | "
            f"prefill {med_p:8.2f} ms [{lo_p:.1f},{hi_p:.1f}] (+{peak_p:6.0f} MB) | "
            f"{tokens / med_p * 1e3:>9,.0f} tok/s | "
            f"decode {med_d:8.2f} ms [{lo_d:.1f},{hi_d:.1f}] (+{peak_d:6.0f} MB) | "
            f"{B / med_d * 1e3:>8,.0f} tok/s | "
            f"first prefill {first_p:.0f} ms, first decode {first_d:.0f} ms"
        )
        del session, prefill_embeds, step_embeds
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
