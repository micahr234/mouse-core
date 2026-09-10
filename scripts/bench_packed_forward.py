"""Benchmark the packed training forward (``packed_forward``) on CUDA.

Measures, per workload: steady-state forward, forward+backward, a complete
LoRA training step (AdamW), real tokens/second, and peak allocated memory
above the parameter baseline, for every ``--train-kernel`` (``varlen``:
flash varlen; ``flex``: FlexAttention block mask) in each ``--modes`` body
(eager, compiled via ``install_compiled_decoder``, compiled with
``gradient_checkpointing``). Warmup / compile time is reported separately.
CUDA is synchronized around every measurement; medians and min/max over
``--iters`` iterations. Each kernel's outputs are also checked against the
other's on the same inputs (max abs diff printed per workload).

    PYTHON_GIL=0 .venv/bin/python scripts/bench_packed_forward.py --layers 8
    PYTHON_GIL=0 .venv/bin/python scripts/bench_packed_forward.py --layers 28 --workloads mid long
    PYTHON_GIL=0 .venv/bin/python scripts/bench_packed_forward.py --train-kernel flex --modes compiled

Default shape is Qwen3-0.6B (hidden 1024, 16 q / 8 kv heads, head_dim 128,
FFN 3072) with fp32 LoRA rank 16 on a frozen bf16 base; ``--layers`` trims the
depth so long streams fit. Pass ``--no-lora`` for a fully trainable fp32
backbone (fp32 reference attention / unfused Flex).

Reference numbers (RTX 3090 Ti, torch 2.13.0+cu130, 8 layers, LoRA r16,
fwd+bwd median, compiled body) for the two kernels on the same weights and
streams; peak memory is equal within 10 MB:

    workload                 flex       varlen
    short   L=512  4 seqs     18.5 ms    21.8 ms
    mid     L=4096 8 seqs     73.2 ms    69.0 ms
    mid recurring groups      73.6 ms    67.5 ms
    long    L=16384 4 seqs   401.1 ms   326.2 ms
    long recurring groups    284.3 ms   252.1 ms
    long 512 groups of 32    237.2 ms   237.2 ms

Compile warmup for the first fwd+bwd: ~2.5 s with varlen, ~12 s with Flex
(the block-mask kernel is autotuned). Gradient checkpointing (28 layers,
L=4096): 7372 MB -> 779 MB at 1.44x the step time.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from typing import Any, cast

import torch

from mouse_core.models.backbone import TrainKernel, Qwen3Backbone, install_compiled_decoder, packed_forward
from mouse_core.models.lora import LoRAConfig
from mouse_core.optim import AdamW


def _workload(name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    if name == "short":
        L, seqs, grp = 512, 4, None
    elif name == "mid":
        L, seqs, grp = 4096, 8, None
    elif name == "mid_recurring":
        L, seqs = 4096, 4
        grp = torch.randint(0, 3, (L,), generator=g)
    elif name == "long":
        L, seqs, grp = 16384, 4, None
    elif name == "long_recurring":
        L, seqs = 16384, 4
        grp = torch.randint(0, 4, (L,), generator=g)
    elif name == "long_manysmall":
        L, seqs = 16384, 8
        grp = (torch.arange(L) // 32) % 64
    elif name == "high_variance":
        lens = [17, 3000, 5, 900, 4000, 61, 2400, 1]
        seq = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)])
        return seq.to(device), torch.zeros(seq.shape[0], dtype=torch.long, device=device)
    else:
        raise ValueError(f"unknown workload {name!r}")
    seq = torch.arange(L) // (L // seqs)
    if grp is None:
        grp = torch.zeros(L, dtype=torch.long)
    return seq.to(device), grp.to(device)


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
    parser.add_argument("--no-lora", action="store_true", help="fully trainable fp32 backbone")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short", "mid", "mid_recurring", "high_variance", "long", "long_recurring", "long_manysmall"],
    )
    parser.add_argument("--modes", nargs="+", default=["eager", "compiled", "compiled+checkpoint"])
    parser.add_argument("--train-kernel", nargs="+", default=["varlen", "flex"], choices=["varlen", "flex"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs CUDA")
    device = torch.device("cuda")
    torch.manual_seed(0)
    lora = None if args.no_lora else LoRAConfig(rank=16, alpha=32.0)
    kernels = cast(list[TrainKernel], args.train_kernel)
    dtype = torch.float32 if lora is None else torch.bfloat16
    backbone = Qwen3Backbone(
        train_kernel=kernels[0],
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
    for name, p in backbone.named_parameters():
        if ".lora_" in name:
            p.data = p.data.float()
            if ".lora_B." in name:
                torch.nn.init.normal_(p, std=0.02)  # nonzero B so A receives gradients
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=1e-4)
    props = torch.cuda.get_device_properties(device)
    print(
        f"{props.name} | torch {torch.__version__} | layers={args.layers} base dtype={dtype} "
        f"lora={'off' if lora is None else f'r{lora.rank}'} params={sum(p.numel() for p in backbone.parameters())/1e6:.1f}M "
        f"trainable={sum(p.numel() for p in trainable)/1e6:.2f}M"
    )

    for mode in args.modes:
        if "compiled" in mode:
            install_compiled_decoder()
        backbone.gradient_checkpointing = "checkpoint" in mode
        for kernel in kernels:
            other: TrainKernel = "flex" if kernel == "varlen" else "varlen"
            print(f"\n### mode: {mode} | train_kernel: {kernel}")
            for wname in args.workloads:
                seq, grp = _workload(wname, device)
                L = seq.shape[0]
                embeds = torch.randn(L, args.hidden, device=device, dtype=dtype, requires_grad=True)

                def fwd(kernel: TrainKernel = kernel) -> torch.Tensor:
                    with torch.no_grad():
                        return packed_forward(
                            model=backbone.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp,
                            train_kernel=kernel,
                        )

                def fwd_bwd(kernel: TrainKernel = kernel) -> None:
                    out = packed_forward(
                        model=backbone.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp,
                        checkpoint=backbone.gradient_checkpointing, train_kernel=kernel,
                    )
                    out.float().square().mean().backward()
                    embeds.grad = None
                    for p in trainable:
                        p.grad = None

                def train_step(kernel: TrainKernel = kernel) -> None:
                    out = packed_forward(
                        model=backbone.model, embeds=embeds, sequence_ids=seq, grouping_ids=grp,
                        checkpoint=backbone.gradient_checkpointing, train_kernel=kernel,
                    )
                    out.float().square().mean().backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    embeds.grad = None

                iters = args.iters if L <= 4096 else max(3, args.iters // 2)
                first_f, med_f, _, _, peak_f = _timed(fwd, iters)
                first_b, med_b, lo_b, hi_b, peak_b = _timed(fwd_bwd, iters)
                _, med_s, _, _, peak_s = _timed(train_step, iters)
                diff = (fwd().float() - fwd(other).float()).abs().max().item()
                print(
                    f"  {wname:15s} L={L:6d} groups={int(torch.unique(seq * 2**20 + grp - grp.min()).numel()):4d} | "
                    f"fwd {med_f:8.2f} ms (+{peak_f:6.0f} MB) | fwd+bwd {med_b:8.2f} ms [{lo_b:.1f},{hi_b:.1f}] "
                    f"(+{peak_b:6.0f} MB) | step {med_s:8.2f} ms (+{peak_s:6.0f} MB) | "
                    f"{L / med_b * 1e3:>9,.0f} tok/s | first fwd {first_f:.0f} ms, first fwd+bwd {first_b:.0f} ms | "
                    f"vs {other} {diff:.2e}"
                )
                del embeds
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
