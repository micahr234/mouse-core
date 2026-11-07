"""Benchmark FlexAttention packed training on CUDA.

Measure eager, compiled, and checkpointed decoder bodies: forward, backward,
optimizer step, tokens/second, and peak allocation. Synchronized wall times
include CPU work. First-call time is reported separately; later shapes can still
compile. Each sample updates weights in the training-step measurement. Example: python bench/bench_train.py --layers 8 --modes compiled
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

from mouse_core.models.backbone import Backbone, install_compiled_decoder, packed_forward
from mouse_core.models.lora import LoRAConfig
from mouse_core.optim import AdamW

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from bench_profile import add_profile_args, profile_call, wants_profile


def _workload(name: str, device: torch.device) -> torch.Tensor:
    """Flat sequence ids. Recurring names are many short sequences."""
    g = torch.Generator().manual_seed(0)
    if name == "short":
        L, seqs = 512, 4
        seq = torch.arange(L) // (L // seqs)
    elif name == "mid":
        L, seqs = 4096, 8
        seq = torch.arange(L) // (L // seqs)
    elif name == "mid_recurring":
        seq = torch.randint(0, 3, (4096,), generator=g)
    elif name == "long":
        L, seqs = 16384, 4
        seq = torch.arange(L) // (L // seqs)
    elif name == "long_recurring":
        seq = torch.randint(0, 4, (16384,), generator=g)
    elif name == "long_manysmall":
        seq = (torch.arange(16384) // 32) % 64
    elif name == "high_variance":
        lens = [17, 3000, 5, 900, 4000, 61, 2400, 1]
        seq = torch.cat([torch.full((n,), i) for i, n in enumerate(lens)])
    else:
        raise ValueError(f"unknown workload {name!r}")
    return seq.to(device)


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
    parser.add_argument("--no-lora", action="store_true", help="fully trainable fp32 backbone")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-attention-distance-tokens", type=int, default=16383)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["short", "mid", "mid_recurring", "high_variance", "long", "long_recurring", "long_manysmall"],
    )
    parser.add_argument("--modes", nargs="+", choices=["eager", "compiled", "compiled+checkpoint"],
                        default=["eager", "compiled", "compiled+checkpoint"])
    add_profile_args(parser)
    args = parser.parse_args()
    if args.iters < 1 or args.max_attention_distance_tokens < 0:
        parser.error("iters must be positive and attention distance nonnegative")

    if "eager" in args.modes and any(mode != "eager" for mode in args.modes[:args.modes.index("eager")]):
        parser.error("eager must precede compiled modes: compiled decoder installation is process-wide")
    if len(set(args.modes)) != len(args.modes):
        parser.error("modes must not repeat")
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
        max_attention_distance_tokens=args.max_attention_distance_tokens,
    )
    backbone.to(device)
    for name, p in backbone.named_parameters():
        if ".lora_" in name:
            p.data = p.data.float()
            if ".lora_B." in name:
                torch.nn.init.normal_(p, std=0.02)  # nonzero B so A receives gradients
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = AdamW(params=[{'params': trainable, 'lr': 1e-4}])
    props = torch.cuda.get_device_properties(device)
    print(
        f"{props.name} | torch {torch.__version__} | layers={args.layers} base dtype={dtype} "
        f"lora={'off' if lora is None else f'r{lora.rank}'} params={sum(p.numel() for p in backbone.parameters())/1e6:.1f}M "
        f"trainable={sum(p.numel() for p in trainable)/1e6:.2f}M "
        f"attention_distance={args.max_attention_distance_tokens} TF32=high wall timing"
    )

    for mode in args.modes:
        if "compiled" in mode:
            install_compiled_decoder()
        backbone.gradient_checkpointing = "checkpoint" in mode
        print(f"\n### mode: {mode} | FlexAttention")
        for wname in args.workloads:
            seq = _workload(wname, device)
            L = seq.shape[0]
            embeds = torch.randn(L, args.hidden, device=device, dtype=dtype, requires_grad=True)

            def fwd() -> torch.Tensor:
                with torch.no_grad():
                    return packed_forward(
                        model=backbone.model, embeds=embeds, sequence_ids=seq,
                    )

            def fwd_bwd() -> None:
                out = packed_forward(
                    model=backbone.model, embeds=embeds, sequence_ids=seq,
                    checkpoint=backbone.gradient_checkpointing,
                )
                out.float().square().mean().backward()
                embeds.grad = None
                for p in trainable:
                    p.grad = None

            def train_step() -> None:
                out = packed_forward(
                    model=backbone.model, embeds=embeds, sequence_ids=seq,
                    checkpoint=backbone.gradient_checkpointing,
                )
                out.float().square().mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                embeds.grad = None

            iters = args.iters
            first_f, med_f, _, _, peak_f = _timed(fwd, iters)
            first_b, med_b, lo_b, hi_b, peak_b = _timed(fwd_bwd, iters)
            _, med_s, _, _, peak_s = _timed(train_step, iters)
            print(
                f"  {wname:15s} L={L:6d} sequences={int(seq.unique().numel()):4d} | "
                f"fwd {med_f:8.2f} ms (+{peak_f:6.0f} MB) | fwd+bwd {med_b:8.2f} ms [{lo_b:.1f},{hi_b:.1f}] "
                f"(+{peak_b:6.0f} MB) | step {med_s:8.2f} ms (+{peak_s:6.0f} MB) | "
                f"{L / med_b * 1e3:>9,.0f} tok/s | first fwd {first_f:.0f} ms, first fwd+bwd {first_b:.0f} ms | "
            )
            if wants_profile(args):
                tag = f"{mode} flex {wname}"
                profile_call(
                    fwd, label=f"{tag} fwd", steps=iters,
                    cuda=True, trace_dir=args.profile_trace,
                )
                profile_call(
                    fwd_bwd, label=f"{tag} fwd+bwd", steps=iters,
                    cuda=True, trace_dir=args.profile_trace,
                )
                profile_call(
                    train_step, label=f"{tag} step", steps=iters,
                    cuda=True, trace_dir=args.profile_trace,
                )
            del embeds
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
