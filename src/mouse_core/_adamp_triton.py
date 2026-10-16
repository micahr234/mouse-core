"""Three GPU passes per same-shape AdamP batch, with no host projection tests."""

from __future__ import annotations

import math
from typing import Any

import torch
import triton
import triton.language as tl

# Triton's launch syntax adds runtime options not represented by its annotations.
# Keep the GPU function bodies typed; expose launchers as dynamic callables below.


@triton.jit
def _moments_stats(ptrs, stats, beta1, beta2, eps, correction2,
                   ROWS: tl.constexpr, COLS: tl.constexpr,
                   BR: tl.constexpr, BC: tl.constexpr, NESTEROV: tl.constexpr):
    tensor = tl.program_id(1)
    pptr = tl.load(ptrs + tensor * 4).to(tl.pointer_type(tl.float32))
    gptr = tl.load(ptrs + tensor * 4 + 1).to(tl.pointer_type(tl.float32))
    mptr = tl.load(ptrs + tensor * 4 + 2).to(tl.pointer_type(tl.float32))
    vptr = tl.load(ptrs + tensor * 4 + 3).to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BC)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    p = tl.load(pptr + offsets, mask, 0)
    g = tl.load(gptr + offsets, mask, 0)
    m = beta1 * tl.load(mptr + offsets, mask, 0) + (1 - beta1) * g
    v = beta2 * tl.load(vptr + offsets, mask, 0) + (1 - beta2) * g * g
    tl.store(mptr + offsets, m, mask)
    tl.store(vptr + offsets, v, mask)
    numerator = beta1 * m + (1 - beta1) * g if NESTEROV else m
    direction = numerator / (tl.sqrt(v) / correction2 + eps)
    base = tensor * 4 * ROWS + rows
    tl.store(stats + base, tl.sum(p * p, 1), rows < ROWS)
    tl.store(stats + base + ROWS, tl.sum(g * g, 1), rows < ROWS)
    tl.store(stats + base + 2 * ROWS, tl.sum(p * g, 1), rows < ROWS)
    tl.store(stats + base + 3 * ROWS, tl.sum(p * direction, 1), rows < ROWS)


@triton.jit
def _projection(stats, summary, delta, eps, ROWS: tl.constexpr, COLS: tl.constexpr,
                BLOCK: tl.constexpr, PROJECT: tl.constexpr):
    tensor = tl.program_id(0)
    r = tl.arange(0, BLOCK)
    base = tensor * 4 * ROWS + r
    p2 = tl.load(stats + base, r < ROWS, 0)
    g2 = tl.load(stats + base + ROWS, r < ROWS, 0)
    pg = tl.load(stats + base + 2 * ROWS, r < ROWS, 0)
    pd = tl.load(stats + base + 3 * ROWS, r < ROWS, 0)
    pnorm = tl.sqrt(tl.sum(p2, 0))
    gnorm = tl.sqrt(tl.sum(g2, 0))
    cosine = tl.abs(pg) / (tl.maximum(tl.sqrt(p2), eps) * tl.maximum(tl.sqrt(g2), eps))
    channel = tl.max(cosine, 0) < delta / tl.sqrt(float(COLS))
    layer_cosine = tl.abs(tl.sum(pg, 0)) / (tl.maximum(pnorm, eps) * tl.maximum(gnorm, eps))
    layer = layer_cosine < delta / tl.sqrt(float(ROWS * COLS))
    mode = tl.where(channel, 1, tl.where(layer, 2, 0)) if PROJECT else 0
    tl.store(summary + tensor * 2, mode)
    tl.store(summary + tensor * 2 + 1, tl.sum(pd, 0) / ((pnorm + eps) * (pnorm + eps)))


@triton.jit
def _update(ptrs, stats, summary, lr, step_size, beta1, eps, correction2, weight_decay, wd_ratio,
            ROWS: tl.constexpr, COLS: tl.constexpr,
            BR: tl.constexpr, BC: tl.constexpr, NESTEROV: tl.constexpr):
    tensor = tl.program_id(1)
    pptr = tl.load(ptrs + tensor * 4).to(tl.pointer_type(tl.float32))
    gptr = tl.load(ptrs + tensor * 4 + 1).to(tl.pointer_type(tl.float32))
    mptr = tl.load(ptrs + tensor * 4 + 2).to(tl.pointer_type(tl.float32))
    vptr = tl.load(ptrs + tensor * 4 + 3).to(tl.pointer_type(tl.float32))
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BC)
    offsets = rows[:, None] * COLS + cols[None, :]
    mask = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    p = tl.load(pptr + offsets, mask, 0)
    m = tl.load(mptr + offsets, mask, 0)
    v = tl.load(vptr + offsets, mask, 0)
    if NESTEROV:
        g = tl.load(gptr + offsets, mask, 0)
        m = beta1 * m + (1 - beta1) * g
    direction = m / (tl.sqrt(v) / correction2 + eps)
    mode = tl.load(summary + tensor * 2)
    layer_factor = tl.load(summary + tensor * 2 + 1)
    base = tensor * 4 * ROWS + rows
    norm = tl.sqrt(tl.load(stats + base, rows < ROWS, 0)) + eps
    row_factor = tl.load(stats + base + 3 * ROWS, rows < ROWS, 0) / (norm * norm)
    factor = tl.where(mode == 1, row_factor, tl.where(mode == 2, layer_factor, 0.0))
    direction = direction - p * factor[:, None]
    decay = tl.where(mode != 0, wd_ratio, 1.0)
    updated = p * (1 - lr * weight_decay * decay) - step_size * direction
    tl.store(pptr + offsets, updated, mask)


def fused_step(tensors: list[tuple[torch.Tensor, ...]], workspace: tuple[torch.Tensor, torch.Tensor],
               *, rows: int, cols: int, project: bool, step: int,
               lr: float, beta1: float, beta2: float, eps: float,
               weight_decay: float, delta: float, wd_ratio: float, nesterov: bool) -> None:
    device = tensors[0][0].device
    # A single small transfer per batch. Tensor pointers are refreshed each step:
    # zero_grad(set_to_none=True), checkpoint loading, and eval parking may replace storage.
    host = torch.tensor([[t.data_ptr() for t in entry] for entry in tensors],
                        dtype=torch.int64, pin_memory=True)
    pointers = host.to(device, non_blocking=True)
    stats, summary = workspace
    bc = triton.next_power_of_2(cols)
    br = min(32, max(1, 1024 // bc))
    warps = 4 if br * bc <= 4096 else 8
    grid = (triton.cdiv(rows, br), len(tensors))
    correction2 = math.sqrt(1 - beta2 ** step)
    moments_kernel: Any = _moments_stats
    projection_kernel: Any = _projection
    update_kernel: Any = _update
    with torch.cuda.device(device):
        moments_kernel[grid](pointers, stats, beta1, beta2, eps, correction2,
                            rows, cols, br, bc, nesterov, num_warps=warps)
        projection_kernel[(len(tensors),)](stats, summary, delta, eps, rows, cols,
                                    triton.next_power_of_2(rows), project)
        update_kernel[grid](pointers, stats, summary, lr, lr / (1 - beta1 ** step), beta1, eps,
                     correction2, weight_decay, wd_ratio, rows, cols, br, bc, nesterov,
                     num_warps=warps)
