"""Fp32 AdamP with a batched Triton CUDA implementation.

``fused=False`` uses eager PyTorch; ``True`` requires CUDA and contiguous
parameters/gradients whose flattened row and column counts are <= 16384.
``None`` selects the fused path on supported CUDA tensors, otherwise eager.
The projection test and update follow Heo et al. (ICLR 2021):
https://github.com/clovaai/AdamP/blob/master/adamp/adamp.py

The CUDA path batches equal shapes/step counts, uses three kernels per batch,
and recomputes the Adam direction in the final kernel rather than retaining
a parameter-sized temporary. Projection decisions stay on the GPU. This is
not a CUDA-graph-capturable or differentiable optimizer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch


def _shape(p: torch.Tensor) -> tuple[int, int]:
    rows = p.shape[0] if p.ndim > 1 else 1
    return rows, p.numel() // rows if rows else 0


def _supported(p: torch.Tensor, g: torch.Tensor) -> bool:
    rows, cols = _shape(p)
    return (p.device.type == "cuda" and p.is_contiguous() and g.is_contiguous()
            and 0 < rows <= 16384 and 0 < cols <= 16384)


@torch.no_grad()
def _eager_step(p: torch.Tensor, g: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                *, step: int, lr: float, beta1: float, beta2: float, eps: float,
                weight_decay: float, delta: float, wd_ratio: float, nesterov: bool) -> None:
    m.mul_(beta1).add_(g, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    denom = v.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(eps)
    direction = ((beta1 * m + (1 - beta1) * g) if nesterov else m) / denom
    decay_ratio = 1.0
    if p.ndim > 1 and p.numel():
        for rows in (p.shape[0], 1):
            pv, gv = p.reshape(rows, -1), g.reshape(rows, -1)
            cosine = torch.nn.functional.cosine_similarity(gv, pv, dim=1, eps=eps).abs()
            if bool(cosine.max() < delta / math.sqrt(pv.shape[1])):
                unit = pv / (pv.norm(dim=1, keepdim=True) + eps)
                dv = direction.reshape(rows, -1)
                direction = (dv - unit * (unit * dv).sum(dim=1, keepdim=True)).reshape_as(p)
                decay_ratio = wd_ratio
                break
    p.mul_(1 - lr * weight_decay * decay_ratio)
    p.add_(direction, alpha=-lr / (1 - beta1 ** step))


class _AdamP(torch.optim.Optimizer):
    def __init__(self, params: Any, *, lr: float, betas: tuple[float, float],
                 eps: float, weight_decay: float, delta: float, wd_ratio: float,
                 nesterov: bool, fused: bool | None) -> None:
        for name, value in (("lr", lr), ("weight_decay", weight_decay), ("delta", delta),
                            ("wd_ratio", wd_ratio), ("eps", eps)):
            if not math.isfinite(value) or value < 0 or (name == "eps" and value == 0):
                raise ValueError(f"{name} must be finite and {'positive' if name == 'eps' else 'nonnegative'}")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas):
            raise ValueError("betas must be in [0, 1)")
        if fused is not None and type(fused) is not bool:
            raise TypeError("fused must be a bool or None")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                     delta=delta, wd_ratio=wd_ratio, nesterov=nesterov))
        self.fused = fused
        self._workspace: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate all active tensors before any parameter or moment changes.
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if p.dtype != torch.float32 or g.dtype != torch.float32:
                    raise TypeError("AdamP trains fp32 parameters only")
                if g.is_sparse or g.layout != torch.strided or p.layout != torch.strided:
                    raise ValueError("AdamP requires dense strided tensors")
                if self.fused is True and not _supported(p, g):
                    raise ValueError("fused AdamP requires contiguous CUDA tensors with row/column counts in [1, 16384]")

        for group_index, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            settings = dict(lr=group["lr"], beta1=beta1, beta2=beta2, eps=group["eps"],
                            weight_decay=group["weight_decay"], delta=group["delta"],
                            wd_ratio=group["wd_ratio"], nesterov=group["nesterov"])
            batches: dict[Any, list[tuple[torch.Tensor, ...]]] = defaultdict(list)
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                state["step"] += 1
                m, v = state["exp_avg"], state["exp_avg_sq"]
                if self.fused is not False and _supported(p, g):
                    key = (p.device, *_shape(p), p.ndim > 1, state["step"])
                    batches[key].append((p, g, m, v))
                else:
                    _eager_step(p, g, m, v, step=state["step"], **settings)
            if batches:
                from mouse_core._adamp_triton import fused_step
                for (device, rows, cols, project, step), tensors in batches.items():
                    key = (group_index, device, rows, cols, project, len(tensors))
                    workspace = self._workspace.get(key)
                    if workspace is None:
                        workspace = (torch.empty((len(tensors), 4, rows), device=device, dtype=torch.float32),
                                     torch.empty((len(tensors), 2), device=device, dtype=torch.float32))
                        self._workspace[key] = workspace
                    fused_step(tensors, workspace, rows=rows, cols=cols, project=project,
                               step=step, **settings)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._workspace.clear()


class AdamP:
    """Mouse-compatible AdamP wrapper; default betas match Mouse's AdamW.

    Supports flat parameter lists or groups with ``params``, ``lr``, and
    ``weight_decay``. Frozen tensors are ignored. CPU/eager and CUDA/fused
    checkpoints are interchangeable; AdamW checkpoints are rejected.
    """

    def __init__(self, *, params: Iterable[Any], lr: float, weight_decay: float = 0.0,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 delta: float = 0.1, wd_ratio: float = 0.1,
                 nesterov: bool = False, fused: bool | None = None) -> None:
        raw = list(params)
        grouped = any(isinstance(p, dict) for p in raw)
        if grouped and not all(isinstance(p, dict) for p in raw):
            raise TypeError("AdamP params must be parameters or param-group dicts, not a mix")
        groups: list[dict[str, Any]] = raw if grouped else [{"params": raw}]
        clean = []
        for group in groups:
            if "params" not in group or set(group) - {"params", "lr", "weight_decay"}:
                raise TypeError("AdamP groups require params and only allow lr/weight_decay overrides")
            trainable = [p for p in group["params"] if p.requires_grad]
            if any(p.dtype != torch.float32 for p in trainable):
                raise TypeError("AdamP trains fp32 parameters only")
            for name, default in (("lr", lr), ("weight_decay", weight_decay)):
                value = group.get(name, default)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be finite and nonnegative")
            clean.append({**group, "params": trainable})
        self._inner = _AdamP(clean, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                             delta=delta, wd_ratio=wd_ratio, nesterov=nesterov, fused=fused)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self._inner.step()

    def state_dict(self) -> dict[str, Any]:
        return {"algorithm": "adamp", "inner": self._inner.state_dict()}

    def clear_workspace(self) -> None:
        """Release reusable CUDA scratch storage, e.g. when parking for evaluation."""
        self._inner._workspace.clear()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != "adamp":
            raise ValueError("expected an AdamP checkpoint; AdamW state is not interchangeable")
        self._inner.load_state_dict(state["inner"])
