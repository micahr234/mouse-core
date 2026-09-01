"""AdamW, and AdamWFp32 when bf16 weights need fp32 masters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def _trainable(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [p for p in params if p.requires_grad]


def _torch_adamw(
    params: list[nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    fused: bool | None,
) -> torch.optim.AdamW:
    if fused is None:
        fused = bool(params) and params[0].device.type == "cuda"
    return torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
    )


def _write_grad(dst: nn.Parameter, src: torch.Tensor | None) -> None:
    if src is None:
        dst.grad = None
        return
    if dst.grad is None:
        dst.grad = src.detach().to(dtype=dst.dtype)
        return
    dst.grad.copy_(src)


class AdamW:
    """Stock AdamW. Same defaults as :class:`AdamWFp32`, no master copies.

    ``fused`` defaults to CUDA from the first trainable parameter.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        fused: bool | None = None,
    ) -> None:
        self._params = _trainable(params)
        self._inner = _torch_adamw(
            self._params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._inner.param_groups

    def zero_grad(self) -> None:
        for param in self._params:
            param.grad = None

    def step(self) -> None:
        self._inner.step()


class AdamWFp32:
    """AdamW with fp32 master weights for non-fp32 compute parameters.

    Forward and backward stay in the parameter dtype (bf16 on CUDA). Each
    trainable non-fp32 parameter has a persistent fp32 master. ``step``
    casts the compute grad onto that master, runs :class:`torch.optim.AdamW`,
    and copies the updated master back. Already-fp32 parameters (output
    heads) are stepped in place.

    Pure-bf16 AdamW re-rounds the weight every step, so updates below half
    a bf16 ULP (~``|w| / 512``) never land. Masters accumulate those
    updates until they cross the threshold.

    ``fused`` defaults to CUDA from the first trainable parameter.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        fused: bool | None = None,
    ) -> None:
        self._params = _trainable(params)
        self._synced: list[tuple[nn.Parameter, nn.Parameter]] = []
        opt_params: list[nn.Parameter] = []
        for compute in self._params:
            if compute.dtype == torch.float32:
                opt_params.append(compute)
                continue
            master = nn.Parameter(compute.detach().to(dtype=torch.float32).clone())
            self._synced.append((compute, master))
            opt_params.append(master)
        self._inner = _torch_adamw(
            opt_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused,
        )

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._inner.param_groups

    def zero_grad(self) -> None:
        for param in self._params:
            param.grad = None

    def step(self) -> None:
        for compute, master in self._synced:
            _write_grad(master, compute.grad)
        self._inner.step()
        with torch.no_grad():
            for compute, master in self._synced:
                compute.copy_(master)
