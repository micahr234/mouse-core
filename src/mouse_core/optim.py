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


def _zero_grad(params: Iterable[nn.Parameter], *, set_to_none: bool) -> None:
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        if set_to_none:
            param.grad = None
            continue
        if grad.grad_fn is not None:
            grad.detach_()
        else:
            grad.requires_grad_(False)
        grad.zero_()


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

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter grads. ``set_to_none`` matches :meth:`torch.optim.Optimizer.zero_grad`."""
        _zero_grad(self._params, set_to_none=set_to_none)

    def step(self) -> None:
        self._inner.step()

    def state_dict(self) -> dict[str, Any]:
        return {"inner": self._inner.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._inner.load_state_dict(state["inner"])


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

    The masters are the source of truth once the optimizer exists: every
    ``step`` writes them back over the compute parameters. Load model
    weights *before* constructing the optimizer; to resume training use
    :meth:`state_dict` / :meth:`load_state_dict`, which carry the masters
    (and re-sync the compute parameters to them) alongside the AdamW moments.

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

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear compute (and master) grads. ``set_to_none`` matches :meth:`torch.optim.Optimizer.zero_grad`."""
        _zero_grad(self._params, set_to_none=set_to_none)
        _zero_grad((master for _, master in self._synced), set_to_none=set_to_none)

    def step(self) -> None:
        for compute, master in self._synced:
            _write_grad(master, compute.grad)
        self._inner.step()
        with torch.no_grad():
            for compute, master in self._synced:
                compute.copy_(master)

    def state_dict(self) -> dict[str, Any]:
        return {
            "inner": self._inner.state_dict(),
            "masters": [master.detach().clone() for _, master in self._synced],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        masters = state["masters"]
        if len(masters) != len(self._synced):
            raise ValueError(
                f"state_dict has {len(masters)} master tensors, optimizer has "
                f"{len(self._synced)} non-fp32 parameters"
            )
        self._inner.load_state_dict(state["inner"])
        with torch.no_grad():
            for (compute, master), saved in zip(self._synced, masters):
                if saved.shape != master.shape:
                    raise ValueError(
                        f"master shape {tuple(saved.shape)} does not match parameter "
                        f"shape {tuple(master.shape)}"
                    )
                master.copy_(saved)
                compute.copy_(master)
