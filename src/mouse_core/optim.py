"""AdamW over fp32 trainable parameters.

Every trainable parameter in a MOUSE model is fp32: heads, projection,
reasoner, and either the whole fp32 backbone (full
fine-tuning) or the fp32 LoRA adapters
of a frozen bf16 backbone. AdamW therefore steps the parameters
directly and needs no fp32 master copies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from math import isfinite
from numbers import Real

import torch
import torch.nn as nn


def _trainable(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    trainable = [p for p in params if p.requires_grad]
    for p in trainable:
        if p.dtype != torch.float32:
            raise TypeError(
                f"AdamW trains fp32 parameters only, got a trainable {p.dtype} parameter "
                f"of shape {tuple(p.shape)}. Build the backbone with dtype=torch.float32 to "
                "fine-tune it, or freeze it with lora= to use a bf16 base."
            )
    return trainable


def _param_groups(
    params: Iterable[dict[str, Any]],
    *,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    """Validate explicit group rates before constructing the torch optimizer."""
    groups: list[dict[str, Any]] = []
    flat: list[nn.Parameter] = []
    for i, group in enumerate(params):
        if not isinstance(group, dict):
            raise TypeError("AdamW params must be param-group dicts with explicit 'params' and 'lr'.")
        missing = {"params", "lr"} - group.keys()
        if missing:
            raise TypeError(f"AdamW param group {i} is missing {', '.join(repr(key) for key in sorted(missing))}.")
        unknown = set(group) - {"params", "lr", "weight_decay"}
        if unknown:
            raise TypeError(f"AdamW param group {i} has unsupported keys {sorted(unknown)}.")
        lr = group["lr"]
        if isinstance(lr, bool) or not isinstance(lr, Real) or not isfinite(lr) or lr < 0:
            raise ValueError(f"AdamW param group {i} lr must be finite and nonnegative.")
        trainable = _trainable(group["params"])
        groups.append({"params": trainable, "lr": float(lr),
                       "weight_decay": group.get("weight_decay", weight_decay)})
        flat.extend(trainable)
    if not groups:
        raise ValueError("AdamW requires at least one explicit parameter group.")
    return groups, flat


def _torch_adamw(
    groups: list[dict[str, Any]],
    *,
    betas: tuple[float, float],
    eps: float,
    fused: bool | None,
) -> torch.optim.AdamW:
    flat = [p for group in groups for p in group["params"]]
    if fused is None:
        fused = bool(flat) and flat[0].device.type == "cuda"
    return torch.optim.AdamW(
        groups,
        betas=betas,
        eps=eps,
        fused=fused,
    )


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
    """:class:`torch.optim.AdamW` over the trainable (fp32) parameters.

    Rejects a trainable non-fp32 parameter: bf16 weights re-round every step,
    so updates below half a bf16 ULP (~``|w| / 512``) would never land.
    ``fused`` defaults to CUDA from the first trainable parameter.

    Every parameter-group dict must specify ``params`` and ``lr``. There is no
    optimizer-wide learning rate or fallback. A group may override the global
    ``weight_decay``. ``model.parameter_groups(lr={module_path: rate, ...})``
    constructs explicit rates for each trainable module.
    """

    def __init__(
        self,
        *,
        params: Iterable[dict[str, Any]],
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        fused: bool | None = None,
    ) -> None:
        groups, self._params = _param_groups(params, weight_decay=weight_decay)
        self._inner = _torch_adamw(
            groups,
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
