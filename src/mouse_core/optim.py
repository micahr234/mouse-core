"""AdamW over fp32 trainable parameters.

Every trainable parameter in a MOUSE model is fp32: heads, encoder,
reasoner / recurrence, and either the whole fp32 backbone (full
fine-tuning) or the fp32 LoRA adapters of a frozen bf16 backbone. AdamW
therefore steps the parameters directly and needs no fp32 master copies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn


def _trainable(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    trainable = [p for p in params if p.requires_grad]
    for p in trainable:
        if p.dtype != torch.float32:
            raise TypeError(
                f"AdamW trains fp32 parameters only, got a trainable {p.dtype} parameter "
                f"of shape {tuple(p.shape)}. Keep the model fp32 (model.to(device)) to "
                "fine-tune the backbone, or freeze it with lora= before casting to bf16."
            )
    return trainable


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
