"""fp32 LoRA adapters on a frozen backbone.

The alternative to fine-tuning the whole backbone in fp32: the base weights
are frozen and, on CUDA, bf16, and the only trainable backbone parameters
are the LoRA factors, which are always fp32. Their rank-``r`` matmuls run
in fp32 and the delta is cast back onto the base output, so updates land
in fp32 tensors and nothing needs fp32 master weights or Polyak shadows.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, kw_only=True)
class LoRAConfig:
    """LoRA hyperparameters for a backbone.

    Every ``nn.Linear`` under the backbone is adapted; there is no subset.

    Args:
        rank: Adapter rank ``r``.
        alpha: Scaling numerator; the delta is scaled by ``alpha / rank``.
        dropout: Dropout on the adapter input (``0`` disables it).
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if int(self.rank) < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}.")
        if float(self.alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))


def _lora_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scale: float,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """``y = W x + scale · B(A(x_fp32))`` with the casts inside one function.

    Packed train and cached decode both call this (via :class:`LoRALinear`),
    so the two paths cannot drift. On CUDA the function is ``torch.compile``d
    into one kernel graph unless a parent decoder body is already compiling,
    in which case the eager math is inlined and fused into that parent.
    """
    y = F.linear(x, weight, bias)
    xd = F.dropout(x, p=dropout_p, training=training) if dropout_p > 0.0 else x
    a = F.linear(xd.to(dtype=torch.float32), lora_A)
    delta = F.linear(a, lora_B) * scale
    return y + delta.to(dtype=y.dtype)


_compiled_lora: Any | None = None
_force_eager_lora = False


def _apply_lora(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scale: float,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Dispatch :func:`_lora_linear` — compiled on CUDA, inlined under compile."""
    if _force_eager_lora or torch.compiler.is_compiling() or x.device.type != "cuda":
        return _lora_linear(x, weight, bias, lora_A, lora_B, scale, dropout_p, training)
    global _compiled_lora
    if _compiled_lora is None:
        _compiled_lora = torch.compile(_lora_linear, dynamic=True)
    return _compiled_lora(x, weight, bias, lora_A, lora_B, scale, dropout_p, training)


class LoRALinear(nn.Module):
    """``base(x) + lora_B(lora_A(x)) * alpha / rank`` with fp32 adapters.

    ``base`` is the frozen wrapped ``nn.Linear`` and keeps its dtype (bf16 on
    CUDA). ``lora_A`` / ``lora_B`` are fp32 and the only trainable
    parameters; the adapter input is cast to fp32 and the delta is cast back
    to the base output dtype. ``lora_B`` starts at zero so the wrapped
    module's output is unchanged at construction.

    The three matmuls live in one function (:func:`_lora_linear`) so packed
    train leftovers and cached decode stay identical; on CUDA that function
    is compiled into a single op.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = config.rank
        self.scale = config.alpha / config.rank
        self.dropout_p = config.dropout
        self.lora_A = nn.Linear(base.in_features, config.rank, bias=False, dtype=torch.float32)
        self.lora_B = nn.Linear(config.rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_lora(
            x,
            self.base.weight,
            self.base.bias,
            self.lora_A.weight,
            self.lora_B.weight,
            self.scale,
            self.dropout_p,
            self.training,
        )


def lora_modules(module: nn.Module) -> Iterator[LoRALinear]:
    """Yield every :class:`LoRALinear` under ``module``."""
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def apply_lora(*, model: nn.Module, config: LoRAConfig) -> int:
    """Freeze ``model`` and wrap every ``nn.Linear`` in :class:`LoRALinear`.

    Every existing parameter is frozen; the LoRA factors are the only
    trainable parameters afterwards. Run this *after* pretrained weights are
    loaded — wrapping renames the base keys to ``<name>.base.weight``.

    Returns:
        Number of modules wrapped.

    Raises:
        ValueError: When no ``nn.Linear`` exists under ``model``.
    """
    model.requires_grad_(False)
    wrapped = 0
    for parent in list(model.modules()):
        if isinstance(parent, LoRALinear):
            continue
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, config))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(f"no nn.Linear found under {type(model).__name__}.")
    return wrapped
