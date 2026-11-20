"""LoRA adapters with explicit storage precision on a frozen backbone.

The alternative to fine-tuning the whole backbone in fp32: the base weights
are frozen and, on CUDA, bf16, and the only trainable backbone parameters
are the LoRA factors, stored in FP32 or BF16. Their rank-``r`` matmuls use
backbone autocast when enabled, otherwise FP32.
The delta is cast back onto the base output. BF16 trainable adapters require
stochastic optimizer updates and, when delayed, stochastic Polyak updates.
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
        dtype: Stored adapter weight dtype. Use ``"float32"`` or ``"bfloat16"``;
            backbone autocast independently controls eligible computation.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("LoRA dtype must be float32 or bfloat16; BF16 training requires stochastic optimizer updates.")
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
    """``y = W x + scale · B(A(x_adapter_dtype))`` with the casts inside one function.

    Packed train and cached decode both call this (via :class:`LoRALinear`),
    so the two paths cannot drift. On CUDA the function is ``torch.compile``d
    into one kernel graph unless a parent decoder body is already compiling,
    in which case the eager math is inlined and fused into that parent.
    """
    y = F.linear(x, weight, bias)
    xd = F.dropout(x, p=dropout_p, training=training) if dropout_p > 0.0 else x
    a = F.linear(xd.to(dtype=lora_A.dtype), lora_A)
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
    """``base(x) + lora_B(lora_A(x)) * alpha / rank`` with FP32 or BF16 adapters.

    ``base`` is the frozen wrapped ``nn.Linear`` and keeps its dtype (bf16 on
    CUDA). ``lora_A`` / ``lora_B`` have the configured dtype and are the only trainable
    parameters; the adapter input is cast to their dtype and the delta is cast back
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
        self.lora_A = nn.Linear(base.in_features, config.rank, bias=False,
                                device=base.weight.device, dtype=getattr(torch, config.dtype))
        self.lora_B = nn.Linear(config.rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=getattr(torch, config.dtype))
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.train(base.training)

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
        ValueError: When no ``nn.Linear`` exists under ``model`` or LoRA has
            already been attached. Validation leaves trainability unchanged.
    """
    parents = list(model.modules())
    if any(isinstance(parent, LoRALinear) for parent in parents):
        raise ValueError("model already contains LoRA adapters.")
    # named_children() deduplicates aliases; replace every registered reference
    # while keeping shared linears tied to the same adapter.
    linears = [(parent, name, child) for parent in parents
               for name, child in parent._modules.items() if isinstance(child, nn.Linear)]
    if not linears:
        raise ValueError(f"no nn.Linear found under {type(model).__name__}.")
    model.requires_grad_(False)
    replacements: dict[int, LoRALinear] = {}
    for parent, name, child in linears:
        if id(child) not in replacements:
            replacements[id(child)] = LoRALinear(child, config)
        setattr(parent, name, replacements[id(child)])
    return len(replacements)


def named_backbone_parameters(backbone: nn.Module, *, adapters: bool) -> Iterator[tuple[str, nn.Parameter]]:
    """Select LoRA factors or base parameters by identity, independent of freezing.

    Names stay relative to the supplied backbone. Tied parameters appear once.
    Only factors owned by LoRALinear are adapters; names alone do not identify one.
    """
    adapter_ids = {id(p) for layer in lora_modules(backbone)
                   for factor in (layer.lora_A, layer.lora_B) for p in factor.parameters()}
    for name, parameter in backbone.named_parameters():
        if (id(parameter) in adapter_ids) == adapters:
            yield name, parameter


def backbone_parameter_view(backbone: nn.Module, *, adapters: bool) -> nn.Module:
    """Non-owning optimizer/target group with original parameter names.

    Do not register this view as a child of the original model. Changing
    requires_grad on its parameters freezes weights without detaching forward
    computation, so gradients can still reach earlier trainable embeddings.
    """
    view = nn.Module()
    for name, parameter in named_backbone_parameters(backbone, adapters=adapters):
        parent = view
        *parts, leaf = name.split(".")
        for part in parts:
            if part not in parent._modules:
                parent.add_module(part, nn.Module())
            parent = parent.get_submodule(part)
        parent.register_parameter(leaf, parameter)
    return view
