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

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA hyperparameters for a backbone.

    Args:
        rank: Adapter rank ``r``.
        alpha: Scaling numerator; the delta is scaled by ``alpha / rank``.
        dropout: Dropout on the adapter input (``0`` disables it).
        targets: Attribute names of the ``nn.Linear`` modules to adapt.
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = _DEFAULT_TARGETS

    def __post_init__(self) -> None:
        if int(self.rank) < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}.")
        if float(self.alpha) <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        targets = tuple(str(t) for t in self.targets)
        if not targets:
            raise ValueError("targets must name at least one nn.Linear attribute.")
        object.__setattr__(self, "rank", int(self.rank))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "targets", targets)


class LoRALinear(nn.Module):
    """``base(x) + lora_B(lora_A(x)) * alpha / rank`` with fp32 adapters.

    ``base`` is the frozen wrapped ``nn.Linear`` and keeps its dtype (bf16 on
    CUDA). ``lora_A`` / ``lora_B`` are fp32 and the only trainable
    parameters; the adapter input is cast to fp32 and the delta is cast back
    to the base output dtype. ``lora_B`` starts at zero so the wrapped
    module's output is unchanged at construction.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}.")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = config.rank
        self.scale = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
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
        y = self.base(x)
        a = F.linear(self.dropout(x).to(dtype=torch.float32), self.lora_A.weight)
        delta = F.linear(a, self.lora_B.weight) * self.scale
        return y + delta.to(dtype=y.dtype)


def lora_modules(module: nn.Module) -> Iterator[LoRALinear]:
    """Yield every :class:`LoRALinear` under ``module``."""
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def apply_lora(model: nn.Module, config: LoRAConfig) -> int:
    """Freeze ``model`` and wrap its target ``nn.Linear`` modules in :class:`LoRALinear`.

    Every existing parameter is frozen; the LoRA factors are the only
    trainable parameters afterwards. Run this *after* pretrained weights are
    loaded — wrapping renames the base keys to ``<target>.base.weight``.

    Returns:
        Number of modules wrapped.

    Raises:
        ValueError: When no attribute named in ``config.targets`` is an
            ``nn.Linear`` anywhere under ``model``.
    """
    model.requires_grad_(False)
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in config.targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, config))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(
            f"no nn.Linear named {config.targets} found under {type(model).__name__}."
        )
    return wrapped
