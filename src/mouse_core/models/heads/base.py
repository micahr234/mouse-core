"""Base classes for all MOUSE output heads.

To add a custom head, subclass :class:`BaseHead` and implement :meth:`forward`.

Example::

    from mouse_core.models.heads.base import BaseHead

    class MyHead(BaseHead):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return self.linear(h)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn


@dataclass
class HeadSpec:
    """Specification for a head to attach to a MOUSE model.

    Supported names:

    - ``"action_value"``: DiscreteActionValueHead
    - ``"action_value_layerwise"``: LayerwiseDiscreteActionValueHead
    - ``"action"``: DiscreteActionHead
    - ``"value"``: SwiGLUHead
    """

    name: str
    # Common
    hidden_dim: int | None = None
    num_layers: int | None = None
    scale: float | None = None
    use_norm: bool | None = None
    # Layerwise action value specific
    num_backbone_layers: int | None = None

    _VALID: ClassVar[tuple[str, ...]] = (
        "action_value",
        "action_value_layerwise",
        "action",
        "value",
    )

    def __post_init__(self) -> None:
        if self.name not in self._VALID:
            raise ValueError(
                f"unknown head name {self.name!r}; expected one of {self._VALID}"
            )
        if self.num_layers is not None and int(self.num_layers) < 0:
            raise ValueError(
                f"head {self.name!r} has negative num_layers ({self.num_layers}); "
                f"use 0 to disable or a positive integer"
            )
        if self.num_backbone_layers is not None and self.name != "action_value_layerwise":
            raise ValueError(
                f"num_backbone_layers is only valid for 'action_value_layerwise' heads, "
                f"got name={self.name!r}"
            )
        if self.num_backbone_layers is not None and int(self.num_backbone_layers) <= 0:
            raise ValueError(f"num_backbone_layers must be positive, got {self.num_backbone_layers!r}")


class BaseHead(nn.Module, ABC):
    """Abstract base for all output heads.

    A head receives the pooled step representations ``[B, S, D]`` from the
    backbone and maps them to a per-step output tensor.

    Subclass this and implement :meth:`forward` to create a custom head.
    The output shape is up to you — ``[B, S, A]`` for logit heads,
    ``[B, S, A, D]`` for vector heads, etc.
    """

    @abstractmethod
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Map step representations to head outputs.

        Args:
            h: Pooled step representations ``[B, S, D]``.

        Returns:
            Output tensor of any shape beginning with ``[B, S, ...]``.
        """
        ...
