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

import torch
import torch.nn as nn


@dataclass
class HeadSpec:
    """Specification for a head to attach to a MOUSE model.

    ``name`` is a caller-chosen key. It is not restricted to a built-in
    list; the head *type* decides architecture.
    """

    name: str
    # Common
    hidden_dim: int | None = None
    num_layers: int | None = None
    scale: float | None = None
    use_norm: bool | None = None
    # Layerwise action value specific
    num_backbone_layers: int | None = None

    def __post_init__(self) -> None:
        if self.num_layers is not None and int(self.num_layers) < 0:
            raise ValueError(
                f"head {self.name!r} has negative num_layers ({self.num_layers}); "
                f"use 0 to disable or a positive integer"
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
