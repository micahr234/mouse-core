"""Base classes for all MOUSE output heads.

To add a custom head, subclass :class:`BaseHead` and implement :meth:`forward`.
If your head uses an EMA target network (like DQN), subclass
:class:`BaseHeadWithTarget` — ``target_forward`` and ``polyak_update`` are
provided automatically; only ``forward`` needs to be implemented.

Example — simple head::

    from mouse_core.models.heads.base import BaseHead

    class MyHead(BaseHead):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return self.linear(h)

Example — head with target network::

    from mouse_core.models.heads.base import BaseHeadWithTarget

    class MyTwinHead(BaseHeadWithTarget):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.online = nn.Linear(in_features, out_features)
            self._init_target(self.online)  # creates self.target, freezes it, syncs weights

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            return self.online(h)
        # target_forward and polyak_update are inherited from BaseHeadWithTarget
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn


@dataclass
class HeadSpec:
    """Specification for a head to attach to a MOUSE model.

    Mirrors the column-declarative style of embedding columns. Each head
    declares its ``name`` (the key it will appear under in model outputs)
    and ``type`` (which head implementation to use). The remaining fields
    are options specific to that head type.

    Supported semantic head types (names):

    - ``"action_value"``: DiscreteActionValueHead — value per discrete action (has target net)
    - ``"action_vector"``: VectorActionValueHead — vector per action (has target net)
    - ``"action"``: DiscreteActionHead — discrete action logits / policy (no target net)
    - ``"value"``: plain head for value regression

    Common options for all:
      - ``num_layers``, ``hidden_dim``, ``scale``, ``use_norm``

    Vector-specific:
      - ``vec_dim``, ``bias_scale`` (only for ``"action_vector"``)

    A head is disabled when ``num_layers`` is 0 (or the spec is omitted).

    Example::

        head_kwargs = {
            "action_head": "action_value",
            "heads": [
                {"name": "action_value", "num_layers": 1, "hidden_dim": 32, "scale": 0.01},
                {"name": "action", "num_layers": 0},
            ]
        }
    """

    name: str
    type: str | None = None
    # Common
    hidden_dim: int | None = None
    num_layers: int | None = None
    scale: float | None = None
    use_norm: bool | None = None
    # Layerwise action value specific
    num_backbone_layers: int | None = None
    # Vector action value specific
    vec_dim: int | None = None
    bias_scale: float | None = None

    _VALID: ClassVar[tuple[str, ...]] = (
        "action_value",
        "action_value_layerwise",
        "action_vector",
        "action",
        "value",
    )

    def __post_init__(self) -> None:
        if self.name not in self._VALID:
            raise ValueError(
                f"unknown head name {self.name!r}; expected one of {self._VALID}"
            )
        if self.type is not None:
            t = str(self.type).lower()
            if t not in self._VALID:
                raise ValueError(
                    f"unknown head type {self.type!r} for head {self.name!r}; expected one of {self._VALID}"
                )
            if t != self.name:
                raise ValueError(
                    f"head 'type' {self.type!r} does not match 'name' {self.name!r}; "
                    f"omit 'type' or set it equal to 'name'"
                )
            if t != self.type:
                object.__setattr__(self, "type", t)
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
        if self.vec_dim is not None and self.name != "action_vector":
            raise ValueError(f"vec_dim is only valid for 'action_vector' heads, got name={self.name!r}")
        if self.vec_dim is not None and int(self.vec_dim) <= 0:
            raise ValueError(f"vec_dim must be positive, got {self.vec_dim!r}")
        if self.bias_scale is not None and self.name != "action_vector":
            raise ValueError(f"bias_scale is only valid for 'action_vector' heads, got name={self.name!r}")

    def effective_type(self) -> str:
        """Return the head implementation type, defaulting to ``name``."""
        return self.type if self.type is not None else self.name


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


class BaseHeadWithTarget(BaseHead):
    """Base for heads that maintain an EMA target network alongside the online head.

    Subclass this when your head needs a slowly-updated target copy for stable
    bootstrap targets (e.g. DQN, VecDQN).

    **Usage:** build ``self.online`` in ``__init__``, then call
    :meth:`_init_target` to create and freeze the target copy automatically.
    :meth:`target_forward` and :meth:`polyak_update` are provided by default
    and operate on ``self.online`` / ``self.target`` — override them only if
    your head needs custom reshaping or update logic.

    Example::

        class MyTwinHead(BaseHeadWithTarget):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.online = nn.Linear(in_features, out_features)
                self._init_target(self.online)

            def forward(self, h):
                return self.online(h)
    """

    online: nn.Module
    target: nn.Module

    def _init_target(self, online: nn.Module) -> None:
        """Deep-copy ``online`` into ``self.target``, freeze it, and sync weights.

        Args:
            online: The online network to copy (typically ``self.online``).
        """
        self.target = copy.deepcopy(online)
        self.target.requires_grad_(False)
        self.polyak_update(tau=1.0)

    def target_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Run the target network with no gradient tracking.

        Args:
            h: Pooled step representations ``[B, S, D]``.

        Returns:
            Output tensor from the target network.
        """
        with torch.no_grad():
            return self.target(h)

    def polyak_update(self, tau: float) -> None:
        """Soft-update target toward online: θ_target ← τ·θ_online + (1−τ)·θ_target.

        Args:
            tau: Interpolation factor in ``[0, 1]``.  ``1.0`` copies online
                 weights into the target; ``0.0`` leaves the target unchanged.
        """
        if tau <= 0.0:
            return
        for online_p, target_p in zip(self.online.parameters(), self.target.parameters()):
            target_p.data.copy_(tau * online_p.data + (1.0 - tau) * target_p.data)
