"""Identity backbone for no-backbone ablations."""

from __future__ import annotations

from typing import Any

import torch

from mouse_core.models.backbone.base import Backbone


class IdentityBackbone(Backbone):
    """A no-op backbone: returns inputs unchanged and never produces a cache.

    It can optionally be told a ``hidden_dim`` so that ``Model`` (and users)
    can perform a consistency check against the encoder's hidden dimension.
    """

    def __init__(self, hidden_dim: int | None = None) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim

    @property
    def hidden_dim(self) -> int | None:
        return self._hidden_dim

    def forward(
        self,
        embeds: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        output_hidden_states = bool(kwargs.pop("output_hidden_states", False))
        if output_hidden_states:
            return embeds, None, (embeds,)
        return embeds, None
