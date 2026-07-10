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
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if output_hidden_states:
            return embeds, (embeds,)
        return embeds
