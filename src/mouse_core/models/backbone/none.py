"""Identity backbone for no-backbone ablations."""

from __future__ import annotations

from typing import Any

import torch

from mouse_core.models.backbone.base import BaseBackbone


class IdentityBackbone(BaseBackbone):
    """A parameterless no-op over already-embedded inputs, without a cache."""

    def __init__(self, *, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        self._hidden_dim = int(hidden_dim)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def _forward_dense(
        self,
        embeds: torch.Tensor,
        *,
        output_hidden_states: bool,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if output_hidden_states:
            return embeds, (embeds,)
        return embeds
