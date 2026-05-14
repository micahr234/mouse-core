"""ModelNone: no-backbone model that feeds embeddings directly to the heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from mouse.models.base import Model, MODEL_CARD_TEMPLATE


class ModelNone(Model, PyTorchModelHubMixin, library_name="MOUSE", tags=["backbone:none"], model_card_template=MODEL_CARD_TEMPLATE):
    """MOUSE model with no backbone; embeddings pass directly to the output heads.

    Useful for ablations or lightweight baselines where no temporal context is
    required. ``backbone_kwargs`` must be empty (or absent) in ``config.json``.
    KV-cache is not supported — always returns ``None`` for the cache.
    """

    def _init_backbone(self, backbone_kwargs: dict) -> None:
        """Assign an ``nn.Identity`` backbone (no-op pass-through)."""
        self.backbone = nn.Identity()

    def backbone_forward(
        self,
        embeds: torch.Tensor,
        token_type: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Pass embeddings through unchanged; always returns ``None`` for cache.

        Args:
            embeds: ``[B, T_total, D]`` embedding tensor.
            token_type: Ignored.
            cache: Ignored.
            use_cache: Ignored.
            cache_position: Ignored.

        Returns:
            Tuple of ``(embeds unchanged, None)``.
        """
        return embeds, None
