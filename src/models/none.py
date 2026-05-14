"""ModelNone: no-backbone model that feeds embeddings directly to the heads."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from mouse.models.base import Model, MODEL_CARD_TEMPLATE


class ModelNone(Model, PyTorchModelHubMixin, library_name="MOUSE", tags=["backbone:none"], model_card_template=MODEL_CARD_TEMPLATE):
    """SAL model with no backbone; embeddings pass directly to the output heads."""

    def _init_backbone(self, backbone_kwargs: dict) -> None:
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
        return embeds, None
