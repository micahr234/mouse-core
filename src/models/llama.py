"""ModelLlama: single-backbone model that processes all tokens in one pass."""

from __future__ import annotations

from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin

from mouse.models.base import Model, MODEL_CARD_TEMPLATE
from mouse.models.backbone import LlamaBackboneConfig
from mouse.models.base import TokenType


class ModelLlama(Model, PyTorchModelHubMixin, library_name="MOUSE", tags=["backbone:llama"], model_card_template=MODEL_CARD_TEMPLATE):
    """SAL model with a single Llama backbone that attends over the full token sequence."""

    def _init_backbone(self, backbone_kwargs: dict) -> None:
        cfg = LlamaBackboneConfig(**backbone_kwargs)
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.max_position_embeddings = cfg.max_position_embeddings
        self.expand = cfg.expand
        self.backbone = cfg.build(self.hidden_dim)

    def backbone_forward(
        self,
        embeds: torch.Tensor,
        token_type: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run the single backbone over all tokens and return hidden states."""
        cache = cache or {}

        has_padding = bool((token_type == TokenType.PAD).any())
        attention_mask = (token_type != TokenType.PAD).long() if has_padding else None

        out = self.backbone(
            inputs_embeds=embeds,
            past_key_values=cache.get("backbone", None),
            use_cache=use_cache,
            position_ids=None,
            attention_mask=attention_mask,
            **kwargs,
        )

        new_cache = {"backbone": out.past_key_values} if use_cache else None
        return out.last_hidden_state, new_cache
