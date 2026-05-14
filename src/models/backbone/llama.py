"""ModelLlama: single-backbone model that processes all tokens in one pass."""

from __future__ import annotations

from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin

from mouse.models.base import Model, MODEL_CARD_TEMPLATE
from mouse.models.backbone.configs import LlamaBackboneConfig
from mouse.models.base import TokenType


class ModelLlama(Model, PyTorchModelHubMixin, library_name="MOUSE", tags=["backbone:llama"], model_card_template=MODEL_CARD_TEMPLATE):
    """MOUSE model with a Llama transformer backbone.

    Attends over the full ``[B, S*T, D]`` token sequence with causal SDPA.
    Supports KV-cache for incremental rollouts (``use_cache=True``).
    """

    def _init_backbone(self, backbone_kwargs: dict) -> None:
        """Build and assign ``self.backbone`` from ``backbone_kwargs``."""
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
        """Run the Llama backbone over the token sequence.

        Args:
            embeds: ``[B, T_total, D]`` embedding tensor from ``StepEmbedder``.
            token_type: ``[B, T_total]`` int64 ``TokenType`` ids; ``PAD`` positions
                are masked out from attention.
            cache: KV-cache dict from a previous call, or ``None`` for full prefill.
                Reads and writes the ``"backbone"`` key.
            use_cache: If ``True``, return an updated KV-cache dict.
            cache_position: Unused; present for interface compatibility.
            **kwargs: Forwarded to the underlying ``LlamaModel``.

        Returns:
            Tuple of ``(hidden_states [B, T_total, D], cache_dict | None)``.
        """
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
