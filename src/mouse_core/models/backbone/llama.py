"""ModelLlama: MOUSE model with a Llama transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import LlamaConfig, LlamaModel

from mouse_core.models.base import Model, MODEL_CARD_TEMPLATE
from mouse_core.models.base import TokenType


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend to avoid driver-specific errors."""
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


@dataclass
class LlamaBackboneConfig:
    """Configuration for a Llama transformer backbone.

    Builds a HuggingFace ``LlamaModel`` with SDPA attention and no token
    embedding or final layer norm (norm is replaced with ``nn.Identity``).

    Args:
        num_layers: Number of transformer decoder layers.
        num_heads: Number of query attention heads.
        num_key_value_heads: Key/value heads for GQA; defaults to ``num_heads``.
        max_position_embeddings: Maximum sequence length for RoPE; should be at
            least ``sequence_length * tokens_per_step``.
        expand: FFN intermediate size multiplier: ``intermediate_size = hidden_dim * expand``.
        intermediate_size: Exact FFN size; overrides ``expand * hidden_dim`` when set.
            Use this when loading from a pretrained model whose FFN size is not
            an integer multiple of the hidden dim.
        rope_parameters: Optional dict forwarded to ``LlamaConfig.rope_parameters``
            for custom RoPE variants (e.g. ``{"rope_type": "llama3"}``).
        rms_norm_eps: Epsilon for RMSNorm layers.
        attention_bias: Whether to add bias to QKV and output projections.
    """

    num_layers: int
    num_heads: int
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 4096
    expand: int = 4
    intermediate_size: int | None = None
    rope_parameters: dict | None = None
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}.")
        self.num_heads = int(self.num_heads)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        else:
            self.num_key_value_heads = int(self.num_key_value_heads)

    def build(self, hidden_dim: int) -> LlamaModel:
        """Instantiate a ``LlamaModel`` with this config.

        Args:
            hidden_dim: Model hidden dimension ``D``; must be divisible by ``num_heads``.

        Returns:
            ``LlamaModel`` with the final norm replaced by ``nn.Identity``.
        """
        _disable_cudnn_sdp()
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        ffn_size = self.intermediate_size if self.intermediate_size is not None else hidden_dim * self.expand
        config_kwargs: dict = dict(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=ffn_size,
            max_position_embeddings=self.max_position_embeddings,
            attention_dropout=0.0,
            attention_bias=self.attention_bias,
            rms_norm_eps=self.rms_norm_eps,
            num_hidden_layers=self.num_layers,
        )
        if self.rope_parameters is not None:
            config_kwargs["rope_parameters"] = self.rope_parameters
        config = LlamaConfig(**config_kwargs)
        config._attn_implementation = "sdpa"
        model = LlamaModel(config)
        model.norm = nn.Identity()  # type: ignore[assignment]
        return model


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
