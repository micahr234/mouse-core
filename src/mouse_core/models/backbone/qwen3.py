"""ModelQwen3: MOUSE model with a Qwen3 transformer backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import Qwen3Config, Qwen3Model

from mouse_core.models.base import Model, MODEL_CARD_TEMPLATE
from mouse_core.models.base import TokenType


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend to avoid driver-specific errors."""
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


@dataclass
class Qwen3BackboneConfig:
    """Configuration for a Qwen3 transformer backbone.

    Builds a HuggingFace ``Qwen3Model`` with SDPA attention and no token
    embedding or final layer norm (norm is replaced with ``nn.Identity``).

    Args:
        num_layers: Number of transformer decoder layers.
        num_heads: Number of query attention heads.
        num_key_value_heads: Key/value heads for GQA; defaults to ``num_heads``.
        head_dim: Per-head attention dimension. When ``None``, defaults to
            ``hidden_dim // num_heads``. Set explicitly to decouple model width
            from attention head size (useful for GQA with small ``num_key_value_heads``).
        max_position_embeddings: Maximum sequence length for RoPE.
        expand: FFN intermediate size multiplier: ``intermediate_size = hidden_dim * expand``.
        intermediate_size: Exact FFN size; overrides ``expand * hidden_dim`` when set.
            Use this when loading from a pretrained model whose FFN size is not
            an integer multiple of the hidden dim.
        rope_parameters: Optional dict forwarded to ``Qwen3Config.rope_parameters``.
        rms_norm_eps: Epsilon for RMSNorm layers.
        attention_bias: Whether to add bias to QKV and output projections.
        use_sliding_window: Enable sliding-window attention (Qwen3 feature).
    """

    num_layers: int
    num_heads: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    expand: int = 3
    intermediate_size: int | None = None
    rope_parameters: dict | None = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    use_sliding_window: bool = False

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}.")
        self.num_heads = int(self.num_heads)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        else:
            self.num_key_value_heads = int(self.num_key_value_heads)

    def build(self, hidden_dim: int) -> Qwen3Model:
        """Instantiate a ``Qwen3Model`` with this config.

        Args:
            hidden_dim: Model hidden dimension ``D``. When ``head_dim`` is ``None``,
                must be divisible by ``num_heads``.

        Returns:
            ``Qwen3Model`` with the final norm replaced by ``nn.Identity``.
        """
        _disable_cudnn_sdp()
        if self.head_dim is None:
            if hidden_dim % self.num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})."
                )
            resolved_head_dim = hidden_dim // self.num_heads
        else:
            resolved_head_dim = int(self.head_dim)
        ffn_size = self.intermediate_size if self.intermediate_size is not None else hidden_dim * self.expand
        config_kwargs: dict = dict(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=resolved_head_dim,
            intermediate_size=ffn_size,
            max_position_embeddings=self.max_position_embeddings,
            attention_dropout=0.0,
            attention_bias=self.attention_bias,
            rms_norm_eps=self.rms_norm_eps,
            num_hidden_layers=self.num_layers,
            use_sliding_window=self.use_sliding_window,
        )
        if self.rope_parameters is not None:
            config_kwargs["rope_parameters"] = self.rope_parameters
        config = Qwen3Config(**config_kwargs)
        config._attn_implementation = "sdpa"
        model = Qwen3Model(config)
        model.norm = nn.Identity()  # type: ignore[assignment]
        return model


class ModelQwen3(Model, PyTorchModelHubMixin, library_name="MOUSE", tags=["backbone:qwen3"], model_card_template=MODEL_CARD_TEMPLATE):
    """MOUSE model with a Qwen3 transformer backbone.

    Attends over the full ``[B, S*T, D]`` token sequence with causal SDPA.
    Supports an explicit ``head_dim`` (set in ``backbone_kwargs``) for grouped-query
    attention with a head size independent of the model width. Supports KV-cache
    for incremental rollouts (``use_cache=True``).
    """

    def _init_backbone(self, backbone_kwargs: dict) -> None:
        """Build and assign ``self.backbone`` from ``backbone_kwargs``."""
        cfg = Qwen3BackboneConfig(**backbone_kwargs)
        self.num_layers = cfg.num_layers
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
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
        """Run the Qwen3 backbone over the token sequence.

        Args:
            embeds: ``[B, T_total, D]`` embedding tensor from ``StepEmbedder``.
            token_type: ``[B, T_total]`` int64 ``TokenType`` ids; ``PAD`` positions
                are masked out from attention.
            cache: KV-cache dict from a previous call, or ``None`` for full prefill.
                Reads and writes the ``"backbone"`` key.
            use_cache: If ``True``, return an updated KV-cache dict.
            cache_position: Unused; present for interface compatibility.
            **kwargs: Forwarded to the underlying ``Qwen3Model``.

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
