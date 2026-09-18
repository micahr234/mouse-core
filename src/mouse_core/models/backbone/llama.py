"""Llama decoder stack builder used by :class:`TransformerBackbone`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import LlamaConfig, LlamaModel

from mouse_core.models.backbone.base import _disable_cudnn_sdp


@dataclass
class _LlamaBackboneConfig:
    """Configuration for a Llama transformer stack.

    Builds a HuggingFace ``LlamaModel`` with SDPA attention. ``vocab_size=``
    is the pretrained vocab, or a stub of 1 when the table is unused.
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

    def build(self, hidden_dim: int, *, vocab_size: int) -> LlamaModel:
        _disable_cudnn_sdp()
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        ffn_size = self.intermediate_size if self.intermediate_size is not None else hidden_dim * self.expand
        config_kwargs: dict = dict(
            vocab_size=int(vocab_size),
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
        return LlamaModel(config)


def llama_config_kwargs(model: Any) -> dict[str, Any]:
    cfg = model.config
    kwargs: dict[str, Any] = dict(
        num_layers=len(model.layers),
        num_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        max_position_embeddings=cfg.max_position_embeddings,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-5),
        attention_bias=getattr(cfg, "attention_bias", False),
        vocab_size=int(cfg.vocab_size),
    )
    rope_parameters = getattr(cfg, "rope_parameters", None)
    if rope_parameters is not None:
        kwargs["rope_parameters"] = rope_parameters
    return kwargs
