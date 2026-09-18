"""Qwen3 decoder stack builder used by :class:`TransformerBackbone`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import Qwen3Config, Qwen3Model

from mouse_core.models.backbone.base import _disable_cudnn_sdp


@dataclass
class _Qwen3BackboneConfig:
    """Configuration for a Qwen3 transformer stack.

    Builds a HuggingFace ``Qwen3Model`` with SDPA attention. ``vocab_size=``
    is the pretrained vocab, or a stub of 1 when the table is unused.
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

    def build(self, hidden_dim: int, *, vocab_size: int) -> Qwen3Model:
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
            vocab_size=int(vocab_size),
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
        return Qwen3Model(config)


def qwen3_config_kwargs(model: Any) -> dict[str, Any]:
    cfg = model.config
    kwargs: dict[str, Any] = dict(
        num_layers=len(model.layers),
        num_heads=cfg.num_attention_heads,
        num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=getattr(cfg, "head_dim", None),
        max_position_embeddings=cfg.max_position_embeddings,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
        attention_bias=getattr(cfg, "attention_bias", False),
        use_sliding_window=getattr(cfg, "use_sliding_window", False),
        vocab_size=int(cfg.vocab_size),
    )
    rope_parameters = getattr(cfg, "rope_parameters", None)
    if rope_parameters is not None:
        kwargs["rope_parameters"] = rope_parameters
    return kwargs
