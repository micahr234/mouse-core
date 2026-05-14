"""Backbone config dataclasses: validate params and build transformer backbone modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel, Qwen3Config, Qwen3Model


def _disable_cudnn_sdp_attention_backend() -> None:
    """Turn off cuDNN as an SDPA backend (safe to call before CUDA is initialized).

    PyTorch may otherwise pick cuDNN scaled-dot-product attention, which can raise
    ``RuntimeError: cuDNN Frontend error: ... No valid execution plans built`` on
    some GPU / driver / toolkit combinations (HF ``attn_implementation="sdpa"``).
    Flash or math SDPA backends are used instead. Call from backbone ``build`` even
    when the module is first created on CPU and moved to GPU later.
    """
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


@dataclass
class LlamaBackboneConfig:
    num_layers: int
    num_heads: int
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 4096
    expand: int = 4
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
        _disable_cudnn_sdp_attention_backend()
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        config_kwargs: dict = dict(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=hidden_dim * self.expand,
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


@dataclass
class Qwen3BackboneConfig:
    num_layers: int
    num_heads: int
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    max_position_embeddings: int = 32768
    expand: int = 3
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
        _disable_cudnn_sdp_attention_backend()
        if self.head_dim is None:
            if hidden_dim % self.num_heads != 0:
                raise ValueError(
                    f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})."
                )
            resolved_head_dim = hidden_dim // self.num_heads
        else:
            resolved_head_dim = int(self.head_dim)
        config_kwargs: dict = dict(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=resolved_head_dim,
            intermediate_size=hidden_dim * self.expand,
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
