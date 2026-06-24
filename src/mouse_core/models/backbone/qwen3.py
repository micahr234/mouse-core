"""Qwen3 transformer backbone.

Provides :class:`Qwen3Backbone`, a thin Backbone adapter around
``transformers.Qwen3Model`` for use with the generic :class:`~mouse_core.models.base.Model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

from mouse_core.models.backbone.base import Backbone, _load_transformer_weights


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend to avoid driver-specific errors."""
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


@dataclass
class _Qwen3BackboneConfig:
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


class Qwen3Backbone(Backbone):
    """Backbone adapter wrapping a ``transformers.Qwen3Model``."""

    def __init__(
        self,
        model: Qwen3Model | None = None,
        *,
        hidden_dim: int | None = None,
        pretrained: str | Path | None = None,
        load_weights: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> None:
        super().__init__()
        if model is not None and pretrained is not None:
            raise TypeError("Qwen3Backbone accepts either model= or pretrained=, not both.")

        if model is not None:
            if not isinstance(model, Qwen3Model):
                raise TypeError(
                    "When passing a model to Qwen3Backbone, it must be a "
                    "transformers.Qwen3Model (with vocab_size=1 and norm=Identity)."
                )
            self.model = model
            self._config_kwargs = self._config_kwargs_from_model(model)
            return

        if pretrained is not None:
            hf_kwargs = hub_kwargs or {}
            extracted_kwargs, extracted_hidden_dim = self._config_from_pretrained(
                pretrained,
                hub_kwargs=hf_kwargs,
                overrides=config_kwargs,
            )
            if hidden_dim is not None and int(hidden_dim) != extracted_hidden_dim:
                raise ValueError(
                    f"hidden_dim={hidden_dim} does not match pretrained hidden size "
                    f"{extracted_hidden_dim} from {pretrained!r}."
                )
            self.model = _Qwen3BackboneConfig(**extracted_kwargs).build(extracted_hidden_dim)
            self._config_kwargs = dict(extracted_kwargs)
            if load_weights:
                self._load_pretrained_weights(pretrained, hub_kwargs=hf_kwargs)
            return

        if hidden_dim is None:
            raise TypeError(
                "Qwen3Backbone requires either a pre-built model, "
                "pretrained=, or hidden_dim plus backbone config arguments "
                "(e.g. Qwen3Backbone(hidden_dim=128, num_layers=2, num_heads=4))."
            )

        self.model = _Qwen3BackboneConfig(**config_kwargs).build(hidden_dim)
        self._config_kwargs = self._config_kwargs_from_model(self.model)

    @staticmethod
    def _config_from_pretrained(
        repo_id_or_path: str | Path,
        *,
        hub_kwargs: dict[str, Any],
        overrides: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(repo_id_or_path, **hub_kwargs)
        model_type = getattr(hf_cfg, "model_type", "").lower()
        if "qwen3" not in model_type and "qwen" not in model_type:
            raise ValueError(
                f"Qwen3Backbone can only load Qwen configs, got model_type={model_type!r} "
                f"from {repo_id_or_path!r}."
            )

        backbone_kwargs: dict[str, Any] = dict(
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=hf_cfg.num_attention_heads,
            num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
            head_dim=getattr(hf_cfg, "head_dim", None),
            max_position_embeddings=hf_cfg.max_position_embeddings,
            intermediate_size=hf_cfg.intermediate_size,
            rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
            attention_bias=getattr(hf_cfg, "attention_bias", False),
            use_sliding_window=getattr(hf_cfg, "use_sliding_window", False),
        )
        backbone_kwargs.update(overrides)
        return backbone_kwargs, int(hf_cfg.hidden_size)

    @staticmethod
    def _config_kwargs_from_model(model: Qwen3Model) -> dict[str, Any]:
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
        )
        rope_parameters = getattr(cfg, "rope_parameters", None)
        if rope_parameters is not None:
            kwargs["rope_parameters"] = rope_parameters
        return kwargs

    def _load_pretrained_weights(
        self,
        repo_id_or_path: str | Path,
        *,
        hub_kwargs: dict[str, Any],
    ) -> None:
        _load_transformer_weights(
            self.model,
            repo_id_or_path,
            hub_kwargs=hub_kwargs,
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(
        self,
        embeds: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        cache = cache or {}

        out = self.model(
            inputs_embeds=embeds,
            past_key_values=cache.get("backbone", None),
            use_cache=use_cache,
            position_ids=None,
            attention_mask=attention_mask,
            **kwargs,
        )

        new_cache = {"backbone": out.past_key_values} if use_cache else None
        return out.last_hidden_state, new_cache

