"""Llama transformer backbone.

Provides :class:`LlamaBackbone`, a thin Backbone adapter around
``transformers.LlamaModel`` for use with the generic :class:`~mouse_core.models.base.Model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel

from mouse_core.models.backbone.base import Backbone, _load_transformer_weights


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend to avoid driver-specific errors."""
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


@dataclass
class _LlamaBackboneConfig:
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


class LlamaBackbone(Backbone):
    """Backbone adapter wrapping a ``transformers.LlamaModel``.

    Construct directly from config args::

           backbone = LlamaBackbone(
               hidden_dim=128,
               num_layers=4,
               num_heads=4,
               max_position_embeddings=256,
           )

    Or load architecture and transformer weights from a pretrained Llama repo::

           backbone = LlamaBackbone(
               pretrained="meta-llama/Llama-3.2-1B",
               num_layers=2,
           )

    The adapter translates the generic MOUSE call into the HF calling
    convention and forwards an optional explicit ``attention_mask``.
    """

    def __init__(
        self,
        model: LlamaModel | None = None,
        *,
        hidden_dim: int | None = None,
        pretrained: str | Path | None = None,
        load_weights: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> None:
        super().__init__()

        if model is not None and pretrained is not None:
            raise TypeError("LlamaBackbone accepts either model= or pretrained=, not both.")

        if model is not None:
            if not isinstance(model, LlamaModel):
                raise TypeError(
                    "When passing a model to LlamaBackbone, it must be a "
                    "transformers.LlamaModel (with vocab_size=1 and norm=Identity)."
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
            self.model = _LlamaBackboneConfig(**extracted_kwargs).build(extracted_hidden_dim)
            self._config_kwargs = dict(extracted_kwargs)
            if load_weights:
                self._load_pretrained_weights(pretrained, hub_kwargs=hf_kwargs)
            return

        if hidden_dim is None:
            raise TypeError(
                "LlamaBackbone requires either a pre-built model, "
                "pretrained=, or hidden_dim plus backbone config arguments "
                "(e.g. LlamaBackbone(hidden_dim=128, num_layers=2, num_heads=4))."
            )

        cfg = _LlamaBackboneConfig(**config_kwargs)
        self.model = cfg.build(hidden_dim)
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
        if "llama" not in model_type:
            raise ValueError(
                f"LlamaBackbone can only load Llama configs, got model_type={model_type!r} "
                f"from {repo_id_or_path!r}."
            )

        backbone_kwargs: dict[str, Any] = dict(
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=hf_cfg.num_attention_heads,
            num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
            max_position_embeddings=hf_cfg.max_position_embeddings,
            intermediate_size=hf_cfg.intermediate_size,
            rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-5),
            attention_bias=getattr(hf_cfg, "attention_bias", False),
        )
        backbone_kwargs.update(overrides)
        return backbone_kwargs, int(hf_cfg.hidden_size)

    @staticmethod
    def _config_kwargs_from_model(model: LlamaModel) -> dict[str, Any]:
        cfg = model.config
        kwargs: dict[str, Any] = dict(
            num_layers=len(model.layers),
            num_heads=cfg.num_attention_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            max_position_embeddings=cfg.max_position_embeddings,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-5),
            attention_bias=getattr(cfg, "attention_bias", False),
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

        output_hidden_states = bool(kwargs.pop("output_hidden_states", False))

        out = self.model(
            inputs_embeds=embeds,
            past_key_values=cache.get("backbone", None),
            use_cache=use_cache,
            position_ids=None,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        new_cache = {"backbone": out.past_key_values} if use_cache else None
        if output_hidden_states:
            if out.hidden_states is None:
                raise RuntimeError("LlamaBackbone expected hidden_states but the model returned None.")
            return out.last_hidden_state, new_cache, out.hidden_states[1:]
        return out.last_hidden_state, new_cache

