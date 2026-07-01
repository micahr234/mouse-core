"""ModernBERT encoder backbone (bidirectional attention).

Provides :class:`ModernBertBackbone`, a thin Backbone adapter around
``transformers.ModernBertModel`` for use with the generic
:class:`~mouse_core.models.base.Model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import ModernBertConfig, ModernBertModel

from mouse_core.models.backbone.base import Backbone, _quiet_transformers_load

DEFAULT_PRETRAINED = "answerdotai/ModernBERT-large"


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend to avoid driver-specific errors."""
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


def _load_modernbert_weights(
    model: nn.Module,
    repo_id_or_path: str | Path,
    *,
    hub_kwargs: dict[str, Any],
) -> None:
    """Load matching ModernBERT encoder weights into a MOUSE backbone."""
    from transformers import AutoModel

    with _quiet_transformers_load():
        pretrained = AutoModel.from_pretrained(repo_id_or_path, **hub_kwargs)
    target_state = model.state_dict()
    loadable = {
        key: value
        for key, value in pretrained.state_dict().items()
        if key in target_state
        and target_state[key].shape == value.shape
        and not key.startswith("embeddings.")
    }
    model.load_state_dict(loadable, strict=False)

    del pretrained


@dataclass
class _ModernBertBackboneConfig:
    """Configuration for a ModernBERT encoder backbone.

    Builds a HuggingFace ``ModernBertModel`` with SDPA attention and no token
    embeddings (MOUSE supplies ``inputs_embeds`` externally).

    Args:
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        max_position_embeddings: Maximum sequence length; should be at least
            ``sequence_length * tokens_per_step``.
        expand: FFN intermediate size multiplier:
            ``intermediate_size = hidden_dim * expand``.
        intermediate_size: Exact FFN size; overrides ``expand * hidden_dim`` when set.
        local_attention: Sliding-window size for local-attention layers.
        layer_types: Per-layer attention pattern (``full_attention`` or
            ``sliding_attention``). When ``None``, alternates every three layers.
        rope_parameters: Optional RoPE settings keyed by attention type.
        norm_eps: Epsilon for normalization layers.
        attention_dropout: Dropout on attention probabilities.
        mlp_dropout: Dropout on MLP activations.
        embedding_dropout: Dropout on embeddings.
        attention_bias: Whether to add bias to attention projections.
        mlp_bias: Whether to add bias to MLP projections.
        norm_bias: Whether to add bias to normalization layers.
        hidden_activation: Activation used in the MLP blocks.
    """

    num_layers: int
    num_heads: int
    max_position_embeddings: int = 8192
    expand: int = 4
    intermediate_size: int | None = None
    local_attention: int = 128
    layer_types: list[str] | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    embedding_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    norm_bias: bool = False
    hidden_activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}.")
        self.num_heads = int(self.num_heads)

    def build(self, hidden_dim: int) -> ModernBertModel:
        """Instantiate a ``ModernBertModel`` with this config."""
        _disable_cudnn_sdp()
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({self.num_heads})."
            )
        ffn_size = self.intermediate_size if self.intermediate_size is not None else hidden_dim * self.expand
        config = ModernBertConfig(
            vocab_size=1,
            hidden_size=hidden_dim,
            num_attention_heads=self.num_heads,
            intermediate_size=ffn_size,
            max_position_embeddings=self.max_position_embeddings,
            num_hidden_layers=self.num_layers,
            local_attention=self.local_attention,
            layer_types=self.layer_types,
            rope_parameters=self.rope_parameters,
            norm_eps=self.norm_eps,
            attention_dropout=self.attention_dropout,
            mlp_dropout=self.mlp_dropout,
            embedding_dropout=self.embedding_dropout,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            norm_bias=self.norm_bias,
            hidden_activation=self.hidden_activation,
            pad_token_id=0,
            bos_token_id=None,
            eos_token_id=None,
            cls_token_id=None,
            sep_token_id=None,
        )
        config._attn_implementation = "sdpa"
        model = ModernBertModel(config)
        model.final_norm = nn.Identity()  # type: ignore[assignment]
        return model


class ModernBertBackbone(Backbone):
    """Backbone adapter wrapping a bidirectional ``transformers.ModernBertModel``.

    Construct directly from config args::

           backbone = ModernBertBackbone(
               hidden_dim=128,
               num_layers=4,
               num_heads=4,
               max_position_embeddings=256,
           )

    Or load architecture and encoder weights from a pretrained ModernBERT repo::

           backbone = ModernBertBackbone(
               pretrained="answerdotai/ModernBERT-large",
               num_layers=12,
           )

    Bidirectional encoders attend to the full sequence on every forward pass.
    KV caching is not supported; pass ``use_cache=False`` (the default).
    """

    def __init__(
        self,
        model: ModernBertModel | None = None,
        *,
        hidden_dim: int | None = None,
        pretrained: str | Path | None = None,
        load_weights: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> None:
        super().__init__()

        if model is not None and pretrained is not None:
            raise TypeError("ModernBertBackbone accepts either model= or pretrained=, not both.")

        if model is not None:
            if not isinstance(model, ModernBertModel):
                raise TypeError(
                    "When passing a model to ModernBertBackbone, it must be a "
                    "transformers.ModernBertModel (with vocab_size=1 and final_norm=Identity)."
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
            self.model = _ModernBertBackboneConfig(**extracted_kwargs).build(extracted_hidden_dim)
            self._config_kwargs = dict(extracted_kwargs)
            if load_weights:
                self._load_pretrained_weights(pretrained, hub_kwargs=hf_kwargs)
            return

        if hidden_dim is None:
            raise TypeError(
                "ModernBertBackbone requires either a pre-built model, "
                "pretrained=, or hidden_dim plus backbone config arguments "
                "(e.g. ModernBertBackbone(hidden_dim=128, num_layers=2, num_heads=4))."
            )

        cfg = _ModernBertBackboneConfig(**config_kwargs)
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
        if model_type != "modernbert":
            raise ValueError(
                f"ModernBertBackbone can only load ModernBERT configs, got model_type={model_type!r} "
                f"from {repo_id_or_path!r}."
            )

        backbone_kwargs: dict[str, Any] = dict(
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=hf_cfg.num_attention_heads,
            max_position_embeddings=hf_cfg.max_position_embeddings,
            intermediate_size=hf_cfg.intermediate_size,
            local_attention=getattr(hf_cfg, "local_attention", 128),
            layer_types=list(getattr(hf_cfg, "layer_types", None) or []),
            rope_parameters=getattr(hf_cfg, "rope_parameters", None),
            norm_eps=getattr(hf_cfg, "norm_eps", 1e-5),
            attention_dropout=getattr(hf_cfg, "attention_dropout", 0.0),
            mlp_dropout=getattr(hf_cfg, "mlp_dropout", 0.0),
            embedding_dropout=getattr(hf_cfg, "embedding_dropout", 0.0),
            attention_bias=getattr(hf_cfg, "attention_bias", False),
            mlp_bias=getattr(hf_cfg, "mlp_bias", False),
            norm_bias=getattr(hf_cfg, "norm_bias", False),
            hidden_activation=getattr(hf_cfg, "hidden_activation", "gelu"),
        )
        if not backbone_kwargs["layer_types"]:
            backbone_kwargs["layer_types"] = None
        backbone_kwargs.update(overrides)
        num_layers = int(backbone_kwargs["num_layers"])
        layer_types = backbone_kwargs.get("layer_types")
        if layer_types is not None and len(layer_types) != num_layers:
            backbone_kwargs["layer_types"] = layer_types[:num_layers]
        return backbone_kwargs, int(hf_cfg.hidden_size)

    @staticmethod
    def _config_kwargs_from_model(model: ModernBertModel) -> dict[str, Any]:
        cfg = model.config
        layer_types = list(getattr(cfg, "layer_types", None) or [])
        return dict(
            num_layers=len(model.layers),
            num_heads=cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            intermediate_size=cfg.intermediate_size,
            local_attention=getattr(cfg, "local_attention", 128),
            layer_types=layer_types or None,
            rope_parameters=getattr(cfg, "rope_parameters", None),
            norm_eps=getattr(cfg, "norm_eps", 1e-5),
            attention_dropout=getattr(cfg, "attention_dropout", 0.0),
            mlp_dropout=getattr(cfg, "mlp_dropout", 0.0),
            embedding_dropout=getattr(cfg, "embedding_dropout", 0.0),
            attention_bias=getattr(cfg, "attention_bias", False),
            mlp_bias=getattr(cfg, "mlp_bias", False),
            norm_bias=getattr(cfg, "norm_bias", False),
            hidden_activation=getattr(cfg, "hidden_activation", "gelu"),
        )

    def _load_pretrained_weights(
        self,
        repo_id_or_path: str | Path,
        *,
        hub_kwargs: dict[str, Any],
    ) -> None:
        _load_modernbert_weights(
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
        if use_cache:
            raise ValueError(
                "ModernBertBackbone uses bidirectional attention and does not support KV caching. "
                "Run full-sequence forwards with use_cache=False."
            )
        if cache is not None:
            raise ValueError(
                "ModernBertBackbone does not accept a KV cache. Pass cache=None and "
                "reprocess the full context on each forward."
            )

        output_hidden_states = bool(kwargs.pop("output_hidden_states", False))

        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        if output_hidden_states:
            if out.hidden_states is None:
                raise RuntimeError(
                    "ModernBertBackbone expected hidden_states but the model returned None."
                )
            return out.last_hidden_state, None, out.hidden_states[1:]
        return out.last_hidden_state, None
