"""Single public transformer backbone.

``TransformerBackbone`` is the one way to attach a decoder stack. Construction
is either ``pretrained=`` (Hub / local checkpoint: inspect, then packed kernels
or HuggingFace ``forward``) or ``architecture=`` plus width (a llama / qwen3
stack from scratch, or ``architecture="hf"`` with a saved ``hf_config``).

Packed Flex / varlen / padded kernels run when every layer is a Llama/Qwen3-
shaped softmax block. Other softmax decoders use HuggingFace ``forward`` with
the grouping mask. Hybrid / linear / sliding-window stacks raise — grouping
would be silently wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from mouse_core.data.token_batch import TokenBatch
from mouse_core.models.backbone.base import (
    Backbone,
    DecodeKernel,
    TrainKernel,
    _apply_final_norm,
    _load_transformer_weights,
    _rope_parameters_from_config,
)
from mouse_core.models.backbone.packed_train import _DecoderStack
from mouse_core.models.backbone.flex_decode import FlexDecodeSession
from mouse_core.models.backbone.llama import _LlamaBackboneConfig, llama_config_kwargs
from mouse_core.models.backbone.qwen3 import _Qwen3BackboneConfig, qwen3_config_kwargs
from mouse_core.models.backbone.embed import embed_token_ids
from mouse_core.models.lora import LoRAConfig

Architecture = Literal["llama", "qwen3", "hf"]


def _as_module(model: _DecoderStack) -> nn.Module:
    """View a decoder stack as ``nn.Module`` for helpers typed that way."""
    return cast(nn.Module, model)

_SOFTMAX_LAYER_TYPES = frozenset({"full_attention", "attention"})
_HYBRID_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _text_cfg(cfg: Any) -> Any:
    text = _cfg_get(cfg, "text_config")
    return text if text is not None else cfg


def grouping_unsupported_reason(cfg: Any) -> str | None:
    """Why this config cannot guarantee ``(sequence_id, grouping_id)`` isolation.

    Returns ``None`` when the stack is either packed-compatible or a maskable
    softmax decoder. Hybrid / linear / sliding-window / encoder-decoder
    configs return a reason string.
    """
    text = _text_cfg(cfg)
    model_type = str(_cfg_get(cfg, "model_type") or _cfg_get(text, "model_type") or "").lower()
    if model_type in _HYBRID_MODEL_TYPES:
        return f"model_type={model_type!r} is a hybrid linear/full-attention stack"
    if _cfg_get(text, "use_sliding_window", False) or _cfg_get(cfg, "use_sliding_window", False):
        return "sliding-window attention"
    layer_types = _cfg_get(text, "layer_types") or _cfg_get(cfg, "layer_types")
    if layer_types:
        bad = [t for t in layer_types if t not in _SOFTMAX_LAYER_TYPES]
        if bad:
            return f"non-softmax layer types {bad}"
    if _cfg_get(cfg, "is_encoder_decoder", False) or _cfg_get(text, "is_encoder_decoder", False):
        return "encoder-decoder"
    return None


def is_packed_compatible(model: nn.Module) -> bool:
    """True when ``packed_forward`` / ``FlexDecodeSession`` can walk this stack."""
    if grouping_unsupported_reason(getattr(model, "config", None)) is not None:
        return False
    if not all(hasattr(model, name) for name in ("layers", "rotary_emb", "norm")):
        return False
    layers = getattr(model, "layers")
    if layers is None or len(layers) < 1:
        return False
    for layer in layers:
        if any(
            not hasattr(layer, name)
            for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
        ):
            return False
        attn = layer.self_attn
        if any(not hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return False
        if not hasattr(attn, "scaling"):
            return False
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "num_attention_heads"):
        return False
    if not hasattr(cfg, "num_key_value_heads"):
        return False
    return True


def _ensure_packed_head_dim(model: nn.Module) -> None:
    cfg = cast(Any, model).config
    if getattr(cfg, "head_dim", None) is None:
        cfg.head_dim = int(cfg.hidden_size) // int(cfg.num_attention_heads)


def _unwrap_decoder(model: nn.Module) -> nn.Module:
    if hasattr(model, "layers") and hasattr(model, "config"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    return model


def _peek_pretrained_config(repo_id_or_path: str | Path, hub_kwargs: dict[str, Any]) -> Any:
    path = Path(repo_id_or_path)
    if path.is_dir() and (path / "config.json").is_file():
        return json.loads((path / "config.json").read_text())
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(repo_id_or_path, **hub_kwargs)


def _hf_config_dict(model: nn.Module) -> dict[str, Any]:
    to_dict = getattr(model.config, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"{type(model.config).__name__} has no to_dict(); cannot save architecture='hf'.")
    return json.loads(json.dumps(to_dict(), default=str))


def _build_hf_from_config(hf_config: dict[str, Any]) -> nn.Module:
    from transformers import AutoModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model_type = hf_config.get("model_type")
    if not isinstance(model_type, str) or model_type not in CONFIG_MAPPING:
        raise ValueError(
            f"architecture='hf' has unknown model_type={model_type!r}; "
            "cannot rebuild the decoder stack."
        )
    cfg = CONFIG_MAPPING[model_type].from_dict(hf_config)
    return _unwrap_decoder(AutoModel.from_config(cfg))


def _hidden_size(cfg: Any) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = _cfg_get(cfg, name)
        if value is not None:
            return int(value)
    raise ValueError("HuggingFace config has no hidden_size / n_embd / d_model.")


class TransformerBackbone(Backbone):
    """Decoder stack with packed kernels when the layout allows, else HF forward.

    ``train_kernel`` (``"varlen"`` / ``"padded"`` / ``"flex"`` / ``"reference"``),
    ``decode_kernel`` (``"flex"``), ``dtype``, and ``use_norm`` are required:
    the uncached-forward kernel, the cached-decode kernel, the dtype of the
    base weights (``torch.float32`` to fine-tune them,
    ``preferred_dtype(device=device)`` for a frozen LoRA base or inference), and
    whether to keep the transformer's final RMSNorm (``False`` replaces it
    with ``Identity``; per-layer norms stay). ``use_norm`` is saved with
    the model.

    Pass **either** ``pretrained=`` **or** ``architecture=``:

    - ``pretrained="Qwen/Qwen3-0.6B"`` inspects the Hub config. A Llama/Qwen3-
      shaped softmax stack uses packed kernels (the speedup). Any other
      maskable softmax decoder uses HuggingFace ``forward`` with the grouping
      mask and requires ``train_kernel="reference"``. Hybrid / linear /
      sliding-window stacks raise.
    - ``architecture="qwen3"`` or ``"llama"`` builds that stack from
      ``hidden_dim``, ``num_layers``, ``num_heads``, … (tests, ablations).
    - ``architecture="hf"`` rebuilds a saved generic stack from ``hf_config``.

    Cached decode (``decode_session``) is the packed/flex path. Generic HF
    stacks raise there; grouping still applies on the full-sequence forward.

    Token embeddings live on this module: ``pretrained=`` loads the
    checkpoint's ``embed_tokens``; from-scratch stacks take ``vocab_size=``
    (or a stub of 1 when the table is unused).
    """

    architecture: Architecture
    uses_packed: bool
    model: _DecoderStack

    def __init__(
        self,
        *,
        train_kernel: TrainKernel,
        decode_kernel: DecodeKernel,
        dtype: torch.dtype,
        use_norm: bool,
        train_autocast_dtype: torch.dtype | None = None,
        decode_autocast_dtype: torch.dtype | None = None,
        hidden_dim: int | None = None,
        pretrained: str | Path | None = None,
        architecture: Architecture | None = None,
        load_weights: bool = True,
        hub_kwargs: dict[str, Any] | None = None,
        lora: LoRAConfig | None = None,
        hf_config: dict[str, Any] | None = None,
        **config_kwargs: Any,
    ) -> None:
        super().__init__()
        self._set_kernels(train_kernel, decode_kernel)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"dtype must be a floating point torch.dtype, got {dtype!r}.")
        self._set_autocast(train_autocast_dtype, decode_autocast_dtype, dtype)
        if pretrained is not None and architecture is not None:
            raise TypeError("TransformerBackbone accepts either pretrained= or architecture=, not both.")
        if pretrained is None and architecture is None:
            raise TypeError(
                "TransformerBackbone requires pretrained= or architecture= "
                "('llama', 'qwen3', or 'hf')."
            )

        if pretrained is not None:
            self._init_from_pretrained(
                dtype=dtype,
                use_norm=use_norm,
                hidden_dim=hidden_dim,
                pretrained=pretrained,
                load_weights=load_weights,
                hub_kwargs=hub_kwargs or {},
                lora=lora,
                config_kwargs=config_kwargs,
            )
            return
        assert architecture is not None
        if config_kwargs:
            # hf rebuild only accepts hf_config / use_norm; extras are a second path.
            extra = set(config_kwargs)
            if architecture == "hf" and extra:
                raise TypeError(
                    f"architecture='hf' does not take {sorted(extra)}; "
                    "pass hf_config= from a saved checkpoint."
                )
        if architecture == "hf":
            if hf_config is None:
                raise TypeError("architecture='hf' requires hf_config= (from save_model).")
            if hidden_dim is not None and int(hidden_dim) != _hidden_size(hf_config):
                raise ValueError(
                    f"hidden_dim={hidden_dim} does not match hf_config hidden size "
                    f"{_hidden_size(hf_config)}."
                )
            self._init_hf_module(
                model=_build_hf_from_config(hf_config),
                dtype=dtype,
                use_norm=use_norm,
                lora=lora,
            )
            return
        if hidden_dim is None:
            raise TypeError(
                f"architecture={architecture!r} requires hidden_dim plus stack "
                "arguments (e.g. TransformerBackbone(architecture='qwen3', "
                "hidden_dim=128, num_layers=2, num_heads=4, use_norm=True))."
            )
        self._init_named_stack(
            architecture=architecture,
            hidden_dim=int(hidden_dim),
            dtype=dtype,
            use_norm=use_norm,
            lora=lora,
            config_kwargs=config_kwargs,
        )

    def _vocab_size_for_stack(self, *, config_vocab: int | None) -> int:
        if config_vocab is None:
            return 1
        return int(config_vocab)

    def _init_named_stack(
        self,
        *,
        architecture: Literal["llama", "qwen3"],
        hidden_dim: int,
        dtype: torch.dtype,
        use_norm: bool,
        lora: LoRAConfig | None,
        config_kwargs: dict[str, Any],
    ) -> None:
        kwargs = dict(config_kwargs)
        vocab_size = self._vocab_size_for_stack(config_vocab=kwargs.pop("vocab_size", None))
        if architecture == "llama":
            self.model = _LlamaBackboneConfig(**kwargs).build(
                hidden_dim, vocab_size=vocab_size
            )
        else:
            self.model = _Qwen3BackboneConfig(**kwargs).build(
                hidden_dim, vocab_size=vocab_size
            )
        self.architecture = architecture
        self.uses_packed = True
        _ensure_packed_head_dim(_as_module(self.model))
        self._config_kwargs = (
            llama_config_kwargs(cast(Any, self.model))
            if architecture == "llama"
            else qwen3_config_kwargs(cast(Any, self.model))
        )
        _apply_final_norm(_as_module(self.model), use_norm)
        self._config_kwargs["use_norm"] = use_norm
        _as_module(self.model).to(dtype)
        self._attach_lora(_as_module(self.model), lora)

    def _init_from_pretrained(
        self,
        *,
        dtype: torch.dtype,
        use_norm: bool,
        hidden_dim: int | None,
        pretrained: str | Path,
        load_weights: bool,
        hub_kwargs: dict[str, Any],
        lora: LoRAConfig | None,
        config_kwargs: dict[str, Any],
    ) -> None:
        raw_cfg = _peek_pretrained_config(pretrained, hub_kwargs)
        reason = grouping_unsupported_reason(raw_cfg)
        if reason is not None:
            raise ValueError(
                f"TransformerBackbone cannot guarantee grouping isolation for "
                f"{str(pretrained)!r}: {reason}."
            )
        model_type = str(_cfg_get(raw_cfg, "model_type") or "").lower()
        if model_type == "qwen3":
            self._init_named_from_pretrained(
                architecture="qwen3",
                dtype=dtype,
                use_norm=use_norm,
                hidden_dim=hidden_dim,
                pretrained=pretrained,
                load_weights=load_weights,
                hub_kwargs=hub_kwargs,
                lora=lora,
                config_kwargs=config_kwargs,
            )
            return
        if "llama" in model_type:
            self._init_named_from_pretrained(
                architecture="llama",
                dtype=dtype,
                use_norm=use_norm,
                hidden_dim=hidden_dim,
                pretrained=pretrained,
                load_weights=load_weights,
                hub_kwargs=hub_kwargs,
                lora=lora,
                config_kwargs=config_kwargs,
            )
            return
        if config_kwargs:
            raise TypeError(
                f"pretrained overrides {sorted(config_kwargs)} are only supported "
                "for llama/qwen3 packed stacks."
            )
        from transformers import AutoModel

        loaded = _unwrap_decoder(AutoModel.from_pretrained(pretrained, **hub_kwargs))
        reason = grouping_unsupported_reason(getattr(loaded, "config", None))
        if reason is not None:
            raise ValueError(
                f"TransformerBackbone cannot guarantee grouping isolation for "
                f"{str(pretrained)!r}: {reason}."
            )
        if hidden_dim is not None and int(hidden_dim) != _hidden_size(loaded.config):
            raise ValueError(
                f"hidden_dim={hidden_dim} does not match pretrained hidden size "
                f"{_hidden_size(loaded.config)} from {str(pretrained)!r}."
            )
        self._init_hf_module(model=loaded, dtype=dtype, use_norm=use_norm, lora=lora)

    def _init_named_from_pretrained(
        self,
        *,
        architecture: Literal["llama", "qwen3"],
        dtype: torch.dtype,
        use_norm: bool,
        hidden_dim: int | None,
        pretrained: str | Path,
        load_weights: bool,
        hub_kwargs: dict[str, Any],
        lora: LoRAConfig | None,
        config_kwargs: dict[str, Any],
    ) -> None:
        if architecture == "qwen3":
            extracted_kwargs, extracted_hidden_dim, extracted_vocab = _qwen3_kwargs_from_pretrained(
                repo_id_or_path=pretrained, hub_kwargs=hub_kwargs, overrides=config_kwargs
            )
            vocab_size = self._vocab_size_for_stack(config_vocab=extracted_vocab)
            model = _Qwen3BackboneConfig(**extracted_kwargs).build(
                extracted_hidden_dim, vocab_size=vocab_size
            )
        else:
            extracted_kwargs, extracted_hidden_dim, extracted_vocab = _llama_kwargs_from_pretrained(
                repo_id_or_path=pretrained, hub_kwargs=hub_kwargs, overrides=config_kwargs
            )
            vocab_size = self._vocab_size_for_stack(config_vocab=extracted_vocab)
            model = _LlamaBackboneConfig(**extracted_kwargs).build(
                extracted_hidden_dim, vocab_size=vocab_size
            )
        if hidden_dim is not None and int(hidden_dim) != extracted_hidden_dim:
            raise ValueError(
                f"hidden_dim={hidden_dim} does not match pretrained hidden size "
                f"{extracted_hidden_dim} from {str(pretrained)!r}."
            )
        self.model = model
        self.architecture = architecture
        self.uses_packed = True
        _ensure_packed_head_dim(_as_module(self.model))
        self._config_kwargs = dict(extracted_kwargs)
        self._config_kwargs["vocab_size"] = vocab_size
        _apply_final_norm(_as_module(self.model), use_norm)
        self._config_kwargs["use_norm"] = use_norm
        if load_weights:
            _load_transformer_weights(
                hub_kwargs=hub_kwargs,
                model=_as_module(self.model),
                repo_id_or_path=pretrained,
            )
        _as_module(self.model).to(dtype)
        self._attach_lora(_as_module(self.model), lora)

    def _init_hf_module(
        self,
        *,
        model: nn.Module,
        dtype: torch.dtype,
        use_norm: bool,
        lora: LoRAConfig | None,
    ) -> None:
        packed = is_packed_compatible(model)
        if packed:
            _ensure_packed_head_dim(model)
        elif grouping_unsupported_reason(getattr(model, "config", None)) is not None:
            raise ValueError(
                "TransformerBackbone cannot guarantee grouping isolation for "
                f"this stack: {grouping_unsupported_reason(model.config)}."
            )
        else:
            if self.train_kernel != "reference":
                raise ValueError(
                    "this architecture has no packed train kernels; use "
                    "train_kernel='reference' (HuggingFace forward + grouping mask), "
                    f"got {self.train_kernel!r}."
                )
        self.model = cast(_DecoderStack, model)
        self.architecture = "hf"
        self.uses_packed = packed
        _apply_final_norm(_as_module(self.model), use_norm)
        self._config_kwargs = {"use_norm": use_norm, "hf_config": _hf_config_dict(_as_module(self.model))}
        _as_module(self.model).to(dtype)
        self._attach_lora(_as_module(self.model), lora)

    @property
    def hidden_dim(self) -> int:
        return _hidden_size(self.model.config)

    def embed(self, token_batch: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return embed_token_ids(
            embed_tokens=cast(nn.Embedding, self.model.get_input_embeddings()),
            token_batch=token_batch,
            hidden_dim=self.hidden_dim,
        )

    def decode_session(self, batch_size: int) -> FlexDecodeSession:
        if not self.uses_packed:
            raise NotImplementedError(
                "cached decode is the packed/flex path; this backbone uses the "
                "generic HuggingFace forward (uncached). Grouping still applies "
                "on the full-sequence path."
            )
        return super().decode_session(batch_size)

    def forward(
        self,
        embeds: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        out = self.model(
            inputs_embeds=embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0] if isinstance(out, tuple) else out
        if output_hidden_states:
            states = getattr(out, "hidden_states", None)
            if states is None:
                raise RuntimeError("TransformerBackbone expected hidden_states but the model returned None.")
            # HF includes the embedding output at index 0; drop it so the tuple is post-layer states.
            layer_states = states[1:] if len(states) > 1 else states
            return hidden, tuple(layer_states)
        return hidden


def _qwen3_kwargs_from_pretrained(
    *,
    repo_id_or_path: str | Path,
    hub_kwargs: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(repo_id_or_path, **hub_kwargs)
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
    rope_parameters = _rope_parameters_from_config(hf_cfg)
    if rope_parameters is not None:
        backbone_kwargs["rope_parameters"] = rope_parameters
    backbone_kwargs.update(overrides)
    return backbone_kwargs, int(hf_cfg.hidden_size), int(hf_cfg.vocab_size)


def _llama_kwargs_from_pretrained(
    *,
    repo_id_or_path: str | Path,
    hub_kwargs: dict[str, Any],
    overrides: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(repo_id_or_path, **hub_kwargs)
    backbone_kwargs: dict[str, Any] = dict(
        num_layers=hf_cfg.num_hidden_layers,
        num_heads=hf_cfg.num_attention_heads,
        num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        max_position_embeddings=hf_cfg.max_position_embeddings,
        intermediate_size=hf_cfg.intermediate_size,
        rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-5),
        attention_bias=getattr(hf_cfg, "attention_bias", False),
    )
    rope_parameters = _rope_parameters_from_config(hf_cfg)
    if rope_parameters is not None:
        backbone_kwargs["rope_parameters"] = rope_parameters
    backbone_kwargs.update(overrides)
    return backbone_kwargs, int(hf_cfg.hidden_size), int(hf_cfg.vocab_size)
