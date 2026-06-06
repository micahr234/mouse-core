from __future__ import annotations

from typing import Any

from mouse_core.models.backbone.llama import ModelLlama, LlamaBackboneConfig
from mouse_core.models.backbone.qwen3 import ModelQwen3, Qwen3BackboneConfig
from mouse_core.models.backbone.none import ModelNone


def backbone_kwargs_from_pretrained(
    repo_id_or_path: str,
    **overrides: Any,
) -> tuple[dict[str, Any], int]:
    """Read backbone architecture defaults from a HuggingFace model config.

    Downloads (or reads locally) the model's ``config.json``, maps its fields
    to the corresponding MOUSE backbone kwargs, and returns them together with
    the model's hidden dimension.  Any keyword arguments are applied on top of
    the pretrained values, letting you change individual settings (e.g. reduce
    ``num_layers``) while keeping everything else.

    Supported model types: ``llama``, ``qwen3``.

    Args:
        repo_id_or_path: HF Hub repo id (e.g. ``"meta-llama/Llama-3.2-1B"``)
            or a local path containing ``config.json``.
        **overrides: Fields to override in the extracted backbone kwargs
            (e.g. ``num_layers=8`` to use only 8 layers of a larger model).

    Returns:
        ``(backbone_kwargs, hidden_dim)`` — pass these directly to
        :func:`~mouse_core.models.base.init_from_pretrained_backbone` or to a
        ``Model`` subclass constructor.

    Example::

        backbone_kwargs, hidden_dim = backbone_kwargs_from_pretrained(
            "meta-llama/Llama-3.2-1B",
            num_layers=8,  # use only 8 of the 16 layers
        )
    """
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(repo_id_or_path)
    model_type: str = getattr(hf_cfg, "model_type", "").lower()

    if "llama" in model_type:
        backbone_kwargs: dict[str, Any] = dict(
            num_layers=hf_cfg.num_hidden_layers,
            num_heads=hf_cfg.num_attention_heads,
            num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
            max_position_embeddings=hf_cfg.max_position_embeddings,
            intermediate_size=hf_cfg.intermediate_size,
            rms_norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-5),
            attention_bias=getattr(hf_cfg, "attention_bias", False),
        )
    elif "qwen3" in model_type or "qwen" in model_type:
        backbone_kwargs = dict(
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
    else:
        raise ValueError(
            f"Unsupported model_type {model_type!r} from {repo_id_or_path!r}. "
            "Supported: llama, qwen3."
        )

    hidden_dim: int = hf_cfg.hidden_size
    backbone_kwargs.update(overrides)
    return backbone_kwargs, hidden_dim


__all__ = [
    "ModelLlama",
    "LlamaBackboneConfig",
    "ModelQwen3",
    "Qwen3BackboneConfig",
    "ModelNone",
    "backbone_kwargs_from_pretrained",
]
