from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.discrete_action import DiscreteActionHead
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.heads.layerwise_dqn import LayerwiseDiscreteActionValueHead
from mouse_core.models.heads.swiglu import SwiGLUHead
from mouse_core.models.heads.vec_dqn import VectorActionValueHead, vector_action_scores


def _backbone_num_layers(backbone: nn.Module) -> int | None:
    """Return transformer block count when the backbone exposes block layers."""
    inner = getattr(backbone, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return len(layers)
    encoder = getattr(inner, "encoder", None)
    encoder_layers = getattr(encoder, "layer", None)
    if encoder_layers is not None:
        return len(encoder_layers)
    from mouse_core.models.backbone import IdentityBackbone

    if isinstance(backbone, IdentityBackbone):
        return 1
    return None


def _hub_repo_id_for_user(repo_id: str, token: str | bool | None = None) -> str:
    """Resolve an unscoped Hub repo name under the authenticated user."""
    if "/" in repo_id:
        return repo_id

    from huggingface_hub import HfApi

    user = HfApi().whoami(token=token)["name"]
    return f"{user}/{repo_id}"


def _expand_segment_ids_to_tokens(
    segment_ids: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Expand step-level segment IDs ``[B, S]`` by per-step token counts ``[B, S]``.

    Result is ``[B, L]`` (batch-padded with ``-1`` for unused trailing slots).
    """
    from mouse_core.models.embedding.packing import expand_segment_ids

    return expand_segment_ids(segment_ids, counts)


def _segment_causal_attention_mask(
    segment_token_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Additive 4D mask ``[B, 1, L, L]``: causal and same-segment only.

    Allowed positions are ``0.0``; blocked positions are a large negative value
    compatible with HuggingFace SDPA / float attention masks.

    When ``token_mask`` is provided (``[B, L]`` bool, True = real token), pad
    positions neither attend nor are attended to. Segment id ``-1`` (batch pad)
    is also treated as non-attending.
    """
    # segment_token_ids: [B, L]
    B, L = segment_token_ids.shape
    device = segment_token_ids.device
    q = torch.arange(L, device=device).view(1, L, 1)
    k = torch.arange(L, device=device).view(1, 1, L)
    causal = k <= q
    real_seg = segment_token_ids >= 0
    same_segment = (
        (segment_token_ids.unsqueeze(2) == segment_token_ids.unsqueeze(1))
        & real_seg.unsqueeze(2)
        & real_seg.unsqueeze(1)
    )
    allowed = causal & same_segment  # [B, L, L]
    if token_mask is not None:
        if token_mask.shape != (B, L):
            raise ValueError(
                f"token_mask must have shape [{B}, {L}], got {tuple(token_mask.shape)}"
            )
        allowed = allowed & token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
    neg = torch.finfo(dtype).min
    mask = torch.zeros(B, 1, L, L, device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed.unsqueeze(1), neg)
    return mask


def _segment_position_ids(
    segment_token_ids: torch.Tensor,
    *,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """RoPE position ids ``[B, L]`` reset to 0 at each segment boundary.

    Pad tokens (``token_mask`` False or segment id ``-1``) keep position 0 and
    are masked out of attention separately.
    """
    B, L = segment_token_ids.shape
    if L == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=segment_token_ids.device)
    real = segment_token_ids >= 0
    if token_mask is not None:
        real = real & token_mask
    # Sentinel prev id so the first real token is always treated as a segment start.
    prev = torch.cat(
        [
            segment_token_ids.new_full((B, 1), -1),
            segment_token_ids[:, :-1],
        ],
        dim=1,
    )
    new_seg = (segment_token_ids != prev) & real
    idx = torch.arange(L, device=segment_token_ids.device).view(1, L).expand(B, -1)
    markers = torch.where(new_seg, idx, torch.full_like(idx, -1))
    # Only accumulate starts on real tokens; pads keep markers at -1 until a real start.
    last_start = markers.cummax(dim=1).values.clamp(min=0)
    pos = (idx - last_start).to(torch.long)
    return torch.where(real, pos, torch.zeros_like(pos))


def save_model(model: "Model", path: str | Path) -> None:
    """Save a MOUSE model to a local directory.

    Writes ``pytorch_model.bin`` and ``config.json`` into *path*. The saved
    directory can be passed back to :func:`load_model`.

    Args:
        model: The model instance to save.
        path: Destination directory (created if absent).

    Example::

        save_model(model, "./checkpoints/step-10000")
        model2 = load_model("./checkpoints/step-10000")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w") as fh:
        json.dump(_model_config(model), fh, indent=2, sort_keys=True)
        fh.write("\n")
    torch.save(model.state_dict(), path / "pytorch_model.bin")


def push_model_to_hub(
    model: "Model",
    repo_id: str,
    *,
    commit_message: str = "Upload MOUSE model",
    private: bool = False,
    clear: bool = False,
    **kwargs: Any,
) -> str:
    """Push a MOUSE model to the Hugging Face Hub.

    Creates the repository if needed, uploads the MOUSE checkpoint files plus a
    model card, and returns the Hub URL.

    Args:
        model: The model instance to upload.
        repo_id: Hub repository ID, e.g. ``"my-model"`` or ``"your-org/your-model"``.
            Unscoped names are resolved under the authenticated user.
        commit_message: Commit message written to the Hub.
        private: Create a private repository if it does not already exist.
        clear: Delete all existing files in the repository before uploading.
            Useful to avoid stale files from a previous push.
        **kwargs: Forwarded to ``huggingface_hub.HfApi.upload_folder``.

    Returns:
        The Hub URL string for the uploaded repository.

    Example::

        url = push_model_to_hub(model, "my-model", clear=True)
        print(url)
    """
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
        token=kwargs.get("token"),
    )
    hub_repo_id = repo_url.repo_id
    if clear:
        # huggingface_hub never deletes .gitattributes, so exclude it from the
        # list to avoid a spurious "no files modified" warning when it is the
        # only file present (e.g. a freshly created repo).
        existing = [f for f in api.list_repo_files(hub_repo_id) if f != ".gitattributes"]
        if existing:
            api.delete_files(
                repo_id=hub_repo_id,
                delete_patterns=existing,
                commit_message="Clear repository before upload",
            )
    with tempfile.TemporaryDirectory() as tmp:
        save_model(model, tmp)
        _write_model_card(model, Path(tmp) / "README.md", repo_id=hub_repo_id)
        api.upload_folder(
            repo_id=hub_repo_id,
            folder_path=tmp,
            commit_message=commit_message,
            **kwargs,
        )
    return str(repo_url)


def _write_model_card(model: "Model", path: Path, *, repo_id: str) -> None:
    config = _model_config(model)
    heads = config["heads"]["heads"]
    head_names = ", ".join(head["name"] for head in heads) or "none"
    modalities = config["encoder"]["kwargs"].get("modalities", [])
    modality_table = _model_card_modality_table(modalities)
    objective_data_example = _model_card_step_stream_example(modalities)
    text = f"""---
library_name: mouse-core
tags:
- mouse-core
- reinforcement-learning
---

# {repo_id}

This repository contains a MOUSE model checkpoint.

## Architecture

- Backbone: `{config["backbone"]["type"]}`
- Hidden dimension: `{config["hidden_dim"]}`
- Heads: `{head_names}`
- Action head: `{config["heads"]["action_head"]}`

### Encoder

`NumericEmbedder` reads flat step-record dicts and projects each declared modality
into the shared `{config["hidden_dim"]}`-dimensional token space before the
backbone.

{modality_table}

## Install MouseCore

```bash
pip install mouse-core
```

## Load The Model

```python
import torch
from mouse_core import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("{repo_id}", map_location="cpu").eval().to(device)
```

## Run Inference

The model accepts a `list[list[dict]]` batch of shape `[B][S]` — B sequences,
each containing S step-record dicts with flat keys matching the encoder's
declared modalities above.

```python
{objective_data_example}

with torch.no_grad():
    predictions, _, cache = model(batch)
    action = model.get_action(predictions, temperature=0.0)
```

`model()` returns `(predictions, objective_data, cache)`. `objective_data` is a
`TensorDict[B, S]` of the modality tensors extracted by the encoder — pass it
to objectives during training. For cached incremental rollout, keep `cache` and
pass it back on the next call with `use_cache=True`. Cached batch rows may have
different lengths on every call (e.g. envs emitting different numbers of steps
between model calls): decoding runs through a FlexAttention session carried in
the cache, so each row decodes exactly as it would alone.
"""
    path.write_text(text, encoding="utf-8")


def _model_card_modality_table(modalities: list[dict[str, Any]]) -> str:
    rows = [
        "| Field | Type | Required | Tensor shape | Dtype | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for modality in modalities:
        field = str(modality.get("field", ""))
        modality_type = str(modality["type"])
        required = bool(modality.get("required", True))
        allow_none = bool(modality.get("allow_none", False))
        rows.append(
            "| "
            + " | ".join([
                f"`{field}`" if field else "-",
                f"`{modality_type}`",
                "yes" if required else "no",
                f"`{_model_card_modality_shape(modality)}`",
                f"`{_model_card_modality_dtype(modality)}`",
                _model_card_modality_notes(modality, allow_none=allow_none),
            ])
            + " |"
        )
    return "\n".join(rows)


def _model_card_modality_shape(modality: dict[str, Any]) -> str:
    modality_type = modality["type"]
    if modality_type in ("continuous", "image"):
        dim = modality.get("dim") or modality.get("size") or "D"
        return f"[B, S, {dim}]"
    if modality_type == "learnable":
        return "not read from step_stream"
    return "[B, S]"


def _model_card_modality_dtype(modality: dict[str, Any]) -> str:
    modality_type = modality["type"]
    if modality_type == "discrete":
        return "torch.long"
    if modality_type == "image":
        return "torch.long or torch.float32"
    if modality_type == "learnable":
        return "n/a"
    return "torch.float32"


def _model_card_modality_notes(modality: dict[str, Any], *, allow_none: bool) -> str:
    modality_type = modality["type"]
    parts: list[str] = []
    if modality_type == "discrete":
        vocab_size = modality.get("vocab_size") or modality.get("size")
        if vocab_size is not None:
            parts.append(f"integer ids in `[0, {int(vocab_size) - 1}]`")
    elif modality_type == "rff":
        parts.append("scalar value")
    elif modality_type == "continuous":
        parts.append("vector values")
    elif modality_type == "image":
        parts.append("pixel or patch values")
    elif modality_type == "learnable":
        parts.append("learned tokens; no input field")
    if allow_none:
        parts.append("`None` uses default value")
    return "; ".join(parts) or "-"


def _model_card_step_stream_example(modalities: list[dict[str, Any]]) -> str:
    fields = [
        _model_card_field_example(modality)
        for modality in modalities
        if modality["type"] != "learnable"
    ]
    body = "\n".join(f"    {field}" for field in fields)
    if not body:
        body = "    # This model declares no input-backed modalities."
    return f"""# Batch shape: [B=1][S=1] — one sequence of one step.
batch = [[
    {{
{body}
    }}
]]
predictions, objective_data, cache = model(batch)"""


def _model_card_field_example(modality: dict[str, Any]) -> str:
    field = modality["field"]
    modality_type = modality["type"]
    optional = "" if modality.get("required", True) else "  # optional"
    if modality_type == "discrete":
        return f'"{field}": 0,{optional}'
    if modality_type == "rff":
        return f'"{field}": 0.0,{optional}'
    if modality_type == "continuous":
        dim = int(modality.get("dim") or modality.get("size") or 1)
        return f'"{field}": [0.0] * {dim},{optional}'
    if modality_type == "image":
        dim = int(modality.get("dim") or modality.get("size") or 1)
        return f'"{field}": [0] * {dim},{optional}'
    return f'"{field}": 0,{optional}'


def _model_config(model: "Model") -> dict[str, Any]:
    return {
        "format": "mouse-core-model-v1",
        "hidden_dim": int(model.hidden_dim),
        "encoder": _encoder_config(model.encoder),
        "backbone": _backbone_config(model.backbone),
        "heads": _heads_config(model),
    }


def _encoder_config(encoder: Encoder) -> dict[str, Any]:
    from mouse_core.models.embedding.embedding import NumericEmbedder
    from mouse_core.models.embedding.text import TextEmbedder

    if isinstance(encoder, NumericEmbedder):
        return {
            "type": "numeric",
            "kwargs": {
                "hidden_dim": int(encoder.hidden_dim),
                "modalities": [_public_modality_config(modality) for modality in encoder.modalities],
                "token_data_len": int(encoder.token_data_len),
                "modality_fusion": encoder.modality_fusion,
                "include_type_token": bool(encoder.include_type_token),
                "fourier_min": float(encoder.fourier_min),
                "fourier_max": float(encoder.fourier_max),
                "std": float(encoder.std),
                "type_embedding_std": float(encoder.type_embedding_std),
            },
        }
    if isinstance(encoder, TextEmbedder):
        return {
            "type": "text",
            "kwargs": {
                "hidden_dim": int(encoder.hidden_dim),
                "modalities": [_public_modality_config(modality) for modality in encoder.modalities],
                "pretrained": encoder.pretrained,
                "format": encoder.format,
            },
        }
    raise TypeError(
        "save_model currently supports NumericEmbedder and TextEmbedder encoders. "
        f"Got {type(encoder).__name__}."
    )


def _public_modality_config(modality: Any) -> dict[str, Any]:
    data = _drop_none(asdict(modality))
    if data.get("type") == "learnable" and str(data.get("field", "")).startswith("__learnable_"):
        data.pop("field", None)
    return data


def _backbone_config(backbone: nn.Module) -> dict[str, Any]:
    from mouse_core.models.backbone.llama import LlamaBackbone
    from mouse_core.models.backbone.none import IdentityBackbone
    from mouse_core.models.backbone.qwen3 import Qwen3Backbone

    if isinstance(backbone, IdentityBackbone):
        return {"type": "identity", "hidden_dim": backbone.hidden_dim}
    if isinstance(backbone, LlamaBackbone):
        return {
            "type": "llama",
            "hidden_dim": backbone.hidden_dim,
            "kwargs": dict(backbone._config_kwargs),
        }
    if isinstance(backbone, Qwen3Backbone):
        return {
            "type": "qwen3",
            "hidden_dim": backbone.hidden_dim,
            "kwargs": dict(backbone._config_kwargs),
        }
    raise TypeError(
        "save_model currently supports IdentityBackbone, LlamaBackbone, and Qwen3Backbone. "
        f"Got {type(backbone).__name__}."
    )


def _heads_config(model: "Model") -> dict[str, Any]:
    heads = []
    for name, head in model._heads.items():
        spec = _head_config(name, head)
        if spec is not None:
            heads.append(spec)
    return {"action_head": model.action_head, "heads": heads}


def _head_config(name: str, head: BaseHead) -> dict[str, Any] | None:
    if isinstance(head, LayerwiseDiscreteActionValueHead):
        return {
            "name": name,
            "type": "action_value_layerwise",
            "num_backbone_layers": head.num_backbone_layers,
            "in_features": head.in_features,
            "out_features": head.out_features,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
    if isinstance(head, DiscreteActionValueHead):
        return {
            "name": name,
            "type": "action_value",
            "in_features": head.in_features,
            "out_features": head.out_features,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
    if isinstance(head, VectorActionValueHead):
        spec = {
            "name": name,
            "type": "action_vector",
            "in_features": head.in_features,
            "max_num_actions": head.max_num_actions,
            "vec_dim": head.vec_dim,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
        if head.bias_scale is not None:
            spec["bias_scale"] = head.bias_scale
        return spec
    if isinstance(head, DiscreteActionHead):
        return {
            "name": name,
            "type": "discrete_action",
            "in_features": head.in_features,
            "out_features": head.out_features,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
    if isinstance(head, SwiGLUHead):
        return {
            "name": name,
            "type": "swiglu",
            "in_features": head.in_features,
            "out_features": head.out_features,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
    raise TypeError(f"save_model does not know how to serialize head {name!r} ({type(head).__name__}).")


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def load_model(
    repo_id_or_path: str,
    force_download: bool = True,
    local_dir: str | Path | None = None,
    **kwargs: Any,
) -> "Model":
    """Load a MOUSE model from a local directory or HuggingFace Hub repo.

    Args:
        repo_id_or_path: A local path to a checkpoint directory or a HF Hub
            repo id (e.g. ``"my-model"`` or ``"your-org/your-model"``).
            Unscoped Hub names are resolved under the authenticated user.
        force_download: If ``True`` (default), bypass the HF Hub cache and re-download.
            Ignored for local paths.
        local_dir: Directory where Hub files are saved after download.  When
            set, ``hf_hub_download`` writes files there and
            set, Hub files are saved there before loading. Ignored for local paths.
        **kwargs: Supports ``map_location`` for ``torch.load`` and forwards Hub
            download kwargs such as ``revision`` or ``token``.

    Returns:
        The loaded model instance.
    """
    map_location = kwargs.pop("map_location", "cpu")
    local = Path(repo_id_or_path)
    if local.exists():
        config_path = local / "config.json"
        weights_path = local / "pytorch_model.bin"
        with config_path.open() as fh:
            config = json.load(fh)
    else:
        from huggingface_hub import hf_hub_download
        hf_kwargs: dict[str, Any] = {"force_download": force_download, **kwargs}
        if local_dir is not None:
            hf_kwargs["local_dir"] = str(local_dir)
        hub_repo_id = _hub_repo_id_for_user(repo_id_or_path, token=kwargs.get("token"))
        config_path = Path(hf_hub_download(repo_id=hub_repo_id, filename="config.json", **hf_kwargs))
        weights_path = Path(hf_hub_download(repo_id=hub_repo_id, filename="pytorch_model.bin", **hf_kwargs))
        with config_path.open() as fh:
            config = json.load(fh)

    if config.get("format") != "mouse-core-model-v1":
        raise ValueError(
            "Unsupported model config format. Expected a MOUSE checkpoint saved "
            "with save_model(...)."
        )

    model = _build_model_from_config(config)
    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    return model


def _build_model_from_config(config: dict[str, Any]) -> "Model":
    encoder = _build_encoder_from_config(config["encoder"])
    backbone = _build_backbone_from_config(config["backbone"])
    heads_cfg = config["heads"]
    heads = _build_heads_from_config(heads_cfg["heads"])
    return Model(
        encoder=encoder,
        backbone=backbone,
        heads=heads,
        action_head=heads_cfg.get("action_head"),
    )


def _build_encoder_from_config(config: dict[str, Any]) -> Encoder:
    enc_type = config.get("type")
    if enc_type == "numeric":
        from mouse_core.models.embedding import NumericEmbedder

        return NumericEmbedder(**config["kwargs"])
    if enc_type == "text":
        from mouse_core.models.embedding import TextEmbedder

        kwargs = dict(config["kwargs"])
        # image_processor / tokenizer are not serialized; reload from pretrained.
        return TextEmbedder(**kwargs)
    raise ValueError(f"Unsupported encoder type {enc_type!r}.")


def _build_backbone_from_config(config: dict[str, Any]) -> nn.Module:
    backbone_type = config.get("type")
    if backbone_type == "identity":
        from mouse_core.models.backbone import IdentityBackbone

        return IdentityBackbone(hidden_dim=config.get("hidden_dim"))
    if backbone_type == "llama":
        from mouse_core.models.backbone import LlamaBackbone

        return LlamaBackbone(hidden_dim=config["hidden_dim"], **config["kwargs"])
    if backbone_type == "qwen3":
        from mouse_core.models.backbone import Qwen3Backbone

        return Qwen3Backbone(hidden_dim=config["hidden_dim"], **config["kwargs"])
    raise ValueError(f"Unsupported backbone type {backbone_type!r}.")


def _build_heads_from_config(heads: list[dict[str, Any]]) -> dict[str, BaseHead]:
    built: dict[str, BaseHead] = {}
    for spec in heads:
        name = spec["name"]
        head_type = spec["type"]
        if head_type == "action_value_layerwise":
            built[name] = LayerwiseDiscreteActionValueHead(
                num_backbone_layers=spec["num_backbone_layers"],
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec.get("use_norm", True),
            )
        elif head_type == "action_value":
            built[name] = DiscreteActionValueHead(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec.get("use_norm", True),
            )
        elif head_type == "action_vector":
            built[name] = VectorActionValueHead(
                in_features=spec["in_features"],
                max_num_actions=spec["max_num_actions"],
                vec_dim=spec["vec_dim"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                bias_scale=spec.get("bias_scale"),
                use_norm=spec.get("use_norm", True),
            )
        elif head_type == "discrete_action":
            built[name] = DiscreteActionHead(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec.get("use_norm", True),
            )
        elif head_type == "swiglu":
            built[name] = SwiGLUHead(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec.get("use_norm", True),
            )
        else:
            raise ValueError(f"Unsupported head type {head_type!r}.")
    return built


class Model(nn.Module):
    """Composable MOUSE model: encoder, backbone, and heads as distinct sections.

    The model is assembled from three pluggable parts:

    - ``encoder``: :class:`~mouse_core.models.embedding.embedding.Encoder`
      Converts a ``TensorDict[B, S]`` of step records into token embeddings
      ``[B, T, D]`` and knows how to pool backbone outputs back to per-step
      representations ``[B, S, D]`` (the vectors used for action output).
    - ``backbone``: a :class:`~mouse_core.models.backbone.Backbone`-compatible
      module that maps ``embeds`` plus optional cache/mask args to
      ``(hidden_states, cache)``.
    - ``heads``: heads can be provided in several ergonomic ways:
        - a single :class:`~mouse_core.models.heads.base.BaseHead` (e.g. ``DiscreteActionValueHead(...)``):
          it becomes the only enabled head and the implicit ``action_head``;
        - a list of head instances (e.g. ``[DiscreteActionValueHead(...), VectorActionValueHead(...)]``):
          you **must** also pass ``action_head`` (a canonical name) to select which one
          ``get_action`` uses;
        - a dict mapping canonical names (``"action_value"``, ``"action_vector"``, ``"action"``, ``"value"``)
          to head instances or ``None`` (for full control and/or multiple heads).
      When a plain head (SwiGLUHead) is passed without a name it defaults to ``"action"``;
      use the dict form if you want it under ``"value"``.

    ``action_head`` names which head ``get_action`` consults. If omitted,
    it is auto-selected by preference: ``action_vector`` > ``action_value`` > ``action`` > ``value``.

    The only supported construction is the explicit three-piece composition:

        encoder = NumericEmbedder(...)
        backbone = LlamaBackbone(...)   # or any Backbone
        heads = DiscreteActionValueHead(...)            # or a dict/list of heads

        model = Model(encoder=encoder, backbone=backbone, heads=heads)

    The backbone is independent; it does not know about the encoder or heads.
    """

    _VALID_HEADS = ("action_value", "action_value_layerwise", "action_vector", "action", "value")

    @staticmethod
    def _normalize_heads(
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None] | None,
        action_head: str | None,
    ) -> dict[str, BaseHead]:
        """Convert the flexible ``heads=`` argument into the internal ``name -> head`` dict.

        Supported inputs:
          - dict (canonical names to head or None): passed through with validation.
          - single BaseHead instance: becomes the only head; name is inferred
            (SwiGLUHead defaults to "action"; you can pass action_head="value" to select it).
          - list/tuple of BaseHead: each gets an inferred name; you *must* provide
            action_head= to declare which one is used by get_action().
        """
        if heads is None:
            return {}

        # Dict form gives full control over names.
        if isinstance(heads, Mapping):
            filtered: dict[str, BaseHead] = {}
            for name, h in heads.items():
                if h is not None:
                    if name not in Model._VALID_HEADS:
                        raise ValueError(f"head name {name!r} is not one of {Model._VALID_HEADS}")
                    if not isinstance(h, BaseHead):
                        raise TypeError(f"head {name!r} must be a BaseHead or None, got {type(h)}")
                    filtered[name] = h
            return filtered

        # Single head instance gives an implicit single-head model.
        if isinstance(heads, BaseHead):
            name = Model._infer_head_name(heads, preferred=action_head)
            return {name: heads}

        # List of heads → explicit action_head required
        if isinstance(heads, (list, tuple)):
            if len(heads) == 0:
                return {}
            result: dict[str, BaseHead] = {}
            for h in heads:
                if not isinstance(h, BaseHead):
                    raise TypeError(f"items in heads list must be BaseHead instances, got {type(h)}")
                nm = Model._infer_head_name(h, preferred=None)
                if nm in result:
                    raise ValueError(
                        f"Multiple heads would map to the same canonical name {nm!r}. "
                        "Use a dict form to provide distinct names, e.g. "
                        "heads={'action_value': h1, 'action': h2}."
                    )
                result[nm] = h

            if action_head is None:
                raise TypeError(
                    "When passing heads as a list you must also specify action_head= "
                    "(one of 'action_value', 'action_vector', 'action', 'value') to select the head used by get_action()."
                )
            return result

        raise TypeError(
            f"heads must be a BaseHead, list[BaseHead], or dict[str, BaseHead|None], "
            f"got {type(heads)}"
        )

    @staticmethod
    def _infer_head_name(head: BaseHead, preferred: str | None = None) -> str:
        """Infer the canonical storage / output key for a concrete head instance."""
        if isinstance(head, LayerwiseDiscreteActionValueHead):
            return "action_value_layerwise"
        if isinstance(head, VectorActionValueHead):
            return "action_vector"
        if isinstance(head, DiscreteActionValueHead):
            return "action_value"
        if isinstance(head, SwiGLUHead):
            if preferred in ("action", "value"):
                return preferred
            return "action"
        raise TypeError(
            f"Cannot infer canonical name for head of type {type(head).__name__}. "
            f"Use the dict form with an explicit key from {Model._VALID_HEADS}."
        )

    def __init__(
        self,
        *,
        encoder: Encoder,
        backbone: nn.Module,
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None] | None = None,
        action_head: str | None = None,
    ):
        """Construct a Model from three independent pieces.

        This is the *only* supported construction path.
        """
        super().__init__()

        if encoder is None or backbone is None:
            # Defensive (types make them required)
            raise TypeError("Model requires encoder and backbone.")

        if not isinstance(encoder, Encoder):
            raise TypeError("encoder must be an instance of Encoder (from mouse_core.models.embedding).")

        # Consistency check between encoder and backbone hidden sizes when available.
        enc_dim = getattr(encoder, "hidden_dim", None)
        bb_dim = getattr(backbone, "hidden_dim", None)
        if enc_dim is not None and bb_dim is not None and enc_dim != bb_dim:
            raise ValueError(
                f"hidden_dim mismatch between encoder ({enc_dim}) and backbone ({bb_dim}). "
                "The embedder and the backbone must agree on the hidden dimension."
            )

        self.encoder: Encoder = encoder
        self.backbone: nn.Module = backbone

        if heads is None:
            raise TypeError("Model requires heads (a BaseHead, list of heads, or dict of named heads).")

        # Normalize flexible heads input (single instance, list, or dict) into the
        # canonical internal dict form.
        heads_dict: dict[str, BaseHead] = Model._normalize_heads(heads, action_head)

        # Store heads for both state dict and typed access
        filtered: dict[str, BaseHead] = {}
        for name, head in heads_dict.items():
            if head is not None:
                if name not in self._VALID_HEADS:
                    raise ValueError(f"head name {name!r} is not one of {self._VALID_HEADS}")
                if not isinstance(head, BaseHead):
                    raise TypeError(f"head {name!r} must be a BaseHead or None, got {type(head)}")
                filtered[name] = head
        self.heads = nn.ModuleDict(filtered)  # for parameters/state
        self._heads: dict[str, BaseHead] = filtered  # typed view for calling

        # Determine action head
        if action_head is not None:
            if action_head not in self._VALID_HEADS:
                raise ValueError(f"action_head must be one of {self._VALID_HEADS}, got {action_head!r}.")
            if action_head not in self.heads:
                raise ValueError(f"action_head={action_head!r} but no such head is enabled.")
            self.action_head: str = action_head
        else:
            # Auto-detect preference order
            for candidate in ("action_vector", "action_value_layerwise", "action_value", "action", "value"):
                if candidate in self.heads:
                    self.action_head = candidate
                    break
            else:
                raise ValueError("No output head is enabled; cannot determine action_head.")

        if "action_value_layerwise" in self._heads:
            layerwise_head = self._heads["action_value_layerwise"]
            if not isinstance(layerwise_head, LayerwiseDiscreteActionValueHead):
                raise TypeError("action_value_layerwise head has unexpected type.")
            bb_layers = _backbone_num_layers(self.backbone)
            if bb_layers is None:
                raise ValueError(
                    "action_value_layerwise requires a backbone with a known layer count "
                    "(e.g. Qwen3Backbone or LlamaBackbone)."
                )
            if layerwise_head.num_backbone_layers != bb_layers:
                raise ValueError(
                    f"Layerwise head expects {layerwise_head.num_backbone_layers} backbone layers "
                    f"but backbone has {bb_layers}."
                )

        # Convenience: expose hidden_dim and max_num_actions from encoder/heads
        self.hidden_dim = int(encoder.hidden_dim)
        # Best-effort inference of action cardinality for introspection only.
        self.max_num_actions: int = 0
        for _name, h in self.heads.items():
            out = getattr(h, "A", None)  # VectorActionValueHead stores A
            if out is None:
                out = getattr(h, "out_features", None)
            if out is None and hasattr(h, "online"):
                out = getattr(h.online, "out_features", None)
            if isinstance(out, int) and out > 0:
                self.max_num_actions = out
                break

    def to(self, *args: Any, **kwargs: Any) -> "Model":
        """Move/cast the model; output heads always stay float32."""
        dtype_kw = kwargs.get("dtype")
        dtype_arg = args[0] if len(args) == 1 and isinstance(args[0], torch.dtype) else None
        target_dtype = dtype_kw or dtype_arg
        if target_dtype is not None and target_dtype != torch.float32:
            kwargs_no_dtype = {k: v for k, v in kwargs.items() if k != "dtype"}
            args_no_dtype = () if dtype_arg is not None else args
            super().to(*args_no_dtype, **kwargs_no_dtype)
            self.encoder.to(*args_no_dtype, dtype=target_dtype, **kwargs_no_dtype)
            self.backbone.to(*args_no_dtype, dtype=target_dtype, **kwargs_no_dtype)
            self.heads.to(*args_no_dtype, dtype=torch.float32, **kwargs_no_dtype)
            return self
        return super().to(*args, **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: list[list[dict]],
        segment_ids: list[list[int]] | None = None,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
    ) -> tuple[TensorDict, TensorDict, dict[str, Any] | None]:
        """Run a full forward pass.

        When decoding with a cache, rows in ``batch`` may have **different
        lengths** on every call (including empty): incremental decoding runs
        through a FlexAttention session that keeps each row inside its own
        causal prefix with per-row RoPE positions, so every row decodes exactly
        as it would alone (verified in ``tests/test_kv_cache.py``). The session
        travels inside the returned ``cache``, so batched incremental decoding
        with ragged chunks (e.g. envs emitting different numbers of steps
        between model calls) needs no bookkeeping from the caller. Predictions
        at a row's leading pad positions are meaningless: row ``b``'s real
        steps are the trailing ``len(batch[b])`` entries, and the last step —
        the one :meth:`get_action` reads — is always real for non-empty rows.

        Args:
            batch: ``[B][S]`` list of raw step-record dicts, as returned by
                ``DataLoader.next_batch()`` or assembled manually for rollout.
                Rows may have unequal lengths (including empty) when decoding
                with a cache; at least one row must be non-empty.
            segment_ids: Optional ``[B][S]`` pack-slice IDs parallel to
                ``batch`` (from ``DataLoader.next_batch()``). Training uses them
                to build a same-segment causal attention mask and to reset RoPE
                ``position_ids`` at each new segment. ``None`` (rollouts) treats
                every step in a row as one segment. Injected into
                ``objective_data`` as ``segment_id`` for TD objectives.
            cache: Optional cache from a prior call. Requires ``use_cache=True``
                (the decode session inside is mutated by every call). The batch
                size must match the call that created the cache.
            use_cache: If True, return an updated cache.

        Returns:
            ``(predictions, objective_data, cache)`` where

            * ``predictions`` is a ``TensorDict[B, S]`` with one entry per enabled head.
            * ``objective_data`` is a ``TensorDict[B, S]`` of the modality tensors
              extracted by the encoder — the same values used for embedding —
              plus ``segment_id``.
              Pass this to objectives (e.g. ``objective(objective_data, predictions)``).
            * ``cache`` is the updated cache, or ``None`` when ``use_cache=False``.
        """
        if cache is not None and not use_cache:
            # The decode session in the cache is mutated by every call, so a
            # "read-only" pass over an existing cache cannot exist.
            raise ValueError("Passing cache= requires use_cache=True.")

        B = len(batch)
        lengths = [len(rows) for rows in batch]
        S = max(lengths, default=0)

        ragged = any(n != S for n in lengths)
        if ragged or use_cache:
            if S == 0:
                raise ValueError("Model.forward requires at least one non-empty row in batch.")
            if not use_cache:
                raise ValueError("Ragged batches require use_cache=True (cached decoding).")
            if ragged:
                # Pad rows are never attended to and never enter the KV cache,
                # so any real row works as filler for the encoder.
                pad_row = next(rows[0] for rows in batch if rows)
                batch = [[pad_row] * (S - n) + rows for rows, n in zip(batch, lengths)]

        if segment_ids is None:
            segment_ids = [[0] * len(rows) for rows in batch]
        if len(segment_ids) != B:
            raise ValueError(
                f"segment_ids batch size ({len(segment_ids)}) must match batch ({B})."
            )
        for b, (rows, ids) in enumerate(zip(batch, segment_ids)):
            if len(ids) != len(rows):
                raise ValueError(
                    f"segment_ids[{b}] length ({len(ids)}) must match batch[{b}] "
                    f"length ({len(rows)})."
                )

        embeds, col_values, step_token_indices = self.encoder(batch)
        if any(value.device != embeds.device for value in col_values.values()):
            col_values = {key: value.to(embeds.device) for key, value in col_values.items()}
        segment_id_tensor = torch.tensor(
            segment_ids, device=embeds.device, dtype=torch.long
        )
        if segment_id_tensor.shape != (B, S):
            raise ValueError(
                f"segment_ids must have shape [{B}, {S}], got {tuple(segment_id_tensor.shape)}."
            )
        col_values["segment_id"] = segment_id_tensor
        objective_data = TensorDict(col_values, batch_size=(B, S))

        from mouse_core.models.embedding.packing import (
            counts_from_step_token_indices,
            real_token_lengths_from_indices,
            token_pad_mask,
        )

        counts = counts_from_step_token_indices(step_token_indices)
        pad_mask = token_pad_mask(step_token_indices, embeds.shape[1])

        needs_layerwise = "action_value_layerwise" in self._heads
        if use_cache:
            from mouse_core.models.embedding.packing import left_align_content

            token_lengths = real_token_lengths_from_indices(
                step_token_indices, num_real_steps=lengths
            )
            flex_embeds, pool_indices = left_align_content(embeds, step_token_indices)
            session = cache["session"] if cache else self.backbone.decode_session(
                batch_size=B, capacity=max(flex_embeds.shape[1], 1)
            )
            session_out = session.forward(
                flex_embeds, token_lengths, output_hidden_states=needs_layerwise
            )
            new_cache: dict[str, Any] | None = {"session": session}
            step_token_indices = pool_indices
        else:
            segment_token_ids = _expand_segment_ids_to_tokens(
                segment_id_tensor, counts
            )
            # Batch-pad expand may be shorter/longer than embeds if packing differs;
            # align to embeds length.
            L = embeds.shape[1]
            if segment_token_ids.shape[1] < L:
                pad = segment_token_ids.new_full(
                    (B, L - segment_token_ids.shape[1]), -1
                )
                segment_token_ids = torch.cat([segment_token_ids, pad], dim=1)
            elif segment_token_ids.shape[1] > L:
                segment_token_ids = segment_token_ids[:, :L]
            attention_mask = _segment_causal_attention_mask(
                segment_token_ids, dtype=embeds.dtype, token_mask=pad_mask
            )
            position_ids = _segment_position_ids(
                segment_token_ids, token_mask=pad_mask
            )
            session_out = self.backbone(
                embeds,
                output_hidden_states=needs_layerwise,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            new_cache = None
        if needs_layerwise:
            h, layer_hiddens = session_out
        else:
            h = session_out
            layer_hiddens = None

        h_step = self.encoder.pool_step_reprs(h, step_token_indices)
        if needs_layerwise:
            assert layer_hiddens is not None
            h_layers = torch.stack(
                [
                    self.encoder.pool_step_reprs(layer_h, step_token_indices)
                    for layer_h in layer_hiddens
                ],
                dim=1,
            )
            predictions = self.head(h_step, batch_size=(B, S), h_layers=h_layers)
        else:
            predictions = self.head(h_step, batch_size=(B, S))
        return predictions, objective_data, new_cache

    def head(
        self,
        h: torch.Tensor,
        batch_size: tuple[int, int],
        *,
        h_layers: torch.Tensor | None = None,
    ) -> TensorDict:
        """Run enabled heads on step representations ``[B, S, D]``."""
        h = h.float()
        if h_layers is not None:
            h_layers = h_layers.float()
        tensors: dict[str, torch.Tensor] = {}
        for name, head_fn in self._heads.items():
            if name == "action_value_layerwise":
                if h_layers is None:
                    raise ValueError("action_value_layerwise head requires h_layers from Model.forward.")
                tensors["action_value_layerwise"] = head_fn.forward(h_layers)
                if hasattr(head_fn, "target_forward"):
                    tf = getattr(head_fn, "target_forward")
                    tensors["action_value_layerwise_target"] = tf(h_layers)
            elif name == "action_value":
                tensors["action_value"] = head_fn.forward(h)
                if hasattr(head_fn, "target_forward"):
                    tf = getattr(head_fn, "target_forward")
                    tensors["action_value_target"] = tf(h)
            elif name == "action_vector":
                tensors["action_vector"] = head_fn.forward(h)
                if hasattr(head_fn, "target_forward"):
                    tf = getattr(head_fn, "target_forward")
                    tensors["action_vector_target"] = tf(h)
            else:
                tensors[name] = head_fn.forward(h)
        return TensorDict(tensors, batch_size=batch_size)

    def polyak_update(
        self,
        action_value_tau: float = 0.0,
        action_value_layerwise_tau: float = 0.0,
        action_vector_tau: float = 0.0,
    ) -> None:
        """Soft-update target heads (for heads that support targets)."""
        if "action_value" in self._heads:
            hd = self._heads["action_value"]
            if hasattr(hd, "polyak_update"):
                pu = getattr(hd, "polyak_update")
                pu(tau=action_value_tau)
        if "action_value_layerwise" in self._heads:
            hl = self._heads["action_value_layerwise"]
            if hasattr(hl, "polyak_update"):
                pu = getattr(hl, "polyak_update")
                pu(tau=action_value_layerwise_tau)
        if "action_vector" in self._heads:
            hv = self._heads["action_vector"]
            if hasattr(hv, "polyak_update"):
                pu = getattr(hv, "polyak_update")
                pu(tau=action_vector_tau)

    def get_action(
        self,
        out: TensorDict,
        temperature: float = 1.0,
        num_actions: int | None = None,
    ) -> torch.Tensor:
        """Select an action using ``action_head`` from the last step."""
        raw = cast(torch.Tensor, out[self.action_head])[:, -1]
        if self.action_head == "action_value_layerwise":
            scores = raw[:, -1, :]
        elif self.action_head == "action_vector":
            scores = vector_action_scores(raw)
        else:
            scores = raw
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)