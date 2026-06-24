from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.heads.swiglu import SwiGLUHead
from mouse_core.models.heads.vec_dqn import VectorActionValueHead, vector_action_scores


def _hub_repo_id_for_user(repo_id: str, token: str | bool | None = None) -> str:
    """Resolve an unscoped Hub repo name under the authenticated user."""
    if "/" in repo_id:
        return repo_id

    from huggingface_hub import HfApi

    user = HfApi().whoami(token=token)["name"]
    return f"{user}/{repo_id}"


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
        existing = list(api.list_repo_files(hub_repo_id))
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
    step_stream_example = _model_card_step_stream_example(modalities)
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

`StepEmbedder` reads flat step-record dicts and projects each declared modality
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
{step_stream_example}

with torch.no_grad():
    out, _, cache = model(batch)
    action = model.get_action(out, temperature=0.0)
```

`model()` returns `(out, step_stream, cache)`. `step_stream` is a
`TensorDict[B, S]` of the modality tensors extracted by the encoder — pass it
to objectives during training. For cached one-step rollout, keep `cache` and
pass it back on the next call with `use_cache=True`.
"""
    path.write_text(text, encoding="utf-8")


def _model_card_modality_table(modalities: list[dict[str, Any]]) -> str:
    rows = [
        "| Field | Embed | Required | Tensor shape | Dtype | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for modality in modalities:
        name = str(modality["name"])
        embed = str(modality["embed"])
        required = bool(modality.get("required", True))
        allow_none = bool(modality.get("allow_none", False))
        rows.append(
            "| "
            + " | ".join([
                f"`{name}`",
                f"`{embed}`",
                "yes" if required else "no",
                f"`{_model_card_modality_shape(modality)}`",
                f"`{_model_card_modality_dtype(modality)}`",
                _model_card_modality_notes(modality, allow_none=allow_none),
            ])
            + " |"
        )
    return "\n".join(rows)


def _model_card_modality_shape(modality: dict[str, Any]) -> str:
    embed = modality["embed"]
    if embed in ("continuous", "image"):
        dim = modality.get("dim") or modality.get("size") or "D"
        return f"[B, S, {dim}]"
    if embed == "learnable":
        return "not read from step_stream"
    return "[B, S]"


def _model_card_modality_dtype(modality: dict[str, Any]) -> str:
    embed = modality["embed"]
    if embed == "discrete":
        return "torch.long"
    if embed == "image":
        return "torch.long or torch.float32"
    if embed == "learnable":
        return "n/a"
    return "torch.float32"


def _model_card_modality_notes(modality: dict[str, Any], *, allow_none: bool) -> str:
    embed = modality["embed"]
    parts: list[str] = []
    if embed == "discrete":
        vocab_size = modality.get("vocab_size") or modality.get("size")
        if vocab_size is not None:
            parts.append(f"integer ids in `[0, {int(vocab_size) - 1}]`")
    elif embed == "rff":
        parts.append("scalar value")
    elif embed == "continuous":
        parts.append("vector values")
    elif embed == "image":
        parts.append("pixel or patch values")
    elif embed == "learnable":
        parts.append("learned tokens; no input field")
    if allow_none:
        parts.append("`None` uses default value")
    return "; ".join(parts) or "-"


def _model_card_step_stream_example(modalities: list[dict[str, Any]]) -> str:
    fields = [
        _model_card_field_example(modality)
        for modality in modalities
        if modality["embed"] != "learnable"
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
out, step_stream, cache = model(batch)"""


def _model_card_field_example(modality: dict[str, Any]) -> str:
    name = modality["name"]
    embed = modality["embed"]
    optional = "" if modality.get("required", True) else "  # optional"
    if embed == "discrete":
        return f'"{name}": 0,{optional}'
    if embed == "rff":
        return f'"{name}": 0.0,{optional}'
    if embed == "continuous":
        dim = int(modality.get("dim") or modality.get("size") or 1)
        return f'"{name}": [0.0] * {dim},{optional}'
    if embed == "image":
        dim = int(modality.get("dim") or modality.get("size") or 1)
        return f'"{name}": [0] * {dim},{optional}'
    return f'"{name}": 0,{optional}'


def _model_config(model: "Model") -> dict[str, Any]:
    return {
        "format": "mouse-core-model-v1",
        "hidden_dim": int(model.hidden_dim),
        "encoder": _encoder_config(model.encoder),
        "backbone": _backbone_config(model.backbone),
        "heads": _heads_config(model),
    }


def _encoder_config(encoder: Encoder) -> dict[str, Any]:
    from mouse_core.models.embedding.embedding import StepEmbedder

    if not isinstance(encoder, StepEmbedder):
        raise TypeError(
            "save_model currently supports StepEmbedder encoders. "
            f"Got {type(encoder).__name__}."
        )
    return {
        "type": "step",
        "kwargs": {
            "hidden_dim": int(encoder.hidden_dim),
            "modalities": [_drop_none(asdict(modality)) for modality in encoder.modalities],
            "token_data_len": int(encoder.token_data_len),
            "num_compute_tokens": int(encoder.num_compute_tokens),
            "concat_modalities": bool(encoder.concat_modalities),
            "include_type_token": bool(encoder.include_type_token),
            "fourier_min": float(encoder.fourier_min),
            "fourier_max": float(encoder.fourier_max),
            "std": float(getattr(encoder, "std", 0.02)),
        },
    }


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
    force_download: bool = False,
    local_dir: str | Path | None = None,
    **kwargs: Any,
) -> "Model":
    """Load a MOUSE model from a local directory or HuggingFace Hub repo.

    Args:
        repo_id_or_path: A local path to a checkpoint directory or a HF Hub
            repo id (e.g. ``"my-model"`` or ``"your-org/your-model"``).
            Unscoped Hub names are resolved under the authenticated user.
        force_download: If ``True``, bypass the HF Hub cache and re-download.
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
    if config.get("type") != "step":
        raise ValueError(f"Unsupported encoder type {config.get('type')!r}.")
    from mouse_core.models.embedding import StepEmbedder

    return StepEmbedder(**config["kwargs"])


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
        if head_type == "action_value":
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

        encoder = StepEmbedder(...)
        backbone = LlamaBackbone(...)   # or any Backbone
        heads = DiscreteActionValueHead(...)            # or a dict/list of heads

        model = Model(encoder=encoder, backbone=backbone, heads=heads)

    The backbone is independent; it does not know about the encoder or heads.
    """

    _VALID_HEADS = ("action_value", "action_vector", "action", "value")

    @staticmethod
    def _normalize_heads(
        heads: BaseHead | list[BaseHead] | dict[str, BaseHead | None] | None,
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
        if isinstance(heads, dict):
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
        heads: BaseHead | list[BaseHead] | dict[str, BaseHead | None] | None = None,
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
            for candidate in ("action_vector", "action_value", "action", "value"):
                if candidate in self.heads:
                    self.action_head = candidate
                    break
            else:
                raise ValueError("No output head is enabled; cannot determine action_head.")

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

    # ------------------------------------------------------------------
    # Backbone adapter (delegates to composed backbone)
    # ------------------------------------------------------------------

    def backbone_forward(
        self,
        embeds: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run the composed backbone; provided for compatibility with external callers."""
        return self.backbone(
            embeds=embeds,
            cache=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: list[list[dict]],
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[TensorDict, TensorDict, dict[str, Any] | None]:
        """Run a full forward pass.

        Args:
            batch: ``[B][S]`` list of raw step-record dicts, as returned by
                ``DataLoader.next_batch()`` or assembled manually for rollout.
            cache: Optional KV cache from a prior call.
            use_cache: If True, return an updated cache.
            cache_position: Optional position ids for incremental decode.
            attention_mask: Optional ``[B, T]`` (or broadcastable) mask. Positions
                corresponding to 0/False are ignored by attention. When None the
                full sequence is attended to (subject to causal masking inside
                the backbone).

        Returns:
            ``(out, step_stream, cache)`` where

            * ``out`` is a ``TensorDict[B, S]`` with one entry per enabled head.
            * ``step_stream`` is a ``TensorDict[B, S]`` of the modality tensors
              extracted by the encoder — the same values used for embedding.
              Pass this to objectives (e.g. ``dqn_objective(step_stream, out, cfg)``).
            * ``cache`` is the updated KV cache, or ``None`` when ``use_cache=False``.
        """
        B = len(batch)
        S = len(batch[0]) if B > 0 else 0

        embeds, col_values = self.encoder(batch)
        step_stream = TensorDict(col_values, batch_size=(B, S))

        h, new_cache = self.backbone(
            embeds=embeds,
            cache=cache,
            use_cache=use_cache,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )

        h_step = self.encoder.pool_step_reprs(h, (B, S)).float()
        return self.head(h_step, batch_size=(B, S)), step_stream, new_cache

    def head(self, h: torch.Tensor, batch_size: tuple[int, int]) -> TensorDict:
        """Run enabled heads on step representations ``[B, S, D]``."""
        tensors: dict[str, torch.Tensor] = {}
        for name, head_fn in self._heads.items():
            if name == "action_value":
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

    def polyak_update(self, action_value_tau: float = 0.0, action_vector_tau: float = 0.0) -> None:
        """Soft-update target heads (for heads that support targets)."""
        if "action_value" in self._heads:
            hd = self._heads["action_value"]
            if hasattr(hd, "polyak_update"):
                pu = getattr(hd, "polyak_update")
                pu(tau=action_value_tau)
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
        scores: torch.Tensor = vector_action_scores(raw) if self.action_head == "action_vector" else raw
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)