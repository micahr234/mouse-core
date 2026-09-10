from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.backbone.base import Backbone
from mouse_core.models.backbone.flex_decode import FlexDecodeSession, packed_rope_positions
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.discrete_action import DiscreteActionHead
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.heads.layerwise_dqn import LayerwiseDiscreteActionValueHead
from mouse_core.models.heads.swiglu import SwiGLUHead
from mouse_core.models.reasoner import LatentReasoner, _InsertionPlan, _plan_insertions
from mouse_core.models.recurrence import Recurrence

if TYPE_CHECKING:
    from mouse_core.data.token_batch import TokenBatch

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
    *,
    model: "Model",
    repo_id: str,
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

        url = push_model_to_hub(model=model, repo_id="my-model", clear=True)
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
        _write_model_card(repo_id=hub_repo_id, model=model, path=Path(tmp) / 'README.md')
        api.upload_folder(
            repo_id=hub_repo_id,
            folder_path=tmp,
            commit_message=commit_message,
            **kwargs,
        )
    return str(repo_url)


def _write_model_card(
    *,
    model: "Model",
    path: Path,
    repo_id: str,
) -> None:
    config = _model_config(model)
    heads = config["heads"]["heads"]
    head_names = ", ".join(head["name"] for head in heads) or "none"
    reasoner_cfg = config.get("reasoner")
    reasoner_line = (
        f"\n- Latent reasoner: `num_thoughts={reasoner_cfg['num_thoughts']}`"
        if reasoner_cfg
        else ""
    )
    recurrence_cfg = config.get("recurrence")
    if recurrence_cfg:
        reasoner_line += (
            f"\n- Recurrence: `num_passes={recurrence_cfg['num_passes']}` "
            "(applied on every forward, including cached decode)"
        )
    encoder_section, tokenizer_snippet, objective_data_example = _model_card_encoder_bits(
        config
    )
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
- Action head: `{config["heads"]["action_head"]}`{reasoner_line}

### Encoder

{encoder_section}

## Install MouseCore

```bash
pip install mouse-core
```

## Load The Model

```python
import torch
from mouse_core import load_model
from mouse_core.models import preferred_dtype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (
    load_model("{repo_id}", map_location="cpu")
    .eval()
    .to(device=device, dtype=preferred_dtype(device))
)
```

## Run Inference

Training and inference both take a `TokenBatch`. Training typically uses
`DataLoader(transform=compose(augmenter, tokenizer))`. Online / inference
uses the tokenizer (no augmenter → `StepTokens`) and
`pack_token_batch` when combining steps. The tokenizer is not part of the
saved model.

```python
{tokenizer_snippet}

{objective_data_example}

with torch.no_grad():
    steps = [eval_transform(step) for step in batch[0]]
    inputs, _ = pack_token_batch(steps, sequence_ids=[0] * len(steps))
    out = model(inputs)
    action = model.get_action(out.predictions, temperature=0.0)
```

`model()` returns a `ModelOutput` with `predictions` and
`last_hidden_state` (final pass; `out.passes` has every pass on a
recurrent model). `pack_token_batch` /
`DataLoader.next_batch()` return `(inputs, objective_data)`; pass
`objective_data` to objectives during training. For cached incremental
rollout, pass ``out.cache`` back as ``cache=`` with `use_cache=True`.
Cached batch rows may have different
lengths on every call (e.g. envs emitting different numbers of steps between
model calls): decoding runs through a FlexAttention session carried in the
cache, so each row decodes exactly as it would alone.
"""
    path.write_text(text, encoding="utf-8")


def _model_card_encoder_bits(config: dict[str, Any]) -> tuple[str, str, str]:
    """Return ``(encoder_section, tokenizer_snippet, step_example)`` for the card."""
    enc = config["encoder"]
    hidden = config["hidden_dim"]
    objective_fields = """    objective_fields=[
        {"input_field": "action"},
        {"input_field": "reward"},
        {"input_field": "episode_done"},
        {"input_field": "task_done"},
    ],
    grouping_field="task_index",
)"""
    if enc.get("type") == "text":
        pretrained = enc["kwargs"].get("pretrained") or "..."
        vocab = enc["kwargs"].get("vocab_size")
        vocab_note = f" (`vocab_size={vocab}`)" if vocab is not None else ""
        learnable = enc["kwargs"].get("learnable") or []
        learnable_note = ""
        if learnable:
            names = ", ".join(
                f"`{item.get('field') or 'learnable'}`" for item in learnable
            )
            learnable_note = (
                f" Optional learnable scratch tokens ({names}) are embedded "
                f"from a separate table and aligned by name with the tokenizer."
            )
        encoder_section = (
            f"`TextEmbedder` looks up pretrained token embeddings{vocab_note} "
            f"for `__text__` / `__vision__` ids in a tokenized "
            f":class:`~mouse_core.data.token_batch.TokenBatch`, mapping them into "
            f"the shared `{hidden}`-dimensional token space before the backbone. "
            f"Step templates and field packing live on `TextTokenizer` "
            f"(not saved with the checkpoint).{learnable_note}"
        )
        tokenizer_snippet = (
            "from mouse_core.data import TextTokenizer, pack_token_batch\n"
            "\n"
            "tokenizer = TextTokenizer(\n"
            "    input_fields=[...],  # type/input_field=; text fields require format=;\n"
            "                         # flag exactly one field head_output=True (the Q readout tokens)\n"
            '    format="...",\n'
            f'    pretrained="{pretrained}",\n'
            f"{objective_fields}\n"
            "eval_transform = tokenizer"
        )
        step_example = """# Rebuild the same TextTokenizer used at train time, then pack steps.
batch = [[
    {
        "action": 0,
        "observation": 1,
        "reward": 0.0,
        "episode_done": 0,
        "task_done": 0,
        "task_index": 0,
    }
]]"""
        return encoder_section, tokenizer_snippet, step_example

    modalities = enc.get("kwargs", {}).get("modalities", [])
    encoder_section = (
        f"`NumericEmbedder` maps a tokenized "
        f":class:`~mouse_core.data.token_batch.TokenBatch`\n"
        f"(discrete ids / continuous values) into the shared `{hidden}`-dimensional\n"
        f"token space before the backbone.\n\n"
        f"{_model_card_modality_table(modalities)}"
    )
    tokenizer_snippet = (
        "from mouse_core.data import NumericTokenizer, pack_token_batch\n"
        "\n"
        "tokenizer = NumericTokenizer(\n"
        "    input_fields=[...],  # input_field=; optional output_field= matches embedder field=;\n"
        "                         # flag exactly one field head_output=True (the Q readout tokens)\n"
        f"{objective_fields}\n"
        "eval_transform = tokenizer"
    )
    return encoder_section, tokenizer_snippet, _model_card_step_stream_example(modalities)


def _model_card_modality_table(modalities: list[dict[str, Any]]) -> str:
    rows = [
        "| Field | Type | Tensor shape | Dtype | Notes |",
        "|---|---|---|---|---|",
    ]
    for modality in modalities:
        field = modality.get("field")
        modality_type = str(modality["type"])
        rows.append(
            "| "
            + " | ".join([
                f"`{field}`" if field else "-",
                f"`{modality_type}`",
                f"`{_model_card_modality_shape(modality)}`",
                f"`{_model_card_modality_dtype(modality)}`",
                _model_card_modality_notes(modality),
            ])
            + " |"
        )
    return "\n".join(rows)


def _model_card_modality_shape(modality: dict[str, Any]) -> str:
    modality_type = modality["type"]
    if modality_type in ("continuous", "image"):
        dim = modality.get("dim") or "D"
        return f"[B, S, {dim}]"
    if modality_type == "learnable":
        return "not read from step_stream"
    return "[B, S]"


def _model_card_modality_dtype(modality: dict[str, Any]) -> str:
    modality_type = modality["type"]
    if modality_type == "discrete":
        return "torch.long"
    if modality_type == "image":
        return "torch.long"
    if modality_type == "learnable":
        return "n/a"
    return "torch.float32"


def _model_card_fourier_range(modality: dict[str, Any]) -> list[str]:
    fmin = modality.get("fourier_min")
    fmax = modality.get("fourier_max")
    if fmin is None or fmax is None:
        return []
    return [f"Fourier range `[{fmin}, {fmax}]`"]


def _model_card_modality_notes(modality: dict[str, Any]) -> str:
    modality_type = modality["type"]
    parts: list[str] = []
    if modality_type == "discrete":
        vocab_size = modality.get("vocab_size")
        if vocab_size is not None:
            parts.append(f"integer ids in `[0, {int(vocab_size) - 1}]`")
    elif modality_type == "fourier":
        parts.append("scalar value")
        parts.extend(_model_card_fourier_range(modality))
    elif modality_type == "continuous":
        parts.append("vector values")
        parts.extend(_model_card_fourier_range(modality))
    elif modality_type == "image":
        parts.append("token ids from an image tokenizer")
    elif modality_type == "learnable":
        parts.append("learned tokens; no input field")
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
steps = [eval_transform(batch[0][0])]  # per-step StepTokens; pack_token_batch for many
inputs, objective_data = pack_token_batch(steps, sequence_ids=[0])
out = model(inputs)"""


def _model_card_field_example(modality: dict[str, Any]) -> str:
    field = modality.get("field")
    if field is None:
        return "# learnable modality (no step field)"
    modality_type = modality["type"]
    if modality_type == "discrete":
        return f'"{field}": 0,'
    if modality_type == "fourier":
        return f'"{field}": 0.0,'
    if modality_type == "continuous":
        dim = int(modality.get("dim") or 1)
        return f'"{field}": [0.0] * {dim},'
    if modality_type == "image":
        return f'"{field}": [0, 1, 2],'  # example token ids
    return f'"{field}": 0,'


def _model_config(model: "Model") -> dict[str, Any]:
    if model.encoder is None or model.backbone is None:
        raise TypeError("save_model requires encoder and backbone.")
    config: dict[str, Any] = {
        "format": "mouse-core-model-v1",
        "hidden_dim": int(model.hidden_dim),
        "encoder": _encoder_config(model.encoder),
        "backbone": _backbone_config(model.backbone),
        "heads": _heads_config(model),
    }
    if model.reasoner is not None:
        config["reasoner"] = {"num_thoughts": int(model.reasoner.num_thoughts)}
    if model.recurrence is not None:
        config["recurrence"] = {"num_passes": int(model.recurrence.num_passes)}
    return config


def _encoder_config(encoder: Encoder) -> dict[str, Any]:
    from mouse_core.models.embedding.embedding import NumericEmbedder
    from mouse_core.models.embedding.text import TextEmbedder

    if isinstance(encoder, NumericEmbedder):
        return {
            "type": "numeric",
            "kwargs": {
                "hidden_dim": int(encoder.hidden_dim),
                "modalities": [_public_modality_config(modality) for modality in encoder.modalities],
            },
        }
    if isinstance(encoder, TextEmbedder):
        kwargs: dict[str, Any] = {
            "hidden_dim": int(encoder.hidden_dim),
            "pretrained": encoder.pretrained,
            "vocab_size": encoder.vocab_size,
            "padding_idx": encoder.padding_idx,
        }
        if encoder.learnable:
            kwargs["learnable"] = [
                _public_modality_config(spec) for spec in encoder.learnable
            ]
        return {"type": "text", "kwargs": kwargs}
    raise TypeError(
        "save_model currently supports NumericEmbedder and TextEmbedder encoders. "
        f"Got {type(encoder).__name__}."
    )


def _public_modality_config(modality: Any) -> dict[str, Any]:
    data = _drop_none(asdict(modality))
    if data.get("type") == "learnable":
        value = data.get("field")
        if isinstance(value, str) and value.startswith("__learnable_"):
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
    reasoner_cfg = config.get("reasoner")
    reasoner = (
        LatentReasoner(
            hidden_dim=int(config["hidden_dim"]),
            num_thoughts=int(reasoner_cfg["num_thoughts"]),
        )
        if reasoner_cfg is not None
        else None
    )
    recurrence_cfg = config.get("recurrence")
    recurrence = (
        Recurrence(
            hidden_dim=int(config["hidden_dim"]),
            num_passes=int(recurrence_cfg["num_passes"]),
        )
        if recurrence_cfg is not None
        else None
    )
    return Model(
        encoder=encoder,
        backbone=backbone,
        heads=heads,
        action_head=heads_cfg.get("action_head"),
        reasoner=reasoner,
        recurrence=recurrence,
    )


def _build_encoder_from_config(config: dict[str, Any]) -> Encoder:
    enc_type = config.get("type")
    kwargs = dict(config.get("kwargs") or {})
    kwargs.pop("extra_fields", None)  # removed; objective_fields live on the tokenizer
    if enc_type == "numeric":
        from mouse_core.models.embedding import NumericEmbedder

        return NumericEmbedder(**kwargs)
    if enc_type == "text":
        from mouse_core.models.embedding import TextEmbedder

        # HF tokenizer / image_processor are not part of the embedder; rebuild
        # TextTokenizer separately for the data pipeline after load. The table
        # weights come from the saved state_dict, so build a fresh table of the
        # saved size instead of re-downloading ``pretrained``.
        pretrained = kwargs.pop("pretrained", None)
        encoder = TextEmbedder(**kwargs)
        encoder.pretrained = pretrained
        return encoder
    raise ValueError(f"Unsupported encoder type {enc_type!r}.")


def _build_backbone_from_config(config: dict[str, Any]) -> Backbone:
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


def _last_hidden(
    session_out: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    """Last-layer hidden states (``[L, D]`` train, ``[B, S, D]`` decode)."""
    if isinstance(session_out, tuple):
        return session_out[0]
    return session_out


def _pool_head_outputs(
    h: torch.Tensor, head_output_indices: torch.Tensor
) -> torch.Tensor:
    """Gather head-output tokens from backbone states.

    ``h`` is ``[L, D]`` (flat packed) or ``[B, L, D]`` (decode).
    Train: ``head_output_indices`` is ``[P]`` into ``0 .. L-1``.
    Decode: ``head_output_indices`` is ``[B, S]`` into the token axis of ``h``.
    """
    if h.ndim == 2:
        return h[head_output_indices.reshape(-1)]
    B, S = head_output_indices.shape
    D = h.shape[-1]
    idx = head_output_indices.unsqueeze(-1).expand(B, S, D)
    return h.gather(1, idx)


def _layer_hiddens(
    session_out: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, ...] | None:
    """Per-layer hidden states when the backbone returned them, else ``None``."""
    if isinstance(session_out, tuple):
        return session_out[1]
    return None


def _run_heads(
    heads: dict[str, BaseHead],
    h: torch.Tensor,
    batch_size: tuple[int, ...] | None,
) -> TensorDict:
    """Run ``heads`` on pooled ``h``. Layerwise ``h`` is ``[N, L, D]`` or ``[B, L, S, D]``."""
    if batch_size is None:
        if "action_value_layerwise" in heads:
            if h.ndim == 3:
                batch_size = (int(h.shape[0]),)
            elif h.ndim == 4:
                batch_size = (int(h.shape[0]), int(h.shape[2]))
            else:
                batch_size = tuple(h.shape[:-1])
        else:
            batch_size = tuple(h.shape[:-1])
    h = h.float()
    tensors: dict[str, torch.Tensor] = {}
    for name, head_fn in heads.items():
        tensors[name] = head_fn.forward(h)
    return TensorDict(tensors, batch_size=batch_size)


@dataclass
class DecodeCache:
    """KV state carried between ``use_cache=True`` calls.

    One :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession`
    per backbone pass (a plain model has one; a recurrent model has
    ``num_passes``). Pass ``out.cache`` back as ``cache=``.
    """

    sessions: tuple[FlexDecodeSession, ...]

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``) in every pass."""
        for session in self.sessions:
            session.reset_rows(rows)


@dataclass
class PassOutput:
    """One backbone pass: last-layer states, per-layer states, head predictions.

    ``last_hidden_state`` is the last-layer residual stream (``[L, D]``
    train, ``[B, S, D]`` decode), on the autograd tape. ``hidden_states`` is
    the per-layer tuple when a layerwise head is enabled. ``predictions`` is
    the head TensorDict for this pass.
    """

    last_hidden_state: torch.Tensor
    predictions: TensorDict
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class ModelOutput:
    """Head predictions plus the token states they were read from.

    ``predictions``, ``last_hidden_state``, and ``hidden_states`` are the
    final backbone pass. ``passes`` holds every pass in order (one entry on
    a plain model; ``num_passes`` on a recurrent model) so a training loop
    can supervise each pass::

        out = model(inputs)
        with torch.no_grad():
            targets = [
                delayed_model(
                    last_hidden_state=p.last_hidden_state,
                    head_output_indices=out.head_output_indices,
                    hidden_states=p.hidden_states,
                ).predictions
                for p in out.passes
            ]

    ``head_output_indices`` maps token states to the rows heads read. A
    reasoning forward extends the stream, so those indices and the states
    describe the stream *with* latents inserted. Incremental decode carries
    ``cache`` — pass ``out.cache`` back as ``cache=`` with ``use_cache=True``.
    """

    predictions: TensorDict
    last_hidden_state: torch.Tensor
    passes: tuple[PassOutput, ...]
    head_output_indices: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    cache: DecodeCache | None = None


class Model(nn.Module):
    """Composable MOUSE model: encoder, backbone, and heads as distinct sections.

    The model is assembled from three pluggable parts (a heads-only delayed
    copy from :meth:`delayed_copy` has neither encoder nor backbone):

    - ``encoder``: :class:`~mouse_core.models.embedding.embedding.Encoder`
      Converts a :class:`~mouse_core.data.token_batch.TokenBatch` into token
      embeddings ``[L, D]``.
    - ``backbone``: a :class:`~mouse_core.models.backbone.Backbone`-compatible
      module that maps encodings to last-layer hidden states ``[L, D]``.
    - ``heads``: heads can be provided in several ergonomic ways:
        - a single :class:`~mouse_core.models.heads.base.BaseHead` (e.g. ``DiscreteActionValueHead(...)``):
          it becomes the only enabled head and the implicit ``action_head``;
        - a list of head instances (e.g. ``[DiscreteActionValueHead(...), SwiGLUHead(...)]``):
          you **must** also pass ``action_head`` (a canonical name) to select which one
          ``get_action`` uses;
        - a dict mapping canonical names (``"action_value"``, ``"action"``, ``"value"``)
          to head instances or ``None`` (for full control and/or multiple heads).
      When a plain head (SwiGLUHead) is passed without a name it defaults to ``"action"``;
      use the dict form if you want it under ``"value"``.

    ``action_head`` names which head ``get_action`` consults. If omitted,
    it is auto-selected by preference: ``action_value`` > ``action`` > ``value``.

    Full construction::

        encoder = NumericEmbedder(modalities=..., hidden_dim=...)
        backbone = LlamaBackbone(...)   # or any Backbone
        heads = DiscreteActionValueHead(...)            # or a dict/list of heads

        model = Model(encoder=encoder, backbone=backbone, heads=heads)

    ``forward`` returns a :class:`ModelOutput` with ``predictions``,
    ``last_hidden_state``, and per-pass ``passes``. The delayed DQN model
    comes from :meth:`delayed_copy`: heads-only (run on the online token
    states) or with a delayed encoder / backbone (run on the ``TokenBatch``);
    interpolate it with :class:`~mouse_core.polyak.Polyak`.
    Recurrent-depth refinement is a
    :class:`~mouse_core.models.recurrence.Recurrence` section
    (``num_passes`` backbone passes per forward, saved with the model). See
    ``examples/13_train_offline_recurrent_dqn.ipynb``.
    """

    _VALID_HEADS = ("action_value", "action_value_layerwise", "action", "value")

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
                    "(one of 'action_value', 'action', 'value') to select the head used by get_action()."
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
        encoder: Encoder | None = None,
        backbone: Backbone | None = None,
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None] | None = None,
        action_head: str | None = None,
        reasoner: LatentReasoner | None = None,
        recurrence: Recurrence | None = None,
    ):
        """Construct a Model from encoder, backbone, and heads.

        Encoder and backbone are both present on a trainable model and both
        absent on a heads-only delayed copy (which runs from
        ``last_hidden_state=``). ``reasoner`` enables Coconut-style latent
        reasoning via ``forward(batch, reasoning=...)``. ``recurrence``
        makes every forward run the backbone ``num_passes`` times through
        the adapter; the two extra sections cannot be combined.
        """
        super().__init__()

        if encoder is not None and not isinstance(encoder, Encoder):
            raise TypeError("encoder must be an instance of Encoder (from mouse_core.models.embedding).")
        if (encoder is None) != (backbone is None):
            raise TypeError(
                "encoder and backbone must be given together (or both omitted "
                "for a heads-only delayed copy)."
            )
        if reasoner is not None and backbone is None:
            raise TypeError("reasoner requires backbone.")
        if recurrence is not None and backbone is None:
            raise TypeError("recurrence requires backbone.")
        if reasoner is not None and recurrence is not None:
            raise ValueError("reasoner and recurrence cannot be combined on one model.")

        enc_dim = getattr(encoder, "hidden_dim", None) if encoder is not None else None
        bb_dim = getattr(backbone, "hidden_dim", None) if backbone is not None else None
        if enc_dim is not None and bb_dim is not None and enc_dim != bb_dim:
            raise ValueError(
                f"hidden_dim mismatch between encoder ({enc_dim}) and backbone ({bb_dim}). "
                "The embedder and the backbone must agree on the hidden dimension."
            )

        self.encoder: Encoder | None = encoder
        self.backbone: Backbone | None = backbone

        if reasoner is not None:
            if not isinstance(reasoner, LatentReasoner):
                raise TypeError(
                    f"reasoner must be a LatentReasoner, got {type(reasoner).__name__}."
                )
            dim = enc_dim if enc_dim is not None else bb_dim
            if dim is not None and reasoner.hidden_dim != dim:
                raise ValueError(
                    f"hidden_dim mismatch between reasoner ({reasoner.hidden_dim}) "
                    f"and model ({dim})."
                )
        self.reasoner: LatentReasoner | None = reasoner

        if recurrence is not None:
            if not isinstance(recurrence, Recurrence):
                raise TypeError(
                    f"recurrence must be a Recurrence, got {type(recurrence).__name__}."
                )
            if bb_dim is not None and recurrence.hidden_dim != bb_dim:
                raise ValueError(
                    f"hidden_dim mismatch between recurrence ({recurrence.hidden_dim}) "
                    f"and backbone ({bb_dim})."
                )
        self.recurrence: Recurrence | None = recurrence

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
            for candidate in ("action_value_layerwise", "action_value", "action", "value"):
                if candidate in self.heads:
                    self.action_head = candidate
                    break
            else:
                raise ValueError("No output head is enabled; cannot determine action_head.")

        if "action_value_layerwise" in self._heads:
            layerwise_head = self._heads["action_value_layerwise"]
            if not isinstance(layerwise_head, LayerwiseDiscreteActionValueHead):
                raise TypeError("action_value_layerwise head has unexpected type.")
            if self.backbone is not None:
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

        if encoder is not None:
            self.hidden_dim = int(encoder.hidden_dim)
        elif bb_dim is not None:
            self.hidden_dim = int(bb_dim)
        else:
            in_features = next(
                (getattr(h, "in_features", None) for h in self._heads.values()),
                None,
            )
            if not isinstance(in_features, int) or in_features < 1:
                raise ValueError(
                    "Cannot infer hidden_dim without encoder, backbone, or a head with in_features."
                )
            self.hidden_dim = int(in_features)
        # Best-effort inference of action cardinality for introspection only.
        self.max_num_actions: int = 0
        for _name, h in self.heads.items():
            out = getattr(h, "out_features", None)
            if isinstance(out, int) and out > 0:
                self.max_num_actions = out
                break

    def delayed_copy(
        self, *, encoder: bool = False, backbone: bool = False, heads: bool = False
    ) -> "Model":
        """Build the delayed model for TD targets.

        Each flag chooses whether that section is delayed — a section is
        delayed exactly when its Polyak ``tau`` is not ``1``. A section set to
        ``True`` is a frozen deep copy (``train()``) that
        :class:`~mouse_core.polyak.Polyak` interpolates; a section left
        ``False`` is the online module itself, shared by reference
        (equivalent to ``tau = 1`` every step). At least one flag must be
        ``True``. The reasoner / recurrence section follows ``backbone``.

        - ``delayed_copy(heads=True)`` — heads-only model. Called as
          ``delayed(last_hidden_state=out.last_hidden_state,
          head_output_indices=out.head_output_indices,
          hidden_states=out.hidden_states)``: the delayed heads read the
          online token states, so encoder, backbone, reasoning latents, and
          recurrent passes are shared with the online forward.
        - ``encoder=True`` and/or ``backbone=True`` — full model. Called as
          ``delayed(inputs)`` (same ``TokenBatch`` and ``reasoning=`` as the
          online forward, under ``torch.no_grad()``); it re-runs the trunk
          through the delayed / shared sections and then the delayed or
          shared heads.

        Construct after ``model.to(...)``. Do not call ``requires_grad_`` /
        ``to`` on a delayed model that shares sections: they would hit the
        online modules.
        """
        if self.backbone is None:
            raise ValueError("delayed_copy is for the online model, not a delayed copy.")
        if not (encoder or backbone or heads):
            raise ValueError(
                "delayed_copy needs at least one delayed section: pass encoder=True, "
                "backbone=True, and/or heads=True (a section with tau = 1 is shared)."
            )

        def _copy(module: nn.Module) -> nn.Module:
            delayed = copy.deepcopy(module)
            delayed.requires_grad_(False)
            delayed.train()
            return delayed

        delayed_heads = {
            name: (cast(BaseHead, _copy(head)) if heads else head)
            for name, head in self._heads.items()
        }
        if not encoder and not backbone:
            return Model(heads=delayed_heads, action_head=self.action_head)

        assert self.encoder is not None
        delayed_encoder = cast(Encoder, _copy(self.encoder)) if encoder else self.encoder
        delayed_backbone = cast(Backbone, _copy(self.backbone)) if backbone else self.backbone
        delayed_reasoner = (
            None
            if self.reasoner is None
            else (cast(LatentReasoner, _copy(self.reasoner)) if backbone else self.reasoner)
        )
        delayed_recurrence = (
            None
            if self.recurrence is None
            else (cast(Recurrence, _copy(self.recurrence)) if backbone else self.recurrence)
        )
        return Model(
            encoder=delayed_encoder,
            backbone=delayed_backbone,
            heads=delayed_heads,
            action_head=self.action_head,
            reasoner=delayed_reasoner,
            recurrence=delayed_recurrence,
        )

    def to(self, *args: Any, **kwargs: Any) -> "Model":
        """Move/cast the model; output heads always stay float32.

        Accepts every ``nn.Module.to`` form (``to(device)``, ``to(dtype)``,
        ``to(device, dtype)``, ``to(tensor)``, keyword variants). Only the
        encoder and backbone take the requested dtype; heads are cast to
        float32 so ``_run_heads`` (which feeds them fp32 features) matches.

        On CUDA, prefer ``model.to(device=device, dtype=preferred_dtype(device))``
        so the encoder/backbone run in bfloat16 and FlexAttention compiles.
        """
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)
        if not dtype.is_floating_point:
            raise TypeError(f"Model.to only accepts floating point dtypes, got {dtype}.")
        common: dict[str, Any] = {"non_blocking": non_blocking}
        if device is not None:
            common["device"] = device
        if memory_format is not None:
            common["memory_format"] = memory_format
        if self.encoder is not None:
            self.encoder.to(dtype=dtype, **common)
        if self.backbone is not None:
            self.backbone.to(dtype=dtype, **common)
        if self.reasoner is not None:
            self.reasoner.to(dtype=dtype, **common)
        if self.recurrence is not None:
            self.recurrence.to(dtype=dtype, **common)
        self.heads.to(dtype=torch.float32, **common)
        return self

    def half(self) -> "Model":
        return self.to(torch.float16)

    def bfloat16(self) -> "Model":
        return self.to(torch.bfloat16)

    def double(self) -> "Model":
        return self.to(torch.float64)

    def _train_backbone_forward(
        self,
        backbone: Backbone,
        embeds: torch.Tensor,
        sequence_ids: torch.Tensor,
        grouping_ids: torch.Tensor,
        needs_layerwise: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Training backbone pass: Flex packed on CUDA, SDPA mask on CPU."""
        transformer = getattr(backbone, "model", None)
        use_flex = (
            transformer is not None
            and hasattr(transformer, "layers")
            and embeds.device.type == "cuda"
        )
        if use_flex:
            from mouse_core.models.backbone.flex_train import flex_packed_forward

            assert transformer is not None
            return flex_packed_forward(
                output_hidden_states=needs_layerwise,
                model=cast(nn.Module, transformer),
                embeds=embeds,
                sequence_ids=sequence_ids,
                grouping_ids=grouping_ids,
            )
        attention_mask = _flat_sequence_causal_mask(
            dtype=embeds.dtype,
            sequence_ids=sequence_ids,
            grouping_ids=grouping_ids,
        )
        position_ids = _flat_sequence_position_ids(
            sequence_ids=sequence_ids,
            grouping_ids=grouping_ids,
        )
        session_out = backbone(
            embeds.unsqueeze(0),
            output_hidden_states=needs_layerwise,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if needs_layerwise:
            h0, layers = session_out
            if h0.ndim == 3:
                h0 = h0.squeeze(0)
            layers = tuple(x.squeeze(0) if x.ndim == 3 else x for x in layers)
            return (h0, layers)
        if isinstance(session_out, torch.Tensor) and session_out.ndim == 3:
            return session_out.squeeze(0)
        return session_out

    def _pool_backbone_out(
        self,
        session_out: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]],
        head_output_indices: torch.Tensor,
        needs_layerwise: bool,
    ) -> torch.Tensor:
        """Pool backbone hidden states to the tensor :meth:`head` consumes."""
        if needs_layerwise:
            _, layer_hiddens = cast(
                tuple[torch.Tensor, tuple[torch.Tensor, ...]], session_out
            )
            return torch.stack(
                [
                    _pool_head_outputs(layer_h, head_output_indices)
                    for layer_h in layer_hiddens
                ],
                dim=1,
            )
        h = cast(torch.Tensor, session_out)
        return _pool_head_outputs(h, head_output_indices)

    def _generate_latents(
        self,
        *,
        embeds: torch.Tensor,
        sequence_ids: torch.Tensor,
        grouping_ids: torch.Tensor,
        plan: _InsertionPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate latent thoughts on the autograd tape and assemble the extended stream.

        Runs ``R`` extra backbone passes over the growing per-burst prefixes
        (all bursts in lockstep; attention isolation makes dropping the other
        sequences exact). Thought ``r``'s input embedding is the reasoner
        adapter applied to the backbone output at the previous position, so
        gradients flow through the whole latent chain. Returns
        ``(ext_embeds, ext_sequence_ids, ext_grouping_ids,
        ext_head_output_indices, token_indices)`` where ``token_indices`` maps
        each original token to its extended-stream position.
        """
        reasoner = self.reasoner
        assert reasoner is not None
        assert self.backbone is not None
        R = plan.num_thoughts
        device = embeds.device
        nb = int(plan.burst_rows.size)

        prefix_bounds = list(
            zip(plan.prefix_starts.tolist(), plan.anchors.tolist())
        )
        thoughts: list[torch.Tensor] = []  # thought r → [nb, D]
        for r in range(R):
            parts: list[torch.Tensor] = []
            seq_parts: list[torch.Tensor] = []
            group_parts: list[torch.Tensor] = []
            last_positions: list[int] = []
            offset = 0
            for j, (start, anchor) in enumerate(prefix_bounds):
                parts.append(embeds[start:anchor])
                seq_parts.append(sequence_ids[start:anchor])
                group_parts.append(grouping_ids[start:anchor])
                if r > 0:
                    parts.append(torch.stack([thoughts[q][j] for q in range(r)]))
                    seq_parts.append(
                        sequence_ids.new_full((r,), int(plan.burst_rows[j]))
                    )
                    group_parts.append(
                        grouping_ids.new_full((r,), int(plan.latent_groups[j]))
                    )
                block = (anchor - start) + r
                last_positions.append(offset + block - 1)
                offset += block
            gen_out = self._train_backbone_forward(
                self.backbone,
                torch.cat(parts),
                torch.cat(seq_parts),
                torch.cat(group_parts),
                False,
            )
            h_last = cast(torch.Tensor, gen_out)[
                torch.as_tensor(last_positions, device=device)
            ]
            thoughts.append(reasoner(h_last))

        latent_embeds = torch.stack(thoughts, dim=1).reshape(nb * R, embeds.shape[-1])
        token_indices = torch.as_tensor(plan.token_positions, device=device)
        latent_indices = torch.as_tensor(plan.latent_positions, device=device)
        ext_embeds = (
            embeds.new_zeros(plan.ext_length, embeds.shape[-1])
            .index_copy(0, token_indices, embeds)
            .index_copy(0, latent_indices, latent_embeds)
        )
        ext_sequence_ids = torch.as_tensor(plan.ext_sequence_ids, device=device)
        ext_grouping_ids = torch.as_tensor(plan.ext_grouping_ids, device=device)
        ext_head_output_indices = torch.as_tensor(
            plan.ext_head_output_indices, device=device
        )
        return (
            ext_embeds,
            ext_sequence_ids,
            ext_grouping_ids,
            ext_head_output_indices,
            token_indices,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: TokenBatch | None = None,
        *,
        last_hidden_state: torch.Tensor | None = None,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
        head_output_indices: torch.Tensor | None = None,
        cache: DecodeCache | None = None,
        use_cache: bool = False,
        reasoning: Sequence[int] | np.ndarray | None = None,
    ) -> ModelOutput:
        """Run a forward pass over a :class:`TokenBatch`, or run heads on given states.

        Training: ``inputs, objective_data = loader.next_batch()`` then
        ``out = model(inputs)``. Delayed DQN with heads-only delay:
        ``delayed_model = model.delayed_copy(heads=True)`` then
        ``delayed_model(last_hidden_state=out.last_hidden_state,
        head_output_indices=out.head_output_indices,
        hidden_states=out.hidden_states)``. With a delayed encoder and/or
        backbone (``delayed_copy(encoder=..., backbone=...)``) run
        ``delayed_model(inputs)`` instead. Interpolate with
        ``Polyak(model, delayed_model)`` and ``polyak.update(tau_heads=...)``
        (plus ``tau_encoder`` / ``tau_backbone`` for delayed trunk sections).
        Online / inference: ``inputs, _ = pack_token_batch([eval_transform(step)],
        sequence_ids=[0])`` then ``model(inputs, use_cache=True)``
        (optionally ragged; empty-only batches raise). Pass ``out.cache``
        back as ``cache=``.

        ``last_hidden_state=`` (heads-only delayed copy) skips encoder and
        backbone and runs the heads after pooling with
        ``head_output_indices``; ``hidden_states=`` supplies the per-layer
        states a layerwise head needs. ``head_output_indices=`` and
        ``hidden_states=`` are only accepted with ``last_hidden_state=``.

        A model with a :class:`~mouse_core.models.recurrence.Recurrence`
        section runs the backbone ``num_passes`` times (pass ``k`` reads
        ``recurrence(encodings, pass_{k-1}.last_hidden_state)``), with and
        without cache. ``predictions`` / ``last_hidden_state`` are the final
        pass; ``passes`` has every pass.

        ``reasoning`` (training only, requires ``Model(reasoner=...)``) is a
        ``[B]`` array of local step indices from
        :func:`~mouse_core.models.reasoner.sample_reasoning_splits`
        (``-1`` skips a row). For each selected step the model generates
        ``reasoner.num_thoughts`` latent thought embeddings on the autograd
        tape — each thought's input is the reasoner adapter applied to the
        backbone output at the previous position — and inserts them
        immediately before that step's first head-output token, so the Q
        readout and all later same-run tokens attend to them. Predictions
        keep the flat ``[P, ...]`` contract; ``last_hidden_state`` /
        ``head_output_indices`` describe the extended stream, so the
        heads-only delayed copy reads the same thoughts.

        Training attention uses FlexAttention over the flat concatenated token
        stream (causal within the same ``(sequence_id, grouping_id)`` run). Cached
        decode keeps one ``FlexDecodeSession`` per backbone pass with per-sequence
        KV caches and the same grouping-id isolation.

        Training predictions are flat over head-output tokens (``[P, ...]``,
        one row per head-output token; ``objective_data["head_output_count"]``
        maps rows to steps). Cached decode returns rectangular ``[B, S]``
        tensors pooled at each step's last head-output token.
        """
        from mouse_core.data.token_batch import TokenBatch as _TokenBatch

        if last_hidden_state is not None:
            if batch is not None:
                raise ValueError("Pass batch= or last_hidden_state=, not both.")
            if use_cache or cache is not None:
                raise ValueError("last_hidden_state= is not supported with use_cache=True.")
            if reasoning is not None:
                raise ValueError("last_hidden_state= is not supported with reasoning=.")
            if head_output_indices is None:
                raise ValueError("last_hidden_state= requires head_output_indices=.")
            return self._forward_from_hidden(
                last_hidden_state,
                hidden_states=hidden_states,
                head_output_indices=head_output_indices,
            )
        if head_output_indices is not None:
            raise ValueError("head_output_indices= is only accepted with last_hidden_state=.")
        if hidden_states is not None:
            raise ValueError("hidden_states= is only accepted with last_hidden_state=.")
        if cache is not None and not use_cache:
            raise ValueError("Passing cache= requires use_cache=True.")
        if batch is None:
            raise TypeError("Model.forward requires a TokenBatch (or last_hidden_state=).")
        if not isinstance(batch, _TokenBatch):
            raise TypeError(
                f"Model.forward expects a TokenBatch, got {type(batch).__name__}. "
                "Use pack_token_batch([transform(step)], ...) "
                "or DataLoader(transform=...)."
            )
        if self.encoder is None or self.backbone is None:
            raise ValueError(
                "this is a heads-only delayed copy; pass last_hidden_state= "
                "and head_output_indices= from the online ModelOutput."
            )

        token_batch = batch
        B = token_batch.B
        step_counts_np = token_batch.step_counts()
        N = token_batch.N
        S_max = token_batch.S

        if use_cache and B > 0 and N == 0:
            raise ValueError("Model.forward requires at least one non-empty row in batch.")

        plan: _InsertionPlan | None = None
        if reasoning is not None:
            if use_cache:
                raise ValueError("reasoning= is not supported with use_cache=True.")
            if self.reasoner is None:
                raise ValueError(
                    "Model.forward(reasoning=...) requires a reasoner; construct "
                    "Model(..., reasoner=LatentReasoner(...))."
                )
            plan = _plan_insertions(
                token_batch,
                np.asarray(reasoning, dtype=np.int64).reshape(-1),
                self.reasoner.num_thoughts,
            )

        embeds, resolved_indices = self.encoder(token_batch)
        # embeds: [L, D]; resolved_indices: [P]
        t = token_batch.to_tensors(embeds.device)
        sequence_ids = t["sequence_ids"]
        grouping_ids = t["grouping_ids"]

        needs_layerwise = "action_value_layerwise" in self._heads
        num_passes = self.recurrence.num_passes if self.recurrence is not None else 1
        new_cache: DecodeCache | None
        pass_outs: list[torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = []

        if use_cache:
            from mouse_core.models.embedding.packing import left_align_content

            # Decode pools one position per step: the step's *last* head-output
            # token (the most informed one when a step has several).
            psteps = t["head_output_steps"]
            last_of_step = torch.ones(
                psteps.shape[0], dtype=torch.bool, device=psteps.device
            )
            last_of_step[:-1] = psteps[1:] != psteps[:-1]
            batched_embeds, token_lengths, local_indices, batched_grouping_ids = (
                _flat_to_batched_left_pad(
                    embeds,
                    sequence_ids,
                    resolved_indices[last_of_step],
                    B,
                    S_max,
                    step_counts_np.tolist(),
                    grouping_ids=grouping_ids,
                )
            )
            if cache is not None:
                if len(cache.sessions) != num_passes:
                    raise ValueError(
                        f"cache has {len(cache.sessions)} sessions but this model "
                        f"runs {num_passes} backbone passes."
                    )
                sessions = cache.sessions
            else:
                capacity = max(batched_embeds.shape[1], 1)
                sessions = tuple(
                    self.backbone.decode_session(batch_size=B, capacity=capacity)
                    for _ in range(num_passes)
                )
            flex_embeds, resolved_indices = left_align_content(
                batched_embeds, local_indices
            )
            # Left-align mask ids to the same trailing-column layout as embeds.
            Lmax = batched_grouping_ids.shape[1]
            flex_grouping_ids = batched_grouping_ids.new_zeros(B, Lmax)
            for b, rl in enumerate(token_lengths):
                if rl == 0:
                    continue
                flex_grouping_ids[b, Lmax - rl :] = batched_grouping_ids[b, :rl]
            # ``token_lengths`` already counts only real tokens (tokenize is ragged;
            # empty rows contribute 0). Do not re-derive from left-padded step indices.
            pass_input = flex_embeds
            for session in sessions:
                session_out = session.forward(
                    output_hidden_states=needs_layerwise,
                    embeds=pass_input,
                    lengths=token_lengths,
                    grouping_ids=flex_grouping_ids,
                )
                pass_outs.append(session_out)
                if self.recurrence is not None:
                    pass_input = self.recurrence(flex_embeds, _last_hidden(session_out))
            new_cache = DecodeCache(sessions=tuple(sessions))
            pred_batch_size: tuple[int, ...] = (B, S_max)
        else:
            if plan is not None:
                # Gradient-taped latent generation; swaps in the extended stream.
                (
                    embeds,
                    sequence_ids,
                    grouping_ids,
                    resolved_indices,
                    _,
                ) = self._generate_latents(
                    embeds=embeds,
                    sequence_ids=sequence_ids,
                    grouping_ids=grouping_ids,
                    plan=plan,
                )
            # Training: Flex packed on CUDA; SDPA mask fallback on CPU (no Flex backward).
            pass_input = embeds
            for _ in range(num_passes):
                session_out = self._train_backbone_forward(
                    self.backbone,
                    pass_input,
                    sequence_ids,
                    grouping_ids,
                    needs_layerwise,
                )
                pass_outs.append(session_out)
                if self.recurrence is not None:
                    pass_input = self.recurrence(embeds, _last_hidden(session_out))
            new_cache = None
            pred_batch_size = (token_batch.P,)

        passes = tuple(
            PassOutput(
                last_hidden_state=_last_hidden(session_out),
                predictions=self.head(
                    h=self._pool_backbone_out(session_out, resolved_indices, needs_layerwise),
                    batch_size=pred_batch_size,
                ),
                hidden_states=_layer_hiddens(session_out) if needs_layerwise else None,
            )
            for session_out in pass_outs
        )
        final = passes[-1]
        return ModelOutput(
            predictions=final.predictions,
            last_hidden_state=final.last_hidden_state,
            passes=passes,
            head_output_indices=resolved_indices,
            hidden_states=final.hidden_states,
            cache=new_cache,
        )

    def _forward_from_hidden(
        self,
        last_hidden_state: torch.Tensor,
        *,
        hidden_states: tuple[torch.Tensor, ...] | None,
        head_output_indices: torch.Tensor,
    ) -> ModelOutput:
        if last_hidden_state.ndim not in (2, 3):
            raise ValueError(
                "last_hidden_state must have shape [L, D] or [B, S, D], "
                f"got {tuple(last_hidden_state.shape)}."
            )
        if last_hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"last_hidden_state last dim must be hidden_dim={self.hidden_dim}, "
                f"got {int(last_hidden_state.shape[-1])}."
            )
        needs_layerwise = "action_value_layerwise" in self._heads
        if needs_layerwise:
            if hidden_states is None:
                raise ValueError(
                    "layerwise heads require hidden_states= from the online ModelOutput."
                )
            h = torch.stack(
                [
                    _pool_head_outputs(layer_h, head_output_indices)
                    for layer_h in hidden_states
                ],
                dim=1,
            )
        else:
            h = _pool_head_outputs(last_hidden_state, head_output_indices)
        predictions = self.head(h=h)
        return ModelOutput(
            predictions=predictions,
            last_hidden_state=last_hidden_state,
            passes=(
                PassOutput(
                    last_hidden_state=last_hidden_state,
                    predictions=predictions,
                    hidden_states=hidden_states,
                ),
            ),
            head_output_indices=head_output_indices,
            hidden_states=hidden_states,
            cache=None,
        )

    def head(
        self,
        *,
        h: torch.Tensor,
        batch_size: tuple[int, ...] | None = None,
    ) -> TensorDict:
        """Run enabled heads on pooled ``h``.

        Regular heads take last-layer ``[N, D]`` or ``[B, S, D]``. A layerwise
        Q head takes stacked layers ``[N, L, D]`` or ``[B, L, S, D]``.
        ``batch_size`` defaults from ``h`` (step axis only for layerwise).
        """
        return _run_heads(self._heads, h, batch_size)

    def get_action(
        self,
        out: TensorDict,
        temperature: float = 1.0,
        num_actions: int | None = None,
    ) -> torch.Tensor:
        """Select an action using ``action_head`` at the last head-output token."""
        raw = cast(torch.Tensor, out[self.action_head])
        if self.action_head == "action_value_layerwise":
            if raw.ndim == 4:
                # Decode: [B, S, L, A] → last step, deepest layer
                scores = raw[:, -1, -1, :]
            elif raw.ndim == 3:
                # Train flat: [N, L, A] → last step, deepest layer
                scores = raw[-1, -1, :].unsqueeze(0)
            else:
                raise ValueError(
                    f"action_value_layerwise expects [B, S, L, A] or [N, L, A], "
                    f"got {tuple(raw.shape)}"
                )
        else:
            if raw.ndim == 3:
                scores = raw[:, -1]
            elif raw.ndim == 2:
                scores = raw[-1].unsqueeze(0)
            else:
                raise ValueError(
                    f"{self.action_head} expects [B, S, A] or [N, A], got {tuple(raw.shape)}"
                )
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def preferred_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """Compute dtype for encoder/backbone: ``bfloat16`` on CUDA, else ``float32``.

    Pass to ``Model.to(device=..., dtype=preferred_dtype(device))``. Heads stay
    float32 via :meth:`Model.to`. Train with :class:`mouse_core.AdamW`, or
    :class:`mouse_core.AdamWFp32` to keep fp32 masters of bf16 weights.
    CUDA FlexAttention only fuses for bf16/fp16.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _flat_to_batched_left_pad(
    embeds: torch.Tensor,
    sequence_ids: torch.Tensor,
    head_output_indices: torch.Tensor,
    B: int,
    S: int,
    step_counts: list[int],
    *,
    grouping_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Scatter flat ``[L, D]`` embeds into a rectangular ``[B, Lmax, D]`` layout.

    Content is packed from index 0 within each row (right-padded).
    ``head_output_indices`` is flat ``[N]`` (one index per step — the caller
    passes each step's last head-output token). Returns local rectangular
    ``head_output_indices`` ``[B, S]`` with real steps in trailing columns
    (left-padded in the step dimension for decode), plus right-padded
    ``grouping_ids`` ``[B, Lmax]`` aligned with the embed rows.
    """
    L, D = embeds.shape
    if grouping_ids.shape != (L,):
        raise ValueError(f"grouping_ids must have shape [{L}], got {tuple(grouping_ids.shape)}")
    token_lengths = [int((sequence_ids == b).sum().item()) for b in range(B)]
    Lmax = max(token_lengths) if token_lengths else 0
    out = embeds.new_zeros(B, Lmax, D)
    out_mask = grouping_ids.new_zeros(B, Lmax)
    local_indices = torch.zeros(B, S, device=embeds.device, dtype=torch.long)

    # Map absolute token index → local index within its sequence.
    local_of_abs = torch.full((L,), -1, device=embeds.device, dtype=torch.long)
    for b in range(B):
        mask = sequence_ids == b
        toks = embeds[mask]
        out[b, : toks.shape[0]] = toks
        out_mask[b, : toks.shape[0]] = grouping_ids[mask]
        abs_idx = torch.where(mask)[0]
        local_of_abs[abs_idx] = torch.arange(toks.shape[0], device=embeds.device)

    flat_offset = 0
    for b in range(B):
        n = int(step_counts[b]) if b < len(step_counts) else S
        for s_local in range(n):
            abs_i = int(head_output_indices[flat_offset + s_local].item())
            # Place into trailing step columns.
            local_indices[b, S - n + s_local] = int(local_of_abs[abs_i].item())
        flat_offset += n
    return out, token_lengths, local_indices, out_mask


def _flat_sequence_causal_mask(
    *,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive attention mask ``[1, 1, L, L]`` for a packed flat sequence."""
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    q = torch.arange(L, device=device)
    kv = torch.arange(L, device=device)
    causal = kv.unsqueeze(0) <= q.unsqueeze(1)
    same_seq = sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(0)
    same_mask = grouping_ids.unsqueeze(1) == grouping_ids.unsqueeze(0)
    allow = causal & same_seq & same_mask
    neg = torch.finfo(dtype).min
    mask = torch.where(
        allow,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), neg, device=device, dtype=dtype),
    )
    return mask.view(1, 1, L, L)


def _flat_sequence_position_ids(
    *,
    sequence_ids: torch.Tensor,
    grouping_ids: torch.Tensor,
) -> torch.Tensor:
    """RoPE positions ``[1, L]``: count of earlier same-``(sequence, grouping_id)`` tokens.

    Same rule as the FlexAttention train path and cached decode, so all three
    agree even when a grouping id recurs after a different one.
    """
    return packed_rope_positions(
        sequence_ids=sequence_ids, grouping_ids=grouping_ids
    ).unsqueeze(0)
