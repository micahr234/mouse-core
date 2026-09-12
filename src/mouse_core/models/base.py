from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.backbone.base import Backbone, _reject_dtype_cast
from mouse_core.models.backbone.flex_decode import DecodeKernel, FlexDecodeSession, packed_rope_positions
from mouse_core.models.backbone.packed_train import TrainKernel
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.discrete_action import DiscreteActionHead
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.heads.layerwise_dqn import LayerwiseDiscreteActionValueHead
from mouse_core.models.heads.swiglu import SwiGLUHead
from mouse_core.models.lora import LoRAConfig
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
        model2 = load_model(
            "./checkpoints/step-10000",
            train_kernel="flex", decode_kernel="flex", dtype=torch.float32,
        )
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
    lora_cfg = config["backbone"].get("lora")
    if lora_cfg:
        reasoner_line += (
            f"\n- LoRA: `rank={lora_cfg['rank']}`, `alpha={lora_cfg['alpha']}` "
            f"on `{', '.join(lora_cfg['targets'])}` (fp32 adapters over frozen base weights)"
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
    load_model(
        "{repo_id}",
        train_kernel="flex",
        decode_kernel="flex",
        dtype=preferred_dtype(device),
        map_location="cpu",
    )
    .eval()
    .to(device)
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
    out = model(inputs, use_cache=True)
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
            f"for `__text__` ids and image-field tokens in a tokenized "
            f":class:`~mouse_core.data.token_batch.TokenBatch`, mapping them into "
            f"the shared `{hidden}`-dimensional token space before the backbone. "
            f"Step templates and field packing live on `Tokenizer` "
            f"(not saved with the checkpoint).{learnable_note}"
        )
        tokenizer_snippet = (
            "from mouse_core.data import Tokenizer, pack_token_batch\n"
            "\n"
            "tokenizer = Tokenizer(\n"
            "    input_fields=[...],  # type/input_field=; text fields require format=\"{field}\";\n"
            "                         # flag exactly one field head_output=True (the Q readout tokens)\n"
            f'    pretrained="{pretrained}",\n'
            f"{objective_fields}\n"
            "eval_transform = tokenizer"
        )
        step_example = """# Rebuild the same Tokenizer used at train time, then pack steps.
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
        "from mouse_core.data import Tokenizer, pack_token_batch\n"
        "\n"
        "tokenizer = Tokenizer(\n"
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
    if isinstance(backbone, (LlamaBackbone, Qwen3Backbone)):
        config: dict[str, Any] = {
            "type": "llama" if isinstance(backbone, LlamaBackbone) else "qwen3",
            "hidden_dim": backbone.hidden_dim,
            "kwargs": dict(backbone._config_kwargs),
        }
        if backbone.lora is not None:
            config["lora"] = asdict(backbone.lora)
        return config
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
    action_head = model.action_head
    if isinstance(action_head, tuple):
        action_head = list(action_head)
    return {"action_head": action_head, "heads": heads}


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
    *,
    train_kernel: TrainKernel,
    decode_kernel: DecodeKernel,
    dtype: torch.dtype,
    force_download: bool = True,
    local_dir: str | Path | None = None,
    **kwargs: Any,
) -> "Model":
    """Load a MOUSE model from a local directory or HuggingFace Hub repo.

    Args:
        repo_id_or_path: A local path to a checkpoint directory or a HF Hub
            repo id (e.g. ``"my-model"`` or ``"your-org/your-model"``).
            Unscoped Hub names are resolved under the authenticated user.
        train_kernel: Kernel for the uncached (packed) forward of a
            transformer backbone, ``"varlen"``, ``"padded"``, or ``"flex"``.
        decode_kernel: Kernel for cached decode, ``"flex"``.
        dtype: Dtype of the transformer backbone's base weights
            (``preferred_dtype(device)`` for inference or a LoRA base,
            ``torch.float32`` to fine-tune them). The saved weights are cast
            into it.

            All three are execution choices for the loading machine, not
            model properties, so they are never stored in the checkpoint and
            must be given here (ignored by backbones without a transformer
            stack).
        force_download: If ``True`` (default), bypass the HF Hub cache and re-download.
            Ignored for local paths.
        local_dir: Directory where Hub files are saved after download.
            Ignored for local paths.
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

    model = _build_model_from_config(config, train_kernel=train_kernel, decode_kernel=decode_kernel, dtype=dtype)
    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    return model


def _build_model_from_config(
    config: dict[str, Any], *, train_kernel: TrainKernel, decode_kernel: DecodeKernel, dtype: torch.dtype
) -> "Model":
    encoder = _build_encoder_from_config(config["encoder"])
    backbone = _build_backbone_from_config(
        config["backbone"], train_kernel=train_kernel, decode_kernel=decode_kernel, dtype=dtype
    )
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
        action_head=heads_cfg["action_head"],
        reasoner=reasoner,
        recurrence=recurrence,
    )


def _build_encoder_from_config(config: dict[str, Any]) -> Encoder:
    enc_type = config.get("type")
    kwargs = dict(config.get("kwargs") or {})
    if enc_type == "numeric":
        from mouse_core.models.embedding import NumericEmbedder

        return NumericEmbedder(**kwargs)
    if enc_type == "text":
        from mouse_core.models.embedding import TextEmbedder

        # HF tokenizer / image_tokenizer are not part of the embedder; rebuild
        # Tokenizer separately for the data pipeline after load. The table
        # weights come from the saved state_dict, so build a fresh table of the
        # saved size instead of re-downloading ``pretrained``.
        pretrained = kwargs.pop("pretrained", None)
        encoder = TextEmbedder(**kwargs)
        encoder.pretrained = pretrained
        return encoder
    raise ValueError(f"Unsupported encoder type {enc_type!r}.")


def _build_backbone_from_config(
    config: dict[str, Any], *, train_kernel: TrainKernel, decode_kernel: DecodeKernel, dtype: torch.dtype
) -> Backbone:
    backbone_type = config.get("type")
    if backbone_type == "identity":
        from mouse_core.models.backbone import IdentityBackbone

        return IdentityBackbone(hidden_dim=config.get("hidden_dim"))
    lora_cfg = config.get("lora")
    lora = LoRAConfig(**lora_cfg) if lora_cfg is not None else None
    runtime: dict[str, Any] = dict(train_kernel=train_kernel, decode_kernel=decode_kernel, dtype=dtype)
    if backbone_type == "llama":
        from mouse_core.models.backbone import LlamaBackbone

        return LlamaBackbone(hidden_dim=config["hidden_dim"], lora=lora, **runtime, **config["kwargs"])
    if backbone_type == "qwen3":
        from mouse_core.models.backbone import Qwen3Backbone

        return Qwen3Backbone(hidden_dim=config["hidden_dim"], lora=lora, **runtime, **config["kwargs"])
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
    ``num_passes``). Pass ``out.cache`` back as ``cache=``. Call
    :meth:`close` when the rollout is finished so VMM pages unmap before
    the next train step (``__del__`` is too late if a CUDA graph still
    holds views).
    """

    sessions: tuple[FlexDecodeSession, ...]

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``) in every pass.

        The rows' next tokens start at position 0 and their KV pages return
        to the shared pool.
        """
        for session in self.sessions:
            session.reset_rows(rows)

    def close(self) -> None:
        """Unmap every session's VMM pool and drop captured CUDA graphs."""
        for session in self.sessions:
            session.close()


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
    can supervise each pass against the delayed model's matching pass::

        out = model(inputs)
        with torch.no_grad():
            delayed_out = delayed_model(inputs)
        losses = [
            objective(data, online.predictions, delayed.predictions)[0]
            for online, delayed in zip(out.passes, delayed_out.passes)
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

    The model is assembled from three pluggable parts:

    - ``encoder``: :class:`~mouse_core.models.embedding.embedding.Encoder`
      Converts a :class:`~mouse_core.data.token_batch.TokenBatch` into token
      embeddings ``[L, D]``.
    - ``backbone``: a :class:`~mouse_core.models.backbone.Backbone`-compatible
      module that maps encodings to last-layer hidden states ``[L, D]``.
    - ``heads``: heads can be provided in several ergonomic ways:
        - a single :class:`~mouse_core.models.heads.base.BaseHead` (e.g. ``DiscreteActionValueHead(...)``):
          it becomes the only enabled head;
        - a list of head instances (e.g. ``[DiscreteActionValueHead(...), SwiGLUHead(...)]``):
          names are inferred from type; ``action_head`` selects which
          ``get_action`` uses;
        - a dict mapping caller-chosen names to head instances or ``None``.
      When a plain head (SwiGLUHead) is passed without a name it defaults to ``"action"``;
      use the dict form to pick the key.

    ``action_head`` names which head(s) ``get_action`` consults. Required.
    A string selects one head; a sequence of names sums those heads'
    scores. ``reasoner`` and ``recurrence`` are required
    (pass ``None`` when unused) and cannot be combined.

    Full construction::

        encoder = NumericEmbedder(modalities=..., hidden_dim=...)
        backbone = LlamaBackbone(...)   # or any Backbone
        heads = DiscreteActionValueHead(...)            # or a dict/list of heads

        model = Model(
            encoder=encoder,
            backbone=backbone,
            heads=heads,
            action_head="action_value",
            reasoner=None,
            recurrence=None,
        )

    ``forward`` returns a :class:`ModelOutput` with ``predictions``,
    ``last_hidden_state``, and per-pass ``passes``. The delayed DQN model
    comes from :meth:`delayed_copy` (a copy of every trainable parameter;
    frozen weights shared by reference), runs on the same ``TokenBatch``,
    and is interpolated per section with :class:`~mouse_core.polyak.Polyak`.
    Recurrent-depth refinement is a
    :class:`~mouse_core.models.recurrence.Recurrence` section
    (``num_passes`` backbone passes per forward, saved with the model). See
    ``examples/12_train_offline_recurrent_dqn.ipynb``.
    """

    @staticmethod
    def _normalize_heads(
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None] | None,
        action_head: str | Sequence[str] | None,
    ) -> dict[str, BaseHead]:
        """Convert the flexible ``heads=`` argument into the internal ``name -> head`` dict.

        Supported inputs:
          - dict (caller-chosen names to head or None): passed through.
          - single BaseHead instance: becomes the only head; name is inferred
            from type (SwiGLUHead defaults to "action"; pass a string
            ``action_head`` to store it under that key).
          - list/tuple of BaseHead: each gets an inferred name; you *must* provide
            action_head= to declare which heads get_action() uses.
        """
        if heads is None:
            return {}

        # Dict form gives full control over names.
        if isinstance(heads, Mapping):
            filtered: dict[str, BaseHead] = {}
            for name, h in heads.items():
                if h is not None:
                    if not isinstance(name, str) or not name:
                        raise ValueError(
                            f"head name must be a non-empty string, got {name!r}"
                        )
                    if not isinstance(h, BaseHead):
                        raise TypeError(f"head {name!r} must be a BaseHead or None, got {type(h)}")
                    filtered[name] = h
            return filtered

        # Single head instance gives an implicit single-head model.
        if isinstance(heads, BaseHead):
            preferred = action_head if isinstance(action_head, str) else None
            name = Model._infer_head_name(heads, preferred=preferred)
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
                        f"Multiple heads would map to the same inferred name {nm!r}. "
                        "Use a dict form to provide distinct names, e.g. "
                        "heads={'action_value': h1, 'action': h2}."
                    )
                result[nm] = h

            if action_head is None:
                raise TypeError(
                    "When passing heads as a list you must also specify action_head= "
                    "to select the head(s) used by get_action()."
                )
            return result

        raise TypeError(
            f"heads must be a BaseHead, list[BaseHead], or dict[str, BaseHead|None], "
            f"got {type(heads)}"
        )

    @staticmethod
    def _infer_head_name(head: BaseHead, preferred: str | None = None) -> str:
        """Infer a default storage key from the head type when no dict key is given."""
        if isinstance(head, LayerwiseDiscreteActionValueHead):
            return "action_value_layerwise"
        if isinstance(head, DiscreteActionValueHead):
            return "action_value"
        if isinstance(head, SwiGLUHead):
            if preferred is not None:
                return preferred
            return "action"
        raise TypeError(
            f"Cannot infer a name for head of type {type(head).__name__}. "
            "Use the dict form with an explicit key."
        )

    def __init__(
        self,
        *,
        encoder: Encoder,
        backbone: Backbone,
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None],
        action_head: str | Sequence[str],
        reasoner: LatentReasoner | None,
        recurrence: Recurrence | None,
    ):
        """Construct a Model from encoder, backbone, and heads.

        Every argument is required. ``reasoner`` enables Coconut-style latent
        reasoning via ``forward(batch, reasoning=...)``. ``recurrence`` makes
        every forward run the backbone ``num_passes`` times through the
        adapter. Pass ``None`` for either unused section; the two cannot be
        combined.
        """
        super().__init__()

        if not isinstance(encoder, Encoder):
            raise TypeError("encoder must be an instance of Encoder (from mouse_core.models.embedding).")
        if not isinstance(backbone, Backbone):
            raise TypeError("backbone must be a Backbone (from mouse_core.models.backbone).")
        if reasoner is not None and recurrence is not None:
            raise ValueError("reasoner and recurrence cannot be combined on one model.")

        enc_dim = int(encoder.hidden_dim)
        bb_dim = getattr(backbone, "hidden_dim", None)
        if bb_dim is not None and enc_dim != bb_dim:
            raise ValueError(
                f"hidden_dim mismatch between encoder ({enc_dim}) and backbone ({bb_dim}). "
                "The embedder and the backbone must agree on the hidden dimension."
            )

        self.encoder: Encoder = encoder
        self.backbone: Backbone = backbone

        if reasoner is not None:
            if not isinstance(reasoner, LatentReasoner):
                raise TypeError(
                    f"reasoner must be a LatentReasoner, got {type(reasoner).__name__}."
                )
            if reasoner.hidden_dim != enc_dim:
                raise ValueError(
                    f"hidden_dim mismatch between reasoner ({reasoner.hidden_dim}) "
                    f"and model ({enc_dim})."
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

        # Normalize flexible heads input (single instance, list, or dict) into the
        # canonical internal dict form.
        heads_dict: dict[str, BaseHead] = Model._normalize_heads(heads, action_head)

        # Store heads for both state dict and typed access
        filtered: dict[str, BaseHead] = {}
        for name, head in heads_dict.items():
            if head is not None:
                if not isinstance(head, BaseHead):
                    raise TypeError(f"head {name!r} must be a BaseHead or None, got {type(head)}")
                filtered[name] = head
        self.heads = nn.ModuleDict(filtered)  # for parameters/state
        self._heads: dict[str, BaseHead] = filtered  # typed view for calling

        action_names = _action_head_names(action_head)
        missing = [name for name in action_names if name not in self.heads]
        if missing:
            raise ValueError(
                f"action_head names {missing} are not enabled; "
                f"heads are {tuple(self.heads)}."
            )
        self.action_head: str | tuple[str, ...] = (
            action_names[0] if len(action_names) == 1 else action_names
        )

        bb_layers: int | None = None
        for name, head in self._heads.items():
            if not isinstance(head, LayerwiseDiscreteActionValueHead):
                continue
            if bb_layers is None:
                bb_layers = _backbone_num_layers(self.backbone)
                if bb_layers is None:
                    raise ValueError(
                        f"{name} is layerwise and needs a backbone with a known "
                        "layer count (e.g. Qwen3Backbone or LlamaBackbone)."
                    )
            if head.num_backbone_layers != bb_layers:
                raise ValueError(
                    f"Layerwise head {name!r} expects {head.num_backbone_layers} "
                    f"backbone layers but backbone has {bb_layers}."
                )

        self.hidden_dim = enc_dim
        # Best-effort inference of action cardinality for introspection only.
        self.max_num_actions: int = 0
        for _name, h in self.heads.items():
            out = getattr(h, "out_features", None)
            if isinstance(out, int) and out > 0:
                self.max_num_actions = out
                break

    def delayed_copy(self) -> "Model":
        """Build the delayed model for TD targets: a frozen copy of this model.

        Every trainable parameter gets its own copy; every frozen parameter
        (``requires_grad=False`` — the base weights of a LoRA backbone) is
        shared by reference with the online model, so a delayed LoRA
        backbone costs one extra copy of the adapters, not of the base. A
        fully trainable fp32 backbone is copied whole. The copy has every
        parameter frozen and is left in ``train()`` mode.

        Run it as ``delayed(inputs)`` with the same ``TokenBatch`` (and
        ``reasoning=``) as the online forward, under ``torch.no_grad()``.
        Interpolate it with :class:`~mouse_core.polyak.Polyak`, which takes
        one ``tau`` per section (heads, encoder, backbone) on every update.

        Construct after ``model.to(...)``. Do not call ``requires_grad_`` /
        ``to`` on the delayed model: shared frozen parameters belong to the
        online model too.
        """
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError(
                "delayed_copy needs a trainable online model (no parameter requires grad)."
            )

        def _copy(module: nn.Module) -> nn.Module:
            shared = {id(p): p for p in module.parameters() if not p.requires_grad}
            delayed = copy.deepcopy(module, memo=shared)
            delayed.requires_grad_(False)
            delayed.train()
            return delayed

        return Model(
            encoder=cast(Encoder, _copy(self.encoder)),
            backbone=cast(Backbone, _copy(self.backbone)),
            heads={name: cast(BaseHead, _copy(head)) for name, head in self._heads.items()},
            action_head=self.action_head,
            reasoner=None if self.reasoner is None else cast(LatentReasoner, _copy(self.reasoner)),
            recurrence=(
                None if self.recurrence is None else cast(Recurrence, _copy(self.recurrence))
            ),
        )

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move the model to a device; never casts.

        Dtypes are fixed when the pieces are built: the transformer backbone
        takes ``dtype=`` (``torch.float32`` to fine-tune the base weights,
        ``preferred_dtype(device)`` for a frozen LoRA base or inference) and
        every other section — LoRA adapters, encoder, reasoner, recurrence,
        heads — is float32, which ``AdamW`` and ``Polyak`` require of every
        trainable parameter. Inputs are cast to the backbone dtype at the
        backbone boundary and its output back to fp32 for the heads. Passing
        a dtype here raises ``TypeError``.
        """
        _reject_dtype_cast("Model", *args, **kwargs)
        return super().to(*args, **kwargs)

    def half(self) -> "Model":
        raise TypeError("Model.half() is not supported; set the backbone dtype when building it.")

    def bfloat16(self) -> "Model":
        raise TypeError("Model.bfloat16() is not supported; set the backbone dtype when building it.")

    def double(self) -> "Model":
        raise TypeError("Model.double() is not supported; set the backbone dtype when building it.")

    def float(self) -> "Model":
        raise TypeError("Model.float() is not supported; set the backbone dtype when building it.")

    def _train_backbone_forward(
        self,
        backbone: Backbone,
        embeds: torch.Tensor,
        sequence_ids: torch.Tensor,
        grouping_ids: torch.Tensor,
        needs_layerwise: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Uncached backbone pass over the flat packed stream.

        Transformer backbones run :func:`packed_forward` with the backbone's
        ``train_kernel`` (``"varlen"``: flash varlen on CUDA bf16/fp16,
        masked SDPA otherwise; ``"padded"``: dense causal SDPA on
        segments padded to ``max_seqlen``; ``"flex"``: FlexAttention). Backbones
        without a decoder stack (``IdentityBackbone``, custom) take the
        rectangular route with a dense sequence/grouping mask. ``embeds``
        come from the fp32 encoder / adapters and are cast to the backbone's
        base dtype here.
        """
        embeds = embeds.to(dtype=backbone.dtype)
        transformer = getattr(backbone, "model", None)
        if transformer is not None and hasattr(transformer, "layers"):
            from mouse_core.models.backbone.packed_train import packed_forward

            return packed_forward(
                model=cast(nn.Module, transformer),
                embeds=embeds,
                sequence_ids=sequence_ids,
                grouping_ids=grouping_ids,
                output_hidden_states=needs_layerwise,
                checkpoint=backbone.gradient_checkpointing,
                train_kernel=backbone.train_kernel,
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
        batch: TokenBatch,
        *,
        cache: DecodeCache | None = None,
        use_cache: bool = False,
        reasoning: Sequence[int] | np.ndarray | None = None,
    ) -> ModelOutput:
        """Run a forward pass over a :class:`TokenBatch`.

        Training: ``inputs, objective_data = loader.next_batch()`` then
        ``out = model(inputs)``. Delayed DQN: ``delayed_model =
        model.delayed_copy()`` then ``delayed_model(inputs)`` under
        ``torch.no_grad()`` (same ``TokenBatch`` and ``reasoning=`` as the
        online forward); interpolate with ``Polyak(model, delayed_model)``
        and ``polyak.update(tau_heads=..., tau_encoder=..., tau_backbone=...)``.
        Online / inference: ``inputs, _ = pack_token_batch([eval_transform(step)],
        sequence_ids=[0])`` then ``model(inputs, use_cache=True)``
        (optionally ragged; empty-only batches raise). Pass ``out.cache``
        back as ``cache=``.

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
        ``head_output_indices`` describe the extended stream.

        Training attention runs the packed stream forward over the flat
        concatenated token stream (causal within the same ``(sequence_id,
        grouping_id)`` class; kernel chosen by ``backbone.train_kernel``). Cached
        decode keeps one ``FlexDecodeSession`` per backbone pass, a paged KV
        pool in which each sequence owns only the pages its own history needs,
        with the same grouping-id isolation. On CUDA the pool grows by mapping
        more physical pages (no copy of existing K/V). Call
        ``out.cache.close()`` when the rollout ends.

        Training predictions are flat over head-output tokens (``[P, ...]``,
        one row per head-output token; ``objective_data["head_output_count"]``
        maps rows to steps). Cached decode returns rectangular ``[B, S]``
        tensors pooled at each step's last head-output token.
        """
        from mouse_core.data.token_batch import TokenBatch as _TokenBatch

        if cache is not None and not use_cache:
            raise ValueError("Passing cache= requires use_cache=True.")
        if not isinstance(batch, _TokenBatch):
            raise TypeError(
                f"Model.forward expects a TokenBatch, got {type(batch).__name__}. "
                "Use pack_token_batch([transform(step)], ...) "
                "or DataLoader(transform=...)."
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
                sessions = tuple(
                    self.backbone.decode_session(batch_size=B) for _ in range(num_passes)
                )
            flex_embeds, resolved_indices = left_align_content(
                batched_embeds, local_indices, token_lengths
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
        *,
        temperature: float,
        num_actions: int | None = None,
    ) -> torch.Tensor:
        """Select an action at the last head-output token of each decode row.

        ``out`` must be cached-decode scores ``[B, S, A]`` (or layerwise
        ``[B, S, L, A]``). Flat training outputs ``[N, A]`` are rejected
        unless ``N == 1``.

        Scores come from ``action_head``: one name, or the sum of each
        named head.
        """
        names = _action_head_names(self.action_head)
        scores = _last_action_scores(
            cast(torch.Tensor, out[names[0]]),
            name=names[0],
            head=self._heads[names[0]],
        )
        for name in names[1:]:
            scores = scores + _last_action_scores(
                cast(torch.Tensor, out[name]),
                name=name,
                head=self._heads[name],
            )
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _action_head_names(action_head: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize ``action_head`` to one or more non-empty names."""
    if isinstance(action_head, str):
        names = (action_head,)
    else:
        names = tuple(action_head)
    if not names:
        raise ValueError("action_head must name at least one head.")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"action_head names must be non-empty strings, got {name!r}."
            )
    if len(set(names)) != len(names):
        raise ValueError(f"action_head has duplicate names: {names}.")
    return names


def _last_action_scores(
    raw: torch.Tensor, *, name: str, head: BaseHead
) -> torch.Tensor:
    """Last-step action scores from a head tensor.

    Layerwise Q: ``[B, S, L, A]`` / ``[N, L, A]`` → last step, deepest
    layer. Other Q / logit heads: ``[B, S, A]`` / ``[N, A]``.
    """
    if isinstance(head, LayerwiseDiscreteActionValueHead):
        if raw.ndim == 4:
            return raw[:, -1, -1, :]
        if raw.ndim == 3:
            if raw.shape[0] != 1:
                raise ValueError(
                    f"{name} has shape {tuple(raw.shape)}; "
                    "get_action on flat [N, L, A] training outputs needs N=1. "
                    "Use cached-decode [B, S, L, A] outputs for a batch."
                )
            return raw[-1, -1, :].unsqueeze(0)
        raise ValueError(
            f"{name} expects [B, S, L, A] or [N, L, A], "
            f"got {tuple(raw.shape)}"
        )
    if raw.ndim == 3:
        return raw[:, -1]
    if raw.ndim == 2:
        if raw.shape[0] != 1:
            raise ValueError(
                f"{name} has shape {tuple(raw.shape)}; get_action on flat "
                "[N, A] training outputs needs N=1 (one step). Use "
                "cached-decode [B, S, A] outputs for a batch."
            )
        return raw[-1].unsqueeze(0)
    raise ValueError(f"{name} expects [B, S, A] or [N, A], got {tuple(raw.shape)}")


def preferred_dtype(device: torch.device | str | None = None) -> torch.dtype:
    """Dtype for a frozen backbone base: ``bfloat16`` on CUDA, else ``float32``.

    Pass as the backbone ``dtype`` (``Qwen3Backbone(dtype=preferred_dtype(device),
    ...)`` or ``load_model(..., dtype=preferred_dtype(device))``) for a LoRA
    backbone (frozen base) or for inference; the CUDA flash varlen kernel
    needs bf16/fp16. Every trainable section is float32 regardless. To
    fine-tune the whole backbone, build it with ``dtype=torch.float32``.
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

    Same rule as the packed train forward and cached decode, so all three
    agree even when a grouping id recurs after a different one.
    """
    return packed_rope_positions(
        sequence_ids=sequence_ids, grouping_ids=grouping_ids
    ).unsqueeze(0)
