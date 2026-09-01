from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.backbone.base import Backbone
from mouse_core.models.backbone.flex_decode import packed_rope_positions
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.discrete_action import DiscreteActionHead
from mouse_core.models.heads.dqn import DiscreteActionValueHead
from mouse_core.models.heads.layerwise_dqn import LayerwiseDiscreteActionValueHead
from mouse_core.models.heads.swiglu import SwiGLUHead

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

`NumericEmbedder` maps a tokenized :class:`~mouse_core.data.token_batch.TokenBatch`
(discrete ids / continuous values) into the shared `{config["hidden_dim"]}`-dimensional
token space before the backbone.

{modality_table}

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
`DataLoader(transform=train_transform)`. Online / inference uses a per-step
`eval_transform` (compose selector/tokenizer, no augmenter → `StepTokens`) and
`pack_token_batch` when combining steps. The tokenizer is not part of the
saved model.

```python
from mouse_core.data import NumericTokenizer, compose, pack_token_batch

tokenizer = NumericTokenizer(
    input_fields=[...],  # input_field=; optional output_field= matches embedder field=
    objective_fields=[
        {{"input_field": "action"}},
        {{"input_field": "reward"}},
        {{"input_field": "episode_done"}},
        {{"input_field": "task_done"}},
    ],
    grouping_field="task_index",
)
eval_transform = tokenizer  # plus compose(selector, tokenizer) as needed

{objective_data_example}

with torch.no_grad():
    steps = [eval_transform(step) for step in batch[0]]
    inputs, _ = pack_token_batch(steps, sequence_ids=[0] * len(steps))
    predictions, averager_inputs = model(inputs)
    action = model.get_action(predictions, temperature=0.0)
```

`model()` returns `(predictions, averager_inputs)`. `pack_token_batch` /
`DataLoader.next_batch()` return `(inputs, objective_data)`; pass
`objective_data` to objectives during training. For cached incremental
rollout, pass ``averager_inputs`` back as ``cache=`` with `use_cache=True`.
Cached batch rows may have different
lengths on every call (e.g. envs emitting different numbers of steps between
model calls): decoding runs through a FlexAttention session carried in the
cache, so each row decodes exactly as it would alone.
"""
    path.write_text(text, encoding="utf-8")


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


def _model_card_modality_notes(modality: dict[str, Any]) -> str:
    modality_type = modality["type"]
    parts: list[str] = []
    if modality_type == "discrete":
        vocab_size = modality.get("vocab_size")
        if vocab_size is not None:
            parts.append(f"integer ids in `[0, {int(vocab_size) - 1}]`")
    elif modality_type == "fourier":
        parts.append("scalar value")
    elif modality_type == "continuous":
        parts.append("vector values")
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
predictions, averager_inputs = model(inputs)"""


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
                "fourier_min": float(encoder.fourier_min),
                "fourier_max": float(encoder.fourier_max),
                "std": float(encoder.std),
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
                "vocab_size": encoder.vocab_size,
                "padding_idx": encoder.padding_idx,
            },
        }
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
    return Model(
        encoder=encoder,
        backbone=backbone,
        heads=heads,
        action_head=heads_cfg.get("action_head"),
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
class AveragerInputs:
    """Per-forward extras for delayed Q and incremental decode.

    Pass to :class:`~mouse_core.polyak.PolyakAverager`: head scope uses ``h``,
    model scope uses ``batch``. Incremental decode uses ``cache`` (or pass this
    object back as ``cache=``). ``h`` is always detached so gradients cannot
    flow back through the averager.
    """

    h: torch.Tensor
    batch: TokenBatch
    cache: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.h = self.h.detach()


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
        - a list of head instances (e.g. ``[DiscreteActionValueHead(...), SwiGLUHead(...)]``):
          you **must** also pass ``action_head`` (a canonical name) to select which one
          ``get_action`` uses;
        - a dict mapping canonical names (``"action_value"``, ``"action"``, ``"value"``)
          to head instances or ``None`` (for full control and/or multiple heads).
      When a plain head (SwiGLUHead) is passed without a name it defaults to ``"action"``;
      use the dict form if you want it under ``"value"``.

    ``action_head`` names which head ``get_action`` consults. If omitted,
    it is auto-selected by preference: ``action_value`` > ``action`` > ``value``.

    The only supported construction is the explicit three-piece composition:

        encoder = NumericEmbedder(modalities=..., hidden_dim=...)
        backbone = LlamaBackbone(...)   # or any Backbone
        heads = DiscreteActionValueHead(...)            # or a dict/list of heads

        model = Model(encoder=encoder, backbone=backbone, heads=heads)

    The backbone is independent; it does not know about the encoder or heads.
    DQN target Q is ``averager(averager_inputs)`` where ``averager_inputs``
    comes from that same ``model(inputs)`` call. Pooled head input
    is ``averager_inputs.h`` (last-layer ``[N, D]``, or stacked layers
    ``[N, L, D]`` when a layerwise head is enabled).
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
        encoder: Encoder,
        backbone: Backbone,
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
        self.backbone: Backbone = backbone

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
            out = getattr(h, "out_features", None)
            if isinstance(out, int) and out > 0:
                self.max_num_actions = out
                break

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
        self.encoder.to(dtype=dtype, **common)
        self.backbone.to(dtype=dtype, **common)
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
        encoder: Encoder,
        session_out: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]],
        prediction_indices: torch.Tensor,
        needs_layerwise: bool,
    ) -> torch.Tensor:
        """Pool backbone hidden states to the tensor :meth:`head` consumes."""
        if needs_layerwise:
            _, layer_hiddens = cast(
                tuple[torch.Tensor, tuple[torch.Tensor, ...]], session_out
            )
            return torch.stack(
                [
                    encoder.pool_step_reprs(layer_h, prediction_indices)
                    for layer_h in layer_hiddens
                ],
                dim=1,
            )
        h = cast(torch.Tensor, session_out)
        return encoder.pool_step_reprs(h, prediction_indices)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: TokenBatch,
        cache: dict[str, Any] | AveragerInputs | None = None,
        use_cache: bool = False,
    ) -> tuple[TensorDict, AveragerInputs]:
        """Run a full forward pass over a :class:`TokenBatch`.

        Training: ``inputs, objective_data = loader.next_batch()`` then
        ``predictions, averager_inputs = model(inputs)``. Delayed DQN
        targets are ``averager(averager_inputs)``. Online / inference:
        ``inputs, _ = pack_token_batch([eval_transform(step)],
        sequence_ids=[0])`` then ``model(inputs, use_cache=True)``
        (optionally ragged; empty-only batches raise). Pass the returned
        :class:`AveragerInputs` (or its ``.cache``) back as ``cache=``.

        Training attention uses FlexAttention over the flat concatenated token
        stream (causal within the same ``(sequence_id, grouping_id)`` run). Cached
        decode keeps ``FlexDecodeSession`` with per-sequence KV caches and the
        same grouping-id isolation.

        Predictions are flat over steps (``N`` steps) for training; cached
        decode returns rectangular ``[B, S]`` tensors.
        """
        from mouse_core.data.token_batch import TokenBatch as _TokenBatch

        if not isinstance(batch, _TokenBatch):
            raise TypeError(
                f"Model.forward expects a TokenBatch, got {type(batch).__name__}. "
                "Use pack_token_batch([transform(step)], ...) "
                "or DataLoader(transform=...)."
            )
        if isinstance(cache, AveragerInputs):
            cache = cache.cache
        if cache is not None and not use_cache:
            raise ValueError("Passing cache= requires use_cache=True.")

        token_batch = batch
        B = token_batch.B
        step_counts_np = token_batch.step_counts()
        N = token_batch.N
        S_max = token_batch.S

        if use_cache and B > 0 and N == 0:
            raise ValueError("Model.forward requires at least one non-empty row in batch.")

        embeds, prediction_indices = self.encoder(token_batch)
        # embeds: [L, D]; prediction_indices: [N]

        t = token_batch.to_tensors(embeds.device)
        sequence_ids = t["sequence_ids"]
        grouping_ids = t["grouping_ids"]

        needs_layerwise = "action_value_layerwise" in self._heads
        new_cache: dict[str, Any] | None

        if use_cache:
            from mouse_core.models.embedding.packing import left_align_content

            batched_embeds, token_lengths, local_indices, batched_grouping_ids = (
                _flat_to_batched_left_pad(
                    embeds,
                    sequence_ids,
                    prediction_indices,
                    B,
                    S_max,
                    step_counts_np.tolist(),
                    grouping_ids=grouping_ids,
                )
            )
            session = cache["session"] if cache else self.backbone.decode_session(
                batch_size=B, capacity=max(batched_embeds.shape[1], 1)
            )
            flex_embeds, prediction_indices = left_align_content(
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
            session_out = session.forward(
                output_hidden_states=needs_layerwise,
                embeds=flex_embeds,
                lengths=token_lengths,
                grouping_ids=flex_grouping_ids,
            )
            new_cache = {"session": session}
            pred_batch_size: tuple[int, ...] = (B, S_max)
        else:
            # Training: Flex packed on CUDA; SDPA mask fallback on CPU (no Flex backward).
            session_out = self._train_backbone_forward(
                self.backbone,
                embeds,
                sequence_ids,
                grouping_ids,
                needs_layerwise,
            )
            new_cache = None
            pred_batch_size = (N,)

        h = self._pool_backbone_out(
            self.encoder, session_out, prediction_indices, needs_layerwise
        )
        predictions = self.head(h=h, batch_size=pred_batch_size)
        return predictions, AveragerInputs(h=h, batch=token_batch, cache=new_cache)

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
        """Select an action using ``action_head`` from the last step."""
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
    prediction_indices: torch.Tensor,
    B: int,
    S: int,
    step_counts: list[int],
    *,
    grouping_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Scatter flat ``[L, D]`` embeds into a rectangular ``[B, Lmax, D]`` layout.

    Content is packed from index 0 within each row (right-padded).
    ``prediction_indices`` is flat ``[N]``. Returns local rectangular
    ``prediction_indices`` ``[B, S]`` with real steps in trailing columns
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
            abs_i = int(prediction_indices[flat_offset + s_local].item())
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
