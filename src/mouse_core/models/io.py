"""Model checkpoints, configuration serialization, and Hugging Face Hub uploads."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from mouse_core.models.backbone.base import Backbone
from mouse_core.models.backbone.flex_decode import DecodeKernel
from mouse_core.models.backbone.packed_train import TrainKernel
from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.classification import ClassificationHead
from mouse_core.models.heads.discretized_action import DiscretizedActionHead
from mouse_core.models.heads.regression import RegressionHead
from mouse_core.models.lora import LoRAConfig
from mouse_core.models.projection import ProjectionHead
from mouse_core.models.reasoner import LatentReasoner

if TYPE_CHECKING:
    from mouse_core.data.tokenizer import Tokenizer
    from mouse_core.models.base import Model


def save_model(*, model: "Model", path: str | Path) -> None:
    """Save a MOUSE model checkpoint to a directory.

    Writes ``pytorch_model.bin`` and ``config.json`` into *path*. The
    packing spec is a separate object — :func:`~mouse_core.data.tokenizer.save_tokenizer`
    / :func:`~mouse_core.data.tokenizer.load_tokenizer`.

    Args:
        model: The model instance to save.
        path: Destination directory (created if absent).

    Example::

        save_model(model=model, path="./checkpoints/step-10000")
        save_tokenizer(tokenizer=tokenizer, path="./checkpoints/step-10000-tokenizer")
        model2 = load_model(
            repo_id_or_path="./checkpoints/step-10000",
            train_kernel="flex", decode_kernel="flex", dtype=torch.float32,
        )
        tokenizer2 = load_tokenizer(repo_id_or_path="./checkpoints/step-10000-tokenizer")
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.json").open("w") as fh:
        json.dump(_model_config(model), fh, indent=2, sort_keys=True)
        fh.write("\n")
    torch.save(model.state_dict(), path / "pytorch_model.bin")


def load_model(
    *,
    repo_id_or_path: str,
    train_kernel: TrainKernel,
    decode_kernel: DecodeKernel,
    dtype: torch.dtype,
    train_autocast_dtype: torch.dtype | None = None,
    decode_autocast_dtype: torch.dtype | None = None,
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
            transformer backbone: ``"varlen"``, ``"padded"``, ``"flex"``, or
            ``"reference"``. Strict — a kernel that cannot run on the current
            device/dtype raises instead of falling back.
        decode_kernel: Kernel for cached decode, ``"flex"``.
        dtype: Dtype of the transformer backbone's base weights
            (``preferred_dtype(device=device)`` for inference or a LoRA base,
            ``torch.float32`` to fine-tune them). The saved weights are cast
            into it.
        train_autocast_dtype: ``torch.bfloat16`` / ``torch.float16``
            declares bf16/fp16 mixed precision for the packed training
            forward of a fp32 backbone; ``None`` (default) trains in the
            base dtype.
        decode_autocast_dtype: The same declaration for cached decode
            (which also allocates its KV pool in the autocast dtype);
            ``None`` (default) decodes in the base dtype.

            All of these are execution choices for the loading machine, not
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

    state = torch.load(weights_path, map_location=map_location)
    model = _build_model_from_config(
        config,
        state=state,
        train_kernel=train_kernel,
        decode_kernel=decode_kernel,
        dtype=dtype,
        train_autocast_dtype=train_autocast_dtype,
        decode_autocast_dtype=decode_autocast_dtype,
    )
    model.load_state_dict(state)
    return model


def push_model_to_hub(
    *,
    model: "Model",
    tokenizer: "Tokenizer",
    repo_id: str,
    tokenizer_repo_id: str,
    commit_message: str = "Upload MOUSE model",
    tokenizer_commit_message: str = "Upload MOUSE tokenizer",
    private: bool = False,
    clear: bool = False,
    **kwargs: Any,
) -> tuple[str, str]:
    """Push a MOUSE model and its tokenizer to two Hugging Face Hub repos.

    The tokenizer is a separate object and a separate repository — it is
    not written into the model checkpoint. :func:`load_model` reads
    *repo_id*; :func:`~mouse_core.data.tokenizer.load_tokenizer` reads
    *tokenizer_repo_id*.

    Args:
        model: The model instance to upload.
        tokenizer: The :class:`~mouse_core.data.tokenizer.Tokenizer` used
            to pack steps for this model. Required — a different object
            from *model*, uploaded to *tokenizer_repo_id*.
        repo_id: Hub repository ID for the model, e.g. ``"my-model"`` or
            ``"your-org/your-model"``. Unscoped names are resolved under
            the authenticated user.
        tokenizer_repo_id: Hub repository ID for the packing spec. Must
            differ from *repo_id*.
        commit_message: Commit message for the model repo.
        tokenizer_commit_message: Commit message for the tokenizer repo.
        private: Create private repositories if they do not already exist.
        clear: Delete all existing files in each repository before uploading.
        **kwargs: Forwarded to ``huggingface_hub.HfApi.upload_folder``.

    Returns:
        ``(model_url, tokenizer_url)`` Hub URL strings.

    Example::

        model_url, tokenizer_url = push_model_to_hub(
            model=model,
            tokenizer=tokenizer,
            repo_id="my-model",
            tokenizer_repo_id="my-tokenizer",
            clear=True,
        )
    """
    from huggingface_hub import HfApi

    from mouse_core.data.tokenizer import Tokenizer, save_tokenizer

    if not isinstance(tokenizer, Tokenizer):
        raise TypeError(
            f"push_model_to_hub requires tokenizer= to be a Tokenizer, got {type(tokenizer).__name__}."
        )
    if repo_id == tokenizer_repo_id:
        raise ValueError(
            "tokenizer_repo_id must be a different Hub repo than repo_id; "
            f"got {repo_id!r} for both."
        )
    token = kwargs.get("token")
    model_url, hub_model_id = _create_hub_repo(
        repo_id=repo_id, private=private, clear=clear, token=token
    )
    tokenizer_url, hub_tok_id = _create_hub_repo(
        repo_id=tokenizer_repo_id, private=private, clear=clear, token=token
    )
    if hub_model_id == hub_tok_id:
        raise ValueError(
            "tokenizer_repo_id resolved to the same Hub repo as repo_id "
            f"({hub_model_id!r}); pass a different tokenizer_repo_id."
        )
    api = HfApi()
    with tempfile.TemporaryDirectory() as model_tmp, tempfile.TemporaryDirectory() as tok_tmp:
        save_model(model=model, path=model_tmp)
        _write_model_card(
            repo_id=hub_model_id,
            tokenizer_repo_id=hub_tok_id,
            model=model,
            path=Path(model_tmp) / "README.md",
        )
        save_tokenizer(tokenizer=tokenizer, path=tok_tmp)
        _write_tokenizer_card(repo_id=hub_tok_id, path=Path(tok_tmp) / "README.md")
        api.upload_folder(
            repo_id=hub_model_id,
            folder_path=model_tmp,
            commit_message=commit_message,
            **kwargs,
        )
        api.upload_folder(
            repo_id=hub_tok_id,
            folder_path=tok_tmp,
            commit_message=tokenizer_commit_message,
            **kwargs,
        )
    return model_url, tokenizer_url


def _model_config(model: "Model") -> dict[str, Any]:
    config: dict[str, Any] = {
        "format": "mouse-core-model-v1",
        "hidden_dim": int(model.hidden_dim),
        "backbone": _backbone_config(model.backbone),
        "heads": _heads_config(model),
    }
    if model.reasoner is not None:
        config["reasoner"] = {"num_thoughts": int(model.reasoner.num_thoughts)}
    if model.projection is not None:
        config["projection"] = {
            "in_features": model.projection.in_features,
            "hidden_dim": model.projection.hidden_dim,
            "num_layers": model.projection.num_layers,
        }
    return config


def _build_model_from_config(
    config: dict[str, Any],
    *,
    state: Mapping[str, torch.Tensor],
    train_kernel: TrainKernel,
    decode_kernel: DecodeKernel,
    dtype: torch.dtype,
    train_autocast_dtype: torch.dtype | None,
    decode_autocast_dtype: torch.dtype | None,
) -> "Model":
    from mouse_core.models.base import Model

    backbone = _build_backbone_from_config(
        config["backbone"],
        train_kernel=train_kernel,
        decode_kernel=decode_kernel,
        dtype=dtype,
        train_autocast_dtype=train_autocast_dtype,
        decode_autocast_dtype=decode_autocast_dtype,
    )
    heads_cfg = config["heads"]
    heads = _build_heads_from_config(heads_cfg["heads"], state=state)
    projection_cfg = config.get("projection")
    projection = ProjectionHead(**projection_cfg) if projection_cfg is not None else None
    reasoner_cfg = config.get("reasoner")
    reasoner = (
        LatentReasoner(
            hidden_dim=int(config["hidden_dim"]),
            num_thoughts=int(reasoner_cfg["num_thoughts"]),
        )
        if reasoner_cfg is not None
        else None
    )
    action_name = heads_cfg["action_source"]
    if action_name not in heads:
        raise ValueError(
            f"saved action_source {action_name!r} is not among loaded heads {tuple(heads)}."
        )
    return Model(
        backbone=backbone,
        heads=heads,
        action_source=action_name,
        reasoner=reasoner,
        projection=projection,
    )


def _backbone_config(backbone: nn.Module) -> dict[str, Any]:
    from mouse_core.models.backbone.none import IdentityBackbone

    if isinstance(backbone, IdentityBackbone):
        config: dict[str, Any] = {
            "type": "identity",
            "hidden_dim": backbone.hidden_dim,
            "vocab_size": backbone.vocab_size,
        }
        if backbone.num_frequencies is not None:
            config["num_frequencies"] = backbone.num_frequencies
        return config
    from mouse_core.models.backbone.transformer import TransformerBackbone

    if isinstance(backbone, TransformerBackbone):
        config = {
            "type": "transformer",
            "architecture": backbone.architecture,
            "hidden_dim": backbone.hidden_dim,
            "kwargs": dict(backbone._config_kwargs),
        }
        if backbone.lora is not None:
            config["lora"] = asdict(backbone.lora)
        return config
    raise TypeError(
        "save_model currently supports IdentityBackbone and TransformerBackbone. "
        f"Got {type(backbone).__name__}."
    )


def _build_backbone_from_config(
    config: dict[str, Any],
    *,
    train_kernel: TrainKernel,
    decode_kernel: DecodeKernel,
    dtype: torch.dtype,
    train_autocast_dtype: torch.dtype | None,
    decode_autocast_dtype: torch.dtype | None,
) -> Backbone:
    backbone_type = config.get("type")
    if backbone_type == "identity":
        from mouse_core.models.backbone import IdentityBackbone

        identity_kwargs: dict[str, Any] = {}
        if "num_frequencies" in config:
            identity_kwargs["num_frequencies"] = int(config["num_frequencies"])
        return IdentityBackbone(
            hidden_dim=int(config["hidden_dim"]),
            vocab_size=int(config["vocab_size"]),
            **identity_kwargs,
        )
    lora_cfg = config.get("lora")
    lora = LoRAConfig(**lora_cfg) if lora_cfg is not None else None
    runtime: dict[str, Any] = dict(
        train_kernel=train_kernel,
        decode_kernel=decode_kernel,
        dtype=dtype,
        train_autocast_dtype=train_autocast_dtype,
        decode_autocast_dtype=decode_autocast_dtype,
    )
    if backbone_type == "transformer":
        from mouse_core.models.backbone import TransformerBackbone

        backbone_kwargs = dict(config["kwargs"])
        # Older checkpoints recorded this flag; a retained native norm
        # has the same weights and behavior as today's unconditional norm.
        if backbone_kwargs.pop("use_norm", True) is not True:
            raise ValueError(
                "Backbone checkpoints with use_norm=False are no longer supported; "
                "the transformer's native final normalization is always retained."
            )
        return TransformerBackbone(
            hidden_dim=config["hidden_dim"],
            architecture=config["architecture"],
            lora=lora,
            **runtime,
            **backbone_kwargs,
        )
    raise ValueError(f"Unsupported backbone type {backbone_type!r}.")


def _heads_config(model: "Model") -> dict[str, Any]:
    return {
        "action_source": model.action_source,
        "heads": [_head_config(name, head) for name, head in model._heads.items()],
    }


def _head_config(name: str, head: BaseHead) -> dict[str, Any]:
    if not isinstance(head, (ClassificationHead, RegressionHead)):
        raise TypeError(f"save_model does not know how to serialize head {name!r} ({type(head).__name__}).")
    spec = {
        "name": name,
        "type": "classification" if isinstance(head, ClassificationHead) else "regression",
        "in_features": head.in_features,
        "out_features": head.out_features,
        "hidden_dim": head.hidden_dim,
        "num_layers": head.num_layers,
        "scale": head.scale,
        "propagate_gradient": head.propagate_gradient,
    }
    if isinstance(head, DiscretizedActionHead):
        spec.update(type="discretized_action", action_dim=head.action_dim)
    return spec


def _build_heads_from_config(
    heads: list[dict[str, Any]], *, state: Mapping[str, torch.Tensor],
) -> dict[str, BaseHead]:
    built: dict[str, BaseHead] = {}
    for spec in heads:
        name = spec["name"]
        if spec.get("use_norm", True) is not True:
            raise ValueError(
                f"Head {name!r} was saved with use_norm=False; "
                "prediction heads now require input RMSNorm."
            )
        head_type = spec["type"]
        kwargs = {key: spec[key] for key in ("in_features", "hidden_dim", "num_layers", "propagate_gradient")}
        kwargs["scale"] = spec.get("scale", 1.0)
        if head_type == "discretized_action":
            # The table lives once in the checkpoint, as a persistent buffer.
            table = state[f"heads.{name}.action_vectors"]
            if tuple(table.shape) != (spec["out_features"], spec["action_dim"]):
                raise ValueError(f"Action table shape for head {name!r} does not match its config.")
            built[name] = DiscretizedActionHead(action_vectors=table, **kwargs)
        elif head_type in ("classification", "regression"):
            head_cls = ClassificationHead if head_type == "classification" else RegressionHead
            built[name] = head_cls(out_features=spec["out_features"], **kwargs)
        else:
            raise ValueError(f"Unsupported head type {head_type!r}.")
    return built


def _hub_repo_id_for_user(repo_id: str, token: str | bool | None = None) -> str:
    """Resolve an unscoped Hub repo name under the authenticated user."""
    if "/" in repo_id:
        return repo_id

    from huggingface_hub import HfApi

    user = HfApi().whoami(token=token)["name"]
    return f"{user}/{repo_id}"


def _create_hub_repo(
    *,
    repo_id: str,
    private: bool,
    clear: bool,
    token: str | bool | None,
) -> tuple[str, str]:
    """Create or open a Hub repo, optionally clearing it. Returns ``(url, hub_repo_id)``."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        private=private,
        exist_ok=True,
        token=token,
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
    return str(repo_url), hub_repo_id


def _write_model_card(
    *,
    model: "Model",
    path: Path,
    repo_id: str,
    tokenizer_repo_id: str,
) -> None:
    config = _model_config(model)
    heads = config["heads"]["heads"]
    head_names = ", ".join(head["name"] for head in heads) or "none"
    reasoner_cfg = config.get("reasoner")
    architecture_details = (
        f"\n- Latent reasoner: `num_thoughts={reasoner_cfg['num_thoughts']}`"
        if reasoner_cfg
        else ""
    )
    projection_cfg = config.get("projection")
    if projection_cfg is not None:
        architecture_details += (
            f"\n- Shared projection: RMSNorm + SwiGLU MLP, "
            f"`hidden_dim={projection_cfg['hidden_dim']}`, "
            f"`num_layers={projection_cfg['num_layers']}`"
        )
    lora_cfg = config["backbone"].get("lora")
    if lora_cfg:
        architecture_details += (
            f"\n- LoRA: `rank={lora_cfg['rank']}`, `alpha={lora_cfg['alpha']}` "
            f"on every backbone `nn.Linear` (fp32 adapters over frozen base weights)"
        )
    encoder_section, tokenizer_snippet, objective_data_example = _model_card_encoder_bits(
        config, tokenizer_repo_id=tokenizer_repo_id
    )
    text = f"""---
library_name: mouse-core
tags:
- mouse-core
- reinforcement-learning
---

# {repo_id}

This repository contains a MOUSE model checkpoint. The packing spec
lives in a separate repo (`{tokenizer_repo_id}`).

## Architecture

- Backbone: `{config["backbone"]["type"]}`
- Hidden dimension: `{config["hidden_dim"]}`
- Heads: `{head_names}`
- Action source: `{config["heads"]["action_source"]}`{architecture_details}

### Token embeddings

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
        repo_id_or_path="{repo_id}",
        train_kernel="flex",
        decode_kernel="flex",
        dtype=preferred_dtype(device=device),
        map_location="cpu",
    )
    .eval()
    .to(device)
)
```

## Run Inference

Training and inference both take a `TokenBatch`. Training typically uses
`DataLoader(transform=compose(stages=(augmenter, tokenizer)))`. Online / inference
uses the tokenizer (no augmenter → `StepTokens`) and
`pack_token_batch` when combining steps. The packing spec is a
separate Hub repo — `load_tokenizer` on `{tokenizer_repo_id}`.

```python
{tokenizer_snippet}

{objective_data_example}

with torch.no_grad():
    steps = [eval_transform(step) for step in batch[0]]
    inputs, _, _ = pack_token_batch(steps=steps, group_ids=[0] * len(steps), continuing=None)
    out = model(inputs, use_cache=True)
    action = model.get_action(out=out, temperature=0.0)
```

`model()` returns a `ModelOutput` with `predictions` and
`last_hidden_state`. `pack_token_batch` /
`DataLoader.next_batch()` return `(inputs, objective_data, group_id)`;
pass `objective_data` and `group_id` to DQN, PPO, and GRPO during
training. For cached incremental
rollout, pass ``out.cache`` back as ``cache=`` with `use_cache=True`.
Cached batch rows may have different
lengths on every call (e.g. envs emitting different numbers of steps between
model calls): decoding runs through a FlexAttention session carried in the
cache, so each row decodes exactly as it would alone.
"""
    path.write_text(text, encoding="utf-8")


def _write_tokenizer_card(*, path: Path, repo_id: str) -> None:
    path.write_text(
        f"""---
library_name: mouse-core
tags:
- mouse-core
- tokenizer
---

# {repo_id}

MOUSE tokenizer packing spec (`tokenizer.json`). Load with:

```python
from mouse_core.data import load_tokenizer

tokenizer = load_tokenizer(repo_id_or_path="{repo_id}")
```
""",
        encoding="utf-8",
    )


def _config_num_frequencies(config: dict[str, Any]) -> int | None:
    backbone = config["backbone"]
    if backbone.get("type") == "identity":
        value = backbone.get("num_frequencies")
    else:
        value = backbone.get("kwargs", {}).get("num_frequencies")
    if value is None:
        return None
    return int(value)


def _model_card_encoder_bits(
    config: dict[str, Any], *, tokenizer_repo_id: str
) -> tuple[str, str, str]:
    """Return ``(encoder_section, tokenizer_snippet, step_example)`` for the card."""
    hidden = config["hidden_dim"]
    tokenizer_snippet = (
        "from mouse_core.data import load_tokenizer, pack_token_batch\n"
        "\n"
        f'tokenizer = load_tokenizer(repo_id_or_path="{tokenizer_repo_id}")\n'
        "eval_transform = tokenizer"
    )
    num_frequencies = _config_num_frequencies(config)
    fourier_note = ""
    if num_frequencies is not None:
        fourier_note = (
            f" `__numeric__` tokens use the same table, then add a Fourier "
            f"projection of the scaled value (`num_frequencies={num_frequencies}`)."
        )
    encoder_section = (
        f"The backbone looks up `__text__` and image-field token ids in a "
        f"tokenized :class:`~mouse_core.data.token_batch.TokenBatch` through "
        f"its `embed_tokens` table ({hidden}-dimensional).{fourier_note} Step templates "
        f"and field packing live on `Tokenizer` (a separate Hub repo, "
        f"`{tokenizer_repo_id}`)."
    )
    step_example = """# load_tokenizer(repo) is the same packing spec used at train time.
batch = [[
    {
        "action": 0,
        "observation": 1,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "task_index": 0,
    }
]]"""
    return encoder_section, tokenizer_snippet, step_example
