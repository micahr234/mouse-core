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

from mouse_core.models.backbone.base import Backbone, _reject_dtype_cast
from mouse_core.models.backbone.flex_decode import DecodeKernel, FlexDecodeSession, packed_rope_positions
from mouse_core.models.backbone.packed_train import TrainKernel
from mouse_core.models.heads.base import BaseHead, _bind_prediction_key
from mouse_core.models.heads.classification import ClassificationHead
from mouse_core.models.heads.layerwise_regression import LayerwiseRegressionHead
from mouse_core.models.heads.regression import RegressionHead
from mouse_core.models.lora import LoRAConfig
from mouse_core.models.reasoner import LatentReasoner, _InsertionPlan, _plan_insertions

if TYPE_CHECKING:
    from mouse_core.data.token_batch import TokenBatch
    from mouse_core.data.tokenizer import Tokenizer

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
    reasoner_line = (
        f"\n- Latent reasoner: `num_thoughts={reasoner_cfg['num_thoughts']}`"
        if reasoner_cfg
        else ""
    )
    lora_cfg = config["backbone"].get("lora")
    if lora_cfg:
        reasoner_line += (
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
- Action source: `{config["heads"]["action_source"]}`{reasoner_line}

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
    inputs, _ = pack_token_batch(steps=steps, sequence_ids=[0] * len(steps))
    out = model(inputs, use_cache=True)
    action = model.get_action(out=out, temperature=0.0)
```

`model()` returns a `ModelOutput` with `predictions` and
`last_hidden_state`. `pack_token_batch` /
`DataLoader.next_batch()` return `(inputs, objective_data)`; pass
`objective_data` to objectives during training. For cached incremental
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
    encoder_section = (
        f"The backbone looks up `__text__` and image-field token ids in a "
        f"tokenized :class:`~mouse_core.data.token_batch.TokenBatch` through "
        f"its `embed_tokens` table ({hidden}-dimensional). Step templates "
        f"and field packing live on `Tokenizer` (a separate Hub repo, "
        f"`{tokenizer_repo_id}`)."
    )
    step_example = """# load_tokenizer(repo) is the same packing spec used at train time.
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


def _model_config(model: "Model") -> dict[str, Any]:
    config: dict[str, Any] = {
        "format": "mouse-core-model-v1",
        "hidden_dim": int(model.hidden_dim),
        "backbone": _backbone_config(model.backbone),
        "heads": _heads_config(model),
    }
    if model.reasoner is not None:
        config["reasoner"] = {"num_thoughts": int(model.reasoner.num_thoughts)}
    return config


def _backbone_config(backbone: nn.Module) -> dict[str, Any]:
    from mouse_core.models.backbone.none import IdentityBackbone

    if isinstance(backbone, IdentityBackbone):
        return {
            "type": "identity",
            "hidden_dim": backbone.hidden_dim,
            "vocab_size": backbone.vocab_size,
        }
    from mouse_core.models.backbone.transformer import TransformerBackbone

    if isinstance(backbone, TransformerBackbone):
        config: dict[str, Any] = {
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


def _heads_config(model: "Model") -> dict[str, Any]:
    heads = []
    for name, head in model._heads.items():
        spec = _head_config(name, head)
        if spec is not None:
            heads.append(spec)
    return {"action_source": model.action_source, "heads": heads}


def _head_config(name: str, head: BaseHead) -> dict[str, Any] | None:
    if isinstance(head, LayerwiseRegressionHead):
        return {
            "name": name,
            "type": "regression_layerwise",
            "num_backbone_layers": head.num_backbone_layers,
            "in_features": head.in_features,
            "out_features": head.out_features,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
    if isinstance(head, ClassificationHead):
        return {
            "name": name,
            "type": "classification",
            "in_features": head.in_features,
            "out_features": head.out_features,
            "hidden_dim": head.hidden_dim,
            "num_layers": head.num_layers,
            "scale": head.scale,
            "use_norm": head.use_norm,
        }
    if isinstance(head, RegressionHead):
        return {
            "name": name,
            "type": "regression",
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

    model = _build_model_from_config(
        config,
        train_kernel=train_kernel,
        decode_kernel=decode_kernel,
        dtype=dtype,
        train_autocast_dtype=train_autocast_dtype,
        decode_autocast_dtype=decode_autocast_dtype,
    )
    state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state)
    return model


def _build_model_from_config(
    config: dict[str, Any],
    *,
    train_kernel: TrainKernel,
    decode_kernel: DecodeKernel,
    dtype: torch.dtype,
    train_autocast_dtype: torch.dtype | None,
    decode_autocast_dtype: torch.dtype | None,
) -> "Model":
    backbone = _build_backbone_from_config(
        config["backbone"],
        train_kernel=train_kernel,
        decode_kernel=decode_kernel,
        dtype=dtype,
        train_autocast_dtype=train_autocast_dtype,
        decode_autocast_dtype=decode_autocast_dtype,
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
    action_name = heads_cfg["action_source"]
    if action_name not in heads:
        raise ValueError(
            f"saved action_source {action_name!r} is not among loaded heads {tuple(heads)}."
        )
    return Model(
        backbone=backbone,
        heads=heads,
        action_source=heads[action_name],
        reasoner=reasoner,
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

        return IdentityBackbone(
            hidden_dim=int(config["hidden_dim"]),
            vocab_size=int(config["vocab_size"]),
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

        return TransformerBackbone(
            hidden_dim=config["hidden_dim"],
            architecture=config["architecture"],
            lora=lora,
            **runtime,
            **config["kwargs"],
        )
    raise ValueError(f"Unsupported backbone type {backbone_type!r}.")


def _build_heads_from_config(heads: list[dict[str, Any]]) -> dict[str, BaseHead]:
    built: dict[str, BaseHead] = {}
    for spec in heads:
        name = spec["name"]
        head_type = spec["type"]
        if head_type == "regression_layerwise":
            built[name] = LayerwiseRegressionHead(
                num_backbone_layers=spec["num_backbone_layers"],
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec["use_norm"],
            )
        elif head_type == "classification":
            built[name] = ClassificationHead(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec["use_norm"],
            )
        elif head_type == "regression":
            built[name] = RegressionHead(
                in_features=spec["in_features"],
                out_features=spec["out_features"],
                hidden_dim=spec["hidden_dim"],
                num_layers=spec["num_layers"],
                scale=spec.get("scale", 1.0),
                use_norm=spec["use_norm"],
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
        if any(isinstance(head, LayerwiseRegressionHead) for head in heads.values()):
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

    Holds one :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession`.
    Pass ``out.cache`` back as ``cache=``. Call :meth:`close` when the
    rollout is finished so VMM pages unmap before the next train step
    (``__del__`` is too late if a CUDA graph still holds views).
    """

    session: FlexDecodeSession

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``).

        The rows' next tokens start at position 0 and their KV pages return
        to the shared pool.
        """
        self.session.reset_rows(rows)

    def close(self) -> None:
        """Unmap the session's VMM pool and drop captured CUDA graphs."""
        self.session.close()


@dataclass
class ModelOutput:
    """Head predictions plus the token states they were read from.

    ``predictions``, ``last_hidden_state``, and ``hidden_states`` come from
    the single backbone forward.

    ``head_output_indices`` maps token states to the rows heads read. A
    reasoning forward extends the stream, so those indices and the states
    describe the stream *with* latents inserted. Cached decode also sets
    ``head_output_valid`` (``[B, S]``): True where a step slot is a real
    last-head-output readout, False on left-padded idle columns.
    Incremental decode carries ``cache`` — pass ``out.cache`` back as
    ``cache=`` with ``use_cache=True``.
    """

    predictions: TensorDict
    last_hidden_state: torch.Tensor
    head_output_indices: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    cache: DecodeCache | None = None
    head_output_valid: torch.Tensor | None = None


class Model(nn.Module):
    """Composable MOUSE model: backbone and heads as distinct sections.

    The model is assembled from two pluggable parts:

    - ``backbone``: a :class:`~mouse_core.models.backbone.Backbone` that
      embeds a :class:`~mouse_core.data.token_batch.TokenBatch` and maps
      those tokens to last-layer hidden states ``[L, D]``. Token
      embeddings live on the backbone: ``TransformerBackbone`` loads
      native ``embed_tokens``;
      :class:`~mouse_core.models.backbone.IdentityBackbone` owns a
      ``vocab_size`` × ``hidden_dim`` table.
    - ``heads``: heads can be provided in several ergonomic ways:
        - a single :class:`~mouse_core.models.heads.base.BaseHead` (e.g. ``RegressionHead(...)``):
          it becomes the only enabled head;
        - a list of head instances (e.g. ``[RegressionHead(...), ClassificationHead(...)]``):
          names are inferred from type;
        - a dict mapping caller-chosen names to head instances or ``None``.
      When a plain head is passed without a name the key is inferred from
      type (``action_value`` / ``action`` / ``action_value_layerwise``);
      use the dict form to pick the key.

    ``action_source`` is the head instance ``get_action`` consults. It must
    be one of the objects in ``heads``. Required.
    ``reasoner`` is required (pass ``None`` when unused).

    Full construction::

        backbone = TransformerBackbone(pretrained=..., ...)
        head = RegressionHead(...)            # or a dict/list of heads

        model = Model(
            backbone=backbone,
            heads=head,
            action_source=head,
            reasoner=None,
        )

    ``forward`` returns a :class:`ModelOutput` with ``predictions`` and
    ``last_hidden_state``.
    The delayed DQN model
    comes from :meth:`delayed_copy` (a copy of every trainable parameter;
    frozen weights shared by reference), runs on the same ``TokenBatch``,
    and is interpolated per section with :class:`~mouse_core.polyak.Polyak`.
    """

    @staticmethod
    def _normalize_heads(
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None] | None,
    ) -> dict[str, BaseHead]:
        """Convert the flexible ``heads=`` argument into the internal ``name -> head`` dict.

        Supported inputs:
          - dict (caller-chosen names to head or None): passed through.
          - single BaseHead instance: becomes the only head; name is inferred
            from type (``action_value`` / ``action`` /
            ``action_value_layerwise``).
          - list/tuple of BaseHead: each gets an inferred name.
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
            return {Model._infer_head_name(heads): heads}

        # List of heads → names inferred from type.
        if isinstance(heads, (list, tuple)):
            if len(heads) == 0:
                return {}
            result: dict[str, BaseHead] = {}
            for h in heads:
                if not isinstance(h, BaseHead):
                    raise TypeError(f"items in heads list must be BaseHead instances, got {type(h)}")
                nm = Model._infer_head_name(h)
                if nm in result:
                    raise ValueError(
                        f"Multiple heads would map to the same inferred name {nm!r}. "
                        "Use a dict form to provide distinct names, e.g. "
                        "heads={'action_value': h1, 'action': h2}."
                    )
                result[nm] = h
            return result

        raise TypeError(
            f"heads must be a BaseHead, list[BaseHead], or dict[str, BaseHead|None], "
            f"got {type(heads)}"
        )

    @staticmethod
    def _infer_head_name(head: BaseHead) -> str:
        """Infer a default storage key from the head type when no dict key is given.

        ``RegressionHead`` → ``action_value``, ``ClassificationHead``
        → ``action``, ``LayerwiseRegressionHead`` → ``action_value_layerwise``.
        """
        if isinstance(head, LayerwiseRegressionHead):
            return "action_value_layerwise"
        if isinstance(head, ClassificationHead):
            return "action"
        if isinstance(head, RegressionHead):
            return "action_value"
        raise TypeError(
            f"Cannot infer a name for head of type {type(head).__name__}. "
            "Use the dict form with an explicit key."
        )

    @staticmethod
    def _head_name(heads: Mapping[str, BaseHead], head: BaseHead, *, what: str) -> str:
        """Resolve a head instance to the storage key of that same object."""
        if not isinstance(head, BaseHead):
            raise TypeError(f"{what} must be a BaseHead instance, got {type(head).__name__}.")
        for name, existing in heads.items():
            if existing is head:
                return name
        raise ValueError(
            f"{what} is not one of the heads passed to heads=; pass the same instance."
        )

    @staticmethod
    def _action_source_name(heads: Mapping[str, BaseHead], action_source: BaseHead) -> str:
        """Resolve ``action_source`` to the storage key of that same instance."""
        return Model._head_name(heads, action_source, what="action_source")

    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None],
        action_source: BaseHead,
        reasoner: LatentReasoner | None,
    ):
        """Construct a Model from backbone and heads.

        Every argument is required. ``reasoner`` enables Coconut-style latent
        reasoning via ``forward(batch, reasoning=...)``. Pass ``None`` when
        unused.
        """
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError("backbone must be a Backbone (from mouse_core.models.backbone).")

        bb_dim = backbone.hidden_dim
        if bb_dim is None:
            raise ValueError("backbone.hidden_dim is required.")
        hidden_dim = int(bb_dim)

        self.backbone: Backbone = backbone

        if reasoner is not None:
            if not isinstance(reasoner, LatentReasoner):
                raise TypeError(
                    f"reasoner must be a LatentReasoner, got {type(reasoner).__name__}."
                )
            if reasoner.hidden_dim != hidden_dim:
                raise ValueError(
                    f"hidden_dim mismatch between reasoner ({reasoner.hidden_dim}) "
                    f"and model ({hidden_dim})."
                )
        self.reasoner: LatentReasoner | None = reasoner

        # Normalize flexible heads input (single instance, list, or dict) into the
        # canonical internal dict form.
        heads_dict: dict[str, BaseHead] = Model._normalize_heads(heads)

        # Store heads for both state dict and typed access
        filtered: dict[str, BaseHead] = {}
        for name, head in heads_dict.items():
            if head is not None:
                if not isinstance(head, BaseHead):
                    raise TypeError(f"head {name!r} must be a BaseHead or None, got {type(head)}")
                filtered[name] = head
        self.heads = nn.ModuleDict(filtered)  # for parameters/state
        self._heads: dict[str, BaseHead] = filtered  # typed view for calling
        for name, head in self._heads.items():
            _bind_prediction_key(head, name)

        self.action_source = Model._action_source_name(self._heads, action_source)

        bb_layers: int | None = None
        for name, head in self._heads.items():
            if not isinstance(head, LayerwiseRegressionHead):
                continue
            if bb_layers is None:
                bb_layers = _backbone_num_layers(self.backbone)
                if bb_layers is None:
                    raise ValueError(
                        f"{name} is layerwise and needs a backbone with a known "
                        "layer count (e.g. TransformerBackbone)."
                    )
            if head.num_backbone_layers != bb_layers:
                raise ValueError(
                    f"Layerwise head {name!r} expects {head.num_backbone_layers} "
                    f"backbone layers but backbone has {bb_layers}."
                )

        self.hidden_dim = hidden_dim
        # Best-effort inference of action cardinality for introspection only.
        self.max_num_actions: int = 0
        for _name, h in self.heads.items():
            out = getattr(h, "out_features", None)
            if isinstance(out, int) and out > 0:
                self.max_num_actions = out
                break

    def delayed_copy(self, *, heads: Sequence[BaseHead]) -> "Model":
        """Build the delayed model for TD targets: a frozen copy of this model.

        ``heads`` is the head instances the delayed model carries — only
        those the objective reads from ``delayed_predictions`` (the Q
        head for ``DqnObjective`` / ``RetraceObjective``, each n-step Q
        head, the layerwise Q head). Heads left out (a policy or
        behavior head whose delayed values nothing uses) are neither
        copied, run, nor Polyak-interpolated. Every instance must be one
        of this model's heads and the list must not be empty. The copy's
        ``action_source`` is this model's when it is among ``heads``,
        else the first head listed (the delayed model does not pick
        actions).

        Every trainable parameter gets its own copy; every frozen parameter
        (``requires_grad=False`` — the base weights of a LoRA backbone) is
        shared by reference with the online model, so a delayed LoRA
        backbone costs one extra copy of the adapters, not of the base. A
        fully trainable fp32 backbone is copied whole. The copy has every
        parameter frozen and is left in ``train()`` mode.

        Run it as ``delayed(inputs)`` with the same ``TokenBatch`` (and
        ``reasoning=``) as the online forward, under ``torch.no_grad()``.
        Interpolate it with :class:`~mouse_core.polyak.Polyak`, which takes
        one ``tau`` per section (heads, backbone) on every update
        and pairs only the heads the delayed model has. Token embeddings
        ride with the backbone ``embed_tokens``.

        Construct after ``model.to(...)``. Do not call ``requires_grad_`` /
        ``to`` on the delayed model: shared frozen parameters belong to the
        online model too.
        """
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError(
                "delayed_copy needs a trainable online model (no parameter requires grad)."
            )
        if isinstance(heads, (str, BaseHead)) or not isinstance(heads, Sequence):
            raise TypeError(
                f"delayed_copy heads must be a sequence of head instances, got {type(heads).__name__}."
            )
        head_list = tuple(heads)
        if not head_list:
            raise ValueError("delayed_copy heads must include at least one head.")
        names = tuple(
            Model._head_name(self._heads, head, what="delayed_copy heads")
            for head in head_list
        )
        if len(set(names)) != len(names):
            raise ValueError(f"delayed_copy heads has duplicate heads: {names}.")

        def _copy(module: nn.Module) -> nn.Module:
            shared = {id(p): p for p in module.parameters() if not p.requires_grad}
            delayed = copy.deepcopy(module, memo=shared)
            delayed.requires_grad_(False)
            delayed.train()
            return delayed

        copied_heads = {name: cast(BaseHead, _copy(self._heads[name])) for name in names}
        action_name = self.action_source if self.action_source in names else names[0]
        return Model(
            backbone=cast(Backbone, _copy(self.backbone)),
            heads=copied_heads,
            action_source=copied_heads[action_name],
            reasoner=None if self.reasoner is None else cast(LatentReasoner, _copy(self.reasoner)),
        )

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move the model to a device; never casts.

        Dtypes are fixed when the pieces are built: the transformer backbone
        takes ``dtype=`` (``torch.float32`` to fine-tune the base weights,
        ``preferred_dtype(device=device)`` for a frozen LoRA base or inference) and
        every other section — LoRA adapters, Identity ``embed_tokens``,
        reasoner, heads — is float32, which ``AdamW`` and
        ``Polyak`` require of every
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
        ``train_kernel`` (``"varlen"``: flash varlen, CUDA bf16/fp16 q/k/v
        only — raises otherwise; ``"padded"``: dense causal SDPA on segments
        padded to ``max_seqlen``; ``"flex"``: FlexAttention; ``"reference"``:
        masked SDPA, O(L^2)) and ``train_autocast_dtype`` (bf16/fp16 mixed
        precision over fp32 weights, ``None`` for the base dtype). Backbones
        without packed kernels (``IdentityBackbone``, generic HuggingFace
        stacks) take the rectangular route with a dense sequence/grouping
        mask. ``embeds`` come from ``backbone.embed`` (fp32 tables /
        adapters) and are cast to the backbone's base dtype here.
        """
        embeds = embeds.to(dtype=backbone.dtype)
        if getattr(backbone, "uses_packed", False):
            from mouse_core.models.backbone.packed_train import packed_forward

            transformer = getattr(backbone, "model", None)
            if transformer is None:
                raise TypeError(f"{type(backbone).__name__}.uses_packed is True but it has no .model.")
            return packed_forward(
                model=cast(nn.Module, transformer),
                embeds=embeds,
                sequence_ids=sequence_ids,
                grouping_ids=grouping_ids,
                output_hidden_states=needs_layerwise,
                checkpoint=backbone.gradient_checkpointing,
                train_kernel=backbone.train_kernel,
                autocast_dtype=backbone.train_autocast_dtype,
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
            thoughts.append(reasoner(h_last).to(dtype=embeds.dtype))

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
        model.delayed_copy(heads=(head,))`` then ``delayed_model(inputs)`` under
        ``torch.no_grad()`` (same ``TokenBatch`` and ``reasoning=`` as the
        online forward); interpolate with ``Polyak(online=model, delayed=delayed_model)``
        and ``polyak.update(tau_heads=..., tau_backbone=...)``.
        Online / inference: ``inputs, _ = pack_token_batch(steps=[eval_transform(step)],
        sequence_ids=[0])`` then ``model(inputs, use_cache=True)``
        (optionally ragged; empty-only batches raise). Pass ``out.cache``
        back as ``cache=``.

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
        decode keeps one ``FlexDecodeSession``, a paged KV pool in which each
        sequence owns only the pages its own history needs, with the same
        grouping-id isolation. On CUDA the pool grows by mapping more physical
        pages (no copy of existing K/V). Call ``out.cache.close()`` when the
        rollout ends.

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
                "Use pack_token_batch(steps=[transform(step)], ...) "
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

        embeds, resolved_indices = self.backbone.embed(token_batch)
        # embeds: [L, D]; resolved_indices: [P]
        t = token_batch.to_tensors(embeds.device)
        sequence_ids = t["sequence_ids"]
        grouping_ids = t["grouping_ids"]

        needs_layerwise = any(
            isinstance(head, LayerwiseRegressionHead) for head in self._heads.values()
        )
        new_cache: DecodeCache | None

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
            session = (
                cache.session
                if cache is not None
                else self.backbone.decode_session(batch_size=B)
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
            session_out = session.forward(
                output_hidden_states=needs_layerwise,
                embeds=flex_embeds,
                lengths=token_lengths,
                grouping_ids=flex_grouping_ids,
            )
            new_cache = DecodeCache(session=session)
            pred_batch_size: tuple[int, ...] = (B, S_max)
            counts = torch.as_tensor(
                step_counts_np.tolist(), device=embeds.device, dtype=torch.long
            )
            # Left-padded steps: trailing ``n`` columns of row ``b`` are real.
            head_output_valid = (
                torch.arange(S_max, device=embeds.device).unsqueeze(0)
                >= (S_max - counts).clamp(min=0).unsqueeze(1)
            )
            head_output_valid = head_output_valid & counts.unsqueeze(1).gt(0)
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
            session_out = self._train_backbone_forward(
                self.backbone,
                embeds,
                sequence_ids,
                grouping_ids,
                needs_layerwise,
            )
            new_cache = None
            pred_batch_size = (token_batch.P,)
            head_output_valid = None

        return ModelOutput(
            predictions=self.head(
                h=self._pool_backbone_out(session_out, resolved_indices, needs_layerwise),
                batch_size=pred_batch_size,
            ),
            last_hidden_state=_last_hidden(session_out),
            head_output_indices=resolved_indices,
            hidden_states=_layer_hiddens(session_out) if needs_layerwise else None,
            cache=new_cache,
            head_output_valid=head_output_valid,
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
        *,
        out: TensorDict | ModelOutput,
        temperature: float,
        num_actions: int | None = None,
    ) -> torch.Tensor:
        """Select an action from the value head at the last valid head-output.

        Cached decode writes Q at each step's last head-output token
        (left-padded ``[B, S, A]``, or layerwise ``[B, S, L, A]``). Pass the
        :class:`ModelOutput` from ``forward(..., use_cache=True)`` so this
        reads the last-layer residual stream at the last *valid* (non-pad)
        head-output of each row — not a padded step column and not a token
        after the head-output field. A scores ``TensorDict`` is treated as
        already aligned: the last step axis is used.

        Flat training outputs ``[N, A]`` are rejected unless ``N == 1``.

        Scores come from the head passed as ``action_source``.
        """
        if isinstance(out, ModelOutput):
            preds = out.predictions
            valid = out.head_output_valid
        else:
            preds = out
            valid = None
        scores = _last_action_scores(
            cast(torch.Tensor, preds[self.action_source]),
            name=self.action_source,
            head=self._heads[self.action_source],
            valid=valid,
        )
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _last_valid_step(valid: torch.Tensor) -> torch.Tensor:
    """Last True index along dim=-1. Every row must have a valid head-output."""
    if valid.ndim != 2:
        raise ValueError(
            f"head_output_valid must have shape [B, S], got {tuple(valid.shape)}"
        )
    has = valid.any(dim=-1)
    if not bool(has.all()):
        empty = (~has).nonzero(as_tuple=True)[0]
        raise ValueError(
            "get_action needs a valid head-output token in every row; "
            f"empty rows: {empty.tolist()}."
        )
    idx = valid.long() * torch.arange(valid.shape[1], device=valid.device)
    return idx.max(dim=-1).values


def _last_action_scores(
    raw: torch.Tensor,
    *,
    name: str,
    head: BaseHead,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Action scores at the last valid head-output of each row.

    Layerwise Q: ``[B, S, L, A]`` / ``[N, L, A]`` → last valid step,
    deepest layer. Other Q / logit heads: ``[B, S, A]`` / ``[N, A]``.
    ``valid`` is the decode ``[B, S]`` mask; omitted, the last step
    column is used (left-padded decode).
    """
    if valid is not None and raw.ndim >= 3 and valid.shape != raw.shape[:2]:
        raise ValueError(
            f"{name} scores {tuple(raw.shape)} do not match "
            f"head_output_valid {tuple(valid.shape)}"
        )
    step = None if valid is None else _last_valid_step(valid)
    batch = None if step is None else torch.arange(raw.shape[0], device=raw.device)
    if isinstance(head, LayerwiseRegressionHead):
        if raw.ndim == 4:
            if step is None:
                return raw[:, -1, -1, :]
            return raw[batch, step, -1, :]
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
        if step is None:
            return raw[:, -1]
        return raw[batch, step]
    if raw.ndim == 2:
        if raw.shape[0] != 1:
            raise ValueError(
                f"{name} has shape {tuple(raw.shape)}; get_action on flat "
                "[N, A] training outputs needs N=1 (one step). Use "
                "cached-decode [B, S, A] outputs for a batch."
            )
        return raw[-1].unsqueeze(0)
    raise ValueError(f"{name} expects [B, S, A] or [N, A], got {tuple(raw.shape)}")


def preferred_dtype(*, device: torch.device | str | None = None) -> torch.dtype:
    """Dtype for a frozen backbone base: ``bfloat16`` on CUDA, else ``float32``.

    Pass as the backbone ``dtype`` (``TransformerBackbone(dtype=preferred_dtype(device=device),
    ...)`` or ``load_model(..., dtype=preferred_dtype(device=device))``) for a LoRA
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
