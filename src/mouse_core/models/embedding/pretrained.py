"""Load an input token table without allocating a pretrained decoder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_token_table(pretrained: str | Path, hub_kwargs: dict[str, Any]) -> nn.Embedding:
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import load_state_dict
    from transformers.utils.hub import cached_file

    config = AutoConfig.from_pretrained(pretrained, **hub_kwargs)
    # Only inspect the architecture: decoder parameters never allocate storage.
    with torch.device("meta"):
        layout = AutoModel.from_config(config)
    table = layout.get_input_embeddings()
    if not isinstance(table, nn.Embedding):
        raise TypeError("pretrained input embeddings must be an nn.Embedding table.")
    name = next(name for name, parameter in layout.named_parameters() if parameter is table.weight)
    candidates = (name, f"{layout.base_model_prefix}.{name}")
    shape, padding_idx = table.weight.shape, table.padding_idx
    del layout, table

    for filename in ("model.safetensors", "model.safetensors.index.json",
                     "pytorch_model.bin", "pytorch_model.bin.index.json"):
        path = cached_file(pretrained, filename, _raise_exceptions_for_missing_entries=False, **hub_kwargs)
        if path is None:
            continue
        if filename.endswith(".index.json"):
            weight_map = json.loads(Path(path).read_text())["weight_map"]
            key = next((key for key in candidates if key in weight_map), None)
            if key is None:
                raise ValueError(f"checkpoint has no input embedding weight matching {candidates}.")
            path = cached_file(pretrained, weight_map[key], **hub_kwargs)
            assert path is not None
        state = load_state_dict(path)
        key = next((key for key in candidates if key in state), None)
        if key is None:
            raise ValueError(f"checkpoint has no input embedding weight matching {candidates}.")
        weight = state[key]
        if weight.shape != shape:
            raise ValueError(f"input embedding weight shape {tuple(weight.shape)} does not match config {tuple(shape)}.")
        weight = weight.to(dtype=torch.float32).clone()
        table = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=padding_idx)
        return table
    raise FileNotFoundError(f"no PyTorch or safetensors checkpoint found for {str(pretrained)!r}.")
