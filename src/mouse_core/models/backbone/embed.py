"""Lookup ``__text__`` / image ids through a backbone ``embed_tokens`` table."""

from __future__ import annotations

import torch
from torch import nn

from mouse_core.data.modality import NAME_TEXT
from mouse_core.data.token_batch import ModalityInfo, TokenBatch


def embed_token_ids(
    *,
    embed_tokens: nn.Embedding,
    token_batch: TokenBatch,
    hidden_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up ``__text__`` and image token ids."""
    device = embed_tokens.weight.device
    dtype = embed_tokens.weight.dtype
    t = token_batch.to_tensors(device)
    ids = t["ids"]
    modality_ids = t["modality_ids"]
    names: tuple[str, ...] = t["modality_names"]
    batch_map: dict[str, ModalityInfo] = t["modality_map"]

    for name in names:
        info = batch_map[name]
        if name == NAME_TEXT:
            if info.type not in ("token", "text"):
                raise TypeError(
                    f"modality {name!r} type mismatch: batch={info.type!r} "
                    "expected token/text"
                )
            continue
        if info.type == "image":
            continue
        raise KeyError(
            f"TokenBatch modality {name!r} is not a text/image id "
            f"(expected {NAME_TEXT!r} / type=image)"
        )

    L = ids.shape[0]
    embeds = torch.zeros(L, hidden_dim, device=device, dtype=dtype)
    if L > 0:
        for local_id, name in enumerate(names):
            mask = modality_ids == local_id
            if not bool(mask.any()):
                continue
            embeds[mask] = embed_tokens(ids[mask]).to(dtype=dtype)

    return embeds, t["head_output_indices"]
