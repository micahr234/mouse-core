"""TextEmbedder — pretrained token embeddings over a flat TokenBatch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mouse_core.data.modality import NAME_TEXT
from mouse_core.data.token_batch import ModalityInfo, TokenBatch
from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.embedding.linear import ScaledEmbedding
from mouse_core.models.embedding.modality import (
    NumericEmbedderModalitySpec,
    expand_embedder_numeric_spec,
)


class TextEmbedder(Encoder):
    """Pretrained token embeddings over a flat :class:`TokenBatch`.

    Token packing lives in :class:`~mouse_core.data.tokenizer.Tokenizer`
    (constructed separately). This module looks up ``embed_tokens`` for
    ``__text__`` ids and ``type="image"`` modalities, plus optional
    ``learnable=`` scratch tables (aligned by name with tokenizer
    ``output_field``).

    The pretrained table comes from exactly one of ``embed_tokens=`` (an
    existing ``nn.Embedding``), ``pretrained=`` (copied from a Hub
    checkpoint), or ``vocab_size=`` (a fresh table, used by ``load_model``
    so the saved ``state_dict`` provides the weights without re-downloading
    the checkpoint).
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        pretrained: str | Path | None = None,
        embed_tokens: nn.Embedding | None = None,
        vocab_size: int | None = None,
        padding_idx: int | None = None,
        hub_kwargs: dict | None = None,
        freeze_embeddings: bool = False,
        learnable: Sequence[dict[str, Any] | NumericEmbedderModalitySpec] | None = None,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self._hub_kwargs = dict(hub_kwargs or {})
        self._pretrained = str(pretrained) if pretrained is not None else None

        sources = [
            name
            for name, given in (
                ("embed_tokens", embed_tokens is not None),
                ("pretrained", pretrained is not None),
                ("vocab_size", vocab_size is not None),
            )
            if given
        ]
        if len(sources) != 1:
            raise TypeError(
                "TextEmbedder requires exactly one of embed_tokens=, pretrained=, "
                f"or vocab_size= (got {sources or 'none'})"
            )
        if embed_tokens is not None:
            if embed_tokens.embedding_dim != hidden_dim:
                raise ValueError(
                    f"embed_tokens dim {embed_tokens.embedding_dim} != hidden_dim {hidden_dim}"
                )
            self.embed_tokens = embed_tokens
        elif pretrained is not None:
            self.embed_tokens = _load_embed_tokens(
                pretrained=pretrained, hidden_dim=hidden_dim, hub_kwargs=self._hub_kwargs
            )
        else:
            assert vocab_size is not None
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
            self.embed_tokens = nn.Embedding(int(vocab_size), hidden_dim, padding_idx=padding_idx)

        if freeze_embeddings:
            self.embed_tokens.weight.requires_grad_(False)

        self.learnable: list[NumericEmbedderModalitySpec] = _coerce_text_learnable(
            learnable
        )
        self._learnable_tables = nn.ModuleDict()
        self._learnable_type_vectors = nn.ParameterDict()
        for spec in self.learnable:
            assert isinstance(spec.field, str)
            assert spec.std is not None
            assert spec.positions is not None
            n = int(spec.tokens or 1)
            self._learnable_tables[spec.field] = ScaledEmbedding(
                n, hidden_dim, scale=float(spec.std)
            )
            self._learnable_type_vectors[spec.field] = nn.Parameter(
                torch.randn(int(spec.positions), hidden_dim) * float(spec.std)
            )

    @property
    def vocab_size(self) -> int:
        return int(self.embed_tokens.num_embeddings)

    @property
    def padding_idx(self) -> int | None:
        idx = self.embed_tokens.padding_idx
        return None if idx is None else int(idx)

    @property
    def pretrained(self) -> str | Path | None:
        return self._pretrained

    @pretrained.setter
    def pretrained(self, value: str | Path | None) -> None:
        self._pretrained = str(value) if value is not None else None

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        return sum(int(spec.positions or 0) for spec in self.learnable)

    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        t = token_batch.to_tensors(device)
        ids = t["ids"]
        positions = t["positions"]
        modality_ids = t["modality_ids"]
        names: tuple[str, ...] = t["modality_names"]
        batch_map: dict[str, ModalityInfo] = t["modality_map"]
        learnable_names = {str(spec.field) for spec in self.learnable}

        for name in names:
            info = batch_map[name]
            if name == NAME_TEXT:
                if info.type not in ("token", "text"):
                    raise TypeError(
                        f"modality {name!r} type mismatch: batch={info.type!r} expected token/text"
                    )
                continue
            if info.type == "image":
                continue
            if name not in learnable_names:
                raise KeyError(
                    f"TokenBatch modality {name!r} not expected by TextEmbedder "
                    f"(expected {NAME_TEXT!r} / type=image and/or "
                    f"learnable {sorted(learnable_names)})"
                )
            if info.type != "learnable":
                raise TypeError(
                    f"modality {name!r} type mismatch: batch={info.type!r} expected learnable"
                )

        L = ids.shape[0]
        D = self._hidden_dim
        embeds = torch.zeros(L, D, device=device, dtype=dtype)
        if L > 0:
            for local_id, name in enumerate(names):
                mask = modality_ids == local_id
                if not bool(mask.any()):
                    continue
                if name == NAME_TEXT or batch_map[name].type == "image":
                    embeds[mask] = self.embed_tokens(ids[mask]).to(dtype=dtype)
                    continue
                type_table = self._learnable_type_vectors[name]
                pos = positions[mask]
                if int(pos.max()) >= type_table.shape[0]:
                    raise ValueError(
                        f"modality {name!r} emitted {int(pos.max()) + 1} tokens in a "
                        f"step but the embedder declares positions={type_table.shape[0]}"
                    )
                content = self._learnable_tables[name](ids[mask]).to(dtype=dtype)
                embeds[mask] = content + type_table[pos].to(dtype=dtype)

        return embeds, t["head_output_indices"]


def _coerce_text_learnable(
    learnable: Sequence[dict[str, Any] | NumericEmbedderModalitySpec] | None,
) -> list[NumericEmbedderModalitySpec]:
    raw = learnable or []
    specs: list[NumericEmbedderModalitySpec] = []
    n_learnable = 0
    for m in raw:
        if isinstance(m, NumericEmbedderModalitySpec):
            spec = m
        else:
            data = dict(m)
            data.setdefault("type", "learnable")
            if data.get("type") != "learnable":
                raise TypeError(
                    "TextEmbedder learnable= entries must be type='learnable' "
                    "(text/token/image packing lives on Tokenizer)"
                )
            spec = NumericEmbedderModalitySpec(**data)
        if spec.type != "learnable":
            raise TypeError(
                "TextEmbedder learnable= entries must be type='learnable' "
                "(text/token/image packing lives on Tokenizer)"
            )
        specs.extend(expand_embedder_numeric_spec(spec, learnable_index=n_learnable))
        n_learnable += 1
    seen: set[str] = set()
    for spec in specs:
        name = str(spec.field)
        if name in seen:
            raise ValueError(f"duplicate TextEmbedder learnable name {name!r}")
        seen.add(name)
    return specs


def _load_embed_tokens(
    *,
    pretrained: str | Path,
    hidden_dim: int,
    hub_kwargs: dict | None,
) -> nn.Embedding:
    from transformers import AutoModel

    from mouse_core.models.backbone.base import _quiet_transformers_load

    with _quiet_transformers_load():
        model = AutoModel.from_pretrained(pretrained, **dict(hub_kwargs or {}))
    src = model.get_input_embeddings()
    if src.embedding_dim != hidden_dim:
        raise ValueError(
            f"pretrained embedding dim {src.embedding_dim} != hidden_dim {hidden_dim}"
        )
    emb = nn.Embedding(
        src.num_embeddings, hidden_dim, padding_idx=getattr(src, "padding_idx", None)
    )
    with torch.no_grad():
        emb.weight.copy_(src.weight)
    del model
    return emb
