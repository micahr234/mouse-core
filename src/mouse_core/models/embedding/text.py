"""TextEmbedder — pretrained token embeddings over a flat TokenBatch."""

from __future__ import annotations

from pathlib import Path
from string import Formatter
from typing import Any

import torch
import torch.nn as nn

from mouse_core.data.text_tokenizer import NAME_TEXT, NAME_VISION
from mouse_core.data.token_batch import ModalityInfo, TokenBatch
from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.embedding.modality import (
    TextEmbedderModalitySpec,
    expand_embedder_text_spec,
)


class TextEmbedder(Encoder):
    """Pretrained token embeddings over a flat :class:`TokenBatch`.

    Token packing lives in :class:`~mouse_core.data.text_tokenizer.TextTokenizer`
    (constructed separately). Alignment is by modality name (``__text__`` /
    ``__vision__``). This module only looks up ``embed_tokens``.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        modalities: list[dict | TextEmbedderModalitySpec] | None = None,
        format: str | None = None,
        pretrained: str | Path | None = None,
        embed_tokens: nn.Embedding | None = None,
        hub_kwargs: dict | None = None,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self.format = format
        self._hub_kwargs = dict(hub_kwargs or {})
        self._pretrained = str(pretrained) if pretrained is not None else None

        raw = modalities or []
        specs: list[TextEmbedderModalitySpec] = []
        for m in raw:
            if isinstance(m, TextEmbedderModalitySpec):
                spec = m
            else:
                data = dict(m)
                if "input_field" in data or "output_field" in data:
                    raise TypeError(
                        "embedder modalities use field= "
                        "(not input_field=/output_field=); "
                        "rename with Selector before tokenize"
                    )
                for banned in ("skip", "required"):
                    if banned in data:
                        raise TypeError(
                            f"embedder modalities do not accept {banned}= "
                            "(tokenizer packing knob)"
                        )
                spec = TextEmbedderModalitySpec(**data)
            specs.extend(expand_embedder_text_spec(spec))
        self.modalities: list[TextEmbedderModalitySpec] = specs

        has_text = any(s.type == "text" for s in specs)
        has_token = any(s.type == "token" for s in specs)
        has_image = any(s.type == "image" for s in specs)
        needs_format = has_text or has_image or has_token
        if needs_format and format is None:
            raise TypeError(
                "TextEmbedder requires format= when text, token, or image modalities "
                "are declared"
            )
        if format is not None and not (has_text or has_token or has_image):
            raise TypeError("format= requires at least one text, token, or image modality")

        if format is not None:
            text_by_field = {
                s.field
                for s in specs
                if s.type == "text" and isinstance(s.field, str)
            }
            token_by_field = {
                s.field
                for s in specs
                if s.type == "token" and isinstance(s.field, str)
            }
            image_by_field = {
                s.field
                for s in specs
                if s.type == "image" and isinstance(s.field, str)
            }
            for _, name, _, _ in Formatter().parse(format):
                if name is None or name == "":
                    continue
                if (
                    name not in text_by_field
                    and name not in token_by_field
                    and name not in image_by_field
                ):
                    raise ValueError(
                        f"format placeholder {{{name}}} has no matching text/token/image modality"
                    )

        expected_names: set[str] = set()
        if has_text or has_token:
            expected_names.add(NAME_TEXT)
        if has_image:
            expected_names.add(NAME_VISION)
        self._expected_names = frozenset(expected_names)

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
            raise TypeError("TextEmbedder requires pretrained= or embed_tokens=")

        if freeze_embeddings:
            self.embed_tokens.weight.requires_grad_(False)

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
        return 0

    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        t = token_batch.to_tensors(device)
        ids = t["ids"]
        modality_ids = t["modality_ids"]
        names: tuple[str, ...] = t["modality_names"]
        batch_map: dict[str, ModalityInfo] = t["modality_map"]

        for name in names:
            if name not in self._expected_names:
                raise KeyError(
                    f"TokenBatch modality {name!r} not expected by TextEmbedder "
                    f"(have {sorted(self._expected_names)})"
                )
            info = batch_map[name]
            if name == NAME_TEXT and info.type not in ("token", "text"):
                raise TypeError(
                    f"modality {name!r} type mismatch: batch={info.type!r} expected token/text"
                )
            if name == NAME_VISION and info.type != "image":
                raise TypeError(
                    f"modality {name!r} type mismatch: batch={info.type!r} expected image"
                )

        L = ids.shape[0]
        D = self._hidden_dim
        embeds = torch.zeros(L, D, device=device, dtype=dtype)
        if L > 0:
            for local_id, name in enumerate(names):
                mask = modality_ids == local_id
                if bool(mask.any()):
                    embeds[mask] = self.embed_tokens(ids[mask]).to(dtype=dtype)

        step_fields = dict(t["step_fields"])
        prediction_indices = t["prediction_indices"]
        return embeds, step_fields, prediction_indices

    def pool_step_reprs(self, h: torch.Tensor, prediction_indices: torch.Tensor) -> torch.Tensor:
        D = self._hidden_dim
        if h.ndim == 2:
            return h[prediction_indices.reshape(-1)]
        B, S = prediction_indices.shape
        idx = prediction_indices.unsqueeze(-1).expand(B, S, D)
        return h.gather(1, idx)


def _load_embed_tokens(
    *,
    pretrained: str | Path,
    hidden_dim: int,
    hub_kwargs: dict | None,
) -> nn.Embedding:
    from transformers import AutoModel

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
