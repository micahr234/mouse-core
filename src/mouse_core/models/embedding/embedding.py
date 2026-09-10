"""NumericEmbedder — typed GPU maps over a flat TokenBatch.

Token packing lives in :class:`~mouse_core.data.numeric_tokenizer.NumericTokenizer`
(constructed separately). Alignment is by modality **name** (not list order).
This module only applies embedding tables and static Fourier features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from mouse_core.data.token_batch import ModalityInfo, TokenBatch
from mouse_core.models.embedding.encoding import StaticFourierFeatures
from mouse_core.models.embedding.linear import ScaledEmbedding
from mouse_core.models.embedding.modality import (
    KIND_DISCRETE,
    KIND_FOURIER,
    KIND_IMAGE,
    KIND_LEARNABLE,
    EmbedderModalityMeta,
    NumericEmbedderModalitySpec,
    resolve_embedder_numeric_modalities,
)


class Encoder(nn.Module, ABC):
    """Abstract base for embedders over :class:`~mouse_core.data.token_batch.TokenBatch`."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def tokens_per_step(self) -> int:
        """Capacity hint; real layout comes from ``head_output_indices``."""
        ...

    @abstractmethod
    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed ``TokenBatch`` → ``(embeds [L, D], head_output_indices [P])``."""
        ...


def _validate_batch_modalities(
    batch_map: Mapping[str, ModalityInfo],
    embedder_meta: Sequence[EmbedderModalityMeta],
) -> dict[str, EmbedderModalityMeta]:
    by_name = {m.name: m for m in embedder_meta}
    for name, info in batch_map.items():
        if name not in by_name:
            raise KeyError(
                f"TokenBatch modality {name!r} has no matching embedder modality "
                f"(have {sorted(by_name)})"
            )
        emb = by_name[name]
        # Tokenizer may emit type=fourier for both fourier/continuous packing.
        batch_type = info.type
        emb_type = emb.kind
        if batch_type == "fourier" and emb_type == KIND_FOURIER:
            pass
        elif batch_type != emb_type and not (
            batch_type == "continuous" and emb_type == KIND_FOURIER
        ):
            raise TypeError(
                f"modality {name!r} type mismatch: batch={batch_type!r} "
                f"embedder={emb_type!r}"
            )
    return by_name


class NumericEmbedder(Encoder):
    """Named embedding tables + static Fourier over a :class:`TokenBatch`.

    Every modality also has a learnable type table of shape
    ``[positions, D]``; token ``t`` gets row ``TokenBatch.positions[t]`` added
    to its content embedding, so the tokens of a multi-token modality
    (continuous coordinates, learnable slots, image patches) are
    distinguishable even when their content embeddings coincide.

    Each modality spec must set ``std`` (init scale of its content embeddings
    and type vectors) and ``positions`` (max tokens per step, i.e. type table
    rows). ``fourier`` / ``continuous`` specs must also set ``fourier_min``
    and ``fourier_max``. There are no embedder-wide defaults.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        modalities: Sequence[dict[str, Any] | NumericEmbedderModalitySpec]
        | Mapping[str, dict[str, Any]]
        | None = None,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)

        specs, meta = resolve_embedder_numeric_modalities(modalities)
        self.modalities: list[NumericEmbedderModalitySpec] = list(specs)
        self._meta: list[EmbedderModalityMeta] = list(meta)
        self._meta_by_name = {m.name: m for m in self._meta}

        self._tables = nn.ModuleDict()
        self._type_vectors = nn.ParameterDict()
        self.fourier = nn.ModuleDict()
        self._fourier_std: dict[str, float] = {}
        for m in self._meta:
            assert m.spec.std is not None
            mod_std = float(m.spec.std)
            self._type_vectors[m.name] = nn.Parameter(
                torch.randn(m.n_positions, hidden_dim) * mod_std
            )
            if m.kind in (KIND_DISCRETE, KIND_LEARNABLE, KIND_IMAGE):
                vs = m.vocab_size if m.kind != KIND_LEARNABLE else m.n_learnable
                if vs <= 0:
                    kind = "learnable tokens" if m.kind == KIND_LEARNABLE else "vocab_size"
                    raise ValueError(
                        f"{m.kind} modality {m.name!r} requires {kind}="
                    )
                self._tables[m.name] = ScaledEmbedding(vs, hidden_dim, scale=mod_std)
            elif m.kind == KIND_FOURIER:
                assert m.spec.fourier_min is not None
                assert m.spec.fourier_max is not None
                self._fourier_std[m.name] = mod_std
                # ``cos`` has variance 1/2; scale so each feature has unit std,
                # then multiply by the modality's ``std`` in ``forward``.
                self.fourier[m.name] = StaticFourierFeatures(
                    num_features=hidden_dim,
                    in_min=float(m.spec.fourier_min),
                    in_max=float(m.spec.fourier_max),
                    num_freq_sets=m.freq_sets,
                    output_scale=1.0 / (0.5 ** 0.5),
                )
        # Follows ``.to(device/dtype)`` so an encoder with no learnable tables
        # (fourier-only) still knows its compute dtype and device.
        self.register_buffer("_anchor", torch.zeros(0), persistent=False)
        self._anchor: torch.Tensor

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        """Max tokens per step: the sum of every modality's declared ``positions``."""
        return sum(m.n_positions for m in self._meta)

    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self.get_buffer("_anchor")
        device = anchor.device
        dtype = anchor.dtype
        t = token_batch.to_tensors(device)
        modality_ids = t["modality_ids"]
        ids = t["ids"]
        values = t["values"]
        positions = t["positions"]
        names: tuple[str, ...] = t["modality_names"]
        batch_map: dict[str, ModalityInfo] = t["modality_map"]
        _validate_batch_modalities(batch_map, self._meta)

        L = modality_ids.shape[0]
        D = self._hidden_dim
        embeds = torch.zeros(L, D, device=device, dtype=dtype)

        if L > 0:
            for local_id, name in enumerate(names):
                mask = modality_ids == local_id
                if not bool(mask.any()):
                    continue
                meta = self._meta_by_name[name]
                pos = positions[mask]
                type_table = self._type_vectors[name]
                if int(pos.max()) >= type_table.shape[0]:
                    raise ValueError(
                        f"modality {name!r} emitted {int(pos.max()) + 1} tokens in a "
                        f"step but the embedder declares positions={type_table.shape[0]}"
                    )
                type_vec = type_table[pos].to(dtype=dtype)
                if meta.kind in (KIND_DISCRETE, KIND_LEARNABLE, KIND_IMAGE):
                    content = self._tables[name](ids[mask]).to(dtype=dtype)
                elif meta.kind == KIND_FOURIER:
                    feat = self.fourier[name](values[mask], ids[mask]) * self._fourier_std[name]
                    content = feat.to(dtype=dtype)
                else:
                    continue
                embeds[mask] = content + type_vec

        return embeds, t["head_output_indices"]
