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
        """Capacity hint; real layout comes from ``prediction_indices``."""
        ...

    @abstractmethod
    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed ``TokenBatch`` → ``(embeds [L, D], prediction_indices [N])``."""
        ...

    @abstractmethod
    def pool_step_reprs(self, h: torch.Tensor, prediction_indices: torch.Tensor) -> torch.Tensor:
        """Gather prediction tokens → ``[N, D]`` (train) or ``[B, S, D]`` (decode).

        ``h`` is ``[L, D]`` (flat packed) or ``[B, L, D]`` (decode).
        Train: ``prediction_indices`` is ``[N]`` absolute indices into ``0 .. L-1``.
        Decode: ``prediction_indices`` is ``[B, S]`` into the token axis of ``h``.
        """
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
    """Named embedding tables + static Fourier over a :class:`TokenBatch`."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        modalities: list[dict[str, Any] | NumericEmbedderModalitySpec]
        | Mapping[str, dict[str, Any]]
        | None = None,
        fourier_min: float = 0.01,
        fourier_max: float = 10.0,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self.fourier_min = float(fourier_min)
        self.fourier_max = float(fourier_max)
        self.std = float(std)

        specs, meta = resolve_embedder_numeric_modalities(modalities)
        self.modalities: list[NumericEmbedderModalitySpec] = list(specs)
        self._meta: list[EmbedderModalityMeta] = list(meta)
        self._meta_by_name = {m.name: m for m in self._meta}

        self._tables = nn.ModuleDict()
        max_freq_sets = 1
        for m in self._meta:
            max_freq_sets = max(max_freq_sets, m.freq_sets)
            if m.kind in (KIND_DISCRETE, KIND_LEARNABLE, KIND_IMAGE):
                vs = m.vocab_size if m.kind != KIND_LEARNABLE else m.n_learnable
                if vs <= 0:
                    kind = "learnable tokens" if m.kind == KIND_LEARNABLE else "vocab_size"
                    raise ValueError(
                        f"{m.kind} modality {m.name!r} requires {kind}="
                    )
                scale = m.spec.std if m.spec.std is not None else std
                self._tables[m.name] = ScaledEmbedding(vs, hidden_dim, scale=scale)

        self._fourier_std: dict[str, float] = {
            m.name: float(m.spec.std if m.spec.std is not None else std)
            for m in self._meta
            if m.kind == KIND_FOURIER
        }
        fourier_scale = float(std) / (0.5 ** 0.5)
        self.fourier = StaticFourierFeatures(
            num_features=hidden_dim,
            in_min=fourier_min,
            in_max=fourier_max,
            num_freq_sets=max_freq_sets,
            output_scale=fourier_scale,
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        """Max tokens if nothing is skipped (capacity hint)."""
        total = 0
        for m in self._meta:
            if m.kind == KIND_DISCRETE:
                total += 1
            elif m.kind == KIND_FOURIER:
                total += m.dim
            elif m.kind == KIND_LEARNABLE:
                total += m.n_learnable
            elif m.kind == KIND_IMAGE:
                total += 1  # unknown a priori; hint only
        return total

    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            device = self.fourier.get_buffer("freqs").device
            dtype = self.fourier.get_buffer("freqs").dtype
        t = token_batch.to_tensors(device)
        modality_ids = t["modality_ids"]
        ids = t["ids"]
        values = t["values"]
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
                if meta.kind in (KIND_DISCRETE, KIND_LEARNABLE, KIND_IMAGE):
                    embeds[mask] = self._tables[name](ids[mask]).to(dtype=dtype)
                elif meta.kind == KIND_FOURIER:
                    feat = self.fourier(values[mask], ids[mask])
                    mod_std = self._fourier_std[name]
                    if self.std != 0.0 and mod_std != self.std:
                        feat = feat * (mod_std / self.std)
                    embeds[mask] = feat.to(dtype=dtype)

        return embeds, t["prediction_indices"]

    def pool_step_reprs(self, h: torch.Tensor, prediction_indices: torch.Tensor) -> torch.Tensor:
        D = self._hidden_dim
        if h.ndim == 2:
            return h[prediction_indices.reshape(-1)]
        B, S = prediction_indices.shape
        idx = prediction_indices.unsqueeze(-1).expand(B, S, D)
        return h.gather(1, idx)
