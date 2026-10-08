"""Identity backbone for no-backbone ablations."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mouse_core.data.token_batch import TokenBatch
from mouse_core.models.backbone.base import Backbone
from mouse_core.models.backbone.embed import embed_token_ids
from mouse_core.models.backbone.fourier import FourierFeatures, check_num_frequencies


class IdentityBackbone(Backbone):
    """A no-op backbone: returns embeddings unchanged and never produces a cache.

    Owns a vocab table (``hidden_dim=`` × ``vocab_size=``) used by
    :meth:`embed`. There is no separate embedder.

    ``num_frequencies`` builds a Fourier module added onto
    ``__numeric__`` token embeddings. Omit it and a batch that
    contains those tokens raises.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        vocab_size: int,
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        self._hidden_dim = int(hidden_dim)
        self.vocab_size = int(vocab_size)
        self.embed_tokens = nn.Embedding(self.vocab_size, self._hidden_dim)
        self.num_frequencies = check_num_frequencies(num_frequencies)
        count = self.num_frequencies
        self.fourier: FourierFeatures | None = (
            None
            if count is None
            else FourierFeatures(num_frequencies=count, hidden_dim=self._hidden_dim)
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def embed(self, token_batch: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        return embed_token_ids(
            embed_tokens=self.embed_tokens,
            token_batch=token_batch,
            hidden_dim=self._hidden_dim,
            fourier=self.fourier,
        )

    def forward(
        self,
        embeds: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if output_hidden_states:
            return embeds, (embeds,)
        return embeds
