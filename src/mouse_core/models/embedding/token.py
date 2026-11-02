"""Token embeddings with independent optional measurement and position additions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from mouse_core.models.embedding.features import FeatureAdditions
from mouse_core.models.embedding.pretrained import load_token_table


class Embedding(nn.Module):
    """Prepared token tensors to embeddings ``[L, D]``; no device transfer.

    The table and optional numeric Fourier module belong to this stage.
    Load only the token table with ``pretrained=`` or create one with
    ``vocab_size=`` and ``hidden_dim=``. ``num_frequencies=`` adds trainable
    numeric Fourier features with one learned projection per numeric input field.
    Supply numeric_field_names from the tokenizer. Initial weights match the token table RMS.
    ``position_num_frequencies=`` enables a separate trainable projection of scalar
    context, added wherever ``position_mask`` is true. Missing values add nothing;
    zero and negative values are valid. It is independent of the numeric projection.
    All embedding parameters are float32.
    The caller moves inputs to the model's device before calling this module.
    """

    def __init__(
        self, *, pretrained: str | Path | None = None,
        vocab_size: int | None = None, hidden_dim: int | None = None,
        num_frequencies: int | None = None, position_num_frequencies: int | None = None,
        numeric_field_names: Sequence[str] = (),
        hub_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if pretrained is not None:
            table = load_token_table(pretrained, hub_kwargs or {})
            if vocab_size is not None and vocab_size != table.num_embeddings:
                raise ValueError("vocab_size does not match the pretrained token table.")
            if hidden_dim is not None and hidden_dim != table.embedding_dim:
                raise ValueError("hidden_dim does not match the pretrained token table.")
        else:
            for name, value in (("vocab_size", vocab_size), ("hidden_dim", hidden_dim)):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer when pretrained is omitted.")
            assert vocab_size is not None and hidden_dim is not None
            table = nn.Embedding(vocab_size, hidden_dim, dtype=torch.float32)
        self.embed_tokens = table
        self.hidden_dim = table.embedding_dim
        self.vocab_size = table.num_embeddings
        self.features = FeatureAdditions(
            hidden_dim=self.hidden_dim, numeric_field_names=numeric_field_names,
            num_frequencies=num_frequencies, position_num_frequencies=position_num_frequencies,
        )
        rms = (torch.linalg.vector_norm(table.weight.detach()) / table.weight.numel() ** 0.5)
        self.features.scale_initial_weights(rms)

    def resize_token_embeddings(self, vocab_size: int) -> None:
        """Grow for registered special tokens before constructing an optimizer.

        Existing rows are preserved. New rows start at the pretrained mean.
        Call this before loading a checkpoint that contains the expanded table.
        """
        if type(vocab_size) is not int or vocab_size < self.vocab_size:
            raise ValueError("vocab_size must be an integer at least as large as the existing table")
        if vocab_size == self.vocab_size:
            return
        old = self.embed_tokens.weight
        table = nn.Embedding(vocab_size, self.hidden_dim, device=old.device, dtype=old.dtype)
        with torch.no_grad():
            table.weight[:self.vocab_size].copy_(old)
            table.weight[self.vocab_size:].copy_(old.mean(dim=0))
        table.weight.requires_grad_(old.requires_grad)
        self.embed_tokens = table
        self.vocab_size = vocab_size

    def forward(
        self, token_ids: torch.Tensor, *, numeric_values: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
        numeric_field_ids: torch.Tensor | None = None, numeric_field_names: Sequence[str] = (),
        position_values: torch.Tensor | None = None, position_mask: torch.Tensor | None = None,
        tokenizer_ids: torch.Tensor | None = None, tokenizer_names: Sequence[str] = ("default",),
    ) -> torch.Tensor:
        """Look up IDs and add each optional feature only at its masked positions."""
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("Embedding expects tensor token_ids.")
        if len(tokenizer_names) != 1:
            raise ValueError("Multiple tokenizers require MultiEmbedding with one table per tokenizer")
        if tokenizer_ids is not None:
            if (tokenizer_ids.shape != token_ids.shape or tokenizer_ids.dtype != torch.long
                    or tokenizer_ids.device != token_ids.device or bool((tokenizer_ids != 0).any())):
                raise ValueError("Single Embedding requires tokenizer_ids of zero matching token_ids")
        if token_ids.device != self.embed_tokens.weight.device:
            raise ValueError("input tensors and embedding weights must be on the same device")
        return self.features(
            self.embed_tokens(token_ids), numeric_values=numeric_values, numeric_mask=numeric_mask,
            numeric_field_ids=numeric_field_ids, numeric_field_names=numeric_field_names,
            position_values=position_values, position_mask=position_mask,
        )
