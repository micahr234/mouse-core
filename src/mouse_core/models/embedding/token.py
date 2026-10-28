"""Look up token IDs and add Fourier features at numeric-token positions."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from mouse_core.models.embedding.fourier import FourierFeatures, check_num_frequencies
from mouse_core.models.embedding.pretrained import load_token_table


class Embedding(nn.Module):
    """Prepared token tensors to embeddings ``[L, D]``; no device transfer.

    The table and optional numeric Fourier module belong to this stage.
    Load only the token table with ``pretrained=`` or create one with
    ``vocab_size=`` and ``hidden_dim=``. ``num_frequencies=`` adds trainable
    numeric Fourier features, initialized to match the token table RMS.
    All embedding parameters are float32.
    The caller moves inputs to the model's device before calling this module.
    """

    def __init__(
        self, *, pretrained: str | Path | None = None,
        vocab_size: int | None = None, hidden_dim: int | None = None,
        num_frequencies: int | None = None, hub_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_frequencies = check_num_frequencies(num_frequencies)
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
        self.fourier = (None if self.num_frequencies is None else
                        FourierFeatures(num_frequencies=self.num_frequencies, hidden_dim=self.hidden_dim))
        if self.fourier is not None:
            with torch.no_grad():
                token_rms = torch.linalg.vector_norm(table.weight) / table.weight.numel() ** 0.5
                self.fourier.proj.weight.mul_(token_rms)

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
        tokenizer_ids: torch.Tensor | None = None, tokenizer_names: Sequence[str] = ("default",),
    ) -> torch.Tensor:
        """Look up IDs and add Fourier features only at numeric-token positions."""
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
        if (numeric_values is None) != (numeric_mask is None):
            raise ValueError("numeric_values and numeric_mask must be supplied together")
        if numeric_values is not None and numeric_mask is not None:
            if not isinstance(numeric_values, torch.Tensor) or not isinstance(numeric_mask, torch.Tensor):
                raise TypeError("Embedding expects tensor numeric_values and numeric_mask.")
            if numeric_values.device != token_ids.device or numeric_mask.device != token_ids.device:
                raise ValueError("input tensors and embedding weights must be on the same device")
            if numeric_values.shape != token_ids.shape or numeric_mask.shape != token_ids.shape:
                raise ValueError("numeric_values and numeric_mask must have the same shape as token_ids")
            if numeric_mask.dtype != torch.bool:
                raise TypeError("numeric_mask must be a boolean tensor")
        embeddings = self.embed_tokens(token_ids)
        if numeric_mask is not None and bool(numeric_mask.any()):
            if self.fourier is None:
                raise RuntimeError("numeric tokens require Fourier features; pass num_frequencies= when building the embedding")
            assert numeric_values is not None
            embeddings[numeric_mask] = embeddings[numeric_mask] + self.fourier(numeric_values[numeric_mask])
        return embeddings
