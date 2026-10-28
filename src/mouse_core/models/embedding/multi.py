"""Route each token to the embedding owned by its named tokenizer."""
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from mouse_core.models.embedding.token import Embedding


class MultiEmbedding(nn.Module):
    """Independent embedding tables, one per tokenizer, producing ``[L, D]``.

    Build ``embeddings={name: Embedding(...)}`` with the same names used by
    ``Tokenizer(tokenizers=...)``. Tables must have the same hidden dimension,
    but may have different vocabulary sizes and numeric Fourier features.
    Token IDs remain local; no vocabulary offsets or new tokens are added.
    Each table can be trained or frozen independently using normal PyTorch.
    """

    def __init__(self, *, embeddings: Mapping[str, Embedding]) -> None:
        super().__init__()
        if not embeddings or any(not isinstance(name, str) or not name or '.' in name for name in embeddings):
            raise ValueError("embeddings must map nonempty names without dots to Embedding modules")
        if any(not isinstance(module, Embedding) for module in embeddings.values()):
            raise TypeError("Each tokenizer needs an Embedding module")
        if len({module.hidden_dim for module in embeddings.values()}) != 1:
            raise ValueError("All tokenizer embeddings must have the same hidden_dim")
        parameters = [id(parameter) for module in embeddings.values() for parameter in module.parameters()]
        if len(parameters) != len(set(parameters)):
            raise ValueError("Each tokenizer must own independent embedding parameters")
        self.tokenizer_names = tuple(sorted(embeddings))
        self.embeddings = nn.ModuleDict({name: embeddings[name] for name in self.tokenizer_names})
        self.hidden_dim = embeddings[self.tokenizer_names[0]].hidden_dim

    def forward(
        self, token_ids: torch.Tensor, *, tokenizer_ids: torch.Tensor,
        tokenizer_names: Sequence[str], numeric_values: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route prepared tensors without transferring them between devices."""
        if tuple(tokenizer_names) != self.tokenizer_names:
            raise ValueError("Input tokenizer_names must match the embedding tables in order")
        if token_ids.ndim != 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 [L]")
        if (tokenizer_ids.shape != token_ids.shape or tokenizer_ids.dtype != torch.long
                or tokenizer_ids.device != token_ids.device):
            raise ValueError("tokenizer_ids must be int64 [L] on the token_ids device")
        if bool(((tokenizer_ids < 0) | (tokenizer_ids >= len(self.tokenizer_names))).any()):
            raise ValueError("tokenizer_ids must identify an embedding table")
        if (numeric_values is None) != (numeric_mask is None):
            raise ValueError("numeric_values and numeric_mask must be supplied together")
        for values in (numeric_values, numeric_mask):
            if values is not None and (values.shape != token_ids.shape or values.device != token_ids.device):
                raise ValueError("Numeric tensors must match token_ids shape and device")
        if numeric_mask is not None and numeric_mask.dtype != torch.bool:
            raise TypeError("numeric_mask must be a boolean tensor")
        first = next(self.parameters())
        if any(parameter.device != token_ids.device for parameter in self.parameters()):
            raise ValueError("Input tensors and all embedding tables must be on the same device")
        output = first.new_zeros((token_ids.numel(), self.hidden_dim))
        for index, name in enumerate(self.tokenizer_names):
            selected = tokenizer_ids == index
            if not bool(selected.any()):
                continue
            output[selected] = self.embeddings[name](
                token_ids[selected],
                numeric_values=None if numeric_values is None else numeric_values[selected],
                numeric_mask=None if numeric_mask is None else numeric_mask[selected],
            )
        return output
