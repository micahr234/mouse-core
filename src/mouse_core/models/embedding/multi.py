"""Route each token to the embedding owned by its named tokenizer."""
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from mouse_core.models.embedding.token import Embedding
from mouse_core.models.embedding.features import FeatureAdditions


class MultiEmbedding(nn.Module):
    """Independent embedding tables, one per tokenizer, producing ``[L, D]``.

    Build ``embeddings={name: Embedding(...)}`` with the same names used by
    ``Tokenizer(tokenizers=...)``. Tables must have the same hidden dimension,
    but may have different vocabulary sizes. Numeric fields own distinct projections
    on this module; a single position projection is shared across all tables.
    Token IDs remain local; no vocabulary offsets or new tokens are added.
    Each table can be trained or frozen independently using normal PyTorch.
    """

    def __init__(self, *, embeddings: Mapping[str, Embedding],
                 numeric_field_names: Sequence[str] = (), num_frequencies: int | None = None,
                 position_num_frequencies: int | None = None) -> None:
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
        if any(module.features.numeric or module.features.position is not None for module in embeddings.values()):
            raise ValueError("Configure numeric and position features on MultiEmbedding, not its token tables")
        self.tokenizer_names = tuple(sorted(embeddings))
        self.embeddings = nn.ModuleDict({name: embeddings[name] for name in self.tokenizer_names})
        self.hidden_dim = embeddings[self.tokenizer_names[0]].hidden_dim
        self.features = FeatureAdditions(
            hidden_dim=self.hidden_dim, numeric_field_names=numeric_field_names,
            num_frequencies=num_frequencies, position_num_frequencies=position_num_frequencies,
        )
        tables = [module.embed_tokens.weight.detach() for module in embeddings.values()]
        rms = torch.sqrt(sum(table.square().sum() for table in tables) / sum(table.numel() for table in tables))
        self.features.scale_initial_weights(rms)

    def forward(
        self, token_ids: torch.Tensor, *, tokenizer_ids: torch.Tensor,
        tokenizer_names: Sequence[str], numeric_values: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
        numeric_field_ids: torch.Tensor | None = None, numeric_field_names: Sequence[str] = (),
        position_values: torch.Tensor | None = None, position_mask: torch.Tensor | None = None,
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
        first = next(self.parameters())
        if any(parameter.device != token_ids.device for parameter in self.parameters()):
            raise ValueError("Input tensors and all embedding tables must be on the same device")
        output = first.new_zeros((token_ids.numel(), self.hidden_dim))
        for index, name in enumerate(self.tokenizer_names):
            selected = tokenizer_ids == index
            if not bool(selected.any()):
                continue
            output[selected] = self.embeddings[name](token_ids[selected])
        return self.features(
            output, numeric_values=numeric_values, numeric_mask=numeric_mask,
            numeric_field_ids=numeric_field_ids, numeric_field_names=numeric_field_names,
            position_values=position_values, position_mask=position_mask,
        )
