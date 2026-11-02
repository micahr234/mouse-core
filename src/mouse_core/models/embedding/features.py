"""Field-specific numeric projections and one shared position projection."""
from collections.abc import Sequence
import torch
from torch import nn
from .fourier import FourierFeatures, check_num_frequencies
from .position import PositionFeatures, validate_position_tensors


class FeatureAdditions(nn.Module):
    """Add scalar features once at the input of the transformer.

    Each numeric input identity owns a learned projection. Repeated values,
    events, and vector elements from that input reuse it. Position tokens share
    one separate projection, including across named token embedding tables.
    """
    def __init__(self, *, hidden_dim: int, numeric_field_names: Sequence[str] = (),
                 num_frequencies: int | None = None, position_num_frequencies: int | None = None):
        super().__init__()
        self.numeric_field_names = tuple(numeric_field_names)
        if (any(not isinstance(name, str) or not name for name in self.numeric_field_names)
                or len(set(self.numeric_field_names)) != len(self.numeric_field_names)):
            raise ValueError("numeric_field_names must contain unique nonempty identities")
        self.num_frequencies = check_num_frequencies(num_frequencies)
        self.position_num_frequencies = check_num_frequencies(position_num_frequencies)
        if self.numeric_field_names and self.num_frequencies is None:
            raise ValueError("Numeric input fields require num_frequencies")
        self.numeric = nn.ModuleList([
            FourierFeatures(num_frequencies=self.num_frequencies, hidden_dim=hidden_dim)
            for _ in self.numeric_field_names
        ])
        self.position = (None if self.position_num_frequencies is None else
                        PositionFeatures(num_frequencies=self.position_num_frequencies, hidden_dim=hidden_dim))

    def scale_initial_weights(self, rms):
        with torch.no_grad():
            for module in (*self.numeric, self.position):
                if module is not None:
                    module.proj.weight.mul_(rms)

    def forward(self, embeddings, *, numeric_values=None, numeric_mask=None,
                numeric_field_ids=None, numeric_field_names=(), position_values=None, position_mask=None):
        # Shape/device validation uses one representative value per token.
        tokens = embeddings[:, 0]
        if (numeric_values is None) != (numeric_mask is None):
            raise ValueError("numeric_values and numeric_mask must be supplied together")
        if numeric_values is not None:
            for value in (numeric_values, numeric_mask):
                if not isinstance(value, torch.Tensor):
                    raise TypeError("Numeric values and mask must be tensors")
                if value.shape != tokens.shape or value.device != tokens.device:
                    raise ValueError("Numeric tensors must match token shape and device")
            if numeric_mask.dtype != torch.bool:
                raise TypeError("numeric_mask must be a boolean tensor")
            if bool(numeric_mask.any()):
                if not self.numeric or tuple(numeric_field_names) != self.numeric_field_names:
                    raise ValueError("Numeric input field identities must match configured projections")
                if not isinstance(numeric_field_ids, torch.Tensor):
                    raise ValueError("Numeric tokens require numeric_field_ids")
                if (numeric_field_ids.shape != tokens.shape or numeric_field_ids.device != tokens.device
                        or numeric_field_ids.dtype != torch.long):
                    raise ValueError("numeric_field_ids must be int64 matching token shape and device")
                if (bool((numeric_field_ids[~numeric_mask] != -1).any())
                        or bool((numeric_field_ids[numeric_mask] < 0).any())
                        or bool((numeric_field_ids[numeric_mask] >= len(self.numeric)).any())):
                    raise ValueError("numeric_field_ids must identify numeric projections; other tokens use -1")
                for index, projection in enumerate(self.numeric):
                    selected = numeric_mask & (numeric_field_ids == index)
                    if bool(selected.any()):
                        embeddings[selected] = embeddings[selected] + projection(numeric_values[selected])
        validate_position_tensors(tokens, position_values, position_mask)
        if position_mask is not None and bool(position_mask.any()):
            if self.position is None:
                raise RuntimeError("Position values require position_num_frequencies on the embedding")
            embeddings[position_mask] = embeddings[position_mask] + self.position(position_values[position_mask])
        return embeddings
