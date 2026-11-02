"""Trainable additive embeddings for optional numeric context."""
import torch
from torch import nn
from mouse_core.models.embedding.fourier import check_num_frequencies


def validate_position_tensors(token_ids, values, mask):
    if (values is None) != (mask is None):
        raise ValueError("position_values and position_mask must be supplied together")
    if values is None:
        return
    if not isinstance(values, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("position_values and position_mask must be tensors")
    if values.shape != token_ids.shape or mask.shape != token_ids.shape:
        raise ValueError("Position tensors must match token_ids shape")
    if values.device != token_ids.device or mask.device != token_ids.device:
        raise ValueError("Position tensors must match token_ids device")
    if mask.dtype != torch.bool:
        raise TypeError("position_mask must be boolean")
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError("Position values must be finite numeric scalars")


class PositionFeatures(nn.Module):
    """Sine/cosine at powers of two radians per normalized input unit.

    A separate learned projection adds shared context to tokens from a field.
    Tokenizer bounds map the input scalar to [-1, 1]. The lowest band is
    one radian, so the two endpoints have distinct features (unlike pi-based
    harmonics). Numeric tokens use a separate feature bank and learned matrix.
    Float64 angles preserve small differences at large magnitudes. Missing
    values are masked by the caller; zero has nonzero cosine features and
    negative values are supported. Values are never clipped or saturated.
    """
    def __init__(self, *, num_frequencies, hidden_dim):
        super().__init__()
        checked = check_num_frequencies(num_frequencies)
        if checked is None:
            raise TypeError("PositionFeatures requires num_frequencies")
        self.num_frequencies = checked
        self.proj = nn.Linear(2 * checked, hidden_dim, bias=False, dtype=torch.float32)
        nn.init.normal_(self.proj.weight, std=checked ** -0.5)

    def forward(self, values):
        if values.ndim != 1:
            raise ValueError("PositionFeatures expects a 1-D scalar tensor")
        frequencies = torch.pow(2.0, torch.arange(self.num_frequencies,
                                                   device=values.device, dtype=torch.float64))
        angles = values.to(torch.float64).unsqueeze(-1) * frequencies
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.proj(features.to(self.proj.weight.dtype))
