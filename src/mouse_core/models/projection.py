"""Shared projection between the backbone and prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from mouse_core.models.mlp import swiglu_blocks


class ProjectionHead(nn.Module):
    """Width-preserving SwiGLU MLP with its own affine input RMSNorm.

    The backbone keeps its native final norm. Both the projection and
    prediction heads always prenormalize their inputs. The readout is::

        normalized backbone → RMSNorm → projection MLP → RMSNorm → prediction heads

    ``num_layers`` counts the linear stages: ``num_layers - 1`` SwiGLU
    blocks followed by a linear projection back to ``in_features``.
    This input RMSNorm belongs to the projection and uses its learning
    rate and Polyak rate. Each prediction head owns its separate input
    norm. Parameters are float32.
    """

    def __init__(
        self,
        *,
        in_features: int,
        hidden_dim: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("in_features", in_features, 1),
            ("hidden_dim", hidden_dim, 1),
            ("num_layers", num_layers, 2),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
        self.in_features = in_features
        self.out_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_norm = nn.RMSNorm(in_features, eps=1e-5)
        self.layers = nn.Sequential(
            *swiglu_blocks(in_features=in_features, hidden_dim=hidden_dim, num_blocks=num_layers - 1),
            nn.Linear(hidden_dim, in_features),
        )
        self.to(device="cpu", dtype=torch.float32)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.layers(self.input_norm(h.float()))
