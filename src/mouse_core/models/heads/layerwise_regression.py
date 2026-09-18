"""Layerwise regression heads — one RegressionHead per backbone layer."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from mouse_core.models.heads.base import BaseHead
from mouse_core.models.heads.regression import RegressionHead


class LayerwiseRegressionHead(BaseHead):
    """One :class:`RegressionHead` per backbone layer.

    Expects pooled step representations stacked over layers as:

    * train (flat): ``[N, L, D]`` → values ``[N, L, A]``
    * decode (rect): ``[B, L, S, D]`` → values ``[B, S, L, A]``
    """

    def __init__(
        self,
        *,
        num_backbone_layers: int,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int,
        use_norm: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_backbone_layers < 1:
            raise ValueError(
                f"num_backbone_layers must be >= 1, got {num_backbone_layers}."
            )
        if type(use_norm) is not bool:
            raise TypeError(f"use_norm must be a bool, got {use_norm!r}.")
        self.num_backbone_layers = int(num_backbone_layers)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.scale = float(scale)
        self.use_norm = use_norm
        self.layer_heads = nn.ModuleList(
            [
                RegressionHead(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    scale=scale,
                    use_norm=use_norm,
                )
                for _ in range(self.num_backbone_layers)
            ]
        )

    def _heads(self) -> list[RegressionHead]:
        return [cast(RegressionHead, head) for head in self.layer_heads]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns per-output values ``[N, L, A]`` or ``[B, S, L, A]``."""
        if h.ndim == 3:
            if h.shape[1] != self.num_backbone_layers:
                raise ValueError(
                    f"Expected {self.num_backbone_layers} backbone layers in h, got {h.shape[1]}."
                )
            outputs = [
                head.forward(h[:, layer_idx]) for layer_idx, head in enumerate(self._heads())
            ]
            return torch.stack(outputs, dim=1)
        if h.ndim != 4:
            raise ValueError(
                f"LayerwiseRegressionHead expects h shape [N, L, D] or "
                f"[B, L, S, D], got {tuple(h.shape)}."
            )
        if h.shape[1] != self.num_backbone_layers:
            raise ValueError(
                f"Expected {self.num_backbone_layers} backbone layers in h, got {h.shape[1]}."
            )
        outputs = [
            head.forward(h[:, layer_idx]) for layer_idx, head in enumerate(self._heads())
        ]
        return torch.stack(outputs, dim=2)
