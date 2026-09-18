"""Heads that only exist so an objective can resolve a predictions key."""

from __future__ import annotations

import torch

from mouse_core.models.heads.base import BaseHead, _bind_prediction_key


class BoundHead(BaseHead):
    def __init__(self, key: str) -> None:
        super().__init__()
        _bind_prediction_key(self, key)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h
