"""SwiGLU building blocks shared by projection and prediction MLPs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Apply ``silu(gate) * value`` using one fused linear projection."""

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 2 * hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.linear(x).chunk(2, dim=-1)
        return F.silu(gate) * value


def swiglu_blocks(*, in_features: int, hidden_dim: int, num_blocks: int) -> list[nn.Module]:
    """Build the hidden stages; each caller owns its norm and final linear."""
    return [
        SwiGLU(in_features if index == 0 else hidden_dim, hidden_dim)
        for index in range(num_blocks)
    ]
