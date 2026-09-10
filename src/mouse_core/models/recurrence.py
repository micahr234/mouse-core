"""Recurrent-depth refinement: re-run the backbone on its own output.

A :class:`Recurrence` section makes ``Model.forward`` run the backbone
``num_passes`` times over the same token stream. Pass 1 reads the encoder
output ``e``. Pass ``k`` reads ``e + proj(RMSNorm(h_{k-1}))`` where
``h_{k-1}`` is the previous pass's last-layer residual stream. Every pass
runs the heads, so a training loop can supervise each pass and average the
losses; ``get_action`` reads the final pass.

Why the adapter: the backbone output passes through the pretrained final
RMSNorm, whose learned per-dim gain leaves it on a very different scale
from the encoder embeddings (on Qwen3-0.6B the gain carries outlier dims,
versus encoder embeddings of RMS ~0.03). Feeding it straight back as the
next input lets those outliers compound pass over pass and in bf16 rounds
the later passes' layer contributions away. The adapter re-normalizes the
recycled state without a gain, re-injects the original encodings so no
pass loses the input, and starts with a zero projection so at construction
every pass equals pass 1 — the recurrence is an exact no-op until the
optimizer turns it on.

Cached decode keeps one KV session per pass (pass ``k`` of a new token
attends to pass ``k`` states of earlier tokens), so a recurrent model
decodes incrementally with the same recurrence it trained with.
Recurrence and :class:`~mouse_core.models.reasoner.LatentReasoner` cannot
be combined on one model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Recurrence(nn.Module):
    """Input adapter for recurrent backbone passes.

    Attach with ``Model(recurrence=Recurrence(hidden_dim=D, num_passes=K))``.
    ``forward(encodings, last_hidden_state)`` returns the next pass's
    backbone input ``encodings + proj(RMSNorm(last_hidden_state))``; ``proj``
    is zero-initialised. Stays float32 under ``Model.to``; the backbone
    output is cast to the encodings' dtype.

    Args:
        hidden_dim: Backbone hidden dimension ``D``.
        num_passes: Total backbone passes per forward (``>= 2``).
    """

    def __init__(self, *, hidden_dim: int, num_passes: int) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        if int(num_passes) < 2:
            raise ValueError(
                f"num_passes must be >= 2 (1 is a plain forward), got {num_passes}."
            )
        self.hidden_dim = int(hidden_dim)
        self.num_passes = int(num_passes)
        self.norm = nn.RMSNorm(self.hidden_dim, eps=1e-5, elementwise_affine=False)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(
        self, encodings: torch.Tensor, last_hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Next pass input from the encodings and the previous pass's output."""
        if last_hidden_state.shape != encodings.shape:
            raise ValueError(
                "last_hidden_state and encodings must have the same shape, got "
                f"{tuple(last_hidden_state.shape)} and {tuple(encodings.shape)}."
            )
        recycled = self.proj(self.norm(last_hidden_state.to(dtype=encodings.dtype)))
        return encodings + recycled
