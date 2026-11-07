"""Full-attention test oracle; not a Mouse runtime backend."""

import torch
from mouse_core.models.backbone.flex_utils import packed_rope_positions


def _flat_sequence_causal_mask(
    *,
    sequence_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive attention mask ``[1, 1, L, L]`` for a packed flat sequence."""
    L = sequence_ids.shape[0]
    device = sequence_ids.device
    q = torch.arange(L, device=device)
    kv = torch.arange(L, device=device)
    causal = kv.unsqueeze(0) <= q.unsqueeze(1)
    same_seq = sequence_ids.unsqueeze(1) == sequence_ids.unsqueeze(0)
    allow = causal & same_seq
    neg = torch.finfo(dtype).min
    mask = torch.where(
        allow,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), neg, device=device, dtype=dtype),
    )
    mask = mask.view(1, 1, L, L)
    return mask


def _flat_sequence_position_ids(*, sequence_ids: torch.Tensor) -> torch.Tensor:
    """RoPE positions ``[1, L]``: count of earlier same-sequence tokens.

    Same rule as the packed train forward and cached decode.
    """
    position_ids = packed_rope_positions(sequence_ids=sequence_ids).unsqueeze(0)
    return position_ids
