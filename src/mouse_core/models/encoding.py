"""Decode-cache ownership and tensor layout helpers.

The model implementation calls embedding, backbone, and prediction modules.
This module does not run or own model stages.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mouse_core.data.token_batch import TokenBatch
from mouse_core.models.backbone.flex_decode import FlexDecodeSession, packed_rope_positions


@dataclass
class DecodeCache:
    """KV state carried between ``use_cache=True`` calls.

    Holds one session per backbone stage: :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession`.
    Pass ``out["cache"]`` back as ``cache=``. Call :meth:`close` when the
    rollout is finished so VMM pages unmap before the next train step
    (``__del__`` is too late if a CUDA graph still holds views).
    """

    sessions: tuple[FlexDecodeSession, ...]

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``).

        The rows' next tokens start at position 0 and their KV pages return
        to the shared pool.
        """
        for session in self.sessions:
            session.reset_rows(rows)

    def close(self) -> None:
        """Unmap every session's VMM pool and drop captured CUDA graphs."""
        for session in self.sessions:
            session.close()


def prepare_decode(
    *, embeds: torch.Tensor, tokens: TokenBatch, sequence_ids: torch.Tensor,
    head_token_ids: torch.Tensor, head_step_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Prepare left-padded embeddings and head-token IDs for cached decoding.

    Returns embeddings, token lengths, head-token IDs, and a valid-head mask.
    Each step contributes its last head token. Both tokens and steps occupy
    trailing columns; empty rows have zero lengths and no valid heads.
    Cache creation and backbone execution remain in the agent's forward.
    """
    if tokens.B > 0 and tokens.L == 0:
        raise ValueError("cached decode requires at least one non-empty row in batch.")

    last_head_in_step = torch.ones_like(head_step_ids, dtype=torch.bool)
    last_head_in_step[:-1] = head_step_ids[1:] != head_step_ids[:-1]
    step_head_token_ids = head_token_ids[last_head_in_step]
    step_counts = torch.bincount(sequence_ids[step_head_token_ids], minlength=tokens.B).tolist()
    token_lengths = torch.bincount(sequence_ids, minlength=tokens.B).tolist()
    max_steps = max(step_counts, default=0)
    max_tokens = max(token_lengths, default=0)

    padded_embeds = embeds.new_zeros(tokens.B, max_tokens, embeds.shape[-1])
    padded_head_token_ids = head_token_ids.new_zeros(tokens.B, max_steps)
    token_offset = 0
    step_offset = 0
    for row, (token_count, step_count) in enumerate(zip(token_lengths, step_counts, strict=True)):
        if token_count:
            padding = max_tokens - token_count
            padded_embeds[row, padding:] = embeds[token_offset:token_offset + token_count]
            padded_head_token_ids[row].fill_(padding)
            if step_count:
                row_head_token_ids = step_head_token_ids[step_offset:step_offset + step_count]
                padded_head_token_ids[row, max_steps - step_count:] = row_head_token_ids - token_offset + padding
        token_offset += token_count
        step_offset += step_count

    step_counts_tensor = head_step_ids.new_tensor(step_counts)
    step_slots = torch.arange(max_steps, device=embeds.device)
    head_valid_mask = step_slots.unsqueeze(0) >= (max_steps - step_counts_tensor).unsqueeze(1)
    return padded_embeds, token_lengths, padded_head_token_ids, head_valid_mask


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
