"""Decode-cache ownership and tensor layout helpers.

The model implementation calls embedding, backbone, and prediction modules.
This module does not run or own model stages.
"""
from __future__ import annotations
from tensordict import TensorDict

from collections.abc import Sequence
from dataclasses import dataclass

import torch


from mouse_core.models.backbone.continuous_decode import ContinuousDecodeSession


@dataclass
class DecodeCache:
    """KV state carried between ``use_cache=True`` calls.

    Holds one Hugging Face continuous decode session per backbone stage.
    Pass ``out["cache"]`` back as ``cache=``. Call :meth:`close` when the
    rollout is finished to release the HF pool before the next train step.
    """

    sessions: tuple[ContinuousDecodeSession, ...]

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``).

        The rows' next tokens start at position 0 and their KV pages return
        to the shared pool.
        """
        for session in self.sessions:
            session.reset_rows(rows)

    def close(self) -> None:
        """Release every session's KV pool and execution resources."""
        for session in self.sessions:
            session.close()


def prepare_decode(
    *, embeds: torch.Tensor, tokens: TensorDict, sequence_ids: torch.Tensor,
    head_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor]:
    """Prepare left-padded embeddings and head-token IDs for cached decoding.

    Returns embeddings, token lengths, head-token IDs, and a valid-head mask.
    Every marked token contributes a readout. Both tokens and readouts occupy
    trailing columns; empty rows have zero lengths and no valid heads.
    Cache creation and backbone execution remain in the agent's forward.
    """
    if tokens["num_sequences"] > 0 and tokens["token_ids"].shape[0] == 0:
        raise ValueError("cached decode requires at least one non-empty row in batch.")

    head_token_ids = head_mask.nonzero(as_tuple=True)[0]
    head_counts = torch.bincount(sequence_ids[head_token_ids], minlength=tokens["num_sequences"]).tolist()
    token_lengths = torch.bincount(sequence_ids, minlength=tokens["num_sequences"]).tolist()
    max_heads = max(head_counts, default=0)
    max_tokens = max(token_lengths, default=0)

    padded_embeds = embeds.new_zeros(tokens["num_sequences"], max_tokens, embeds.shape[-1])
    padded_head_token_ids = head_token_ids.new_zeros(tokens["num_sequences"], max_heads)
    token_offset = 0
    head_offset = 0
    for row, (token_count, head_count) in enumerate(zip(token_lengths, head_counts, strict=True)):
        if token_count:
            padding = max_tokens - token_count
            padded_embeds[row, padding:] = embeds[token_offset:token_offset + token_count]
            padded_head_token_ids[row].fill_(padding)
            if head_count:
                row_head_token_ids = head_token_ids[head_offset:head_offset + head_count]
                padded_head_token_ids[row, max_heads - head_count:] = row_head_token_ids - token_offset + padding
        token_offset += token_count
        head_offset += head_count

    head_counts_tensor = head_token_ids.new_tensor(head_counts)
    head_slots = torch.arange(max_heads, device=embeds.device)
    head_valid_mask = head_slots.unsqueeze(0) >= (max_heads - head_counts_tensor).unsqueeze(1)
    return padded_embeds, token_lengths, padded_head_token_ids, head_valid_mask
