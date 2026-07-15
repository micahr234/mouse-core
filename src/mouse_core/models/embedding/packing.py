"""Ragged token packing helpers shared by NumericEmbedder and TextEmbedder."""

from __future__ import annotations

import torch


def counts_from_step_token_indices(indices: torch.Tensor) -> torch.Tensor:
    """Derive per-step token counts ``[B, S]`` from prediction-token indices.

    Assumes steps are packed contiguously (no holes) with at least one token
    each. Works for left- or right-padded flat sequences.
    """
    if indices.ndim != 2:
        raise ValueError(f"step_token_indices must be [B, S], got shape {tuple(indices.shape)}")
    B, S = indices.shape
    if S == 0:
        return indices.new_zeros(B, 0)
    counts = torch.empty_like(indices)
    # Recover start of step 0: for left-pad, steps may not start at 0.
    # counts[0] = indices[0] - start0 + 1, but start0 = indices[0] - (unknown).
    # Contiguous pack: start_{s+1} = indices[s] + 1, so counts[s+1] = indices[s+1] - indices[s].
    # For step 0, start0 = indices[0] - counts[0] + 1. We need another signal for counts[0].
    #
    # With left-padding, content is contiguous at the end: start0 = L - row_len.
    # We require callers to use pack_and_pad_rows which sets indices such that
    # steps are contiguous: indices[s] = start0 + cumsum(counts)[s] - 1.
    # Then counts[0] cannot be recovered from indices alone without knowing start0.
    #
    # Convention used by pack_and_pad_rows (left-pad): store indices relative to
    # the flat sequence; steps are contiguous so
    #   counts[s] = indices[s] - indices[s-1]  (s>0)
    #   counts[0] = indices[0] - first_content_index + 1
    # where first_content_index = indices[0] - (we store via: first step starts at
    # indices[0] - c0 + 1, and c0 is encoded by requiring the content block to be
    # contiguous ending at indices[-1], with no gaps).
    #
    # If steps are contiguous: start[s+1] = indices[s] + 1.
    # start[0] can be inferred only if we know there is no gap before step 0
    # within the content — for left-pad content, start[0] = indices[-1] - total + 1
    # but total = sum(counts) is circular.
    #
    # Practical convention: pack_and_pad_rows writes indices so that
    #   indices[s] - indices[s-1] == counts[s] for s>0, and
    #   indices[0] + 1 == counts[0] + start0, with start0 known as
    #   start0 = indices[-1] - (indices[-1] - indices[0] + counts[0]) + counts[0] ...
    #
    # Simplest robust approach used here: require step 0 to start such that
    # counts[0] = indices[0] - prev_end, where prev_end = indices[0] - counts[0],
    # i.e. we define counts via adjacent differences AFTER inserting a virtual
    # sentinel start-1 = first_index - counts[0].
    #
    # pack_and_pad_rows guarantees contiguous content with known starts:
    #   starts[s] = offset + sum(counts[:s])
    #   indices[s] = starts[s] + counts[s] - 1
    # So counts[s] = indices[s] - starts[s] + 1 = indices[s] - (indices[s-1]+1) + 1
    #              = indices[s] - indices[s-1]  for s>0
    # and counts[0] = indices[0] - offset + 1.  offset is NOT in indices alone
    # unless offset==0 (right-pad / no pad).
    #
    # For Model we need counts for segment expand. With left-pad, offset > 0.
    # Recover offset: content has no internal gaps, so
    #   offset = indices[0] - counts[0] + 1
    # Still circular for counts[0].
    #
    # FIX: also return starts from pack, OR store counts in step_token_indices
    # derivation by assuming the content block is gap-free from the first
    # prediction-index chain:
    #   Let d[s] = indices[s] - indices[s-1] for s>0  (= counts[s])
    #   Let span = indices[-1] - indices[0] + 1  (= sum(counts) - counts[0] + counts[0] = sum(counts)? 
    #   indices[-1] - indices[0] + 1 = sum_{s=1}^{S-1} counts[s] + counts[0] = sum(counts)
    #   Yes! total = indices[-1] - indices[0] + 1 ONLY if start0 == indices[0]-counts[0]+1
    #   and indices[-1] = start0 + total - 1 ⇒ indices[-1] - indices[0] + 1 = total - counts[0] + counts[0] = total.
    #   So total = indices[-1] - indices[0] + 1 works when steps are contiguous!
    #   counts[0] = total - sum(d[1:]) = (indices[-1]-indices[0]+1) - sum(indices[s]-indices[s-1] for s>0)
    #            = indices[-1]-indices[0]+1 - (indices[-1]-indices[0]) = 1.
    #
    # WAIT that forces counts[0]==1 always. That's wrong.
    #
    # indices[-1] - indices[0] + 1 = (start0+total-1) - (start0+counts[0]-1) + 1
    #   = total - counts[0] + 1
    # So NOT equal to total unless counts[0]==1.
    #
    # Correct recovery for contiguous steps:
    #   counts[s] = indices[s] - indices[s-1] for s>0  (since start[s]=indices[s-1]+1)
    #   counts[0] cannot be recovered from indices alone without start0 or total.
    #
    # Therefore pack_and_pad_rows must either return counts, or we change the
    # encoder contract. Plan says derive from indices alone with >=1 token and
    # contiguous from 0 — that implies offset=0 / start0=0, i.e. RIGHT pad or
    # no pad, with step 0 starting at 0: counts[0] = indices[0] + 1.
    #
    # Flex needs trailing real tokens ⇒ left-pad the TENSOR for Flex only in
    # Model.forward when use_cache, without changing encoder indices (encoder
    # right-pads; Model left-aligns a copy for Flex).
    #
    # Encoder pack: RIGHT-pad, start0=0, counts[0]=indices[0]+1. ✓
    # Model use_cache: build left-padded view for Flex from right-padded embeds.
    counts[:, 0] = indices[:, 0] + 1
    if S > 1:
        counts[:, 1:] = indices[:, 1:] - indices[:, :-1]
    if (counts < 1).any():
        raise ValueError(
            "invalid step_token_indices: derived counts must be >= 1 for every step "
            "(steps must be packed contiguously starting at index 0)"
        )
    return counts


def token_pad_mask(indices: torch.Tensor, length: int) -> torch.Tensor:
    """Boolean mask ``[B, L]`` — True for real tokens, False for batch-end pads.

    Assumes right-padding: real tokens occupy ``0 .. indices[b, -1]``.
    """
    if indices.ndim != 2:
        raise ValueError(f"step_token_indices must be [B, S], got shape {tuple(indices.shape)}")
    B, S = indices.shape
    device = indices.device
    if S == 0:
        return torch.zeros(B, length, dtype=torch.bool, device=device)
    last = indices[:, -1]
    t = torch.arange(length, device=device).unsqueeze(0)
    return t <= last.unsqueeze(1)


def expand_segment_ids(segment_ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Expand step-level segment IDs ``[B, S]`` by per-step token counts ``[B, S]``.

    Result is ``[B, L]`` right-padded with ``-1``.
    """
    if segment_ids.shape != counts.shape:
        raise ValueError(
            f"segment_ids shape {tuple(segment_ids.shape)} must match counts "
            f"{tuple(counts.shape)}"
        )
    B, S = segment_ids.shape
    if S == 0:
        return segment_ids.new_zeros(B, 0)

    if (counts < 1).any():
        raise ValueError("every step must contribute at least one token (counts >= 1)")

    row_lengths = counts.sum(dim=1)
    L = int(row_lengths.max().item())
    out = segment_ids.new_full((B, L), -1)
    for b in range(B):
        pieces = [
            segment_ids[b, s].expand(int(counts[b, s].item()))
            for s in range(S)
        ]
        flat = torch.cat(pieces, dim=0)
        out[b, : flat.shape[0]] = flat
    return out


def real_token_lengths_from_indices(
    indices: torch.Tensor,
    *,
    num_real_steps: list[int],
) -> list[int]:
    """Token counts for trailing real steps (cached decode with leading step pads)."""
    B, S = indices.shape
    if len(num_real_steps) != B:
        raise ValueError(
            f"num_real_steps length ({len(num_real_steps)}) must match batch ({B})"
        )
    lengths: list[int] = []
    for b, n in enumerate(num_real_steps):
        if n < 0 or n > S:
            raise ValueError(f"num_real_steps[{b}]={n} out of range for S={S}")
        if n == 0:
            lengths.append(0)
        elif n == S:
            lengths.append(int(indices[b, S - 1].item()) + 1)
        else:
            start = int(indices[b, S - n - 1].item()) + 1
            end = int(indices[b, S - 1].item()) + 1
            lengths.append(end - start)
    return lengths


def left_align_content(
    embeds: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad rows so content sits in trailing columns (FlexDecode layout).

    Encoder output is right-padded. Returns ``(aligned_embeds, pool_indices)``
    where ``pool_indices`` are ``step_token_indices`` shifted into the aligned
    layout for :meth:`Encoder.pool_step_reprs`.
    """
    B, L, _D = embeds.shape
    row_lens = indices[:, -1] + 1
    aligned = embeds.new_zeros(embeds.shape)
    pool_indices = indices.clone()
    for b in range(B):
        rl = int(row_lens[b].item())
        if rl == 0:
            continue
        offset = L - rl
        aligned[b, offset:] = embeds[b, :rl]
        pool_indices[b] = indices[b] + offset
    return aligned, pool_indices


def pack_and_pad_rows(
    row_step_spans: list[list[torch.Tensor]],
    *,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack per-step spans into a right-padded flat sequence starting at index 0.

    Args:
        row_step_spans: ``row_step_spans[b][s]`` is ``[n_{b,s}, D]`` (n >= 1).

    Returns:
        ``(embeds [B, L, D], step_token_indices [B, S])``.
    """
    B = len(row_step_spans)
    if B == 0:
        empty = torch.zeros(0, 0, hidden_dim, device=device, dtype=dtype)
        return empty, torch.zeros(0, 0, device=device, dtype=torch.long)

    S = len(row_step_spans[0])
    for b, steps in enumerate(row_step_spans):
        if len(steps) != S:
            raise ValueError(
                f"all rows must have the same number of steps; row 0 has {S}, "
                f"row {b} has {len(steps)}"
            )

    row_flats: list[torch.Tensor] = []
    indices = torch.empty(B, S, device=device, dtype=torch.long)
    for b in range(B):
        spans = row_step_spans[b]
        for s, span in enumerate(spans):
            if span.ndim != 2 or span.shape[-1] != hidden_dim:
                raise ValueError(
                    f"step span [{b}][{s}] must be [n, {hidden_dim}], got {tuple(span.shape)}"
                )
            if span.shape[0] < 1:
                raise ValueError(f"step [{b}][{s}] must have at least one token")
        flat = torch.cat([sp.to(device=device, dtype=dtype) for sp in spans], dim=0)
        row_flats.append(flat)
        cursor = 0
        for s, span in enumerate(spans):
            cursor += span.shape[0]
            indices[b, s] = cursor - 1

    L = max(f.shape[0] for f in row_flats)
    embeds = torch.zeros(B, L, hidden_dim, device=device, dtype=dtype)
    for b, flat in enumerate(row_flats):
        embeds[b, : flat.shape[0]] = flat
    return embeds, indices
