"""Optional discrete action sampling for application-owned Agent.get_action methods."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from mouse_core.data.token_batch import TokenBatch
from mouse_core.sampling import check_temperature_range


def action_readouts(
    scores: torch.Tensor, inputs: TokenBatch, *, sequence_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Read scores on the current decision step of each requested sequence.

    Each selected input sequence must end at the step requesting an action.
    Match that step against ``head_step_ids``; an older readout cannot stand in
    for a missing current readout. Unselected passive/empty sequences are OK.
    Packed scores are ``[P, A]``; cached scores are left-padded ``[B, S, A]``
    as produced by ``prepare_decode``. Return ``[requested sequences, A]``
    in the requested order, using the last readout if a step has several.
    """
    selected = np.asarray(list(range(inputs.B)) if sequence_ids is None else sequence_ids)
    if selected.ndim != 1 or (selected.size and selected.dtype.kind not in "iu"):
        raise ValueError("sequence_ids must be a sequence of integer IDs")
    selected = selected.astype(np.int64)
    if selected.size and (selected.min() < 0 or selected.max() >= inputs.B):
        raise ValueError("sequence_ids must identify sequences in the input batch")
    counts = inputs.step_counts()
    if np.any(counts[selected] == 0):
        raise ValueError("Action requested for a sequence without a current step")
    current_steps = np.cumsum(counts)[selected] - 1
    # Sorted head_step_ids allow gaps (passive steps) and duplicate readouts.
    positions = np.searchsorted(inputs.head_step_ids, current_steps, side="right") - 1
    if np.any(positions < 0) or np.any(inputs.head_step_ids[positions] != current_steps):
        raise ValueError("Action requested without a readout on the current step; an older readout cannot be used")
    if scores.ndim == 2:
        if scores.shape[0] != inputs.P:
            raise ValueError("Packed scores must have one row per input readout")
        return scores[torch.as_tensor(positions, device=scores.device)]
    if scores.ndim == 3:
        steps_with_heads = np.unique(inputs.head_step_ids)
        assert inputs.step_sequence_ids is not None
        readout_counts = np.bincount(inputs.step_sequence_ids[steps_with_heads], minlength=inputs.B)
        width = int(readout_counts.max()) if inputs.B else 0
        if scores.shape[:2] != (inputs.B, width):
            raise ValueError("Cached scores must match the input batch's left-padded step readouts")
        # The current step is the final readout-bearing step, hence the last
        # column after left padding. The step match above makes this safe.
        rows = torch.as_tensor(selected, device=scores.device)
        return scores[rows, width - 1] if selected.size else scores.new_empty((0, scores.shape[-1]))
    raise ValueError("Action scores must be packed [P, A] or cached [B, S, A]")


def select_discrete_action(
    *, scores: torch.Tensor, temperature: float, valid: torch.Tensor | None = None,
    num_actions: int | None = None,
    inputs: TokenBatch | None = None, sequence_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Select IDs from the final valid readout: greedy at zero, sampled otherwise.

    With ``inputs``, scores are ``[B, S, A]`` for cached decode or ``[P, A]``
    for packed readouts. Without inputs, packed scores must be ``[1, A]``;
    the optional legacy ``valid`` mask marks real cached readouts.
    Pass ``inputs`` to require a readout on the current decision step, with
    optional ``sequence_ids`` selecting only environments requesting actions.
    """
    check_temperature_range(temperature, temperature)
    if inputs is not None:
        if valid is not None:
            raise ValueError("Use inputs for step validation instead of valid")
        scores = action_readouts(scores, inputs, sequence_ids=sequence_ids)
    else:
        if sequence_ids is not None:
            raise ValueError("sequence_ids requires inputs")
        scores = _last_action_scores(scores, name="action", valid=valid)
    if num_actions is not None:
        if type(num_actions) is not int or not 1 <= num_actions <= scores.shape[-1]:
            raise ValueError("num_actions must select at least one available action")
        scores = scores[:, :num_actions]
    if temperature == 0.0:
        return scores.argmax(dim=-1)
    # Match collection's float64 arithmetic: tiny temperatures can underflow
    # in fp32, and centering finite fp32 extremes can otherwise overflow.
    scores = scores.to(torch.float64)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = F.softmax(scores / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _last_valid_step(valid: torch.Tensor) -> torch.Tensor:
    """Last True index along dim=-1. Every row must have a valid head-output."""
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError(
            f"head_valid_mask must be bool [B, S], got {valid.dtype} {tuple(valid.shape)}"
        )
    has = valid.any(dim=-1)
    if not bool(has.all()):
        empty = (~has).nonzero(as_tuple=True)[0]
        raise ValueError(
            "get_action needs a valid head-output token in every row; "
            f"empty rows: {empty.tolist()}."
        )
    idx = valid.long() * torch.arange(valid.shape[1], device=valid.device)
    return idx.max(dim=-1).values


def _last_action_scores(
    raw: torch.Tensor,
    *,
    name: str,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Action scores at the last valid head-output of each row.

    Q / logit heads: ``[B, S, A]`` / ``[N, A]``. ``valid`` is the decode
    ``[B, S]`` mask; omitted, the last step column is used (left-padded
    decode).
    """
    expected_mask_shape = raw.shape[:2] if raw.ndim == 3 else (1, 1)
    if valid is not None and valid.shape != expected_mask_shape:
        raise ValueError(
            f"{name} scores {tuple(raw.shape)} do not match "
            f"head_valid_mask {tuple(valid.shape)}"
        )
    step = None if valid is None else _last_valid_step(valid)
    batch = None if step is None else torch.arange(raw.shape[0], device=raw.device)
    if raw.ndim == 3:
        if step is None:
            return raw[:, -1]
        return raw[batch, step]
    if raw.ndim == 2:
        if raw.shape[0] != 1:
            raise ValueError(
                f"{name} has shape {tuple(raw.shape)}; get_action on flat "
                "[N, A] training outputs needs N=1 (one step). Use "
                "cached-decode [B, S, A] outputs for a batch."
            )
        return raw[-1].unsqueeze(0)
    raise ValueError(f"{name} expects [B, S, A] or [N, A], got {tuple(raw.shape)}")
