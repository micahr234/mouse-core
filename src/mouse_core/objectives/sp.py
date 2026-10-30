"""Supervised policy objective on action logits at head-output positions."""

from __future__ import annotations

import math
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction


def best_action(q_targets: torch.Tensor) -> torch.Tensor:
    """Index of a uniformly random finite maximizer per row.

    Use this for greedy action selection with random tie-breaking.
    ``SpObjective`` accepts Q values directly.

    ``-inf`` padding is never selected. Rows with no finite entry fall through
    to ``argmax`` of an all-``-inf`` mask (index 0), matching ``torch.argmax``.
    """
    finite = torch.isfinite(q_targets)
    q = q_targets.masked_fill(~finite, -torch.inf)
    is_max = finite & (q == q.max(dim=-1, keepdim=True).values)
    scores = torch.rand(q.shape, device=q.device, dtype=torch.float32)
    return scores.masked_fill(~is_max, -torch.inf).argmax(dim=-1)


def _sp_kl(
    q_targets: torch.Tensor,
    logits: torch.Tensor,
    *,
    loss_type: Literal["forward_kl", "reverse_kl"],
    oracle_temperature: float,
) -> torch.Tensor:
    """Per-row KL between softmaxed finite oracle Q values and student logits."""
    compute_dtype = torch.promote_types(
        torch.float32, torch.promote_types(q_targets.dtype, logits.dtype)
    )
    q = q_targets.detach().to(dtype=compute_dtype)
    valid = torch.isfinite(q)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError(
            "SpObjective: each unmasked KL target row must contain a finite Q value."
        )

    # A nonfinite Q value denotes an unavailable action, not probability mass.
    # Mask before scaling so +inf and NaN do not enter the softmax.
    max_q = q.masked_fill(~valid, -torch.inf).max(dim=-1, keepdim=True).values
    if oracle_temperature == 0.0:
        # The T -> 0 limit is uniform over finite maximizers. Construct it
        # directly so neither 0/0 nor gradients through a division occur.
        maxima = valid & (q == max_q)
        oracle_logits = torch.zeros_like(q).masked_fill(~maxima, -torch.inf)
    else:
        oracle_logits = ((q - max_q) / oracle_temperature).masked_fill(~valid, -torch.inf)
    student_logits = logits.to(dtype=compute_dtype).masked_fill(~valid, -torch.inf)
    oracle_log_probs = F.log_softmax(oracle_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    if loss_type == "forward_kl":
        weights = oracle_log_probs.exp()
        active = valid & (weights > 0)
        # Zero-probability entries must not form 0 * (+/-inf).
        oracle_terms = oracle_log_probs.masked_fill(~active, 0)
        student_terms = student_log_probs.masked_fill(~active, 0)
        per_row = (weights.masked_fill(~active, 0) * (oracle_terms - student_terms)).sum(-1)
    else:
        weights = student_log_probs.exp()
        active = valid & (weights > 0)
        student_terms = student_log_probs.masked_fill(~active, 0)
        oracle_terms = oracle_log_probs.masked_fill(~active, 0)
        per_row = (weights.masked_fill(~active, 0) * (student_terms - oracle_terms)).sum(-1)
    return per_row


def _skip_mask(mask: torch.Tensor, *, rows: int, heads: int, batched: bool) -> torch.Tensor:
    """Return a [P, K] skip mask, allowing one mask shared by all heads."""
    if not batched:
        values = mask.reshape(-1)
        if values.shape[0] != rows:
            raise ValueError(
                f"mask length ({values.shape[0]}) must match flattened target rows ({rows})."
            )
        return (values != 0)[:, None]
    if mask.shape == (rows,):
        values = mask[:, None].expand(rows, heads)
    elif mask.shape == (rows, heads):
        values = mask
    else:
        raise ValueError(
            f"SpObjective: skip mask must have shape [{rows}] or [{rows}, {heads}]."
        )
    return values != 0


class SpObjective(Objective[...]):
    """Supervised policy loss on action logits at head-output positions.

    By default, reads action logits ``[..., A]`` and returns a scalar.
    With ``batched_heads=True``, reads independent heads ``[P, K, A]``
    and returns one loss and metric per head, shaped ``[K]``;
    callers choose how to reduce across heads. Targets are floating-point
    oracle Q values shaped like the logits. In batched mode, ``[P, A]``
    values are shared by all heads and ``[P, K, A]`` values are per head.
    Nonfinite Q entries are unavailable actions, excluded from both policies.

    Positive oracle temperatures produce ``softmax(Q / temperature)``.
    At zero, the oracle is uniform over finite Q maxima. Forward KL then
    equals NLL for a unique best action. With k tied maxima, it equals
    the mean NLL over those actions minus log(k), with the same gradients
    as randomly tie-broken NLL in expectation. All-equal Q rows are uniform.
    Reverse KL requires positive temperature because a hard oracle target
    gives infinite divergence for student mass on nonoptimal actions.

    Rows where ``mask_key`` is True or any nonzero number are dropped. The
    mask follows the leading dimensions by default. In batched mode,
    ``[P]`` is shared across heads and ``[P, K]`` is per head. Pass
    ``mask_key="episode_ended"`` to skip steps where the wrapped episode
    ended (``episode_terminated`` or ``episode_truncated`` on a
    ``RepeatTaskEnv`` observation, or mouse-gym ``terminated`` /
    ``truncated`` when the wrapped env's episode is the rollout unit).

    Args:
        loss_type: ``"forward_kl"`` for KL(oracle || student), or
            ``"reverse_kl"`` for KL(student || oracle).
        oracle_temperature: Finite, nonnegative oracle temperature; zero
            is supported only for forward KL. Defaults to zero.
        batched_heads: Treat ``[P, K, A]`` as independent heads and return
            one loss per head. Defaults to the original scalar reduction.
        mask_key: Key in ``objective_data`` for a per-row skip mask (bool True
            or any nonzero number). ``None`` disables the skip. Required.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["forward_kl", "reverse_kl"] = "forward_kl",
        oracle_temperature: float = 0.0,
        batched_heads: bool = False,
        mask_key: str | None,
    ) -> None:
        if loss_type not in {"forward_kl", "reverse_kl"}:
            raise ValueError(
                f"Invalid SpObjective loss_type: {loss_type!r} "
                "(expected 'forward_kl' or 'reverse_kl')."
            )
        if not math.isfinite(oracle_temperature) or oracle_temperature < 0.0:
            raise ValueError("oracle_temperature must be finite and nonnegative.")
        if loss_type == "reverse_kl" and oracle_temperature == 0.0:
            raise ValueError("reverse_kl requires oracle_temperature > 0.")
        self.loss_type: Literal["forward_kl", "reverse_kl"] = cast(
            Literal["forward_kl", "reverse_kl"], loss_type
        )
        self.oracle_temperature = oracle_temperature
        self.batched_heads = batched_heads
        self.mask_key = mask_key

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        backup_value_predictions: None = None,
        backup_action_predictions: None = None,
        value_predictions: None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        backup_value_predictions: torch.Tensor | None = None,
        backup_action_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "SpObjective",
            backup_value_predictions=backup_value_predictions,
            backup_action_predictions=backup_action_predictions,
            value_predictions=value_predictions,
        )
        target_tensor = _require_prediction(
            targets, owner="SpObjective", name="targets"
        )
        logits: torch.Tensor = predictions
        batched = self.batched_heads
        if batched and logits.ndim != 3:
            raise ValueError("SpObjective: batched heads require predictions [P, K, A].")
        if logits.ndim < 1:
            raise ValueError("SpObjective: predictions must have shape [..., A].")
        A = logits.shape[-1]
        if logits.numel() == 0 or A == 0:
            raise ValueError("SpObjective: batch is empty (no tokens).")
        if batched:
            logits_heads = logits
            P, K, _ = logits_heads.shape
        else:
            P, K = logits.numel() // A, 1
            logits_heads = logits.reshape(P, K, A)

        if self.mask_key is not None:
            valid_rows = ~_skip_mask(
                objective_data[self.mask_key], rows=P, heads=K, batched=batched
            )
        else:
            valid_rows = torch.ones((P, K), dtype=torch.bool, device=logits.device)

        if not bool(valid_rows.any(dim=0).all()):
            raise ValueError(
                "SpObjective: no rows left after applying the skip mask for at least one head."
            )
        active = valid_rows.reshape(-1)
        logits_active = logits_heads.reshape(-1, A)[active]
        if not batched and target_tensor.shape == logits.shape:
            q_targets = target_tensor.reshape(P, 1, A)
        elif target_tensor.shape == (P, A) and batched:
            q_targets = target_tensor[:, None, :].expand(P, K, A)
        elif target_tensor.shape == logits.shape and batched:
            q_targets = target_tensor
        else:
            raise ValueError(
                "SpObjective: KL targets must have the same shape as predictions "
                "or shared shape [P, A], "
                f"got {tuple(target_tensor.shape)}."
            )
        if not target_tensor.is_floating_point():
            raise TypeError("SpObjective: KL targets must contain floating-point Q values.")
        q_active = q_targets.reshape(-1, A)[active]
        per_row = _sp_kl(
            q_targets=q_active,
            logits=logits_active,
            loss_type=self.loss_type,
            oracle_temperature=self.oracle_temperature,
        )
        head_index = torch.arange(K, device=logits.device).expand(P, K).reshape(-1)[active]
        loss_by_head = per_row.new_zeros(K).scatter_add(0, head_index, per_row)
        loss_by_head = loss_by_head / valid_rows.sum(dim=0).to(per_row.dtype)
        loss = loss_by_head if batched else loss_by_head.squeeze(0)
        metrics: dict[str, float | torch.Tensor] = {"action": loss.detach()}
        return {"action": loss}, metrics
