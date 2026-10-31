"""Supervised policy and value losses at caller-aligned prediction positions."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction


def best_action(q_targets: torch.Tensor) -> torch.Tensor:
    """Index of a uniformly random finite maximizer per row.

    Use this for greedy action selection with random tie-breaking.
    ``SupervisedObjective`` accepts Q values directly.

    ``-inf`` padding is never selected. Rows with no finite entry fall through
    to ``argmax`` of an all-``-inf`` mask (index 0), matching ``torch.argmax``.
    """
    finite = torch.isfinite(q_targets)
    q = q_targets.masked_fill(~finite, -torch.inf)
    is_max = finite & (q == q.max(dim=-1, keepdim=True).values)
    scores = torch.rand(q.shape, device=q.device, dtype=torch.float32)
    return scores.masked_fill(~is_max, -torch.inf).argmax(dim=-1)


def _policy_kl(
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
            "SupervisedObjective: each unmasked KL target row must contain a finite Q value."
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
            f"SupervisedObjective: skip mask must have shape [{rows}] or [{rows}, {heads}]."
        )
    return values != 0


class SupervisedObjective(Objective[...]):
    """Supervised policy or action-value learning with a common mask/reduction.

    ``forward_kl`` and ``reverse_kl`` read floating-point oracle Q-vector
    ``targets``. Finite Q entries define the available actions. Positive
    ``oracle_temperature`` softmaxes Q; zero gives a uniform distribution over
    finite maxima and is allowed only for forward KL.

    ``mse`` and ``mae`` regress raw predictions onto Q-vector ``targets``.
    Nonfinite target entries are ignored. Without explicit row weights, the
    loss averages all finite entries. With row weights, each row first averages
    its finite entries, then participates in the weighted row mean.

    ``ce`` requires integer ``actions`` identifying the actual target action at
    each prediction position. It never derives labels from Q values or accepts
    probability targets. The caller aligns actions to predictions; there is no
    implicit timestep shift. Logits for unavailable actions may be ``-inf``.

    Predictions have shape ``[..., A]`` and return a scalar by default.
    ``batched_heads=True`` uses ``[P, K, A]`` and returns independent ``[K]``
    losses. Q targets may be shared ``[P, A]`` or per head ``[P, K, A]``;
    actions may be shared ``[P]`` or per head ``[P, K]``. Nonbatched targets
    match predictions, and actions match their leading dimensions.

    ``mask_key`` optionally names a nonzero-means-skip row mask in
    ``objective_data``. ``weight_key`` optionally names nonnegative finite row
    weights. Both accept shared ``[P]`` or per-head ``[P, K]`` tensors in
    batched mode. Masked rows are ignored before inspecting their values.
    Each head requires positive effective weight. Targets and weights are
    detached. Losses and detached metrics use the key ``supervised``.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["forward_kl", "reverse_kl", "mse", "mae", "ce"] = "forward_kl",
        oracle_temperature: float = 0.0,
        batched_heads: bool = False,
        mask_key: str | None = None,
        weight_key: str | None = None,
    ) -> None:
        if loss_type not in {"forward_kl", "reverse_kl", "mse", "mae", "ce"}:
            raise ValueError(f"Invalid SupervisedObjective loss_type: {loss_type!r}.")
        if not math.isfinite(oracle_temperature) or oracle_temperature < 0:
            raise ValueError("oracle_temperature must be finite and nonnegative.")
        if loss_type == "reverse_kl" and oracle_temperature == 0:
            raise ValueError("reverse_kl requires oracle_temperature > 0.")
        self.loss_type = loss_type
        self.oracle_temperature = oracle_temperature
        self.batched_heads = batched_heads
        self.mask_key = mask_key
        self.weight_key = weight_key

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        backup_value_predictions: torch.Tensor | None = None,
        backup_action_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "SupervisedObjective",
            backup_value_predictions=backup_value_predictions,
            backup_action_predictions=backup_action_predictions,
            value_predictions=value_predictions,
        )
        if self.loss_type == "ce":
            _reject_predictions("SupervisedObjective", targets=targets)
            labels = _require_prediction(actions, owner="SupervisedObjective", name="actions")
        else:
            _reject_predictions("SupervisedObjective", actions=actions)
            labels = _require_prediction(targets, owner="SupervisedObjective", name="targets")
        batched = self.batched_heads
        if batched and predictions.ndim != 3:
            raise ValueError("SupervisedObjective: batched heads require predictions [P, K, A].")
        if predictions.ndim < 1:
            raise ValueError("SupervisedObjective: predictions must have shape [..., A].")
        A = predictions.shape[-1]
        if predictions.numel() == 0 or A == 0:
            raise ValueError("SupervisedObjective: batch is empty (no tokens or actions).")
        if batched:
            P, K, _ = predictions.shape
        else:
            P, K = predictions.numel() // A, 1
        prediction_heads = predictions.reshape(P, K, A)
        valid_rows = torch.ones((P, K), dtype=torch.bool, device=predictions.device)
        if self.mask_key is not None:
            valid_rows = ~_skip_mask(objective_data[self.mask_key], rows=P, heads=K, batched=batched)

        counts = torch.ones((P, K), dtype=torch.float32, device=predictions.device)
        q_targets: torch.Tensor | None = None
        action_targets: torch.Tensor | None = None
        if self.loss_type == "ce":
            if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError("SupervisedObjective: CE actions must contain integer action IDs.")
            if labels.shape == predictions.shape[:-1]:
                action_targets = labels.detach().reshape(P, K)
            elif batched and labels.shape == (P,):
                action_targets = labels.detach()[:, None].expand(P, K)
            else:
                raise ValueError("SupervisedObjective: actions must match prediction rows or shared shape [P].")
        else:
            if labels.shape == predictions.shape:
                q_targets = labels.detach().reshape(P, K, A)
            elif batched and labels.shape == (P, A):
                q_targets = labels.detach()[:, None, :].expand(P, K, A)
            else:
                raise ValueError("SupervisedObjective: Q targets must have the same shape as predictions or shared shape [P, A].")
            if self.loss_type in ("forward_kl", "reverse_kl") and not labels.is_floating_point():
                raise TypeError("SupervisedObjective: targets must contain floating-point Q values.")
            if self.loss_type in ("mse", "mae"):
                counts = torch.isfinite(q_targets).sum(-1)
                valid_rows = valid_rows & (counts > 0)

        if not bool(valid_rows.any(dim=0).all()):
            raise ValueError("SupervisedObjective: no finite target rows left after the skip mask for at least one head.")
        active = valid_rows.reshape(-1)
        predicted = prediction_heads.reshape(-1, A)[active]
        dtype = torch.promote_types(torch.float32, predictions.dtype)
        if self.loss_type == "ce":
            assert action_targets is not None
            chosen = action_targets.reshape(-1)[active].long()
            if bool(((chosen < 0) | (chosen >= A)).any()):
                raise ValueError(f"SupervisedObjective: CE action IDs must be in [0, {A}).")
            per_row = F.cross_entropy(predicted.to(dtype), chosen, reduction="none")
        else:
            assert q_targets is not None
            target_rows = q_targets.reshape(-1, A)[active]
            if self.loss_type in ("forward_kl", "reverse_kl"):
                per_row = _policy_kl(
                    target_rows, predicted, loss_type=self.loss_type,
                    oracle_temperature=self.oracle_temperature,
                )
            else:
                dtype = torch.promote_types(dtype, target_rows.dtype)
                finite = torch.isfinite(target_rows)
                difference = (torch.where(finite, predicted.to(dtype), 0)
                              - torch.where(finite, target_rows.to(dtype), 0))
                errors = difference.square() if self.loss_type == "mse" else difference.abs()
                per_row = errors.sum(-1) / counts.reshape(-1)[active]

        if self.weight_key is None:
            # Regression's unweighted mean counts individual finite entries.
            weights = counts.reshape(-1)[active].to(per_row.dtype)
        else:
            raw = objective_data[self.weight_key].detach().to(device=predictions.device, dtype=per_row.dtype)
            if not batched and raw.numel() == P:
                raw = raw.reshape(P, 1)
            elif batched and raw.shape == (P,):
                raw = raw[:, None].expand(P, K)
            elif raw.shape != (P, K):
                raise ValueError("SupervisedObjective: weights must match target rows or batched heads.")
            weights = raw.reshape(-1)[active]
            if not bool((torch.isfinite(weights) & (weights >= 0)).all()):
                raise ValueError("SupervisedObjective: active weights must be finite and nonnegative.")
        head_index = torch.arange(K, device=predictions.device).expand(P, K).reshape(-1)[active]
        totals = per_row.new_zeros(K).scatter_add(0, head_index, weights)
        if not bool((totals > 0).all()):
            raise ValueError("SupervisedObjective: each head requires positive total weight.")
        loss_by_head = per_row.new_zeros(K).scatter_add(0, head_index, weights * per_row) / totals
        loss = loss_by_head if batched else loss_by_head.squeeze(0)
        return {"supervised": loss}, {"supervised": loss.detach()}

