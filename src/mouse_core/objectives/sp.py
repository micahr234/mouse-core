"""Supervised policy objective on action logits at head-output positions."""

from __future__ import annotations

import math
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions, _require_prediction


def _integer_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert integer storage types without silently truncating action labels."""
    if actions.is_floating_point() or actions.is_complex() or actions.dtype == torch.bool:
        raise TypeError("target actions must have an integer dtype.")
    return actions.to(dtype=torch.int64)


def best_action(q_targets: torch.Tensor) -> torch.Tensor:
    """Index of a uniformly random finite maximizer per row.

    Callers that distill from a Q vector (e.g. ``q_star``) run this
    outside ``SpObjective`` and pass the resulting action ids as
    ``targets=``.

    ``-inf`` padding is never selected. Rows with no finite entry fall through
    to ``argmax`` of an all-``-inf`` mask (index 0), matching ``torch.argmax``.
    """
    finite = torch.isfinite(q_targets)
    q = q_targets.masked_fill(~finite, -torch.inf)
    is_max = finite & (q == q.max(dim=-1, keepdim=True).values)
    scores = torch.rand(q.shape, device=q.device, dtype=torch.float32)
    return scores.masked_fill(~is_max, -torch.inf).argmax(dim=-1)


def sp_ce(
    target_actions: torch.Tensor,
    logits: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    invalid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard CE onto integer action labels (aligned rows).

    ``invalid`` is an optional ``[N, A]`` bool mask of action slots that do
    not exist. Those slots are excluded from the student softmax and from
    label smoothing.

    Args:
        target_actions: ``[N]`` integer action ids.
        logits: ``[N, A]`` student action logits.
        label_smoothing: Mixes uniform mass over the *valid* actions into the
            hard label.
        invalid: Optional ``[N, A]`` bool; ``True`` marks a padded action.
    """
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be in [0, 1].")
    target_actions = _integer_actions(target_actions)
    if target_actions.ndim != 1:
        raise ValueError(
            f"sp_ce target_actions must be 1-D [N], got shape {tuple(target_actions.shape)}."
        )
    if logits.ndim != 2:
        raise ValueError(
            f"sp_ce logits must be 2-D [N, A], got shape {tuple(logits.shape)}."
        )
    if target_actions.shape[0] != logits.shape[0]:
        raise ValueError(
            f"sp_ce row count mismatch: target_actions {target_actions.shape[0]} "
            f"vs logits {logits.shape[0]}."
        )
    if invalid is not None:
        if invalid.shape != logits.shape:
            raise ValueError(
                f"sp_ce invalid shape {tuple(invalid.shape)} must match "
                f"logits {tuple(logits.shape)}."
            )
        if invalid.dtype != torch.bool:
            raise TypeError(
                f"sp_ce invalid must be bool, got {invalid.dtype}."
            )
        if bool(invalid.gather(dim=-1, index=target_actions.unsqueeze(-1)).any()):
            raise ValueError("sp_ce target labels must not select an invalid action.")
    if invalid is None:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target_actions.unsqueeze(-1)).squeeze(-1)
        if label_smoothing > 0.0:
            smooth = -log_probs.mean(dim=-1)
            nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
        return nll.mean()
    log_probs = F.log_softmax(logits.masked_fill(invalid, -torch.inf), dim=-1)
    nll = -log_probs.gather(dim=-1, index=target_actions.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0.0:
        valid = (~invalid).to(dtype=log_probs.dtype)
        num_valid = valid.sum(dim=-1).clamp(min=1.0)
        smooth = -log_probs.masked_fill(invalid, 0).sum(dim=-1) / num_valid
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    return nll.mean()


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
    callers choose how to reduce across heads. With ``loss_type="ce"``
    (the default), targets are integer actions matching the leading
    dimensions. In batched mode, ``[P]`` labels are shared by all heads
    and ``[P, K]`` labels are per head. Filter Q* outside with
    ``best_action`` (uniform among tied maxima) and pass those ids, or pass
    dataset / behavior actions. With either KL loss, ``targets`` are oracle
    Q values shaped like the logits. In batched mode, ``[P, A]`` values
    are shared by all heads and ``[P, K, A]`` values are per head.
    Finite Q entries are softmaxed at ``oracle_temperature`` inside the
    loss; nonfinite entries (including
    ``-inf`` padding) are excluded from both oracle and student policies.
    All-zero Q rows produce uniform oracle policies. Callers pass the tensor
    at call time; there is no ``targets_key`` batch lookup.

    Rows where ``mask_key`` is True or any nonzero number are dropped. The
    mask follows the leading dimensions by default. In batched mode,
    ``[P]`` is shared across heads and ``[P, K]`` is per head. Pass
    ``mask_key="episode_ended"`` to skip steps where the wrapped episode
    ended (``episode_terminated`` or ``episode_truncated`` on a
    ``RepeatTaskEnv`` observation, or mouse-gym ``terminated`` /
    ``truncated`` when the wrapped env's episode is the rollout unit).

    Args:
        loss_type: ``"ce"`` for hard-label cross entropy,
            ``"forward_kl"`` for KL(oracle || student), or
            ``"reverse_kl"`` for KL(student || oracle).
        label_smoothing: For CE only, mixes uniform mass over all action
            slots into the hard label.
        oracle_temperature: Positive temperature for softmaxing oracle Q
            values in KL modes. Smaller values emphasize high-Q actions.
        batched_heads: Treat ``[P, K, A]`` as independent heads and return
            one loss per head. Defaults to the original scalar reduction.
        mask_key: Key in ``objective_data`` for a per-row skip mask (bool True
            or any nonzero number). ``None`` disables the skip. Required.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["ce", "forward_kl", "reverse_kl"] = "ce",
        label_smoothing: float = 0.0,
        oracle_temperature: float = 1.0,
        batched_heads: bool = False,
        mask_key: str | None,
    ) -> None:
        if loss_type not in {"ce", "forward_kl", "reverse_kl"}:
            raise ValueError(
                f"Invalid SpObjective loss_type: {loss_type!r} "
                "(expected 'ce', 'forward_kl', or 'reverse_kl')."
            )
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be in [0, 1].")
        if not math.isfinite(oracle_temperature) or oracle_temperature <= 0.0:
            raise ValueError("oracle_temperature must be finite and positive.")
        if loss_type != "ce" and label_smoothing != 0.0:
            raise ValueError("label_smoothing is supported only with loss_type='ce'.")
        self.loss_type: Literal["ce", "forward_kl", "reverse_kl"] = cast(
            Literal["ce", "forward_kl", "reverse_kl"], loss_type
        )
        self.label_smoothing = label_smoothing
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
        if self.loss_type == "ce":
            if not batched:
                leading = logits.shape[:-1]
                if target_tensor.shape != leading and target_tensor.shape != (*leading, 1):
                    raise ValueError(
                        "SpObjective: CE targets must have shape "
                        f"{tuple(leading)} or {(*leading, 1)}, got {tuple(target_tensor.shape)}."
                    )
                target_actions = target_tensor.reshape(P, 1)
            elif target_tensor.shape == (P,):
                target_actions = target_tensor[:, None].expand(P, K)
            elif target_tensor.shape == (P, K):
                target_actions = target_tensor
            elif target_tensor.shape == (P, K, 1):
                target_actions = target_tensor.squeeze(-1)
            else:
                raise ValueError(
                    "SpObjective: CE targets must have shape [P] or [P, K] "
                    f"(optionally with trailing singleton), got {tuple(target_tensor.shape)}."
                )
            target_actions = _integer_actions(target_actions).reshape(-1)[active]
            if bool(((target_actions < 0) | (target_actions >= A)).any()):
                raise ValueError(
                    f"SpObjective: targets action ids must be in "
                    f"[0, {A}), got min={int(target_actions.min())} "
                    f"max={int(target_actions.max())}."
                )
            per_row = F.cross_entropy(
                logits_active,
                target_actions,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
        else:
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
