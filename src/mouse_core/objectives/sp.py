"""Supervised policy objective over action predictions at head-output positions."""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective, _reject_predictions


def best_action(q_targets: torch.Tensor) -> torch.Tensor:
    """Index of a uniformly random finite maximizer per row.

    Callers that distill hard CE from a Q vector (e.g. ``info_q_star``) run
    this outside ``SpObjective`` and pass the resulting action ids as
    ``targets_key``. Soft SP losses still take the full Q vector.

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
    not exist (same role as ``-inf`` padding in soft SP losses). Those slots
    are excluded from the student softmax and from label smoothing.

    Args:
        target_actions: ``[N]`` integer action ids.
        logits: ``[N, A]`` student action logits.
        label_smoothing: Mixes uniform mass over the *valid* actions into the
            hard label.
        invalid: Optional ``[N, A]`` bool; ``True`` marks a padded action.
    """
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
    fill = torch.finfo(logits.dtype).min / 4
    if invalid is None:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target_actions.unsqueeze(-1)).squeeze(-1)
        if label_smoothing > 0.0:
            smooth = -log_probs.mean(dim=-1)
            nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
        return nll.mean()
    log_probs = F.log_softmax(logits.masked_fill(invalid, fill), dim=-1)
    nll = -log_probs.gather(dim=-1, index=target_actions.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0.0:
        valid = (~invalid).to(dtype=log_probs.dtype)
        num_valid = valid.sum(dim=-1).clamp(min=1.0)
        smooth = -(log_probs * valid).sum(dim=-1) / num_valid
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    return nll.mean()


def _soft_distributions(
    q_targets: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher and student log-probs renormalized over valid actions only.

    ``-inf`` entries in ``q_targets`` are padding sentinels for actions that do
    not exist. Both distributions assign them exactly zero probability (label
    smoothing included): otherwise backward-direction losses, where the student
    puts mass on actions with zero teacher mass, are infinite.

    Padded positions are masked with a very negative *finite* value rather than
    ``-inf`` so that ``exp`` underflows to exactly 0 while every intermediate
    tensor and jacobian stays finite — masking with ``-inf`` makes the loss
    finite but its gradient NaN.
    """
    invalid = ~torch.isfinite(q_targets)
    fill = torch.finfo(logits.dtype).min / 4
    log_teacher = F.log_softmax((q_targets / temperature).masked_fill(invalid, fill), dim=-1)
    log_student = F.log_softmax(logits.masked_fill(invalid, fill), dim=-1)
    if label_smoothing > 0.0:
        valid = (~invalid).to(log_teacher.dtype)
        num_valid = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        smoothed = (1.0 - label_smoothing) * log_teacher.exp() + label_smoothing * valid / num_valid
        # Fill with 1.0 before the log (not after) so no -inf is ever created.
        log_teacher = smoothed.masked_fill(invalid, 1.0).log().masked_fill(invalid, fill)
    return log_teacher, log_student


def sp_js(
    q_targets: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Jensen–Shannon loss between teacher ``q_targets`` and student ``logits`` (aligned rows).

    Builds temperature-scaled soft distributions, optional label smoothing on the teacher only,
    then ``JS = 0.5 KL(P‖M) + 0.5 KL(Q‖M)`` with ``M = 0.5 (P + Q)``, mean over rows, × T².

    Args:
        q_targets: ``[N, A]`` teacher Q-values (e.g. ``q_star`` at PREDICTION rows).
        logits: ``[N, A]`` student action logits at the same rows.
        temperature: Must be ``> 0``; applied to both teacher and student logits.
        label_smoothing: Mixes uniform mass into the teacher distribution (probability space).
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"sp_js temperature must be > 0, got {temp}.")
    log_teacher, log_student = _soft_distributions(q_targets, logits, temp, label_smoothing)

    log_m = torch.logaddexp(log_teacher, log_student) - math.log(2.0)
    # KL(P‖M) and KL(Q‖M) via kl_div(input=log M, target=log P, log_target=True)
    # -> exp(log P) * (log P - log M). nan_to_num is defensive: padded actions are
    # already finite-masked in _soft_distributions, but rows that are entirely
    # padded (possible when calling this helper directly) would still yield NaN.
    kl_pm = torch.nan_to_num(
        F.kl_div(log_m, log_teacher, log_target=True, reduction="none"),
        nan=0.0,
    ).sum(dim=-1)
    kl_qm = torch.nan_to_num(
        F.kl_div(log_m, log_student, log_target=True, reduction="none"),
        nan=0.0,
    ).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js.mean() * (temp * temp)


def sp_kl(
    q_targets: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
    label_smoothing: float = 0.0,
    direction: str = "fwd",
) -> torch.Tensor:
    """Temperature-scaled KL loss between teacher ``q_targets`` and student ``logits``.

    Args:
        q_targets: ``[N, A]`` teacher Q-values.
        logits: ``[N, A]`` student logits.
        temperature: Must be ``> 0``.
        label_smoothing: Optional smoothing applied to teacher distribution only.
        direction: ``"fwd"`` computes ``KL(P_teacher || Q_student)``;
            ``"bwd"`` computes ``KL(Q_student || P_teacher)``.
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"sp_kl temperature must be > 0, got {temp}.")
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"sp_kl direction must be 'fwd' or 'bwd', got {direction!r}.")
    log_teacher, log_student = _soft_distributions(q_targets, logits, temp, label_smoothing)

    if direction == "fwd":
        kl = torch.nan_to_num(
            F.kl_div(log_student, log_teacher, log_target=True, reduction="none"),
            nan=0.0,
        ).sum(dim=-1)
    else:
        kl = torch.nan_to_num(
            F.kl_div(log_teacher, log_student, log_target=True, reduction="none"),
            nan=0.0,
        ).sum(dim=-1)
    return kl.mean()


def sp_soft_ce(
    q_targets: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
    label_smoothing: float = 0.0,
    direction: str = "fwd",
) -> torch.Tensor:
    """Directional soft cross-entropy between teacher ``q_targets`` and student ``logits``.

    Teacher targets are ``softmax(q_targets / temperature)``. Optional label
    smoothing is applied on the teacher distribution only.

    - ``direction="fwd"`` computes ``H(P_teacher, Q_student) = -sum P log Q``.
    - ``direction="bwd"`` computes ``H(Q_student, P_teacher) = -sum Q log P``.
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"sp_soft_ce temperature must be > 0, got {temp}.")
    if direction not in ("fwd", "bwd"):
        raise ValueError(f"sp_soft_ce direction must be 'fwd' or 'bwd', got {direction!r}.")
    log_teacher, log_student = _soft_distributions(q_targets, logits, temp, label_smoothing)
    if direction == "fwd":
        teacher = log_teacher.exp()
        per_row = torch.nan_to_num(-(teacher * log_student), nan=0.0).sum(dim=-1)
    else:
        student = log_student.exp()
        per_row = torch.nan_to_num(-(student * log_teacher), nan=0.0).sum(dim=-1)
    return per_row.mean()


def _skip_mask(mask: torch.Tensor, n_rows: int) -> torch.Tensor:
    """True where the mask is nonzero (bool True counts as skip)."""
    values = mask.reshape(-1)
    if values.shape[0] != n_rows:
        raise ValueError(
            f"mask length ({values.shape[0]}) must match flattened target rows ({n_rows})."
        )
    if values.dtype == torch.bool:
        return values
    return values != 0


class SpObjective(Objective):
    """Supervised policy objective over action logits at head-output positions.

    Reads ``predictions`` (shape ``[B, S, A]``) and compares against
    ``objective_data[targets_key]``.

    Hard ``"ce"`` expects **integer action ids** shaped like ``predictions``
    without the action axis (``[B, S]``). Filter Q* outside with
    ``best_action`` (uniform among tied maxima) and pass those ids, or pass
    dataset / behavior actions for a data-policy head. Soft losses still take
    a full ``[B, S, A]`` Q teacher (e.g. ``info_q_star``).

    Rows where ``mask_key`` is True or any nonzero number are dropped. Pass
    ``mask_key="episode_done"`` to skip both terminated and truncated steps
    (the episode is over; there is no next action to imitate). Soft losses
    also drop rows with no finite teacher entry.

    Args:
        loss_type: Which distillation loss to apply.  ``"ce"`` uses hard
            labels from ``targets_key``; the soft variants treat a Q vector as
            a distribution.
        temperature: Softmax temperature applied to targets before soft losses
            (ignored for ``"ce"``).
        label_smoothing: Mixes uniform mass over the valid (finite-target)
            actions into the teacher. On hard ``"ce"`` it smooths the hard
            label over all action slots; on soft losses it is applied to the
            teacher distribution only (excluding ``-inf`` padding).
        targets_key: Key in ``objective_data``. For ``"ce"``, integer actions
            ``[B, S]``; for soft losses, Q targets ``[B, S, A]``.
        mask_key: Key in ``objective_data`` for a per-row skip mask (bool True
            or any nonzero number). ``None`` disables the skip (e.g. teacher-logit
            distillation with no mask column).
    """

    def __init__(
        self,
        *,
        loss_type: Literal["ce", "ce-soft-fwd", "ce-soft-bwd", "js", "kl-fwd", "kl-bwd"] = "ce",
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        targets_key: str = "info_q_star",
        mask_key: str | None = "episode_done",
    ) -> None:
        self.loss_type = loss_type
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.targets_key = targets_key
        self.mask_key = mask_key

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]: ...

    @overload
    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        delayed_predictions: None = None,
        value_predictions: None = None,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]: ...

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        delayed_predictions: torch.Tensor | None = None,
        value_predictions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        _reject_predictions(
            "SpObjective",
            delayed_predictions=delayed_predictions,
            value_predictions=value_predictions,
        )
        logits: torch.Tensor = predictions
        temp = float(self.temperature)

        A = logits.shape[-1]
        leading = logits.shape[:-1]
        logits_flat = logits.reshape(-1, A)
        n_rows = logits_flat.shape[0]

        if n_rows == 0:
            raise ValueError("SpObjective: batch is empty (no tokens).")

        if self.mask_key is not None:
            valid_rows = ~_skip_mask(objective_data[self.mask_key], n_rows)
        else:
            valid_rows = torch.ones(n_rows, dtype=torch.bool, device=logits.device)

        if self.loss_type == "ce":
            targets = objective_data[self.targets_key]
            if targets.shape != leading and targets.shape != (*leading, 1):
                raise ValueError(
                    f"SpObjective: {self.targets_key!r} for loss_type='ce' must have "
                    f"shape {tuple(leading)} or {(*leading, 1)}, got {tuple(targets.shape)}."
                )
            target_actions = targets.reshape(-1).to(dtype=torch.long)
            if not valid_rows.any():
                raise ValueError(
                    "SpObjective: no rows left after applying the skip mask."
                )
            logits_flat = logits_flat[valid_rows]
            target_actions = target_actions[valid_rows]
            if (target_actions < 0).any() or (target_actions >= A).any():
                raise ValueError(
                    f"SpObjective: {self.targets_key!r} action ids must be in "
                    f"[0, {A}), got min={int(target_actions.min())} "
                    f"max={int(target_actions.max())}."
                )
            loss = sp_ce(
                target_actions=target_actions,
                logits=logits_flat,
                label_smoothing=self.label_smoothing,
            )
        else:
            q_targets = objective_data[self.targets_key].reshape(-1, A).to(dtype=logits.dtype)
            if torch.isnan(q_targets).any():
                raise ValueError(f"SpObjective: {self.targets_key!r} contains NaN values.")
            if torch.isposinf(q_targets).any():
                raise ValueError(f"SpObjective: {self.targets_key!r} contains +inf values.")
            valid_rows = valid_rows & torch.isfinite(q_targets).any(dim=-1)
            if not valid_rows.any():
                raise ValueError(
                    "SpObjective: no rows left after applying the skip mask "
                    f"and dropping non-finite {self.targets_key!r} targets."
                )
            logits_flat = logits_flat[valid_rows]
            q_targets = q_targets[valid_rows]
            if self.loss_type == "ce-soft-fwd":
                loss = sp_soft_ce(q_targets=q_targets, logits=logits_flat, temperature=temp, label_smoothing=self.label_smoothing, direction="fwd")
            elif self.loss_type == "ce-soft-bwd":
                loss = sp_soft_ce(q_targets=q_targets, logits=logits_flat, temperature=temp, label_smoothing=self.label_smoothing, direction="bwd")
            elif self.loss_type == "js":
                loss = sp_js(q_targets=q_targets, logits=logits_flat, temperature=temp, label_smoothing=self.label_smoothing)
            elif self.loss_type == "kl-fwd":
                loss = sp_kl(q_targets=q_targets, logits=logits_flat, temperature=temp, label_smoothing=self.label_smoothing, direction="fwd")
            elif self.loss_type == "kl-bwd":
                loss = sp_kl(q_targets=q_targets, logits=logits_flat, temperature=temp, label_smoothing=self.label_smoothing, direction="bwd")
            else:
                raise ValueError(
                    f"Invalid SpObjective loss_type: {self.loss_type!r} "
                    "(expected 'ce', 'ce-soft-fwd', 'ce-soft-bwd', 'js', 'kl-fwd', or 'kl-bwd')."
                )

        metrics: dict[str, float | torch.Tensor] = {"action": float(loss.detach().item())}
        return loss, metrics
