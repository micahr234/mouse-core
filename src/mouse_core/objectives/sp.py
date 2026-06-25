"""Supervised policy objective over action predictions at PREDICTION positions."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from mouse_core.objectives.base import Objective


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
    log_teacher = F.log_softmax(q_targets / temp, dim=-1)
    log_student = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        num_actions = q_targets.shape[-1]
        log_teacher = ((1.0 - label_smoothing) * log_teacher.exp() + label_smoothing / num_actions).log()

    log_m = torch.logaddexp(log_teacher, log_student) - math.log(2.0)
    # KL(P‖M) and KL(Q‖M) via kl_div(input=log M, target=log P, log_target=True)
    # -> exp(log P) * (log P - log M). nan_to_num: -inf padding in info_env_q_star gives 0*(-inf) -> NaN otherwise.
    kl_pm = torch.nan_to_num(
        F.kl_div(log_m, log_teacher, log_target=True, reduction="none"),
        nan=0.0,
    ).sum(dim=-1)
    kl_qm = torch.nan_to_num(
        F.kl_div(log_m, log_student, log_target=True, reduction="none"),
        nan=0.0,
    ).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return js.mean()


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
    log_teacher = F.log_softmax(q_targets / temp, dim=-1)
    log_student = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0:
        num_actions = q_targets.shape[-1]
        log_teacher = ((1.0 - label_smoothing) * log_teacher.exp() + label_smoothing / num_actions).log()

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
    log_teacher = F.log_softmax(q_targets / temp, dim=-1)
    if label_smoothing > 0.0:
        num_actions = q_targets.shape[-1]
        log_teacher = ((1.0 - label_smoothing) * log_teacher.exp() + label_smoothing / num_actions).log()
    log_student = F.log_softmax(logits, dim=-1)
    if direction == "fwd":
        teacher = log_teacher.exp()
        per_row = torch.nan_to_num(-(teacher * log_student), nan=0.0).sum(dim=-1)
    else:
        student = log_student.exp()
        per_row = torch.nan_to_num(-(student * log_teacher), nan=0.0).sum(dim=-1)
    return per_row.mean()


class SpObjective(Objective):
    """Supervised policy objective distilling ``info_env_q_star`` expert targets into action logits.

    Reads ``predictions["action"]`` (shape ``[B, S, A]``) from the model's action head.

    Args:
        loss_type: Which distillation loss to apply.  ``"ce"`` uses the argmax of
            ``info_env_q_star`` as a hard label; the soft variants treat it as a
            distribution.
        temperature: Softmax temperature applied to ``info_env_q_star`` before soft losses
            (ignored for ``"ce"``).
        label_smoothing: Label-smoothing coefficient (applied to hard ``"ce"`` only).
        predictions_key: Key in ``predictions`` that holds the ``[B, S, A]`` action logits.
    """

    def __init__(
        self,
        *,
        loss_type: Literal["ce", "ce-soft-fwd", "ce-soft-bwd", "js", "kl-fwd", "kl-bwd"] = "ce",
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        predictions_key: str = "action",
    ) -> None:
        self.loss_type = loss_type
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.predictions_key = predictions_key

    def __call__(
        self,
        objective_data: TensorDict,
        predictions: TensorDict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits: torch.Tensor = predictions[self.predictions_key]
        temp = float(self.temperature)

        A = logits.shape[-1]
        logits = logits.reshape(-1, A)
        q_targets = objective_data["info_env_q_star"].reshape(-1, A).to(dtype=logits.dtype)

        if q_targets.shape[0] == 0:
            raise ValueError("SpObjective: batch is empty (no tokens).")
        if not torch.isfinite(q_targets).all():
            raise ValueError("SpObjective: info_env_q_star contains non-finite values (NaN or inf).")

        if self.loss_type == "ce":
            target_actions = q_targets.argmax(dim=-1).to(dtype=torch.long)
            loss = F.cross_entropy(logits, target_actions, label_smoothing=self.label_smoothing)
        elif self.loss_type == "ce-soft-fwd":
            loss = sp_soft_ce(q_targets=q_targets, logits=logits, temperature=temp, label_smoothing=self.label_smoothing, direction="fwd")
        elif self.loss_type == "ce-soft-bwd":
            loss = sp_soft_ce(q_targets=q_targets, logits=logits, temperature=temp, label_smoothing=self.label_smoothing, direction="bwd")
        elif self.loss_type == "js":
            loss = sp_js(q_targets=q_targets, logits=logits, temperature=temp, label_smoothing=self.label_smoothing)
        elif self.loss_type == "kl-fwd":
            loss = sp_kl(q_targets=q_targets, logits=logits, temperature=temp, label_smoothing=self.label_smoothing, direction="fwd")
        elif self.loss_type == "kl-bwd":
            loss = sp_kl(q_targets=q_targets, logits=logits, temperature=temp, label_smoothing=self.label_smoothing, direction="bwd")
        else:
            raise ValueError(
                f"Invalid SpObjective loss_type: {self.loss_type!r} "
                "(expected 'ce', 'ce-soft-fwd', 'ce-soft-bwd', 'js', 'kl-fwd', or 'kl-bwd')."
            )

        metrics: dict[str, float] = {"action": float(loss.detach().item())}
        return loss, metrics
