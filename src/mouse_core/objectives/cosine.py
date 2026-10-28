"""Cosine matching of aligned latent encodings with a stop-gradient target."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import _expand_policy_readouts
from mouse_core.objectives._actions import executed_actions


class CosineObjective(Objective[...]):
    """Minimize mean ``1 - cosine_similarity(predictions, targets)``.

    Inputs have identical shape ``[..., D]``; the final dimension holds
    features. Targets are always detached. Callers own encoding, transition
    prediction, pair alignment, and target-network updates. There are no
    projection or predictor heads inside the objective.
    Use ``latent_cosine_loss`` to pair sequential encodings and run a supplied
    action-conditioned transition model inside the objective.

    ``mask_key`` names an optional per-pair skip mask in ``objective_data``:
    True/nonzero drops a pair. Pass None explicitly to compare every pair.
    The mask must match the leading dimensions, without broadcasting.
    Empty inputs or an entirely masked batch raise ValueError.

    ``eps`` is the positive norm floor. Half-precision inputs are promoted
    to float32 for stable normalization, including zero-vector inputs.
    Add the returned ``cosine`` term to DQN with a caller-chosen weight.
    """

    def __init__(self, *, mask_key: str | None, eps: float = 1e-8) -> None:
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.mask_key = mask_key
        self.eps = eps

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        if predictions.ndim < 1 or predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have identical shape [..., D].")
        if predictions.numel() == 0:
            raise ValueError("CosineObjective: encodings must be nonempty.")
        if not predictions.is_floating_point() or not targets.is_floating_point():
            raise TypeError("predictions and targets must be floating-point encodings.")
        if predictions.device != targets.device:
            raise ValueError("predictions and targets must share a device.")

        target = targets.detach()
        if self.mask_key is not None:
            mask = objective_data[self.mask_key]
            if mask.shape != predictions.shape[:-1]:
                raise ValueError("skip mask shape must match encoding leading dimensions.")
            if mask.device != predictions.device:
                raise ValueError("skip mask and encodings must share a device.")
            valid = mask.reshape(-1) == 0
            if not valid.any():
                raise ValueError("CosineObjective: no pairs left after applying the skip mask.")
            predictions = predictions.reshape(-1, predictions.shape[-1])[valid]
            target = target.reshape(-1, target.shape[-1])[valid]

        if predictions.dtype in (torch.float16, torch.bfloat16):
            predictions = predictions.float()
        if target.dtype in (torch.float16, torch.bfloat16):
            target = target.float()
        similarity = F.cosine_similarity(predictions, target, dim=-1, eps=self.eps).mean()
        loss = 1 - similarity
        return {"cosine": loss}, {
            "cosine": loss.detach(),
            "cosine_similarity": similarity.detach(),
        }

    def latent_cosine_loss(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_id: torch.Tensor,
        transition: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        reset_key: str | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        """Predict the next encoding from each current encoding and incoming action.

        Encodings are aligned ``[N, D]`` readouts, one per step. The action at
        row i+1 belongs to the transition from i to i+1. Exclude group breaks,
        transitions out of terminated/truncated rows, and transitions into
        reset rows identified by ``reset_key`` (None disables that check).
        When ``action_executed`` is present, action-independent transitions
        are also excluded; their action slots are only placeholders.
        Only pairs with readouts at both ends are eligible; a terminal
        observation without a readout has no latent target. ``mask_key``, if configured,
        skips source rows. No valid pairs returns a differentiable zero without
        calling the transition. Targets are detached by the cosine loss.
        """
        if predictions.ndim != 2 or predictions.shape != targets.shape:
            raise ValueError("sequential encodings must have identical shape [N, D].")
        if predictions.shape[1] == 0:
            raise ValueError("encoding feature dimension must be nonempty.")
        if not predictions.is_floating_point() or not targets.is_floating_point():
            raise TypeError("predictions and targets must be floating-point encodings.")
        if predictions.device != targets.device:
            raise ValueError("predictions and targets must share a device.")
        predictions, _, readout = _expand_policy_readouts(objective_data, predictions, who="Cosine")
        targets, _, _ = _expand_policy_readouts(objective_data, targets, who="Cosine")
        n = predictions.shape[0]

        def column(name: str, value: torch.Tensor) -> torch.Tensor:
            if value.shape != (n,) or value.device != predictions.device:
                raise ValueError(f"{name} must have shape [N] on the encodings device.")
            return value

        groups = column("group_id", group_id)
        actions = column("action", objective_data["action"])
        if groups.dtype != torch.int64 or actions.dtype != torch.int64:
            raise TypeError("group_id and action must be int64.")
        terminated = column("terminated", objective_data["terminated"]).bool()
        truncated = column("truncated", objective_data["truncated"]).bool()
        valid = (groups[:-1] == groups[1:]) & ~terminated[:-1] & ~truncated[:-1]
        valid &= executed_actions(objective_data, actions)[1:] & readout[:-1] & readout[1:]
        if reset_key is not None:
            valid &= ~column(reset_key, objective_data[reset_key])[1:].bool()
        if self.mask_key is not None:
            valid &= column(self.mask_key, objective_data[self.mask_key])[:-1] == 0
        if not valid.any():
            zero = predictions.reshape(-1)[:0].sum()
            return {"cosine": zero}, {"cosine": zero.detach()}
        predicted_next = transition(predictions[:-1][valid], actions[1:][valid])
        # Rows have already been filtered; retain the direct objective's contract.
        aligned_data = {} if self.mask_key is None else {
            self.mask_key: torch.zeros_like(actions[1:][valid], dtype=torch.bool)
        }
        return self(
            objective_data=aligned_data,
            predictions=predicted_next,
            targets=targets[1:][valid],
        )
