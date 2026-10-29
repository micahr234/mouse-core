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

    By default, inputs have identical shape ``[..., D]``; the final dimension
    holds features. Targets are always detached. Callers own encoding, transition
    prediction, pair alignment, and target-network updates. There are no
    projection or predictor heads inside the objective.
    With ``batched_heads=True``, predictions are ``[P, K, D]`` and the loss
    is ``[K]``, one independent value per head. Targets may be shared
    ``[P, D]`` or per-head ``[P, K, D]``. A head without valid pairs has a
    differentiable zero loss. The default retains scalar loss for any
    leading shape, including ``[B, S, D]``.
    Use ``latent_cosine_loss`` to pair sequential encodings and run a supplied
    action-conditioned transition model inside the objective.

    ``mask_key`` names an optional per-pair skip mask in ``objective_data``:
    True/nonzero drops a pair. Pass None explicitly to compare every pair.
    In batched-head mode, a mask can be shared ``[P]`` or per-head ``[P, K]``.
    Otherwise it must match all leading dimensions, without broadcasting.
    Empty direct inputs raise ValueError. In scalar mode, an entirely masked
    batch also raises ValueError.

    ``eps`` is the positive norm floor. Half-precision inputs are promoted
    to float32 for stable normalization, including zero-vector inputs.
    Add the returned ``cosine`` term to DQN with a caller-chosen weight.
    """

    def __init__(
        self, *, mask_key: str | None, eps: float = 1e-8,
        batched_heads: bool = False,
    ) -> None:
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive.")
        self.mask_key = mask_key
        self.eps = eps
        self.batched_heads = batched_heads

    def _batched_cosine(
        self, predictions: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        """Reduce ``[P, K, D]`` pairs over P without mixing heads."""
        target = targets.detach()
        if predictions.dtype in (torch.float16, torch.bfloat16):
            predictions = predictions.float()
        if target.dtype in (torch.float16, torch.bfloat16):
            target = target.float()
        # Clear masked values before cosine: a skipped NaN must not enter
        # either the forward value or its backward graph.
        predictions = predictions.masked_fill(~valid[..., None], 0)
        target = target.masked_fill(~valid[..., None], 0)
        similarities = F.cosine_similarity(predictions, target, dim=-1, eps=self.eps)
        count = valid.sum(dim=0)
        mean = similarities.sum(dim=0) / count.clamp_min(1)
        zero = predictions[:0].sum(dim=(0, 2))
        loss = torch.where(count > 0, 1 - mean, zero)
        mean = torch.where(count > 0, mean, torch.zeros_like(mean))
        return {"cosine": loss}, {
            "cosine": loss.detach(),
            "cosine_similarity": mean.detach(),
        }

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        if self.batched_heads:
            if predictions.ndim != 3:
                raise ValueError("batched-head predictions must have shape [P, K, D].")
            shared_shape = (predictions.shape[0], predictions.shape[2])
            if targets.shape not in (predictions.shape, shared_shape):
                raise ValueError(
                    "batched-head targets must have shape [P, D] or [P, K, D]."
                )
        elif predictions.ndim < 1 or predictions.shape != targets.shape:
            raise ValueError("predictions and targets must have identical shape [..., D].")
        if predictions.numel() == 0:
            raise ValueError("CosineObjective: encodings must be nonempty.")
        if not predictions.is_floating_point() or not targets.is_floating_point():
            raise TypeError("predictions and targets must be floating-point encodings.")
        if predictions.device != targets.device:
            raise ValueError("predictions and targets must share a device.")

        if self.batched_heads:
            p, k, _ = predictions.shape
            if self.mask_key is None:
                valid = torch.ones((p, k), dtype=torch.bool, device=predictions.device)
            else:
                mask = objective_data[self.mask_key]
                if mask.shape == (p,):
                    valid = (mask == 0)[:, None].expand(p, k)
                elif mask.shape == (p, k):
                    valid = mask == 0
                else:
                    raise ValueError(
                        "batched-head skip mask must have shape [P] or [P, K]."
                    )
                if mask.device != predictions.device:
                    raise ValueError("skip mask and encodings must share a device.")
            aligned_targets = targets[:, None, :].expand_as(predictions) if targets.ndim == 2 else targets
            return self._batched_cosine(predictions, aligned_targets, valid)

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

        Encodings are aligned ``[N, D]`` readouts, one per step. With
        ``batched_heads=True``, predictions are ``[N, K, D]`` and targets
        may be shared ``[N, D]`` or per-head ``[N, K, D]``. The transition
        receives only eligible flattened ``[M, D]`` sources and ``[M]``
        incoming actions. The action at row i+1 belongs to the transition
        from i to i+1. Exclude group breaks,
        transitions out of terminated/truncated rows, and transitions into
        reset rows identified by ``reset_key`` (None disables that check).
        When ``action_executed`` is present, action-independent transitions
        are also excluded; their action slots are only placeholders.
        Only pairs with readouts at both ends are eligible; a terminal
        observation without a readout has no latent target. ``mask_key``, if
        configured, skips source rows. Batched-head masks may be shared
        ``[N]`` or per-head ``[N, K]``. No valid pairs returns a
        differentiable zero without calling the transition. Targets are
        detached by the cosine loss.
        """
        if self.batched_heads:
            if predictions.ndim != 3:
                raise ValueError("batched-head sequential predictions must have shape [N, K, D].")
            shared_shape = (predictions.shape[0], predictions.shape[2])
            if targets.shape not in (predictions.shape, shared_shape):
                raise ValueError(
                    "batched-head sequential targets must have shape [N, D] or [N, K, D]."
                )
            if predictions.shape[1] == 0 or predictions.shape[2] == 0:
                raise ValueError("head and encoding feature dimensions must be nonempty.")
        else:
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
        if self.mask_key is not None and not self.batched_heads:
            valid &= column(self.mask_key, objective_data[self.mask_key])[:-1] == 0

        if self.batched_heads:
            k, d = predictions.shape[1:]
            head_valid = valid[:, None].expand(-1, k)
            if self.mask_key is not None:
                mask = objective_data[self.mask_key]
                if mask.shape == (n,):
                    skip = column(self.mask_key, mask)[:-1, None]
                elif mask.shape == (n, k):
                    if mask.device != predictions.device:
                        raise ValueError("skip mask and encodings must share a device.")
                    skip = mask[:-1]
                else:
                    raise ValueError(
                        "batched-head skip mask must have shape [N] or [N, K]."
                    )
                head_valid = head_valid & (skip == 0)
            if not head_valid.any():
                zero = predictions[:0].sum(dim=(0, 2))
                return {"cosine": zero}, {"cosine": zero.detach()}
            source = predictions[:-1][head_valid]
            incoming_actions = actions[1:, None].expand(-1, k)[head_valid]
            predicted_next = transition(source, incoming_actions)
            if predicted_next.shape != source.shape:
                raise ValueError("transition output must have shape [M, D].")
            if predicted_next.device != predictions.device or not predicted_next.is_floating_point():
                raise ValueError("transition output must be floating-point on the encodings device.")
            predicted_dense = predicted_next.new_zeros((n - 1, k, d))
            predicted_dense[head_valid] = predicted_next
            next_targets = (
                targets[1:, None, :].expand(-1, k, -1)
                if targets.ndim == 2 else targets[1:]
            )
            return self._batched_cosine(predicted_dense, next_targets, head_valid)

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
