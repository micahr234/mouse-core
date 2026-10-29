"""Function-space replication of every action value."""

import torch
from mouse_core.objectives.base import Objective


class ReplicateObjective(Objective[...]):
    """Mean squared error across states and actions; targets never receive gradients.

    Supply only valid action columns, including actions absent from the dataset.
    ``[states, actions]`` returns one scalar. ``[states, heads, actions]``
    returns one loss per head, matching independent single-head calls.
    Batched heads may share ``[states, actions]`` targets or use separate
    ``[states, heads, actions]`` targets.
    """

    def __call__(
        self, *, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        if predictions.ndim not in (2, 3) or predictions.numel() == 0:
            raise ValueError(
                "Replication requires nonempty [states, actions] or "
                "[states, heads, actions] predictions."
            )
        if targets.shape == predictions.shape:
            aligned_targets = targets
        elif predictions.ndim == 3 and targets.shape == (
            predictions.shape[0], predictions.shape[2]
        ):
            aligned_targets = targets[:, None, :]
        else:
            raise ValueError(
                "Replication targets must match predictions or share "
                "[states, actions] across heads."
            )
        squared = (predictions.float() - aligned_targets.detach().float()).square()
        loss = squared.mean(dim=(0, 2)) if predictions.ndim == 3 else squared.mean()
        return {"replicate": loss}, {"replicate": loss.detach()}
