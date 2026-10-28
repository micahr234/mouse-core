"""Function-space replication of every action value."""

import torch
from mouse_core.objectives.base import Objective


class ReplicateObjective(Objective[...]):
    """Mean squared error across states and actions; targets never receive gradients.

    Supply only valid action columns, including actions absent from the dataset.
    """

    def __call__(
        self, *, predictions: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        if predictions.ndim != 2 or predictions.shape != targets.shape or predictions.numel() == 0:
            raise ValueError("Replication requires matching nonempty [states, actions] tensors.")
        loss = (predictions.float() - targets.detach().float()).square().mean()
        return {"replicate": loss}, {"replicate": loss.detach()}
