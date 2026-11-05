"""Action presence comes from the action value itself: NaN means no action."""

import torch


def executed_actions(action: torch.Tensor) -> torch.Tensor:
    """Finite action values are present; NaN represents an automatic transition."""
    if action.dtype == torch.int64:
        return torch.ones_like(action, dtype=torch.bool)
    if not action.is_floating_point():
        raise TypeError(f"action must be int64 or floating point with NaN for missing actions, got {action.dtype}.")
    if bool(torch.isinf(action).any()):
        raise ValueError("action cannot contain infinity; use NaN for a missing action.")
    return ~torch.isnan(action)


def canonical_actions(action: torch.Tensor) -> torch.Tensor:
    """Validate discrete IDs and supply safe integer indices for masked gathers."""
    present = executed_actions(action)
    if action.dtype == torch.int64:
        return action
    safe = torch.where(present, action, 0.)
    if bool(((safe != safe.trunc()) | (safe < -(2**63)) | (safe >= 2**63)).any()):
        raise ValueError("Present action IDs must be integers representable as int64.")
    return safe.to(torch.int64)
