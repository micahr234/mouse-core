"""Validity of incoming actions, separate from observation and episode validity."""

import torch


def executed_actions(columns: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """Read the batch mask derived by packing nullable incoming actions.

    Raw rows store ``action=None`` on resets. Their packed integer slot is
    only a placeholder. Direct tensor callers may omit the mask when all
    actions ran.
    """
    executed = columns.get("action_executed")
    if executed is None:
        return torch.ones_like(action, dtype=torch.bool)
    if executed.dtype not in (torch.bool, torch.int64) or executed.shape != action.shape:
        raise ValueError("action_executed must be bool or binary int64 [N] matching action.")
    if executed.dtype == torch.int64 and bool(((executed < 0) | (executed > 1)).any()):
        raise ValueError("action_executed must be bool or binary int64 [N] matching action.")
    return executed.to(device=action.device, dtype=torch.bool)


def canonical_actions(columns: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """Never validate or gather an unexecuted action's arbitrary placeholder."""
    return torch.where(executed_actions(columns, action), action, 0)
