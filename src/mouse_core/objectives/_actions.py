"""Validity of incoming actions, separate from observation and task validity."""

import torch


def executed_actions(columns: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """A reset frame has no incoming action; its integer slot is a placeholder."""
    executed = columns.get("action_executed")
    if executed is None:
        return torch.ones_like(action, dtype=torch.bool)
    if executed.dtype != torch.bool or executed.shape != action.shape:
        raise ValueError("action_executed must be bool [N] matching action.")
    return executed.to(device=action.device)


def canonical_actions(columns: dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
    """Never validate or gather an unexecuted action's arbitrary placeholder."""
    return torch.where(executed_actions(columns, action), action, 0)
