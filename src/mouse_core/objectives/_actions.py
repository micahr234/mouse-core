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


def replay_action_indices(data, sequence_ids, *, step_ids, readout_shape):
    """Pair prediction rows with recorded outgoing actions, never policy argmaxes.

    ``data["action"][t+1]`` records the action taken from step ``t``.
    ``step_ids`` maps each prediction row to its source step; repeated and
    missing readouts are supported. NaN actions and episode boundaries have
    no outgoing training pair. Return prediction row indices and int64 actions.
    """
    actions = data["action"]
    if actions.ndim != 1 or sequence_ids.shape != actions.shape:
        raise ValueError("actions and sequence_ids must be matching one-dimensional columns")
    if sequence_ids.dtype != torch.int64 or step_ids.dtype != torch.int64:
        raise TypeError("sequence_ids and step_ids must be int64")
    if step_ids.ndim != 1 or step_ids.shape != readout_shape:
        raise ValueError("Replay actions and value readouts are misaligned")
    if actions.device != sequence_ids.device or actions.device != step_ids.device:
        raise ValueError("actions, sequence_ids and step_ids must share a device")
    if bool(((step_ids < 0) | (step_ids >= actions.numel())).any()):
        raise ValueError("step_ids must identify existing steps")
    present = executed_actions(actions)
    ids = canonical_actions(actions)
    if bool((ids[present] < 0).any()):
        raise ValueError("Present action IDs must be nonnegative")
    outgoing = torch.zeros_like(ids)
    valid = torch.zeros_like(present)
    outgoing[:-1] = ids[1:]
    valid[:-1] = present[1:] & (sequence_ids[:-1] == sequence_ids[1:])
    rows = valid[step_ids].nonzero(as_tuple=True)[0]
    return rows, outgoing[step_ids[rows]]
