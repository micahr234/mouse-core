"""One-step Q-learning for internal think actions and environment actions."""

import math
import torch


def value_reasoning_loss(
    *, q_values: torch.Tensor, target_q_values: torch.Tensor,
    lengths: torch.Tensor, environment_actions: torch.Tensor,
    environment_targets: torch.Tensor, weights: torch.Tensor,
    reasoning_step_cost: float, gamma_think: float, min_steps: int,
    value_scale: float = 1., value_shift: float = 0.,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train actual think transitions, then the recorded outgoing environment action.

    Values have shape [decisions, depth cap, environment actions + think].
    ``lengths`` counts generated states, including the first free state.
    Environment targets/weights come from the environment's one-step TD backup,
    using next-observation *first-state* values. The caller must restrict that
    bootstrap to think when min_steps exceeds one. They are in transformed value
    units, as is the per-continuation cost. Internal transitions use gamma_think;
    environment targets retain their independently configured discount.
    Invalid/padded transitions receive no loss. Target values always detach.
    """
    if q_values.ndim != 3 or q_values.shape[-1] < 2 or q_values.shape[1] < 1:
        raise ValueError("q_values must have shape [decisions, depth cap, environment actions + think]")
    count, cap, choices = q_values.shape
    if type(min_steps) is not int or not 1 <= min_steps <= cap:
        raise ValueError("min_steps must be an integer in [1, depth cap]")
    if target_q_values.shape != q_values.shape:
        raise ValueError("Online and target Q shapes must match")
    if any(x.shape != (count,) for x in (lengths, environment_actions, environment_targets, weights)):
        raise ValueError("Lengths, actions, targets and weights must have one entry per decision")
    if lengths.dtype != torch.long or bool(((lengths < min_steps) | (lengths > cap)).any()):
        raise ValueError("Lengths must be int64 counts between the reasoning minimum and cap")
    if (environment_actions.dtype != torch.long
            or bool(((environment_actions < 0) | (environment_actions >= choices - 1)).any())):
        raise ValueError("Environment actions must be int64 indices excluding think")
    if (not math.isfinite(reasoning_step_cost) or reasoning_step_cost < 0
            or not math.isfinite(value_scale) or value_scale <= 0 or not math.isfinite(value_shift)):
        raise ValueError("Cost must be nonnegative and value transform finite with positive scale")
    if type(gamma_think) not in (int, float) or not math.isfinite(gamma_think) or not 0 <= gamma_think <= 1:
        raise ValueError("gamma_think must be a finite number in [0, 1]")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Weights must be finite and nonnegative")
    depth = torch.arange(cap, device=q_values.device)[None]
    continuing = depth < lengths[:, None] - 1
    valid = depth < lengths[:, None]
    actions = torch.where(continuing, choices - 1, environment_actions[:, None])
    predictions = value_scale * q_values.gather(-1, actions[..., None]).squeeze(-1) + value_shift
    with torch.no_grad():
        delayed = value_scale * target_q_values.detach() + value_shift
        # The minimum forces think; the cap forces an environment action.
        # A counterfactual think at an earlier legal stopping state is available.
        # Padded states never become transition targets.
        delayed = delayed.clone()
        delayed[:, :min_steps - 1, :-1] = -torch.inf
        delayed[:, -1, -1] = -torch.inf
        next_values = delayed.amax(-1)
        next_values = torch.cat((next_values[:, 1:], next_values.new_zeros((count, 1))), dim=1)
        targets = torch.where(continuing, gamma_think * next_values - reasoning_step_cost,
                              environment_targets.detach()[:, None])
        td_weights = weights.detach()[:, None] * valid
    errors = (predictions - targets).masked_fill(td_weights <= 0, 0).square()
    loss = (errors * td_weights).sum() / td_weights.sum().clamp_min(1)
    return loss, dict(td_target=targets, td_prediction=predictions, td_weight=td_weights)


def final_reasoning_loss(
    *, q_values: torch.Tensor, environment_targets: torch.Tensor,
    lengths: torch.Tensor, weights: torch.Tensor, reasoning_step_cost: float,
    value_scale: float = 1., value_shift: float = 0.,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only the final selected action, charging this decision's total cost.

    The caller gathers the recorded environment action at the final generated depth.
    Environment targets use next-observation final-state values, already transformed
    into value units. The first reasoning update is free. Raw replay rewards are
    not changed, and targets, costs, and weights never receive gradients.
    """
    if q_values.ndim != 1 or any(x.shape != q_values.shape
                                for x in (environment_targets, lengths, weights)):
        raise ValueError("Final values, targets, lengths and weights must be one-dimensional with matching shapes")
    if lengths.dtype != torch.long or bool((lengths < 1).any()):
        raise ValueError("Lengths must be positive int64 counts")
    if (type(reasoning_step_cost) not in (int, float)
            or not math.isfinite(reasoning_step_cost) or reasoning_step_cost < 0
            or not math.isfinite(value_scale) or value_scale <= 0 or not math.isfinite(value_shift)):
        raise ValueError("Cost must be nonnegative and value transform finite with positive scale")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Weights must be finite and nonnegative")
    predictions = value_scale * q_values + value_shift
    with torch.no_grad():
        targets = environment_targets.detach() - reasoning_step_cost * (lengths - 1)
        td_weights = weights.detach()
    errors = (predictions - targets).masked_fill(td_weights <= 0, 0).square()
    loss = (errors * td_weights).sum() / td_weights.sum().clamp_min(1)
    return loss, dict(td_target=targets, td_prediction=predictions, td_weight=td_weights)
