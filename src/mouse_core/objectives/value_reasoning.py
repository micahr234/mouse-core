"""Hierarchical Bellman learning for EXECUTE/THINK and environment-action heads."""

import math
import torch
from mouse_core.models.think_execute import ThinkExecuteValues


def value_reasoning_loss(
    *, q_values: ThinkExecuteValues, target_q_values: ThinkExecuteValues,
    lengths: torch.Tensor, environment_actions: torch.Tensor,
    environment_targets: torch.Tensor, weights: torch.Tensor,
    reasoning_reward: float, gamma_think: float, min_steps: int,
    value_scale: float = 1., value_shift: float = 0.,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train observed THINK transitions, the final EXECUTE, and its recorded action.

    THINK -> reward + gamma_think * max legal next EXECUTE/THINK value.
    EXECUTE -> max target environment-action value at this same reasoning state.
    Environment action -> supplied environment Bellman target, bootstrapped
    from the next observation's FIRST legal EXECUTE/THINK state.

    The zero-time EXECUTE transition has no reward or discount. Values/targets use
    the same affine value transform. The loss averages over all observed
    transitions (lengths binary choices plus one environment action per decision).
    Fixed-depth final-only learning uses final_reasoning_loss instead.
    """
    if not isinstance(q_values, ThinkExecuteValues) or not isinstance(target_q_values, ThinkExecuteValues):
        raise TypeError('value_reasoning_loss requires separate action and EXECUTE/THINK values')
    if q_values.actions.ndim != 3 or q_values.actions.shape[1] < 1:
        raise ValueError('Action values must have shape [decisions, depth cap, environment actions]')
    count, cap, choices = q_values.actions.shape
    if type(min_steps) is not int or not 1 <= min_steps <= cap:
        raise ValueError('min_steps must be an integer in [1, depth cap]')
    if target_q_values.actions.shape != q_values.actions.shape:
        raise ValueError('Online and target Q shapes must match')
    if any(x.shape != (count,) for x in (lengths, environment_actions, environment_targets, weights)):
        raise ValueError('Lengths, actions, targets and weights must have one entry per decision')
    if lengths.dtype != torch.long or bool(((lengths < min_steps) | (lengths > cap)).any()):
        raise ValueError('Lengths must be int64 counts between the reasoning minimum and cap')
    if (environment_actions.dtype != torch.long
            or bool(((environment_actions < 0) | (environment_actions >= choices)).any())):
        raise ValueError('Environment actions must be int64 environment-action indices')
    if (type(reasoning_reward) not in (int, float) or not math.isfinite(reasoning_reward)
            or not math.isfinite(value_scale) or value_scale <= 0 or not math.isfinite(value_shift)):
        raise ValueError('reasoning_reward must be finite and value transform finite with positive scale')
    if type(gamma_think) not in (int, float) or not math.isfinite(gamma_think) or not 0 <= gamma_think <= 1:
        raise ValueError('gamma_think must be a finite number in [0, 1]')
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError('Weights must be finite and nonnegative')
    depth = torch.arange(cap, device=q_values.device)[None]
    continuing = depth < lengths[:, None] - 1
    valid = depth < lengths[:, None]
    binary_actions = continuing.long()  # EXECUTE=0, THINK=1
    predictions = value_scale * q_values.decisions.gather(-1, binary_actions[..., None]).squeeze(-1) + value_shift
    rows = torch.arange(count, device=q_values.device)
    action_predictions = value_scale * q_values.actions[rows, lengths - 1, environment_actions] + value_shift
    with torch.no_grad():
        delayed = target_q_values.masked(depth + 1, min_steps=min_steps, max_steps=cap)
        next_values = (value_scale * delayed.decisions + value_shift).amax(-1)
        next_values = torch.cat((next_values[:, 1:], next_values.new_zeros((count, 1))), dim=1)
        # EXECUTE changes heads at this token, not depth or environment state.
        # Its zero-time edge has neither reasoning reward nor a discount.
        execute_targets = (value_scale * target_q_values.actions + value_shift).amax(-1)
        targets = torch.where(continuing, reasoning_reward + gamma_think * next_values, execute_targets)
        td_weights = weights.detach()[:, None] * valid
        action_weights = weights.detach()
        action_targets = environment_targets.detach()
    errors = (predictions - targets).masked_fill(td_weights <= 0, 0).square()
    action_errors = (action_predictions - action_targets).masked_fill(action_weights <= 0, 0).square()
    loss = ((errors * td_weights).sum() + (action_errors * action_weights).sum()) / (td_weights.sum() + action_weights.sum()).clamp_min(1)
    return loss, dict(td_target=targets, td_prediction=predictions, td_weight=td_weights,
                      action_target=action_targets, action_prediction=action_predictions, action_weight=action_weights)


def final_reasoning_loss(
    *, q_values: torch.Tensor, environment_targets: torch.Tensor,
    lengths: torch.Tensor, weights: torch.Tensor, reasoning_reward: float,
    value_scale: float = 1., value_shift: float = 0.,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train only the final selected action, adding this decision's total reasoning reward.

    The caller gathers the recorded environment action at the final generated depth.
    Environment targets use next-observation final-state values, already transformed
    into value units. The first reasoning update is free. Raw replay rewards are
    not changed, and targets, rewards, and weights never receive gradients.
    """
    if q_values.ndim != 1 or any(x.shape != q_values.shape
                                for x in (environment_targets, lengths, weights)):
        raise ValueError("Final values, targets, lengths and weights must be one-dimensional with matching shapes")
    if lengths.dtype != torch.long or bool((lengths < 1).any()):
        raise ValueError("Lengths must be positive int64 counts")
    if (type(reasoning_reward) not in (int, float)
            or not math.isfinite(reasoning_reward)
            or not math.isfinite(value_scale) or value_scale <= 0 or not math.isfinite(value_shift)):
        raise ValueError("reasoning_reward must be finite and value transform finite with positive scale")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Weights must be finite and nonnegative")
    predictions = value_scale * q_values + value_shift
    with torch.no_grad():
        targets = environment_targets.detach() + reasoning_reward * (lengths - 1)
        td_weights = weights.detach()
    errors = (predictions - targets).masked_fill(td_weights <= 0, 0).square()
    loss = (errors * td_weights).sum() / td_weights.sum().clamp_min(1)
    return loss, dict(td_target=targets, td_prediction=predictions, td_weight=td_weights)
