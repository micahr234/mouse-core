"""One-step Value Forcing extended to a jointly evolving action vector."""

import torch

from mouse_core.objectives.base import Objective
from mouse_core.objectives._actions import canonical_actions, executed_actions
from mouse_core.objectives.dqn import (
    _column_targets, _head_output_layout, _pair_values_to_rows, _pair_weight,
    _raise_if_cross_group_backup, _require_action_ids, _step_predictions,
    _validate_request_actions,
)
from mouse_core.objectives.transforms import (
    Discount, Reward, _apply_transform, _require_transform,
)


class DiffusionDqnObjective(Objective[...]):
    """Joint action-vector flow loss, with greedy delayed action selection.

    ``prepare`` aligns incoming replay labels and constructs detached flow inputs
    and targets. Networks and noise sampling remain in the caller. Paths and
    velocities are [P, T, A]; action scores are [P, A], times [T], noise [P].
    P counts readouts, T diffusion steps, and A actions. The independent Gaussian
    correction draw is shared across diffusion times for each readout.

    Only the taken action receives a velocity loss. Its input is the scalar
    Q-only Bellman sample; other input coordinates retain their delayed
    current-state samples. The head may read the complete vector, but unobserved
    outputs receive no direct loss. This extends the paper's scalar objective;
    its flow-matching argument does not establish transport of this joint path.

    Only the next decision is bootstrapped. As in one-step DQN, automatic rows
    (including inner resets) carry their rewards/discounts to that decision.
    There is no horizon or continuation gate. Rewards are used as supplied;
    the paper's optional reward normalization is not applied implicitly.
    """

    def __init__(self, *, discount: Discount | None, reward: Reward | None = None) -> None:
        self.discount = _require_transform(discount, name="discount")
        self.reward = _require_transform(reward, name="reward")

    @torch.no_grad()
    def prepare(
        self, *, objective_data: dict[str, torch.Tensor],
        sequence_ids: torch.Tensor, step_ids: torch.Tensor,
        backup_samples: torch.Tensor, backup_value_predictions: torch.Tensor,
        backup_action_predictions: torch.Tensor, times: torch.Tensor, noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (flow inputs [P,T,A], targets [P,T], actions [P], weights [P])."""
        if backup_samples.ndim != 3 or min(backup_samples.shape[1:]) < 1:
            raise ValueError("backup_samples must have shape [P, T, A] with T,A > 0.")
        p, t, a = backup_samples.shape
        if backup_value_predictions.shape != (p, t, a):
            raise ValueError("backup_value_predictions must match backup_samples.")
        if backup_action_predictions.shape != (p, a):
            raise ValueError("backup_action_predictions must have shape [P, A].")
        if times.shape != (t,) or noise.shape != (p,):
            raise ValueError("times must have shape [T] and noise [P].")
        if not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any()):
            raise ValueError("times must be finite and in [0, 1].")
        for tensor in (backup_samples, backup_value_predictions, backup_action_predictions, times, noise):
            if tensor.dtype != torch.float32 or tensor.device != backup_samples.device:
                raise ValueError("Flow tensors must be float32 on the same device.")

        device = backup_samples.device
        action = objective_data["action"]
        if action.ndim != 1:
            raise ValueError("action must have shape [N].")
        n = len(action)
        executed = executed_actions(action)
        action = canonical_actions(action)
        _require_action_ids(action, a)
        step_of, last_rows = _head_output_layout(step_ids, N=n, P=p, device=device)
        pair_weight = _pair_weight(sequence_ids=sequence_ids, N=n, device=device)
        readout = last_rows < p
        _validate_request_actions(readout, executed, sequence_ids)
        automatic = torch.cat([~executed[1:] & (pair_weight > 0), executed.new_zeros(1)]) | ~readout
        reward = _apply_transform(
            transform=self.reward, name="reward", objective_data=objective_data,
            N=n, dtype=torch.float32, device=device, identity=objective_data["reward"],
        )
        discount = _apply_transform(
            transform=self.discount, name="discount", objective_data=objective_data,
            N=n, dtype=torch.float32, device=device, identity=torch.ones(n, device=device),
        )
        if not bool(torch.isfinite(discount).all()) or bool(((discount < 0) | (discount > 1)).any()):
            raise ValueError("discount must be finite and in [0, 1].")

        # Select one next action for its entire path, not a different action
        # at each noise level. Selection scores come from integrated returns.
        chosen = backup_action_predictions.argmax(-1)[:, None, None].expand(p, t, 1)
        samples = backup_samples.gather(-1, chosen).squeeze(-1)
        velocities = backup_value_predictions.gather(-1, chosen).squeeze(-1)
        values = torch.cat([samples.new_zeros(p, 1), samples.new_ones(p, 1), samples, velocities], -1)
        values = _step_predictions(values, last_rows).masked_fill(
            (automatic | (discount == 0))[:, None], 0,
        )
        # One existing DQN scan gives R, G, G*z_next, G*v_next. It only
        # continues through automatic rows; a real decision always stops it.
        rewards = values.new_zeros(n, 2 + 2 * t)
        rewards[:, 0] = reward
        backup, participate, _ = _column_targets(
            reward=rewards, discount=discount, v_step=values,
            pair_weight=pair_weight, columns=automatic.float(), decision_count=None,
        )
        pair_weight = pair_weight * executed[1:]
        _raise_if_cross_group_backup(in_run=pair_weight > 0, participate=participate[:, 0])
        backup = _pair_values_to_rows(backup, step_of)
        weight = _pair_values_to_rows(pair_weight, step_of)
        r, g = backup[:, :1], backup[:, 1:2]
        correction = (1 - g.square()).clamp_min(0).sqrt() * noise[:, None]
        bellman_inputs = times * r + backup[:, 2:2 + t] + (1 - times) * correction
        bellman_targets = r + backup[:, 2 + t:] - correction
        next_step = (step_of + 1).clamp_max(n - 1)
        actions = action[next_step]
        taken = actions[:, None, None].expand(p, t, 1)
        # Retain each unobserved action's current-state sample as context.
        # Broadcasting the scalar Bellman input would destroy the joint prior.
        inputs = backup_samples.scatter(-1, taken, bellman_inputs[:, :, None])
        active = weight[:, None] > 0
        return (
            inputs.masked_fill(~active[:, :, None], 0),
            bellman_targets.masked_fill(~active, 0), actions, weight,
        )

    def __call__(
        self, *, predictions: torch.Tensor, targets: torch.Tensor,
        actions: torch.Tensor, weight: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        """Match only taken-action velocities at every flow step of active readouts."""
        if predictions.ndim != 3 or predictions.shape[1] < 1:
            raise ValueError("predictions must have shape [P, T, A] with T > 0.")
        p, t, _ = predictions.shape
        if targets.shape != (p, t) or actions.shape != (p,) or weight.shape != (p,):
            raise ValueError("targets [P,T], actions [P], and weight [P] must match predictions.")
        selected = predictions.gather(-1, actions[:, None, None].expand(p, t, 1)).squeeze(-1)
        error = torch.where(weight[:, None] > 0, selected - targets.detach(), 0).square()
        per_step = error.sum(0) / weight.sum().clamp_min(1)
        loss = per_step.mean()
        return {"diffusion_dqn": loss}, {"per_diffusion_step": per_step.detach()}
