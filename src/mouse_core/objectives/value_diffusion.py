"""Q-only Value Forcing with DQN-style replay alignment and delayed backups."""

import torch

from mouse_core.objectives._actions import canonical_actions, executed_actions
from mouse_core.objectives.base import Objective
from mouse_core.objectives.dqn import (
    _head_output_layout, _pair_weight, _require_action_ids, _require_done_flags,
    _validate_request_actions,
)
from mouse_core.objectives.transforms import Discount, Reward, _apply_transform, _require_transform


class ValueDiffusionObjective(Objective[...]):
    """Fit Value Forcing velocities using externally evaluated predictions.

    Like DQN, the loss receives replay labels, readout mappings, online predictions,
    and delayed value/action predictions. Here value predictions are velocities
    [P,T,A], while action predictions are endpoint returns [P,A]. The objective
    selects the next action, constructs the velocity target, and reduces the loss.

    Call ``inputs`` to construct the Bellman-transformed online input from
    delayed samples, then evaluate the online network externally and pass its
    predictions to this objective. Input and loss use the same delayed trajectory,
    greedy next action, and independent correction noise N. Share the initial Z
    across readouts in a replay sequence; N is independent of Z. All model
    evaluation, integration, and random sampling stay outside the objective.

    Rewards use the ordinary DQN return convention. The paper's practical
    (1-gamma) reward scaling is not applied implicitly.
    """

    def __init__(
        self,
        *,
        discount: Discount | None,
        reward: Reward | None,
        action_key: str = "action",
        terminated_key: str = "terminated",
        truncated_key: str = "truncated",
    ) -> None:
        self.discount = _require_transform(discount, name="discount")
        self.reward = _require_transform(reward, name="reward")
        self.action_key = action_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key

    @torch.no_grad()
    def _transitions(self, *, objective_data: dict[str, torch.Tensor],
                     sequence_ids: torch.Tensor, step_ids: torch.Tensor,
                     num_readouts: int, num_actions: int,
                     device: torch.device
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align replay into reward, discount, next readout, action, and valid rows."""
        action = objective_data[self.action_key]
        if action.ndim != 1:
            raise ValueError("action must have shape [N].")
        n = len(action)
        present = executed_actions(action)
        actions = canonical_actions(action)
        _require_action_ids(actions, num_actions)
        _require_done_flags(objective_data, terminated_key=self.terminated_key,
                            truncated_key=self.truncated_key, N=n)
        step_of, readout = _head_output_layout(step_ids, N=n, P=num_readouts, device=device)
        pairs = _pair_weight(sequence_ids=sequence_ids, N=n, device=device)
        _validate_request_actions(readout < num_readouts, present, sequence_ids)
        same_sequence = torch.cat((pairs > 0, present.new_zeros(1)))
        decision = same_sequence & torch.cat((present[1:], present.new_zeros(1)))
        discounts = _apply_transform(
            transform=self.discount, name="discount", objective_data=objective_data,
            N=n, dtype=torch.float32, device=device, identity=torch.ones(n, device=device),
        )
        if not bool((torch.isfinite(discounts) & (discounts >= 0) & (discounts <= 1)).all()):
            raise ValueError("Value diffusion requires discounts in [0,1].")
        rewards = _apply_transform(
            transform=self.reward, name="reward", objective_data=objective_data,
            N=n, dtype=torch.float32, device=device, identity=objective_data["reward"],
        )

        valid = decision[step_of]
        next_step = (step_of + 1).clamp_max(n - 1)
        taken_actions = actions[next_step]
        reward = discounts.new_zeros(num_readouts)
        discount = valid.to(discounts.dtype)
        next_rows = step_of.new_zeros(num_readouts)
        pending = valid.clone()
        while bool(pending.any()):
            reward += torch.where(pending, discount * rewards[next_step], 0)
            discount = torch.where(pending, discount * discounts[next_step], discount)
            reached_decision = decision[next_step]
            next_rows = torch.where(pending & reached_decision, readout[next_step], next_rows)
            pending = pending & (discount != 0) & ~reached_decision
            if bool((pending & ~same_sequence[next_step]).any()):
                raise ValueError("A nonterminal backup needs a next decision in the same sequence.")
            next_step = (next_step + 1).clamp_max(n - 1)
        return reward[:, None], discount[:, None], next_rows, taken_actions, valid

    @torch.no_grad()
    def _backup(self, *, objective_data: dict[str, torch.Tensor], sequence_ids: torch.Tensor,
                step_ids: torch.Tensor, values: torch.Tensor,
                backup_action_predictions: torch.Tensor, noise: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or min(values.shape) < 1:
            raise ValueError("Backup samples/velocities must have shape [P,T,A] with nonempty axes.")
        rows, _, actions = values.shape
        if backup_action_predictions.shape != (rows, actions):
            raise ValueError("backup_action_predictions must have shape [P,A].")
        if noise.shape != (rows,):
            raise ValueError("noise must contain one independent correction draw per readout [P].")
        for name, tensor in (("backup values", values), ("backup_action_predictions", backup_action_predictions),
                             ("noise", noise)):
            if tensor.dtype != torch.float32:
                raise TypeError(f"Value diffusion expects float32 {name}.")
            if tensor.device != values.device:
                raise ValueError(f"{name} must share the backup device.")
        if not bool(torch.isfinite(noise).all()):
            raise ValueError("noise must be finite.")
        reward, discount, next_rows, taken, valid = self._transitions(
            objective_data=objective_data, sequence_ids=sequence_ids, step_ids=step_ids,
            num_readouts=rows, num_actions=actions, device=values.device,
        )
        next_actions = backup_action_predictions[next_rows].argmax(-1)
        selected = values[next_rows, :, next_actions]
        selected = torch.where(discount > 0, selected, 0)
        correction = (1 - discount.square()).sqrt() * noise[:, None]
        return selected, reward, discount, correction, taken, valid

    @torch.no_grad()
    def inputs(self, *, objective_data: dict[str, torch.Tensor], sequence_ids: torch.Tensor,
               step_ids: torch.Tensor, backup_samples: torch.Tensor,
               backup_action_predictions: torch.Tensor, noise_levels: torch.Tensor,
               noise: torch.Tensor) -> torch.Tensor:
        """Return x_tau [P,T] for external online evaluation; never draw noise or call a model."""
        samples, reward, discount, correction, _, _ = self._backup(
            objective_data=objective_data, sequence_ids=sequence_ids, step_ids=step_ids,
            values=backup_samples, backup_action_predictions=backup_action_predictions, noise=noise,
        )
        if (noise_levels.shape != (samples.shape[1],) or noise_levels.device != samples.device
                or noise_levels.dtype != torch.float32
                or not bool((torch.isfinite(noise_levels) & (noise_levels >= 0) & (noise_levels <= 1)).all())):
            raise ValueError("noise_levels must be float32 [T] in [0,1] on the backup device.")
        times = noise_levels[None, :]
        return times * reward + discount * samples + (1 - times) * correction

    def __call__(
        self,
        *,
        objective_data: dict[str, torch.Tensor],
        sequence_ids: torch.Tensor,
        step_ids: torch.Tensor,
        predictions: torch.Tensor,
        backup_value_predictions: torch.Tensor,
        backup_action_predictions: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor]]:
        velocities, reward, discount, correction, actions, valid = self._backup(
            objective_data=objective_data, sequence_ids=sequence_ids, step_ids=step_ids,
            values=backup_value_predictions, backup_action_predictions=backup_action_predictions, noise=noise,
        )
        targets = reward + discount * velocities - correction
        if predictions.dtype != torch.float32:
            raise TypeError("Value diffusion expects float32 predictions.")
        if predictions.device != targets.device:
            raise ValueError("predictions and backups must share a device.")
        if predictions.shape == backup_value_predictions.shape:
            selected = predictions.gather(-1, actions[:, None, None].expand(*targets.shape, 1)).squeeze(-1)
        elif predictions.shape == targets.shape:
            selected = predictions
        else:
            raise ValueError("predictions must have shape [P,T,A] or selected-action shape [P,T].")
        weights = valid[:, None].expand_as(targets).float()
        residual = torch.where(valid[:, None], selected, 0) - torch.where(valid[:, None], targets, 0)
        per_flow_step = (residual.square() * weights).sum(0) / weights.sum(0).clamp_min(1)
        loss = per_flow_step.mean()
        return {"value_diffusion": loss}, {
            "value_diffusion": loss.detach(), "per_flow_step": per_flow_step.detach(),
            "backup": targets, "backup_weight": weights, "effective_discount": discount[:, 0],
        }
