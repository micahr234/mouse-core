"""Separate environment-action values from the binary EXECUTE/THINK decision."""
from dataclasses import dataclass
import torch
from mouse_core.sampling import tempered_policy_logits


@dataclass(frozen=True)
class ReasoningActionValues:
    """Environment-action values [..., A] and control values [..., 2], ordered EXECUTE, THINK.

    Selection returns an environment action in [0, A), or A for THINK.
    EXECUTE itself is internal and is never sent to the environment.
    """
    action_values: torch.Tensor
    control_values: torch.Tensor

    def __post_init__(self):
        if (self.action_values.ndim < 1 or self.action_values.shape[-1] < 1
                or self.control_values.shape != (*self.action_values.shape[:-1], 2)
                or self.action_values.device != self.control_values.device):
            raise ValueError('Action values [..., A] and EXECUTE/THINK values [..., 2] must have matching batches and devices')

    def __len__(self):
        return len(self.action_values)

    def __getitem__(self, rows):
        return ReasoningActionValues(self.action_values[rows], self.control_values[rows])

    @property
    def num_actions(self):
        return self.action_values.shape[-1]

    @property
    def device(self):
        return self.action_values.device

    def masked(self, depths, *, min_depth, max_depth):
        decisions = self.control_values.clone()
        decisions[..., 0] = decisions[..., 0].masked_fill(depths < min_depth, -torch.inf)
        decisions[..., 1] = decisions[..., 1].masked_fill(depths >= max_depth, -torch.inf)
        return ReasoningActionValues(self.action_values, decisions)

    def _policy_control_values(self, policy_think_enabled):
        """Apply policy permission without suppressing minimum-required THINK."""
        gate = self.control_values
        if not policy_think_enabled:
            gate = gate.clone()
            gate[..., 1] = gate[..., 1].masked_fill(torch.isfinite(gate[..., 0]), -torch.inf)
        return gate

    def greedy(self, *, policy_think_enabled=True):
        gate = self._policy_control_values(policy_think_enabled)
        return torch.where(gate.argmax(-1) == 1, self.num_actions, self.action_values.argmax(-1))

    def available(self):
        """Availability in the external action space [environment actions, THINK]."""
        return torch.cat((torch.isfinite(self.action_values) & torch.isfinite(self.control_values[..., :1]),
                          torch.isfinite(self.control_values[..., 1:])), dim=-1)

    def policy_logits(self, temperatures, *, policy_think_enabled=True):
        """Two-stage log probabilities; THINK probability is independent of A.

        At zero temperature the binary argmax is resolved before the action
        argmax. Minimum-forced THINK remains available when policy THINK is off.
        """
        gate = self._policy_control_values(policy_think_enabled)
        gate = tempered_policy_logits(gate, temperatures).log_softmax(-1)
        actions = tempered_policy_logits(self.action_values, temperatures).log_softmax(-1)
        return torch.cat((gate[..., :1] + actions, gate[..., 1:]), dim=-1)


class ReasoningActionHeads(torch.nn.Module):
    """Compose environment-action and reasoning-control heads on the same state."""
    def __init__(self, *, action_head: torch.nn.Module, reasoning_control_head: torch.nn.Module):
        super().__init__()
        self.action_head = action_head
        self.reasoning_control_head = reasoning_control_head

    def forward(self, states):
        return ReasoningActionValues(self.action_head(states), self.reasoning_control_head(states))
