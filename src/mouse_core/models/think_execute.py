"""Separate environment-action values from the binary EXECUTE/THINK decision."""
from dataclasses import dataclass
import torch
from mouse_core.sampling import tempered_policy_logits


@dataclass(frozen=True)
class ThinkExecuteValues:
    """Action values [..., A] and decision values [..., 2], ordered EXECUTE, THINK.

    Selection returns an environment action in [0, A), or A for THINK.
    EXECUTE itself is internal and is never sent to the environment.
    """
    actions: torch.Tensor
    decisions: torch.Tensor

    def __post_init__(self):
        if (self.actions.ndim < 1 or self.actions.shape[-1] < 1
                or self.decisions.shape != (*self.actions.shape[:-1], 2)
                or self.actions.device != self.decisions.device):
            raise ValueError('Action values [..., A] and EXECUTE/THINK values [..., 2] must have matching batches and devices')

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, rows):
        return ThinkExecuteValues(self.actions[rows], self.decisions[rows])

    @property
    def num_actions(self):
        return self.actions.shape[-1]

    @property
    def device(self):
        return self.actions.device

    def masked(self, depths, *, min_steps, max_steps):
        decisions = self.decisions.clone()
        decisions[..., 0] = decisions[..., 0].masked_fill(depths < min_steps, -torch.inf)
        decisions[..., 1] = decisions[..., 1].masked_fill(depths >= max_steps, -torch.inf)
        return ThinkExecuteValues(self.actions, decisions)

    def greedy(self, *, allow_policy_think=True):
        gate = self.decisions
        if not allow_policy_think:
            gate = gate.clone()
            gate[..., 1] = gate[..., 1].masked_fill(torch.isfinite(gate[..., 0]), -torch.inf)
        return torch.where(gate.argmax(-1) == 1, self.num_actions, self.actions.argmax(-1))

    def available(self):
        """Availability in the external action space [environment actions, THINK]."""
        return torch.cat((torch.isfinite(self.actions) & torch.isfinite(self.decisions[..., :1]),
                          torch.isfinite(self.decisions[..., 1:])), dim=-1)

    def policy_logits(self, temperatures, *, allow_policy_think=True):
        """Two-stage log probabilities; THINK probability is independent of A.

        At zero temperature the binary argmax is resolved before the action
        argmax. Minimum-forced THINK remains available when policy THINK is off.
        """
        gate = self.decisions.clone()
        if not allow_policy_think:
            gate[..., 1] = gate[..., 1].masked_fill(torch.isfinite(gate[..., 0]), -torch.inf)
        gate = tempered_policy_logits(gate, temperatures).log_softmax(-1)
        actions = tempered_policy_logits(self.actions, temperatures).log_softmax(-1)
        return torch.cat((gate[..., :1] + actions, gate[..., 1:]), dim=-1)


class ThinkExecuteHead(torch.nn.Module):
    """Compose environment-action and reasoning-control heads on the same state."""
    def __init__(self, *, action_head: torch.nn.Module, reasoning_control_head: torch.nn.Module):
        super().__init__()
        self.action_head = action_head
        self.reasoning_control_head = reasoning_control_head

    def forward(self, states):
        return ThinkExecuteValues(self.action_head(states), self.reasoning_control_head(states))
