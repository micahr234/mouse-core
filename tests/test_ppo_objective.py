"""Tests for PPO objective on synthetic tensors."""
from __future__ import annotations
import torch
from tensordict import TensorDict
from mouse_core.objectives import PpoObjective, batch_field, sample_discrete_action

def _ppo_batch(*, n: int=8, a: int=3, with_old_log_prob: bool=True, sequence_id: list[int] | None=None) -> tuple[TensorDict, TensorDict]:
    action = torch.randint(0, a, (n,))
    reward = torch.randn(n)
    done = torch.zeros(n, dtype=torch.long)
    data: dict[str, torch.Tensor] = {'action': action, 'reward': reward, 'done': done, 'sequence_id': torch.tensor(sequence_id if sequence_id is not None else [0] * (n // 2) + [1] * (n - n // 2))}
    if with_old_log_prob:
        data['old_log_prob'] = torch.randn(n)
    objective_data = TensorDict(data, batch_size=[n])
    predictions = TensorDict({'action': torch.randn(n, a), 'value': torch.randn(n, 1)}, batch_size=[n])
    return (objective_data, predictions)

def test_ppo_objective_runs() -> None:
    objective_data, predictions = _ppo_batch()
    loss, metrics = PpoObjective(gamma_step=0.99)(objective_data, predictions)
    assert loss.ndim == 0
    assert 'ppo' in metrics
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics

def test_ppo_objective_runs_without_old_log_prob() -> None:
    objective_data, predictions = _ppo_batch(with_old_log_prob=False)
    loss, metrics = PpoObjective()(objective_data, predictions)
    assert loss.ndim == 0
    assert metrics['clipfrac'] == 0.0

def test_ppo_objective_accepts_squeezed_value() -> None:
    objective_data, predictions = _ppo_batch()
    predictions['value'] = predictions['value'].squeeze(-1)
    loss, _ = PpoObjective()(objective_data, predictions)
    assert loss.ndim == 0

def test_ppo_objective_rejects_wrong_action_shape() -> None:
    objective_data, predictions = _ppo_batch()
    objective_data['action'] = torch.randint(0, 3, (8, 1))
    try:
        PpoObjective()(objective_data, predictions)
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for action shape [N, 1]')

def test_ppo_objective_requires_min_sequence() -> None:
    objective_data = TensorDict({'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'done': torch.zeros(1, dtype=torch.long)}, batch_size=[1])
    predictions = TensorDict({'action': torch.zeros(1, 2), 'value': torch.zeros(1, 1)}, batch_size=[1])
    try:
        PpoObjective()(objective_data, predictions)
    except ValueError as e:
        assert 'N >= 2' in str(e)
    else:
        raise AssertionError('expected ValueError for N < 2')

def test_ppo_objective_closed_form_single_transition() -> None:
    objective_data = TensorDict({'action': torch.tensor([0, 0]), 'reward': torch.tensor([0.0, 4.0]), 'done': torch.tensor([0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0])}, batch_size=[2])
    predictions = TensorDict({'action': torch.tensor([[20.0, -20.0], [20.0, -20.0]]), 'value': torch.tensor([[1.0], [0.0]])}, batch_size=[2])
    objective = PpoObjective(gamma_step=0.0, gae_lambda=1.0, clip_eps=0.2, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)
    loss, metrics = objective(objective_data, predictions)
    assert abs(loss.item() - 6.0) < 0.001
    assert abs(metrics['policy_loss'] - -3.0) < 0.001
    assert abs(metrics['value_loss'] - 9.0) < 0.001

def test_ppo_objective_skips_transitions_across_sequences() -> None:
    objective_data = TensorDict({'action': torch.tensor([0, 0, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'done': torch.tensor([0, 0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0, 0.0]), 'sequence_id': torch.tensor([0, 1, 1])}, batch_size=[3])
    predictions = TensorDict({'action': torch.tensor([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]), 'value': torch.tensor([[0.0], [2.0], [0.0]])}, batch_size=[3])
    loss, _ = PpoObjective(gamma_step=0.0, gae_lambda=1.0, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)(objective_data, predictions)
    assert abs(loss.item() - 6.0) < 0.001

def test_ppo_objective_raises_when_all_pairs_cross_sequences() -> None:
    objective_data = TensorDict({'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor([0, 1, 2])}, batch_size=[3])
    predictions = TensorDict({'action': torch.zeros(3, 2), 'value': torch.zeros(3, 1)}, batch_size=[3])
    try:
        PpoObjective()(objective_data, predictions)
    except ValueError as e:
        assert 'sequence boundary' in str(e)
    else:
        raise AssertionError('expected ValueError when every pair crosses a sequence boundary')

def test_sample_discrete_action_shapes() -> None:
    logits = torch.randn(4, 5)
    actions, log_probs = sample_discrete_action(num_actions=3, logits=logits)
    assert actions.shape == (4,)
    assert log_probs.shape == (4,)

def test_batch_field_flat_ragged() -> None:
    batch = [[{'old_log_prob': 0.1}, {'old_log_prob': 0.2}], [{'old_log_prob': 0.3}]]
    out = batch_field(batch=batch, key='old_log_prob')
    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([0.1, 0.2, 0.3]))
