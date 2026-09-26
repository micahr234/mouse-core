"""Tests for PPO objective on synthetic tensors."""
from __future__ import annotations
import pytest
import torch
from mouse_core.objectives import PpoObjective, affine_reward, affine_value, boundary_discount, sample_discrete_action


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_episode_terminal=0.0,
        gamma_episode_truncated=0.0,
        gamma_task_terminal=0.0,
        gamma_task_truncated=0.0,
    )
    kwargs.update(overrides)
    return boundary_discount(**kwargs)


def _rew(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_reward(**kwargs)  # type: ignore[arg-type]


def _val(**overrides: object):
    kwargs: dict[str, object] = dict(scale=1.0, shift=0.0)
    kwargs.update(overrides)
    return affine_value(**kwargs)  # type: ignore[arg-type]


class _PpoCall:
    def __init__(self, objective: PpoObjective) -> None:
        self.objective = objective

    def __call__(self, *, objective_data: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor]):
        return self.objective(
            objective_data=objective_data,
            predictions=predictions["action"],
            value_predictions=predictions["value"],
        )


def _ppo(**overrides: object) -> _PpoCall:
    kwargs: dict[str, object] = dict(
        discount=_disc(gamma_step=0.99),
        reward=_rew(),
        value=_val(),
        grouping_field=None,
        bootstrap_cutoff=True,
    )
    kwargs.update(overrides)
    return _PpoCall(PpoObjective(**kwargs))  # type: ignore[arg-type]


def _ppo_batch(*, n: int=8, a: int=3, with_old_log_prob: bool=True, sequence_id: list[int] | None=None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    action = torch.randint(0, a, (n,))
    reward = torch.randn(n)
    episode_done = torch.zeros(n, dtype=torch.long)
    task_done = torch.zeros(n, dtype=torch.long)
    data: dict[str, torch.Tensor] = {'action': action, 'reward': reward, 'episode_done': episode_done, 'task_done': task_done, 'sequence_id': torch.tensor(sequence_id if sequence_id is not None else [0] * (n // 2) + [1] * (n - n // 2))}
    if with_old_log_prob:
        data['old_log_prob'] = torch.randn(n)
    objective_data = data
    predictions = {'action': torch.randn(n, a), 'value': torch.randn(n, 1)}
    return (objective_data, predictions)

def test_ppo_objective_runs() -> None:
    objective_data, predictions = _ppo_batch()
    loss, metrics = _ppo()(objective_data=objective_data, predictions=predictions)
    assert loss.ndim == 0
    assert 'ppo' in metrics
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics

def test_ppo_objective_runs_without_old_log_prob() -> None:
    objective_data, predictions = _ppo_batch(with_old_log_prob=False)
    loss, metrics = _ppo()(objective_data=objective_data, predictions=predictions)
    assert loss.ndim == 0
    assert metrics['clipfrac'] == 0.0

def test_ppo_objective_accepts_squeezed_value() -> None:
    objective_data, predictions = _ppo_batch()
    predictions['value'] = predictions['value'].squeeze(-1)
    loss, _ = _ppo()(objective_data=objective_data, predictions=predictions)
    assert loss.ndim == 0

def test_ppo_objective_rejects_wrong_action_shape() -> None:
    objective_data, predictions = _ppo_batch()
    objective_data['action'] = torch.randint(0, 3, (8, 1))
    with pytest.raises(ValueError, match="action shape"):
        _ppo()(objective_data=objective_data, predictions=predictions)

def test_ppo_objective_requires_min_sequence() -> None:
    objective_data = {'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'episode_done': torch.zeros(1, dtype=torch.long), 'task_done': torch.zeros(1, dtype=torch.long)}
    predictions = {'action': torch.zeros(1, 2), 'value': torch.zeros(1, 1)}
    with pytest.raises(ValueError, match="N >= 2"):
        _ppo()(objective_data=objective_data, predictions=predictions)

def test_ppo_objective_closed_form_single_transition() -> None:
    objective_data = {'action': torch.tensor([0, 0]), 'reward': torch.tensor([0.0, 4.0]), 'episode_done': torch.tensor([0, 0]), 'task_done': torch.tensor([0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0])}
    predictions = {'action': torch.tensor([[20.0, -20.0], [20.0, -20.0]]), 'value': torch.tensor([[1.0], [0.0]])}
    objective = _ppo(discount=_disc(gamma_step=0.0), gae_lambda=1.0, clip_eps=0.2, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)
    loss, metrics = objective(objective_data=objective_data, predictions=predictions)
    assert abs(loss.item() - 6.0) < 0.001
    assert abs(metrics['policy_loss'] - -3.0) < 0.001
    assert abs(metrics['value_loss'] - 9.0) < 0.001

def test_ppo_masks_cross_task_pairs_only_when_grouping_field_set() -> None:
    """Same sequence_id + a task change: grouping_field is the only cut.

    ``task_done`` on the last row of task A discounts the *incoming* pair,
    not the outgoing (A, B) pair. ``grouping_field=None`` trains that pair
    (wrong action/reward from the new task). Omitting the argument is an
    error — it must not silently default to no isolation.
    """
    objective_data = {
            "action": torch.tensor([0, 0, 0]),
            "reward": torch.tensor([0.0, 1.0, 5.0]),
            "episode_done": torch.tensor([0, 0, 0]),
            "task_done": torch.tensor([0, 0, 0]),
            "old_log_prob": torch.tensor([0.0, 0.0, 0.0]),
            "sequence_id": torch.tensor([0, 0, 0]),
            "task_index": torch.tensor([0, 1, 1]),
        }
    predictions = {
            "action": torch.tensor([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]),
            "value": torch.tensor([[0.0], [2.0], [0.0]]),
        }
    kwargs = dict(
        discount=_disc(gamma_step=0.0), gae_lambda=1.0, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False
    )
    loss_cut, _ = _ppo(grouping_field="task_index", **kwargs)(
        objective_data=objective_data, predictions=predictions
    )
    assert abs(loss_cut.item() - 6.0) < 0.001
    loss_leak, _ = _ppo(grouping_field=None, **kwargs)(objective_data=objective_data, predictions=predictions)
    assert abs(loss_leak.item() - 6.0) > 0.1


def test_ppo_objective_skips_transitions_across_sequences() -> None:
    objective_data = {'action': torch.tensor([0, 0, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0, 0.0]), 'sequence_id': torch.tensor([0, 1, 1])}
    predictions = {'action': torch.tensor([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]), 'value': torch.tensor([[0.0], [2.0], [0.0]])}
    loss, _ = _ppo(discount=_disc(gamma_step=0.0), gae_lambda=1.0, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)(objective_data=objective_data, predictions=predictions)
    assert abs(loss.item() - 6.0) < 0.001

def test_ppo_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    objective_data = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'episode_done': torch.tensor([0, 0, 0]), 'task_done': torch.tensor([0, 0, 0]), 'sequence_id': torch.tensor([0, 1, 2])}
    predictions = {'action': torch.zeros(3, 2), 'value': torch.zeros(3, 1)}
    loss, metrics = _ppo()(objective_data=objective_data, predictions=predictions)
    assert abs(loss.item()) < 1e-05
    assert abs(metrics['advantage_mean']) < 1e-05

def test_ppo_policy_loss_does_not_backprop_into_values() -> None:
    torch.manual_seed(0)
    objective_data, predictions = _ppo_batch(n=6, a=3)
    predictions['action'] = predictions['action'].clone().requires_grad_(True)
    predictions['value'] = predictions['value'].clone().requires_grad_(True)
    objective = _ppo(vf_coef=0.0, ent_coef=0.0, normalize_advantage=True)
    loss, _ = objective(objective_data=objective_data, predictions=predictions)
    loss.backward()
    assert predictions['action'].grad is not None
    assert predictions['action'].grad.abs().sum() > 0
    assert predictions['value'].grad is None or torch.all(predictions['value'].grad == 0)


def test_sample_discrete_action_shapes() -> None:
    logits = torch.randn(4, 5)
    actions, log_probs = sample_discrete_action(num_actions=3, logits=logits)
    assert actions.shape == (4,)
    assert log_probs.shape == (4,)


def test_ppo_requires_grouping_field_argument() -> None:
    with pytest.raises(TypeError, match="grouping_field"):
        PpoObjective(discount=_disc(), reward=_rew(), value=_val(), bootstrap_cutoff=True)  # type: ignore[call-arg]


def test_ppo_bootstrap_cutoff_is_switchable() -> None:
    """GAE adds V at the batch end unless bootstrap_cutoff is off. A terminal does not."""
    predictions = {
        "action": torch.tensor([[20.0, -20.0], [20.0, -20.0]]),
        "value": torch.tensor([[1.0], [5.0]]),
    }
    common = dict(
        discount=_disc(gamma_step=1.0),
        gae_lambda=1.0,
        vf_coef=1.0,
        ent_coef=0.0,
        normalize_advantage=False,
    )

    def run(*, bootstrap_cutoff: bool, episode_done: torch.Tensor) -> float:
        objective_data = {
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 4.0]),
            "episode_done": episode_done,
            "task_done": torch.tensor([0, 0]),
            "old_log_prob": torch.tensor([0.0, 0.0]),
        }
        loss, _ = _ppo(bootstrap_cutoff=bootstrap_cutoff, **common)(
            objective_data=objective_data, predictions=predictions,
        )
        return float(loss.item())

    # δ = 4 + V(s') - 1 = 8; value loss 64; policy loss -8.
    assert abs(run(bootstrap_cutoff=True, episode_done=torch.tensor([0, 0])) - 56.0) < 0.001
    # Same transition with the cutoff value omitted: δ = 3, as if γV were 0.
    assert abs(run(bootstrap_cutoff=False, episode_done=torch.tensor([0, 0])) - 6.0) < 0.001
    truncated = _disc(gamma_step=1.0, gamma_episode_truncated=1.0)
    on, _ = _ppo(bootstrap_cutoff=True, discount=truncated, gae_lambda=1.0, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)(
        objective_data={
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 4.0]),
            "episode_done": torch.tensor([0, 2]),
            "task_done": torch.tensor([0, 0]),
            "old_log_prob": torch.tensor([0.0, 0.0]),
        },
        predictions=predictions,
    )
    off, _ = _ppo(bootstrap_cutoff=False, discount=truncated, gae_lambda=1.0, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)(
        objective_data={
            "action": torch.tensor([0, 0]),
            "reward": torch.tensor([0.0, 4.0]),
            "episode_done": torch.tensor([0, 2]),
            "task_done": torch.tensor([0, 0]),
            "old_log_prob": torch.tensor([0.0, 0.0]),
        },
        predictions=predictions,
    )
    assert abs(on.item() - 56.0) < 0.001
    assert abs(off.item() - 6.0) < 0.001
    terminal_on = run(bootstrap_cutoff=True, episode_done=torch.tensor([0, 1]))
    terminal_off = run(bootstrap_cutoff=False, episode_done=torch.tensor([0, 1]))
    assert abs(terminal_on - 6.0) < 0.001
    assert abs(terminal_off - terminal_on) < 1e-05


def test_ppo_requires_bootstrap_cutoff_argument() -> None:
    with pytest.raises(TypeError, match="bootstrap_cutoff"):
        PpoObjective(  # type: ignore[call-arg]
            discount=_disc(), reward=_rew(), value=_val(), grouping_field=None,
        )


def test_ppo_rejects_multi_head_output_rows() -> None:
    objective_data, predictions = _ppo_batch(n=4)
    objective_data["head_output_count"] = torch.tensor([2, 2, 1, 1])
    with pytest.raises(ValueError, match="one prediction row per step"):
        _ppo()(objective_data=objective_data, predictions=predictions)
