"""Tests for PPO objective on synthetic tensors."""
from __future__ import annotations

from tests._prediction_layout import fixture_head_steps
import pytest
import torch
from mouse_core.objectives import PpoObjective, affine_reward, affine_value, boundary_discount, sample_discrete_action
from mouse_core.objectives.dqn import _pair_weight
from mouse_core.objectives.ppo import _gae_advantages


def _disc(**overrides: float):
    kwargs = dict(
        gamma_step=1.0,
        gamma_terminated=0.0,
        gamma_truncated=0.0,
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
        sequence_ids = objective_data["sequence_ids"] if "sequence_ids" in objective_data else torch.zeros(
            int(objective_data["action"].shape[0]), dtype=torch.int64
        )
        data = {key: value for key, value in objective_data.items() if key != "sequence_ids"}
        return self.objective(
            objective_data=data,
            sequence_ids=sequence_ids, step_ids=fixture_head_steps(data),
            predictions=predictions["action"],
            value_predictions=predictions["value"],
        )


def _ppo(**overrides: object) -> _PpoCall:
    kwargs: dict[str, object] = dict(
        discount=_disc(gamma_step=0.99),
        reward=_rew(),
        value=_val(),
    )
    kwargs.update(overrides)
    return _PpoCall(PpoObjective(**kwargs))  # type: ignore[arg-type]


def _ppo_batch(*, n: int=8, a: int=3, with_old_log_prob: bool=True, sequence_ids: list[int] | None=None) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    action = torch.randint(0, a, (n,))
    reward = torch.randn(n)
    terminated = torch.zeros(n, dtype=torch.long)
    truncated = torch.zeros(n, dtype=torch.long)
    data: dict[str, torch.Tensor] = {'action': action, 'reward': reward, 'terminated': terminated, 'truncated': truncated, 'sequence_ids': torch.tensor(sequence_ids if sequence_ids is not None else [0] * (n // 2) + [1] * (n - n // 2))}
    ends = torch.ones(n, dtype=torch.bool)
    ends[:-1] = data['sequence_ids'][:-1] != data['sequence_ids'][1:]
    data['terminated'][ends] = 1
    if with_old_log_prob:
        data['old_log_prob'] = torch.randn(n)
    objective_data = data
    predictions = {'action': torch.randn(n, a), 'value': torch.randn(n, 1)}
    return (objective_data, predictions)

def test_ppo_objective_runs() -> None:
    objective_data, predictions = _ppo_batch()
    loss, metrics = _ppo()(objective_data=objective_data, predictions=predictions)
    assert loss["ppo"].ndim == 0
    assert 'ppo' in metrics
    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics

def test_ppo_objective_runs_without_old_log_prob() -> None:
    objective_data, predictions = _ppo_batch(with_old_log_prob=False)
    loss, metrics = _ppo()(objective_data=objective_data, predictions=predictions)
    assert loss["ppo"].ndim == 0
    assert metrics['clipfrac'] == 0.0

def test_ppo_objective_accepts_squeezed_value() -> None:
    objective_data, predictions = _ppo_batch()
    predictions['value'] = predictions['value'].squeeze(-1)
    loss, _ = _ppo()(objective_data=objective_data, predictions=predictions)
    assert loss["ppo"].ndim == 0

def test_ppo_objective_rejects_wrong_action_shape() -> None:
    objective_data, predictions = _ppo_batch()
    objective_data['action'] = torch.randint(0, 3, (8, 1))
    with pytest.raises(ValueError, match="action shape"):
        _ppo()(objective_data=objective_data, predictions=predictions)

def test_ppo_objective_requires_min_sequence() -> None:
    objective_data = {'action': torch.zeros(1, dtype=torch.long), 'reward': torch.zeros(1), 'terminated': torch.zeros(1, dtype=torch.long), 'truncated': torch.zeros(1, dtype=torch.long)}
    predictions = {'action': torch.zeros(1, 2), 'value': torch.zeros(1, 1)}
    with pytest.raises(ValueError, match="N >= 2"):
        _ppo()(objective_data=objective_data, predictions=predictions)

def test_ppo_objective_closed_form_single_transition() -> None:
    objective_data = {'action': torch.tensor([0, 0]), 'reward': torch.tensor([0.0, 4.0]), 'terminated': torch.tensor([0, 0]), 'truncated': torch.tensor([0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0])}
    predictions = {'action': torch.tensor([[20.0, -20.0], [20.0, -20.0]]), 'value': torch.tensor([[1.0], [0.0]])}
    objective = _ppo(discount=_disc(gamma_step=0.0), gae_lambda=1.0, clip_eps=0.2, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)
    loss, metrics = objective(objective_data=objective_data, predictions=predictions)
    assert abs(loss["ppo"].item() - 6.0) < 0.001
    assert abs(metrics['policy_loss'] - -3.0) < 0.001
    assert abs(metrics['value_loss'] - 9.0) < 0.001

def test_ppo_objective_skips_transitions_across_sequences() -> None:
    objective_data = {'action': torch.tensor([0, 0, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'terminated': torch.tensor([0, 0, 0]), 'truncated': torch.tensor([0, 0, 0]), 'old_log_prob': torch.tensor([0.0, 0.0, 0.0]), 'sequence_ids': torch.tensor([0, 1, 1])}
    predictions = {'action': torch.tensor([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]]), 'value': torch.tensor([[0.0], [2.0], [0.0]])}
    loss, _ = _ppo(discount=_disc(gamma_step=0.0), gae_lambda=1.0, vf_coef=1.0, ent_coef=0.0, normalize_advantage=False)(objective_data=objective_data, predictions=predictions)
    assert abs(loss["ppo"].item() - 6.0) < 0.001

def test_ppo_objective_all_out_of_run_pairs_yield_zero_loss() -> None:
    objective_data = {'action': torch.tensor([0, 1, 0]), 'reward': torch.tensor([0.0, 1.0, 5.0]), 'terminated': torch.tensor([0, 0, 0]), 'truncated': torch.tensor([0, 0, 0]), 'sequence_ids': torch.tensor([0, 1, 2])}
    predictions = {'action': torch.zeros(3, 2), 'value': torch.zeros(3, 1)}
    loss, metrics = _ppo()(objective_data=objective_data, predictions=predictions)
    assert abs(loss["ppo"].item()) < 1e-05
    assert abs(metrics['advantage_mean']) < 1e-05

def test_ppo_policy_loss_does_not_backprop_into_values() -> None:
    torch.manual_seed(0)
    objective_data, predictions = _ppo_batch(n=6, a=3)
    predictions['action'] = predictions['action'].clone().requires_grad_(True)
    predictions['value'] = predictions['value'].clone().requires_grad_(True)
    objective = _ppo(vf_coef=0.0, ent_coef=0.0, normalize_advantage=True)
    loss, _ = objective(objective_data=objective_data, predictions=predictions)
    loss["ppo"].backward()
    assert predictions['action'].grad is not None
    assert predictions['action'].grad.abs().sum() > 0
    assert predictions['value'].grad is None or torch.all(predictions['value'].grad == 0)


def test_sample_discrete_action_shapes() -> None:
    logits = torch.randn(4, 5)
    actions, log_probs = sample_discrete_action(num_actions=3, logits=logits)
    assert actions.shape == (4,)
    assert log_probs.shape == (4,)


@pytest.mark.parametrize("counts", [[2, 2, 1, 1], [0, 2, 1, 1]])
def test_ppo_rejects_multi_head_output_rows(counts) -> None:
    objective_data, predictions = _ppo_batch(n=4)
    objective_data["readout_counts"] = torch.tensor(counts)
    with pytest.raises(ValueError, match="one prediction row per step|step_ids must have shape"):
        _ppo()(objective_data=objective_data, predictions=predictions)


def test_ppo_explained_variance_is_invariant_to_constant_value_bias() -> None:
    data = {
        "action": torch.zeros(3, dtype=torch.int64),
        "reward": torch.tensor([0.0, 1.0, 2.0]),
        "terminated": torch.zeros(3, dtype=torch.bool),
        "truncated": torch.zeros(3, dtype=torch.bool),
    }
    _, metrics = _ppo(discount=_disc(gamma_step=0.0))(
        objective_data=data,
        predictions={"action": torch.zeros(3, 2), "value": torch.tensor([5.0, 6.0, 0.0])},
    )
    assert metrics["value_loss"] == pytest.approx(16.0)
    assert metrics["explained_variance"] == pytest.approx(1.0)


@pytest.mark.parametrize("num_actions", [-1, 0, 4])
def test_sample_discrete_action_rejects_invalid_action_count(num_actions) -> None:
    with pytest.raises(ValueError, match="num_actions"):
        sample_discrete_action(logits=torch.zeros(2, 3), num_actions=num_actions)


def test_ppo_behavior_log_probabilities_are_fixed_targets() -> None:
    data, predictions = _ppo_batch(n=4)
    old_log_prob = data['old_log_prob'].requires_grad_()
    predictions['action'].requires_grad_()
    losses, _ = _ppo(normalize_advantage=False)(objective_data=data, predictions=predictions)
    losses['ppo'].backward()
    assert old_log_prob.grad is None
    assert predictions['action'].grad is not None
