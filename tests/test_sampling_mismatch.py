import math
import pytest
import torch
from mouse_core.objectives.grpo import sampling_mismatch_weights, GrpoObjective


@pytest.mark.parametrize('mode,expected', [
    ('none', [1., 1., 1., 1., 1., 1.]),
    ('action_truncate', [1., 2., 2., 1., .5, 3.]),
    ('action_mask', [1., 2., 2., 1., 0., 0.]),
    ('episode_truncate', [1., 3., 3., 1., 1., 1.]),
    ('episode_mask', [1., 0., 0., 1., 1., 1.]),
])
def test_mismatch_modes_exclude_resets_and_separate_branches(mode, expected):
    actual = sampling_mismatch_weights(
        training_log_probs=torch.tensor([float('nan'), math.log(2), math.log(2),
                                         float('nan'), math.log(.25), math.log(4)]),
        sampling_log_probs=torch.zeros(6), group_id=torch.tensor([0, 0, 0, 1, 1, 1]),
        executed=torch.tensor([False, True, True, False, True, True]),
        mode=mode, ratio_min=.5, ratio_max=3.)
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


def test_episode_overflow_is_capped_before_exponentiation():
    actual = sampling_mismatch_weights(training_log_probs=torch.ones(4)*1000,
        sampling_log_probs=torch.zeros(4), group_id=torch.zeros(4), executed=torch.ones(4).bool(),
        mode='episode_truncate', ratio_max=3.)
    torch.testing.assert_close(actual, torch.full((4,), 3., dtype=torch.float64))


def test_mismatch_weight_is_detached_and_separate_from_policy_clipping():
    logits = torch.zeros(2, 2, requires_grad=True)
    weights = torch.tensor([1., 3.], requires_grad=True)
    data = {'action': torch.tensor([0, 0]), 'advantage': torch.tensor([0., 2.]),
            'training_log_prob': torch.tensor([0., math.log(.5)]),
            'old_log_prob': torch.tensor([0., math.log(.1)]), 'sampling_weight': weights}
    objective = GrpoObjective(ent_coef=0., old_log_prob_key='training_log_prob',
                              sampling_weight_key='sampling_weight')
    loss, metrics = objective(objective_data=data, group_id=torch.zeros(2, dtype=torch.long), predictions=logits)
    assert loss['grpo'].item() == pytest.approx(-6.)
    assert metrics['clipfrac'] == 0.
    loss['grpo'].backward()
    assert weights.grad is None
    assert logits.grad is not None
    assert logits.grad[0, 0].item() == pytest.approx(-3.)


def test_corrected_objective_requires_frozen_probabilities():
    objective = GrpoObjective(old_log_prob_key='training_log_prob', sampling_weight_key='sampling_weight')
    data = {'action': torch.tensor([0, 0]), 'advantage': torch.ones(2), 'sampling_weight': torch.ones(2)}
    with pytest.raises(KeyError, match='frozen training'):
        objective(objective_data=data, group_id=torch.zeros(2, dtype=torch.long), predictions=torch.zeros(2, 2))
