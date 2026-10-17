import numpy as np
import pytest
import torch
from mouse_core.sampling import EpisodeTemperature, check_temperature_range, last_action_scores, sample_tempered_actions, tempered_policy_logits


class RecordingRng:
    def __init__(self):
        self.generator = np.random.default_rng(17)
        self.draws = []
        self.probabilities = []

    def uniform(self, low, high):
        value = self.generator.uniform(low, high)
        self.draws.append(value)
        return value

    def choice(self, n, *, p):
        self.probabilities.append(p.copy())
        return self.generator.choice(n, p=p)


def test_uniform_temperature_is_per_task_and_per_stream():
    rng = RecordingRng()
    policy = EpisodeTemperature(0.2, 1.7, rng)
    first = policy.for_episode(0, 0)
    assert all(policy.for_episode(0, 0) == first for _ in range(20))
    second = policy.for_episode(1, 0)
    third = policy.for_episode(0, 1)
    assert policy.for_episode(1, 0) == second
    assert len({first, second, third}) == 3
    assert len(rng.draws) == 3
    values = np.array([policy.for_episode(0, episode) for episode in range(2, 5002)])
    assert values.min() >= 0.2 and values.max() < 1.7
    assert values.mean() == pytest.approx(0.95, abs=0.03)
    assert values.var() == pytest.approx(1.5 ** 2 / 12, abs=0.01)


@pytest.mark.parametrize('bounds', [(-1, 1), (2, 1), (0, float('inf')), (float('nan'), 1)])
def test_invalid_temperature_ranges(bounds):
    with pytest.raises(ValueError, match='temperature bounds'):
        check_temperature_range(*bounds)


def test_tempered_actions_use_correct_probabilities_and_log_probs():
    rng = RecordingRng()
    scores = torch.tensor([[0., 1., 2.], [0., 1., 2.], [0., 1., 2.]])
    actions, logs = sample_tempered_actions(scores, [0., 0.5, 2.], rng)
    assert actions[0] == 2 and logs[0] == 0
    assert len(rng.probabilities) == 2  # Greedy consumes no random action draw.
    for i, temperature in enumerate((0.5, 2.), start=1):
        expected = (scores[i].double() / temperature).softmax(-1)
        np.testing.assert_allclose(rng.probabilities[i - 1], expected.numpy())
        assert logs[i] == pytest.approx(expected[actions[i]].log().item())
    assert rng.probabilities[0][-1] > rng.probabilities[1][-1]


def test_tiny_temperature_and_large_logits_are_stable():
    actions, logs = sample_tempered_actions(
        torch.tensor([[-1e30, 1e30], [1., 2.]]), [1e-200, 0.], np.random.default_rng(1),
    )
    assert actions.tolist() == [1, 1]
    assert logs.tolist() == [0., 0.]


def test_last_readout_masks_padding_and_illegal_actions():
    scores = torch.arange(24.).reshape(2, 3, 4)
    valid = torch.tensor([[True, False, False], [True, True, False]])
    torch.testing.assert_close(last_action_scores(scores, valid, 3), scores[[0, 1], [0, 1], :3])


@pytest.mark.parametrize('temperature', [0., 0.2, 0.7, 2.0])
def test_grpo_likelihood_ratio_matches_collected_episode_policy(temperature):
    from mouse_core.objectives import GrpoObjective
    from mouse_core.sampling import tempered_policy_logits

    scores = torch.tensor([[0.2, 0.8, -0.3]] * 4, requires_grad=True)
    actions, log_probs = sample_tempered_actions(scores[:-1], [temperature] * 3, np.random.default_rng(7))
    data = {'action': torch.cat([torch.tensor([0]), actions]),
            'old_log_prob': torch.cat([torch.tensor([0.]), log_probs]),
            'advantage': torch.tensor([0., 1., -1., 1.])}
    objective = GrpoObjective(clip_ratio_min=0.8, clip_ratio_max=1.2, ent_coef=0.01, num_actions=3)
    losses, metrics = objective(
        objective_data=data, group_id=torch.zeros(4, dtype=torch.int64),
        predictions=tempered_policy_logits(scores, torch.full((4,), temperature)),
    )
    assert metrics['approx_kl'] == pytest.approx(0., abs=1e-7)
    assert metrics['clipfrac'] == 0
    losses['grpo'].backward()
    assert torch.isfinite(scores.grad).all()
    if temperature == 0:
        assert scores.grad.count_nonzero() == 0
    else:
        assert scores.grad.count_nonzero() > 0


@pytest.mark.parametrize('temperature', [-1., float('nan'), float('inf')])
def test_decode_selector_rejects_invalid_temperature(temperature):
    from mouse_core.models import select_discrete_action
    with pytest.raises(ValueError, match='temperature'):
        select_discrete_action(scores=torch.ones(1,2), temperature=temperature)


@pytest.mark.parametrize('count', [0, -1, 3, True])
def test_decode_selector_rejects_invalid_action_count(count):
    from mouse_core.models import select_discrete_action
    with pytest.raises(ValueError, match='num_actions'):
        select_discrete_action(scores=torch.ones(1,2), temperature=0., num_actions=count)


def test_last_readout_rejects_misaligned_mask():
    with pytest.raises(ValueError, match='valid must'):
        last_action_scores(torch.zeros(2,3,4),torch.ones(2,2,dtype=torch.bool),4)
    with pytest.raises(ValueError, match='valid must'):
        last_action_scores(torch.zeros(2,3,4),torch.ones(2,3),4)
