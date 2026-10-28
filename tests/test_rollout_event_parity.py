"""Dataset, online-policy, and inference loops share the same event contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mouse_core.data import Datastore, EventLog, pack_token_batch
from mouse_core.objectives import group_relative_advantages
from mouse_gym import RepeatTaskTracker
from tests.test_collection_events import _collect_outputs, _notebook_transform
from tests.test_grpo_example_collection import _functions


OUTPUTS = [
    (0, 0., False, False, False, False),
    (1, -.25, False, False, False, False),
    (2, 0., True, False, False, False),
    (3, 0., False, False, False, False),
    (4, 1., True, False, True, False),
    (0, 0., False, False, False, False),
    (1, 0., False, True, False, False),
    (2, 0., False, False, False, False),
    (3, 0., False, False, False, True),
]


def assert_matching_events_and_tokens(reference, actual, *, behavior_labels=False):
    # Policy probabilities are agent-owned training labels, not environment data.
    events = actual.events
    for event in events:
        if event['provider'] == 'agent' and behavior_labels:
            assert event['payload'].pop('old_log_prob') == -.5
    assert events == reference.events
    tok = _notebook_transform('02_train_offline_dqn.ipynb', tokenizer_only=True)
    expected = tok.encode(reference)
    for source in (actual, actual.to_dataset()):
        steps = tok.encode(source)
        assert len(steps) == len(expected)
        for left, right in zip(expected, steps, strict=True):
            for key in ('token_ids', 'numeric_values', 'numeric_mask', 'head_token_mask',
                        'group_start_token_ids'):
                np.testing.assert_array_equal(getattr(left, key), getattr(right, key))
            assert left.objective_fields == right.objective_fields
        packed, _, _ = pack_token_batch(steps=steps, continuing=None)
        reference_packed, _, _ = pack_token_batch(steps=expected, continuing=None)
        np.testing.assert_array_equal(packed.token_ids, reference_packed.token_ids)
        np.testing.assert_array_equal(packed.head_step_ids, reference_packed.head_step_ids)


@pytest.mark.parametrize('oracle_values', [False, True])
@pytest.mark.parametrize('filename', ['03_train_online_dqn.ipynb', '07_train_online_ppo.ipynb', '15_inference.ipynb'])
def test_online_and_inference_events_match_dataset_collection(filename, oracle_values):
    reference, _ = _collect_outputs('01_collect_dataset.ipynb', OUTPUTS, oracle_values=oracle_values)
    actual, _ = _collect_outputs(filename, OUTPUTS, oracle_values=oracle_values, explore=False)
    assert_matching_events_and_tokens(reference, actual, behavior_labels=filename.startswith('07_'))


def test_inference_continuation_matches_one_collection_run():
    reference, _ = _collect_outputs('01_collect_dataset.ipynb', OUTPUTS)
    # Pause after an action request and after an inner ending; resume the same env.
    actual, _ = _collect_outputs('15_inference.ipynb', OUTPUTS, inference_chunks=[2, 1, 6])
    assert_matching_events_and_tokens(reference, actual)


@pytest.mark.parametrize('oracle_values', [False, True])
def test_grpo_reset_and_branch_events_match_dataset_collection(oracle_values):
    outputs = OUTPUTS[:5]
    reference, _ = _collect_outputs('01_collect_dataset.ipynb', outputs, oracle_values=oracle_values)

    class Env:
        def __init__(self):
            self.index = 0

        def step(self, action):
            assert action == (None if self.index in (0, 3) else 2)
            obs, reward, inner_terminal, inner_truncated, terminal, truncated = outputs[self.index]
            self.index += 1
            return ({'observation': obs, 'episode_terminated': inner_terminal,
                     'episode_truncated': inner_truncated}, reward, terminal, truncated,
                    {'q_star': [0., 0., 1., 0.]} if oracle_values else {})

        def close(self):
            pass

    namespace = dict(EventLog=EventLog, RepeatTaskTracker=RepeatTaskTracker,
                     Datastore=Datastore, torch=torch, group_relative_advantages=group_relative_advantages,
                     modules=SimpleNamespace(eval=lambda: None), make_episode_env=lambda **kwargs: Env(),
                     act_once=lambda **kwargs: (2, -.5), SEED_OFFSET=0, BRANCHES_PER_ENV=1,
                     stores=[], branch_advantages=[], next_episode_index=0)

    def fork_group(*, env, log, tracker, branches_per_env):
        events, reward = namespace['rollout_branch'](env=env, log=log, tracker=tracker)
        assert env.index == len(outputs)
        return [events], [reward]

    namespace['fork_group'] = fork_group
    _functions({'run_rollout', 'rollout_branch', 'stamp_advantages_with_prefix_len'}, namespace)
    assert namespace['run_rollout'](num_envs=1) == (1, 1)
    actual = EventLog.from_dataset(namespace['stores'][0].to_dataset())
    assert_matching_events_and_tokens(reference, actual, behavior_labels=True)
