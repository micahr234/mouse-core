"""The existing collectors write the sole source format, without Hub access."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from mouse_core.data import Augmenter, Datastore, EventLog, Tokenizer, compose, pack_token_batch
from mouse_core.data.event_transform import environment_decisions, latest_environment_decision
from mouse_gym import RepeatTaskTracker
from mouse_core.models import action_readouts, select_discrete_action
from mouse_core.models.kv_policy import cache_needs_rebuild, rebuild_starts, resolve_cache_bounds
from mouse_core.objectives import DqnObjective, boundary_discount


EXAMPLES = Path(__file__).parents[1] / 'examples'
FROZENLAKE_COLLECTORS = [
    '01_collect_dataset.ipynb', '03_train_online_dqn.ipynb',
    '07_train_online_ppo.ipynb', '15_inference.ipynb',
]
FROZENLAKE_REWARD_TRAINING = [
    '02_train_offline_dqn.ipynb', '03_train_online_dqn.ipynb',
    '07_train_online_ppo.ipynb', '10_train_offline_reasoning_dqn.ipynb',
    '13_train_offline_n_step_dqn.ipynb', '14_train_offline_value_gap_dqn.ipynb',
    '21_train_dqn_cosine.ipynb',
]


def _collect_outputs(filename, outputs, *, oracle_values=True, inference_chunks=None, explore=True):
    """Run the notebook's actual rollout loop against controlled environment outputs."""
    notebook = json.loads((EXAMPLES / filename).read_text())
    names = {'run_rollout', 'run_eval'}
    definitions: list[ast.stmt] = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
        for node in ast.parse(''.join(cell['source'])).body
        if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace: dict[str, Any] = {}

    class ActionSpace(gym.spaces.Discrete):
        def sample(self, mask=None, probability=None):
            return np.int64(2)

    class Env:
        names = ['test']
        action_space = SimpleNamespace(sample=lambda: (2,), spaces=[ActionSpace(4)])
        envs = [SimpleNamespace(render=lambda: None)]

        def __init__(self):
            self.index = 0
            self.requested = False
            self.envs = [SimpleNamespace(
                render=lambda: None,
                action_space=self.action_space.spaces[0],
                step=lambda action: self.step((action,))[0],
            )]

        def step(self, actions):
            log = namespace['logs'][0]
            assert all(event['payload'] for event in log if event['provider'] == 'coordinator')
            if self.index:
                previous = next(event for event in reversed(log.events) if event['provider'] == 'environment')
                assert previous['payload'].get('action_requested', False) is self.requested
            # Agent events are recorded when the action is selected, before stepping.
            if self.requested:
                assert log.last_event['provider'] == 'agent'
                assert log.last_event['payload']['action'] == int(actions[0]) == 2
            else:
                new_task = self.index == 0 or any(outputs[self.index - 1][-2:])
                assert log.last_event['provider'] == ('coordinator' if new_task else 'environment')
            step, reward, inner_terminal, inner_truncated, terminal, truncated = outputs[self.index]
            self.index += 1
            self.requested = not (inner_terminal or inner_truncated or terminal or truncated)
            return [({'observation': step, 'episode_terminated': inner_terminal,
                      'episode_truncated': inner_truncated}, reward, terminal, truncated,
                     {'q_star': np.array([0., 0., 1., 0.])} if oracle_values else {})]

        def close(self):
            pass

    env = Env()
    cache = SimpleNamespace(close=lambda: None, reset_rows=lambda **kw: None)
    def encoder(batch, **kwargs):
        head_steps = np.unique(batch.head_step_ids)
        counts = np.bincount(batch.step_sequence_ids[head_steps], minlength=batch.B)
        return {'z': torch.tensor([0., 0., 1., 0.]).expand(batch.B, int(counts.max()), 4),
                'cache': cache}

    namespace.update(EventLog=EventLog, RepeatTaskTracker=RepeatTaskTracker, gym=gym,
        latest_environment_decision=latest_environment_decision, torch=torch, np=np, json=json,
        env=env, eval_env=env, logs=[EventLog()], trackers=[RepeatTaskTracker()],
        stores=[Datastore(name='test')], contexts=[[]], eval_contexts=[[]],
        modules=SimpleNamespace(eval=lambda: None), device=torch.device('cpu'),
        MAX_CONTEXT=32, MAX_ACTIONS=4, MAX_STEPS_PER_EPISODE=3,
        REPEAT_COUNT=2, ORACLE_PROB=0.5 if oracle_values else 0., EPISODES_PER_ENV=1,
        epsilon_for_env_step=lambda **kw: float(explore), make_rollout_env=lambda **kw: env,
        encoder=encoder, q_head=lambda z: z, policy_head=lambda z: z, value_head=lambda z: z,
        tokenizer=_notebook_transform('02_train_offline_dqn.ipynb', tokenizer_only=True),
        select_discrete_action=select_discrete_action, action_readouts=action_readouts,
        sample_discrete_action=lambda **kw: (torch.tensor([2]), torch.tensor([-.5])),
        cache_needs_rebuild=cache_needs_rebuild, rebuild_starts=rebuild_starts,
        resolve_cache_bounds=resolve_cache_bounds)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, 'exec'), namespace)
    if filename.startswith('01_'):
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if cell['cell_type'] == 'code' and 'outputs = env.step(tuple(actions))' in ''.join(cell['source']))
        # Continuing collection mid-task preserves the event stream.
        split = len(outputs) // 2
        for steps in (split, len(outputs) - split):
            namespace['STEPS_PER_ENV'] = steps
            exec(compile(source, filename, 'exec'), namespace)
    elif filename.startswith('15_'):
        scores = [[]]
        for count in inference_chunks or [len(outputs)]:
            _, _, chunk_scores = namespace['run_eval'](
                num_steps=count, max_cache=4, start_cache=2, temperature=0.)
            scores[0].extend(chunk_scores[0])
        expected_scores = []
        task_reward = 0.
        for _, reward, _, _, terminal, truncated in outputs:
            task_reward += reward
            if terminal or truncated:
                expected_scores.append(task_reward)
                task_reward = 0.
        assert scores == [expected_scores]
    else:
        complete_episodes = sum(terminal or truncated for _, _, _, _, terminal, truncated in outputs)
        namespace['run_rollout'](env_steps=0, num_episodes=complete_episodes, max_cache=32)
        assert env.index == len(outputs)
    log = namespace['logs'][0]
    if filename.startswith(('03_', '07_')):
        assert EventLog.from_dataset(namespace['stores'][0].to_dataset()).events == log.events
        episode = log.last_event['episode_index']
        assert namespace['contexts'][0] == [event for event in log if event['episode_index'] == episode]
    if filename.startswith('15_'):
        history = [event for step in namespace['eval_contexts'][0] for event in step]
        assert history == log.events[-len(history):]
    return log, list(environment_decisions(log.to_dataset()))


@pytest.mark.parametrize('ending', [
    (True, False, False, False), (False, True, False, False),
    (False, False, True, False), (False, False, False, True),
])
def test_collector_logs_oracle_values_only_when_an_action_is_requested(ending):
    log, rows = _collect_outputs('01_collect_dataset.ipynb', [
        (0, 0., False, False, False, False),
        (1, 1., *ending),
        (2, 0., False, False, False, False),
    ])
    for row in rows:
        assert ('q_star' in row) == row.get('action_requested', False)
    payloads = [event['payload'] for event in log if event['provider'] == 'environment']
    assert ['q_star' in payload for payload in payloads] == [True, False, True]
    assert [payload.get('repeat_episode_index') for payload in payloads] == [
        0, None, 0 if ending[2] or ending[3] else 1,
    ]


@pytest.mark.parametrize(('probability', 'expected_actions'), [
    (0., [0, 0, 0, 0, 0]),
    (.5, [2, 0, 2, 0, 2]),
    (1., [2, 2, 2, 2, 2]),
])
def test_collector_uses_constant_oracle_probability_only_for_requested_actions(probability, expected_actions):
    notebook = json.loads((EXAMPLES / '01_collect_dataset.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'outputs = env.step(tuple(actions))' in ''.join(cell['source']))
    log = EventLog()
    draws = []
    random_samples = []
    random_values = iter([.25, .75, .25, .75, .25])

    def random_draw(size):
        assert size == 1 and namespace["env"].requested
        value = next(random_values)
        draws.append(value)
        return torch.tensor([value])

    def sample_random_action():
        assert namespace["env"].requested and draws[-1] >= probability
        random_samples.append(len(draws) - 1)
        return np.int64(0)

    class Env:
        action_space = SimpleNamespace(spaces=[SimpleNamespace(sample=sample_random_action)])

        def __init__(self):
            self.index = 0
            self.requested = False

        def close(self):
            pass

        def step(self, actions):
            index = self.index
            self.index += 1
            reset = index in (0, 3, 5, 7)
            if reset:
                assert actions == (None,)
                assert log.last_event['provider'] == ('coordinator' if index in (0, 5) else 'environment')
            else:
                assert type(actions[0]) is int
                assert log.last_event['provider'] == 'agent'
                assert log.last_event['payload']['action'] == actions[0]
            inner_end = index in (2, 4, 6, 8)
            outer_end = index in (4, 8)
            self.requested = not (inner_end or outer_end)
            return [({'observation': index, 'episode_terminated': inner_end,
                      'episode_truncated': False}, 0., outer_end, False,
                     {'q_star': np.array([0., 0., 1., 0.])})]

    namespace: dict[str, Any] = dict(env=Env(), logs=[log], trackers=[RepeatTaskTracker()], stores=[Datastore(name='oracle')],
        RepeatTaskTracker=RepeatTaskTracker, ORACLE_PROB=probability, REPEAT_COUNT=2,
        torch=SimpleNamespace(rand=random_draw), np=np, json=json)
    for steps in (2, 3, 4):
        namespace['STEPS_PER_ENV'] = steps
        exec(compile(source, '<collector>', 'exec'), namespace)
    assert len(draws) == 5
    assert random_samples == [i for i, action in enumerate(expected_actions) if action == 0]
    decisions = list(environment_decisions(log.to_dataset()))
    assert [row.get('action') for row in decisions] == [
        None, expected_actions[0], expected_actions[1], None, expected_actions[2],
        None, expected_actions[3], None, expected_actions[4],
    ]
    assert [row.get('repeat_episode_index') for row in decisions] == [
        0, None, None, 1, None, 0, None, 1, None,
    ]


def _frozenlake_episode(filename):
    return _collect_outputs(filename, [
        (0, 0., False, False, False, False),
        (1, 0., False, False, False, False),
        (2, 1., True, False, True, False),
    ])


@pytest.mark.parametrize('filename', FROZENLAKE_COLLECTORS)
def test_collectors_save_repeat_indices_only_on_inner_resets(filename):
    log, live_rows = _collect_outputs(filename, [
        (0, 0., False, False, False, False),
        (1, 0., False, False, False, False),
        (2, 0., True, False, False, False),
        (3, 0., False, False, False, False),
        (4, 1., True, False, True, False),
        (0, 0., False, False, False, False),
        (1, 0., False, True, False, False),
        (2, 0., False, False, False, False),
        (3, 0., False, True, False, True),
    ])
    dataset = log.to_store().to_dataset()
    persisted = EventLog.from_dataset(dataset)
    outputs = [event['payload'] for event in persisted.events if event['provider'] == 'coordinator']
    replayed = list(environment_decisions(dataset))
    assert outputs == [{'episode_start': True}, {'episode_start': True}]
    assert persisted.events == log.events
    assert all('repeat_episode_index' not in event['payload']
               for event in persisted if event['provider'] != 'environment')
    assert [event['payload'].get('repeat_episode_index') for event in persisted
            if event['provider'] == 'environment'] == [0, None, None, 1, None, 0, None, 1, None]
    for rows in (live_rows, replayed):
        assert [row.get('repeat_episode_index') for row in rows] == [
            0, None, None, 1, None, 0, None, 1, None,
        ]
        assert all(('repeat_episode_index' in row) == ('action' not in row) for row in rows)
        assert all('repeat_episode_ended' not in row for row in rows)
        assert all('episode_reset' not in row for row in rows)
    assert all('repeat_episode_start' not in row for row in outputs)
    for rows in (live_rows, replayed):
        assert [row.get('action') for row in rows] == [None, 2, 2, None, 2, None, 2, None, 2]
        assert [row['episode_index'] for row in rows] == [0, 0, 0, 0, 0, 1, 1, 1, 1]
        assert [row['step_index'] for row in rows] == [0, 1, 2, 3, 4, 0, 1, 2, 3]
        for row in rows:
            assert all((event['episode_index'], event['step_index']) ==
                       (row['episode_index'], row['step_index'])
                       for event in json.loads(row['events']))


@pytest.mark.parametrize('filename', FROZENLAKE_COLLECTORS)
def test_frozenlake_collectors_omit_zero_rewards_and_false_flags(filename):
    log, live_rows = _frozenlake_episode(filename)
    persisted = EventLog.from_dataset(log.to_store().to_dataset())
    outputs = [event['payload'] for event in persisted.events if event['provider'] == 'environment']
    assert ['reward' in output for output in outputs] == [False, False, True]
    assert outputs[-1]['reward'] == 1.
    for event in persisted:
        for field in ('episode_start', 'terminated', 'truncated',
                      'inner_terminated', 'inner_truncated'):
            if field in event['payload']:
                assert event['payload'][field] is True
    assert [row.get('reward', 0.) for row in live_rows] == [0., 0., 1.]
    assert [row.get('action') for row in environment_decisions(persisted.to_dataset())] == [None, 2, 2]


class _CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([[ord(char) for char in text]])}


def _notebook_transform(filename, *, tokenizer_only=False):
    """Build each actual example's field pipeline without model downloads or I/O."""
    notebook = json.loads((EXAMPLES / filename).read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'group_prefix = ' in ''.join(cell['source']))
    assignments = {'group_prefix', 'augmenter', 'tokenizer', 'train_transform', 'transform'}
    body: list[ast.stmt] = [node for node in ast.parse(source).body if
            isinstance(node, ast.FunctionDef) and
            (node.name.startswith('on_') or node.name in {'frozenlake_fields', 'behavior_fields'})
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                                                    and target.id in assignments for target in node.targets)]

    def tokenizer_factory(**kwargs):
        sources = kwargs.pop('tokenizers')
        return Tokenizer(tokenizers={name: _CharacterTokenizer() for name in sources}, **kwargs)

    namespace: dict[str, Any] = dict(Augmenter=Augmenter, Tokenizer=tokenizer_factory, compose=compose,
                                    MAX_ACTIONS=4, MAX_OBS_DISCRETE=64, MAX_STEPS_PER_EPISODE=3,
                                    REPEAT_COUNT=1, PRETRAINED='unused', TOKENIZER_SOURCES={'text': 'unused'})
    exec(compile(ast.Module(body=body, type_ignores=[]), filename, 'exec'), namespace)
    if tokenizer_only:
        return namespace['tokenizer']
    return namespace['train_transform'] if 'train_transform' in namespace else namespace['transform']


@pytest.mark.parametrize('filename', [
    *FROZENLAKE_REWARD_TRAINING, '05_train_offline_sv.ipynb',
    '08_train_online_grpo.ipynb', '09_train_offline_sp.ipynb',
])
@pytest.mark.parametrize('ending', ['terminated', 'truncated'])
def test_tokenizers_render_saved_repeat_indices_and_preserve_action_masks(filename, ending):
    terminated, truncated = ending == 'terminated', ending == 'truncated'
    log, _ = _collect_outputs('01_collect_dataset.ipynb', [
        (0, 0., False, False, False, False),
        (1, 0., False, False, False, False),
        (2, 0., terminated, truncated, False, False),
        (3, 0., False, False, False, False),
        (4, 1., terminated, truncated, terminated, truncated),
    ])
    # A saved index determines the header even when it cannot be inferred by counting endings.
    saved = EventLog()
    reset_indices = iter((7, 12))
    for event in log:
        payload = event['payload']
        if 'repeat_episode_index' in payload:
            payload = {**payload, 'repeat_episode_index': next(reset_indices)}
        saved.event(**{**event, 'payload': payload})
    log = saved
    before = json.dumps(log.events, sort_keys=True)
    dataset = log.to_store().to_dataset()
    persisted = EventLog.from_dataset(dataset)
    rows = list(environment_decisions(log.to_dataset()))
    transform = _notebook_transform(filename)
    for row in rows:
        if 'grpo' in filename:
            row.update(advantage=1., old_log_prob=row.get('old_log_prob', 0.))
    steps = transform(log)
    assert [int(step.head_token_mask.sum()) for step in steps] == [1, 1, 0, 1, 0]

    def markers(tokens):
        return re.findall(r'episode \d+\n', ''.join(chr(int(token)) for token in tokens))

    assert [markers(step.token_ids) for step in steps] == [
        ['episode 8\n'], [], [], ['episode 13\n'], [],
    ]
    rendered = [''.join(chr(int(token)) for token in step.token_ids) for step in steps]
    assert re.search(r'episode 8\n,\d+\n$', rendered[0])
    assert re.fullmatch(r'episode 13\n,\d+\n', rendered[3])
    assert rendered[2].endswith(f',{ending}\n\n')
    assert rendered[4].endswith(f',{ending}\n\n')
    assert [row.get('repeat_episode_index') for row in rows] == [7, None, None, 12, None]
    assert json.dumps(log.events, sort_keys=True) == before
    assert json.dumps(persisted.events, sort_keys=True) == before
    assert [row.get('repeat_episode_index') for row in environment_decisions(dataset)] == [
        7, None, None, 12, None,
    ]
    replay_steps = _notebook_transform(filename)(persisted)
    assert [markers(step.token_ids) for step in replay_steps] == [markers(step.token_ids) for step in steps]
    assert [int(step.head_token_mask.sum()) for step in replay_steps] == [1, 1, 0, 1, 0]
    assert json.dumps(persisted.events, sort_keys=True) == before
    for start in (0, 1):
        batch, labels, _ = pack_token_batch(steps=steps[start:], sequence_ids=[0] * (len(steps) - start),
                                            batch_size=1, continuing=None)
        assert markers(batch.token_ids) == (['episode 8\n', 'episode 13\n'] if start == 0 else ['episode 13\n'])
        if 'cosine' in filename:
            assert 'episode_reset' not in labels
            assert labels['action_executed'].tolist() == [False, True, True, False, True][start:]
        if 'sp' in filename:
            # Supervised labels describe the resulting observation, not its incoming request.
            active = ~(labels['terminated'] | labels['truncated']
                       | labels['inner_terminated'] | labels['inner_truncated']).bool()
            assert active.tolist() == [True, True, False, True, False][start:]


@pytest.mark.parametrize('filename', FROZENLAKE_REWARD_TRAINING)
def test_rewards_preserve_replay_tokens_labels_and_dqn_updates(filename):
    log, rows = _frozenlake_episode('01_collect_dataset.ipynb')
    replayed = EventLog.from_dataset(log.to_dataset())
    encode = _notebook_transform(filename)
    transform = encode

    def batch(source):
        if hasattr(transform, "reseed"):
            transform.reseed(generation=7)
        return pack_token_batch(steps=transform(source), sequence_ids=[0, 0, 0], batch_size=1, continuing=None)

    inputs, labels, groups = batch(log)
    replay_inputs, replay_labels, replay_groups = batch(replayed.to_dataset())
    np.testing.assert_array_equal(inputs.token_ids, replay_inputs.token_ids)
    assert labels['reward'].tolist() == [0., 0., 1.]
    for key in labels:
        torch.testing.assert_close(labels[key], replay_labels[key])
    torch.testing.assert_close(groups, replay_groups)
    assert 'reward' not in rows[0]
    assert encode(log)[0].objective_fields['reward'] == 0.

    objective = DqnObjective(reward=None, value=None, gate=None, temperature=0.,
        discount=boundary_discount(gamma_step=.9, gamma_terminated=0., gamma_truncated=0.),
        cross_group_backups='fault')
    predictions = torch.arange(inputs.P * 4, dtype=torch.float32).reshape(inputs.P, 4).requires_grad_()
    backup = predictions.detach() / 3
    losses, _ = objective(objective_data=labels, group_id=groups, predictions=predictions,
        backup_value_predictions=backup, backup_action_predictions=backup)
    assert torch.isfinite(losses['action_value'])
    losses['action_value'].backward()
    assert predictions.grad is not None and predictions.grad.abs().sum() > 0


@pytest.mark.parametrize('filename', [
    '05_train_offline_sv.ipynb', '08_train_online_grpo.ipynb', '09_train_offline_sp.ipynb',
])
def test_other_frozenlake_tokenizers_encode_events(filename):
    log, _ = _frozenlake_episode('01_collect_dataset.ipynb')
    transform = _notebook_transform(filename)
    inputs, labels, _ = pack_token_batch(steps=transform(log.to_dataset()),
        sequence_ids=[0, 0, 0], batch_size=1, continuing=None)
    assert inputs.N == 3
    expected = {'action', 'advantage', 'old_log_prob'} if 'grpo' in filename else {'q_star'}
    assert expected <= labels.keys()
    if 'grpo' not in filename:
        assert torch.isfinite(labels['q_star'][:2]).all()
        assert torch.isneginf(labels['q_star'][-1]).all()
    if 'sp' in filename:
        active = ~(labels['terminated'] | labels['truncated']
                   | labels['inner_terminated'] | labels['inner_truncated']).bool()
        assert active.tolist() == [True, True, False]


@pytest.mark.parametrize('filename', ['01_collect_dataset.ipynb', '17_collect_cartpole.ipynb'])
def test_existing_collector_records_events(filename):
    notebook = json.loads((Path(__file__).parents[1]/'examples'/filename).read_text())
    namespace = {}
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if 'url = push_stores_to_hub' in source:
                continue
            module = ast.parse(source)
            # Collect with test-owned limits; never publish.
            exec(compile(module, filename, 'exec'), namespace)
            namespace.update(NUM_ENVS=1, REPEAT_COUNT=2, MAX_STEPS_PER_EPISODE=3, STEPS_PER_ENV=12)
        for store in namespace['stores']:
            dataset = store.to_dataset()
            assert set(dataset.column_names) == {'provider', 'payload', 'episode_index', 'step_index'}
            log = EventLog.from_dataset(dataset)
            decisions = list(environment_decisions(dataset))
            assert len(decisions) == 12
            assert len(log) > len(decisions)
            assert 'action' not in decisions[0]
            agents = [e['payload'] for e in log.events if e['provider'] == 'agent']
            assert all(payload['action'] is not None for payload in agents)
            assert all(not {'episode_index', 'step_index'} & payload.keys() for payload in agents)
            assert all('action' not in e['payload'] for e in log.events if e['provider'] == 'environment')
            if filename.startswith('17_'):
                assert all({'cart_position', 'pole_angle'} <= row.keys() for row in decisions)
                assert all('observation' not in row for row in decisions)
                assert all(not any(key.startswith('repeat_episode') for key in row) for row in decisions)
                transform = _notebook_transform('18_train_offline_cartpole_dqn.ipynb')
                inputs, labels, _ = pack_token_batch(
                    steps=transform(dataset),
                    sequence_ids=[row['episode_index'] for row in decisions],
                    batch_size=len({row['episode_index'] for row in decisions}), continuing=None,
                )
                assert int(inputs.numeric_mask.sum()) == 2 * len(decisions)
                # Physical action ids are preserved by the training pipeline.
                assert labels['action'].tolist() == [row.get('action') or 0 for row in decisions]

    finally:
        if 'env' in namespace: namespace['env'].close()


def test_cartpole_observation_wrapper_hides_velocities():
    notebook = json.loads((EXAMPLES / '17_collect_cartpole.ipynb').read_text())
    factory = next(node for cell in notebook['cells'] if cell['cell_type'] == 'code'
        for node in ast.parse(''.join(cell['source'])).body
        if isinstance(node, ast.FunctionDef) and node.name == 'make_cartpole')
    namespace: dict[str, Any] = {'gym': gym}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), '<cartpole>', 'exec'), namespace)
    env = namespace['make_cartpole']()
    try:
        observation, _ = env.reset(seed=42)
        assert env.observation_space.shape == (2,)
        np.testing.assert_allclose(observation, np.asarray(env.unwrapped.state)[[0, 2]])
        for action in (0, 1, 0):
            observation, _, _, _, _ = env.step(action)
            np.testing.assert_allclose(observation, np.asarray(env.unwrapped.state)[[0, 2]])
    finally:
        env.close()
