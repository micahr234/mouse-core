import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from mouse_core.data import Datastore, EventLog, environment_decisions, latest_environment_decision
from mouse_gym import RepeatTaskTracker
from mouse_core.objectives import group_relative_advantages


PATH = Path(__file__).resolve().parents[1] / 'examples/08_train_online_grpo.ipynb'


def _functions(names, namespace):
    notebook = json.loads(PATH.read_text())
    definitions: list[ast.stmt] = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
        for node in ast.parse(''.join(cell['source'])).body
        if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PATH), 'exec'), namespace)


def _reset_log():
    log = EventLog()
    log.event(provider='coordinator', episode_index=0, step_index=0, payload={
        'episode_start': True,
    })
    log.event(provider='environment', episode_index=0, step_index=0, payload={
        'observation': 0,
        'reward': 0.0,
        'inner_terminated': False,
        'inner_truncated': False,
        'terminated': False,
        'truncated': False,
        'action_requested': True,
    })
    return log


def _reset_tracker():
    tracker = RepeatTaskTracker()
    tracker.update(output=({'observation': 0, 'episode_terminated': False,
                          'episode_truncated': False}, 0., False, False, {}))
    return tracker


def test_example_branches_share_random_state_and_use_terminal_rewards():
    policy_calls = 0

    def act_once(**kwargs):
        nonlocal policy_calls
        action = policy_calls // 60
        policy_calls += 1
        return action, -0.69

    class Env:
        def __init__(self):
            self.rng = np.random.default_rng(42)
            self.steps = 0

        def step(self, action):
            self.steps += 1
            terminal = self.steps == 60
            return ({'observation': int(self.rng.integers(100000)),
                     'episode_terminated': terminal, 'episode_truncated': False},
                    float(action) if terminal else 0., terminal, False, {})

        def close(self):
            pass

    namespace: dict[str, Any] = dict(modules=SimpleNamespace(eval=lambda: None), EventLog=EventLog, RepeatTaskTracker=RepeatTaskTracker,
        latest_environment_decision=latest_environment_decision,
        copy=copy, torch=torch, act_once=act_once, group_relative_advantages=group_relative_advantages)
    _functions({'rollout_branch', 'fork_group', 'stamp_advantages_with_prefix_len'}, namespace)
    env = Env()
    log = _reset_log()
    tracker = _reset_tracker()
    branches, returns = namespace['fork_group'](env=env, log=log, tracker=tracker, branches_per_env=2)
    assert env.steps == 0
    assert len(log) == 2 and tracker.action_requested and tracker.step_index == 0
    assert tracker.tasks == tracker.episodes == ()
    assert returns == [0.0, 1.0]
    assert all(len(events) == 122 and events[-1]['payload']['terminated'] for events in branches)
    event_branches = branches
    branches = [list(environment_decisions(events)) for events in branches]
    assert [r['observation'] for r in branches[0][1:]] == [r['observation'] for r in branches[1][1:]]
    for action, branch in enumerate(branches):
        assert 'action' not in branch[0]
        assert all(row['action'] == action and row['old_log_prob'] == -.69 for row in branch[1:])
        assert all(row.get('reward', 0.) == 0. for row in branch[:-1])
        replay = EventLog()
        for row in branch:
            for event in json.loads(row['events']):
                replay.event(**event)
        assert list(environment_decisions(replay.to_dataset())) == branch
    stamped = namespace['stamp_advantages_with_prefix_len'](branches=event_branches, returns=returns, prefix_len=1)
    assert stamped[0][0]['advantage'] == stamped[1][0]['advantage'] == 0
    assert all(r['advantage'] < 0 for r in stamped[0][2:])
    assert all(r['advantage'] > 0 for r in stamped[1][2:])


def test_example_collection_starts_new_groups_without_prior_history():
    seen = []

    def make_episode_env(seed):
        return SimpleNamespace(step=lambda action: ({'observation': seed,
            'episode_terminated': False, 'episode_truncated': False}, 0., False, False, {}),
            close=lambda: None)

    def fork_group(**kwargs):
        log = kwargs['log']
        prefix = list(environment_decisions(log.to_dataset()))
        assert len(log) == 2 and len(prefix) == 1 and prefix[0]['episode_start']
        seen.append((prefix[0]['observation'], prefix[0]['episode_index']))
        return [log.events, log.events], [0, 0]

    namespace: dict[str, Any] = dict(modules=SimpleNamespace(eval=lambda: None), EventLog=EventLog, RepeatTaskTracker=RepeatTaskTracker,
        latest_environment_decision=latest_environment_decision,
        Datastore=Datastore, SEED_OFFSET=10, BRANCHES_PER_ENV=2, next_episode_index=0, stores=[],
        make_episode_env=make_episode_env, torch=torch, json=json, fork_group=fork_group,
        stamp_advantages_with_prefix_len=lambda **kw: [
            [{**row, 'advantage': 0.} for row in branch] for branch in kw['branches']])
    _functions({'run_rollout'}, namespace)
    assert namespace['run_rollout'](num_envs=2) == (2, 4)
    assert namespace['run_rollout'](num_envs=2) == (2, 4)
    assert seen == [(10, 0), (11, 0), (12, 0), (13, 0)]
    assert len(namespace['stores']) == 4


def test_example_branch_logs_only_executed_actions_and_repeat_boundaries():
    outputs = iter([
        ({'observation': 1, 'episode_terminated': True, 'episode_truncated': False}, 0., False, False, {}),
        ({'observation': 2, 'episode_terminated': False, 'episode_truncated': False}, 0., False, False, {}),
        ({'observation': 3, 'episode_terminated': False, 'episode_truncated': True}, 1., False, True, {}),
    ])
    actions = []
    log = _reset_log()

    def step(action):
        actions.append(action)
        if action is not None:
            assert log.last_event['provider'] == 'agent'
        return next(outputs)

    namespace: dict[str, Any] = dict(modules=SimpleNamespace(eval=lambda: None), EventLog=EventLog, RepeatTaskTracker=RepeatTaskTracker,
        latest_environment_decision=latest_environment_decision,
                                    act_once=lambda **kw: (2, -.5))
    _functions({'rollout_branch'}, namespace)
    env = SimpleNamespace(step=step)
    rows, ret = namespace['rollout_branch'](env=env, log=log, tracker=_reset_tracker())
    rows = list(environment_decisions(rows))
    assert actions == [2, None, 2]
    assert ret == 1.
    assert all('repeat_episode_index' not in row for row in rows)
    assert [row.get('action') for row in rows] == [None, 2, None, 2]
    assert ['old_log_prob' in row for row in rows] == [False, True, False, True]
    assert [row.get('reward', 0.) for row in rows] == [0., 0., 0., 1.]
