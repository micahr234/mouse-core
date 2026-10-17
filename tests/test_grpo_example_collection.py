import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from mouse_core.objectives import group_relative_advantages


def test_example_branches_share_random_state_and_use_terminal_rewards():
    path = Path(__file__).resolve().parents[1] / 'examples/08_train_online_grpo.ipynb'
    notebook = json.loads(path.read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'def rollout_branch(' in ''.join(cell['source']))
    names = {'rollout_branch', 'fork_group', 'stamp_advantages_with_prefix_len'}
    definitions: list[ast.stmt] = [node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
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
            return dict(action=action, reward=float(action) if self.steps == 60 else 0.0,
                        observation=int(self.rng.integers(100000)),
                        terminated=self.steps == 60, truncated=False, repeat_episode_ended=False)
        def close(self):
            pass

    namespace: dict[str, Any] = dict(modules=SimpleNamespace(eval=lambda: None), Model=object, BRANCHES_PER_ENV=2, copy=copy, torch=torch,
                     act_once=act_once, step_row=lambda **kw: kw['output'],
                     group_relative_advantages=group_relative_advantages)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    env = Env()
    prefix = [dict(terminated=False, truncated=False, repeat_episode_ended=False, step_index=0, reward=0)]
    branches, returns = namespace['fork_group'](
        env=env, prefix=prefix, cursor={})
    assert env.steps == 0
    assert returns == [0.0, 1.0]
    assert all(len(rows) == 61 and rows[-1]['terminated'] for rows in branches)
    assert [r['observation'] for r in branches[0][1:]] == [r['observation'] for r in branches[1][1:]]
    stamped = namespace['stamp_advantages_with_prefix_len'](
        branches=branches, returns=returns, prefix_len=1)
    assert stamped[0][0]['advantage'] == stamped[1][0]['advantage'] == 0
    assert all(r['advantage'] < 0 for r in stamped[0][1:])
    assert all(r['advantage'] > 0 for r in stamped[1][1:])


def test_example_collection_starts_new_groups_without_prior_history():
    from mouse_core.data import Datastore
    path = Path(__file__).resolve().parents[1] / 'examples/08_train_online_grpo.ipynb'
    notebook = json.loads(path.read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'def run_rollout(' in ''.join(cell['source']))
    seen = []

    def make_episode_env(seed):
        return SimpleNamespace(step=lambda action: dict(step_index=0, reward=0, seed=seed),
                               close=lambda: None)

    def fork_group(**kwargs):
        prefix = kwargs['prefix']
        assert len(prefix) == 1 and prefix[0]['step_index'] == 0
        seen.append((prefix[0]['seed'], kwargs['cursor']['episode_index']))
        return [prefix, copy.deepcopy(prefix)], [0, 0]

    namespace: dict[str, Any] = dict(modules=SimpleNamespace(eval=lambda: None), Model=object, Datastore=Datastore, SEED_OFFSET=10, BRANCHES_PER_ENV=2,
                     next_episode_index=0, stores=[], make_episode_env=make_episode_env,
                     fresh_cursor=lambda: {}, step_row=lambda **kw: kw['output'],
                     fork_group=fork_group,
                     stamp_advantages_with_prefix_len=lambda **kw: kw['branches'])
    exec(compile(source, str(path), 'exec'), namespace)
    assert namespace['run_rollout'](num_envs=2) == (2, 4)
    assert namespace['run_rollout'](num_envs=2) == (2, 4)
    assert seen == [(10, 0), (11, 1), (12, 2), (13, 3)]
    assert len(namespace['stores']) == 4


def test_example_outer_and_repeat_episode_counters_are_independent():
    for name in ['03_train_online_dqn.ipynb', '07_train_online_ppo.ipynb', '15_inference.ipynb']:
        path = Path(__file__).resolve().parents[1] / 'examples' / name
        notebook = json.loads(path.read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if 'def new_cursor(' in ''.join(cell.get('source', [])))
        definition = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'new_cursor')
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), namespace)
        cursor = namespace['new_cursor'](n=2)
        cursor['repeat_episode_index'][0] += 1
        assert cursor['episode_index'] == [0, 0]
        assert cursor['repeat_episode_index'] == [1, 0]
