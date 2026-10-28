from tests._event_fixtures import measurement_events
"""Exercise the actual notebook encoders, separate heads, and training loops."""
import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mouse_core import AdamW, polyak_update
from mouse_core.data import (environment_episode_ranges, DataLoader, EventLog, Tokenizer, to_device,
                             compose, environment_decisions, latest_environment_decision)
from mouse_core.data.token_batch import TokenBatch
from mouse_gym import RepeatTaskTracker
from mouse_core.models import (
    Backbone, Embedding, MultiEmbedding, DecodeCache, LatentReasoner, MLP, Model,
    DiscretizedActionHead, prepare_decode, select_discrete_action, action_readouts,
    load_model, save_model, sample_reasoning_splits,
)
from mouse_core.objectives import (
    DqnObjective, SvObjective, SpObjective, PpoObjective, GrpoObjective,
    CosineObjective, best_action, boundary_discount, nstep_gate,
    value_gap_gate, general_gate,
)
from tests._token_batch_helpers import batch_to_token_batch, token_tokenizer

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NOTEBOOKS = [path for path in sorted(EXAMPLES.glob("*.ipynb"))
             if "class Encoder(Model):" in path.read_text()]


def _tokens(*actions):
    return replace(batch_to_token_batch(token_tokenizer("action"), [[{"action": action} for action in actions]]),
                   tokenizer_names=("text",))


def _example(path, **overrides):
    """Use the notebook's construction code with small local backend factories."""
    notebook = json.loads(path.read_text())
    namespace: dict[str, Any] = {**globals(), "device": torch.device("cpu")}
    # Preserve symbolic settings; no network, environment, or publication calls.
    for node in ast.parse("".join(notebook["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    namespace.update(
        PRETRAINED="local-test", TOKENIZER_SOURCES={"text": "local-test"}, MAX_ACTIONS=3, HIDDEN_DIM=8, BACKBONE_LAYERS=1,
        BACKBONE_HEADS=2, NUM_FREQUENCIES=2, NUM_THOUGHTS=2,
        vocabulary={str(i): i for i in range(32)},
        text_tokenizer=list(range(32)),
        action_vectors=torch.tensor([[-2.], [0.], [2.]]),
    )
    namespace.update(overrides)
    namespace["tokenizer"] = SimpleNamespace(tokenizer_sources=namespace["TOKENIZER_SOURCES"])
    construction = []

    def embedding_factory(**kwargs):
        construction.append(("embedding", kwargs))
        return Embedding(hidden_dim=8, vocab_size=32,
                         num_frequencies=kwargs.get("num_frequencies"))

    def backbone_factory(**kwargs):
        construction.append(("backbone", kwargs))
        return Backbone(architecture="qwen3", hidden_dim=8, num_layers=1, num_heads=2,
                        train_kernel="reference", decode_kernel="flex", dtype=torch.float32)

    namespace.update(Embedding=embedding_factory, Backbone=backbone_factory)
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "class Encoder(Model):" in "".join(cell["source"]))
    body = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            body.append(node)
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "tokenizer" for t in node.targets):
                continue
            body.append(node)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    table_count = 1 if path.stem.startswith(("19_", "20_")) else len(namespace["TOKENIZER_SOURCES"])
    assert [name for name, _ in construction] == ["embedding"] * table_count + ["backbone"]
    if "modules" not in namespace:  # cosine example saves separate tracker dicts
        namespace["modules"] = nn.ModuleDict({
            name: namespace[name] for name in ("encoder", "q_head", "transition")
        })
    return notebook, namespace


def _heads(ns):
    if "q_heads" in ns:
        return list(ns["q_heads"].values())
    return [ns[name] for name in ("q_head", "policy_head", "value_head") if name in ns]


def test_example_encoder_builds_and_trains_each_named_tokenizer_table():
    from tests._token_batch_helpers import IntIdTokenizer
    _, ns = _example(EXAMPLES / '02_train_offline_dqn.ipynb',
                     TOKENIZER_SOURCES={'text': 'text-model', 'actions': 'action-model'})
    encoder = ns['encoder']
    assert set(encoder.embedding.embeddings) == {'text', 'actions'}
    tokenizer = Tokenizer(tokenizers={'text': IntIdTokenizer(), 'actions': IntIdTokenizer()}, input_fields=[
        dict(type='text', tokenizer='actions', input_field='action', format='{field}'),
        dict(type='text', tokenizer='text', input_field='observation', format='{field}', head_output=True),
    ])
    inputs = tokenizer.pack_rows(rows=[measurement_events([{'action': 2, 'observation': 3}])], continuing=None)
    ns['q_head'](encoder(inputs)['z']).square().sum().backward()
    for table in encoder.embedding.embeddings.values():
        assert table.embed_tokens.weight.grad is not None
        assert table.embed_tokens.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_encoders_and_heads_have_separate_gradients_and_cached_readouts(path):
    _, ns = _example(path)
    encoder = ns["encoder"]
    encoder.eval()
    assert set(dict(encoder.named_children())) == (
        {"embedding", "backbone", "reasoner"} if "reasoning" in path.stem
        else {"embedding", "backbone"}
    )
    assert not {id(p) for p in encoder.embedding.parameters()} & {id(p) for p in encoder.backbone.parameters()}
    tokens = _tokens(0, 1)
    features = []
    hook = encoder.backbone.register_forward_hook(lambda module, args, output: features.append(output))
    try:
        out = encoder(tokens)
    finally:
        hook.remove()
    assert set(out) == {"z", "cache"}
    torch.testing.assert_close(out["z"], features[-1][tokens.head_token_ids])
    heads = _heads(ns)
    torch.stack([head(out["z"]).square().mean() for head in heads]).sum().backward()
    for module in (encoder.embedding, encoder.backbone, *heads):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with torch.no_grad():
        first = encoder(_tokens(0), use_cache=True)
        try:
            last = encoder(_tokens(1), cache=first["cache"], use_cache=True)
            full_z = torch.cat([first["z"], last["z"]], dim=1).squeeze(0)
            torch.testing.assert_close(full_z, out["z"], atol=2e-5, rtol=2e-5)
            for head in heads:
                torch.testing.assert_close(head(full_z), head(out["z"]), atol=2e-5, rtol=2e-5)
        finally:
            first["cache"].close()
    if "delayed_encoder" in ns:
        target = ns["delayed_encoder"]
        assert not {id(p) for p in target.parameters()} & {id(p) for p in encoder.parameters()}
        assert all(not p.requires_grad for p in target.parameters())
        assert not target(_tokens(0, 1))["z"].requires_grad


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if "15_inference" not in p.stem], ids=lambda p: p.stem)
def test_notebook_training_uses_all_online_modules_and_updates_targets(path):
    notebook, ns = _example(path)
    tokens = _tokens(0, 1, 2)
    # Bellman backups use complete episodes and fail on unfinished groups.
    data = {
        "action": torch.tensor([0, 1, 2]), "reward": torch.tensor([0., 0.5, 1.]),
        "terminated": torch.tensor([False, False, True]),
        "truncated": torch.zeros(3, dtype=torch.bool),
        "episode_reset": torch.tensor([True, False, False]),
        "action_requested": torch.tensor([False, True, True]),
        "inner_terminated": torch.tensor([0, 0, 1]),
        "inner_truncated": torch.zeros(3, dtype=torch.int64),
        "episode_ended": torch.tensor([False, False, True]),
        "old_log_prob": torch.full((3,), -1.1), "advantage": torch.tensor([0., 1., -1.]),
        "q_star": torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        "environment_step": torch.ones(3, dtype=torch.bool),
    }
    group_id = torch.zeros(3, dtype=torch.int64)

    class Loader:
        def __init__(self, **kwargs):
            pass
        def next_batch(self):
            return tokens, data, group_id
        def refresh(self):
            pass
        def close(self):
            pass
        def __len__(self):
            return 1
        def set_sample_objective_fields(self, name, values):
            data[name] = values[0]

    ns.update(loader=Loader(), DataLoader=Loader, EpochDataLoader=Loader,
              stores=[[1]], store=[1], MAX_STEPS_PER_EPISODE=30,
              transform=lambda row: row, tokenizer=SimpleNamespace(encode=lambda events: events), ACTION_COMPONENTS=2,
              branch_advantages=[data['advantage']])
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if "def run_train(" in "".join(c["source"]))
    exec(compile(cell, str(path), "exec"), ns)
    optimizer = ns["optimizer"]
    optimized = [id(p) for group in optimizer.param_groups for p in group["params"]]
    expected = [id(p) for p in ns["modules"].parameters() if p.requires_grad]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == set(expected)
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in ns["modules"].items()}
    delayed_before = ([p.detach().clone() for p in ns["delayed_encoder"].parameters()]
                      if "delayed_encoder" in ns else None)
    kwargs = {"epochs": 1} if "grpo" in path.stem else {"num_steps": 1}
    if path.stem.startswith(("19", "20")):
        kwargs["cycle"] = 0
    if path.stem.startswith("09_"):
        supervised_objective = ns["objective"]

        def check_active_observations(**kwargs):
            # Generic repeat flags are integer columns. They must form a boolean
            # mask, including the reset observation and excluding the terminal.
            assert kwargs['predictions'].shape[0] == 2
            assert kwargs['targets'].tolist() == [0, 1]
            return supervised_objective(**kwargs)

        ns["objective"] = check_active_observations
    result = ns["run_train"](**kwargs)
    assert torch.isfinite(torch.as_tensor(result[0]))
    for name, module in ns["modules"].items():
        assert any(not torch.equal(old, new) for old, new in zip(before[name], module.parameters()))
    if delayed_before is not None:
        target = ns["delayed_encoder"]
        assert all(p.grad is None for p in target.parameters())
        assert any(not torch.equal(old, new) for old, new in zip(delayed_before, target.parameters()))
        # Only the value head has a delayed copy; actions use the online head.
        assert "backup_action_head" not in ns and "backup_action_heads" not in ns
        assert "POLYAK_TAU_ACTION" not in ns
        if "q_heads" in ns:
            pairs = [(ns["q_heads"][key], ns["backup_value_heads"][key])
                       for key in ns["head_keys"]]
            initial = before["q_heads"]
        else:
            pairs = [(ns["q_head"], ns["backup_value_head"])]
            initial = before["q_head"]
        offset = 0
        for source, value in pairs:
            for online, v in zip(source.parameters(), value.parameters()):
                old = initial[offset]
                torch.testing.assert_close(v, old.lerp(online.detach(), ns["POLYAK_TAU_VALUE"]))
                assert not v.requires_grad
                offset += 1


def test_training_checkpoint_loads_in_inference_example(tmp_path):
    _, train = _example(EXAMPLES / "02_train_offline_dqn.ipynb")
    _, inference = _example(EXAMPLES / "15_inference.ipynb")
    train["modules"].eval()
    inference["modules"].eval()
    tokens = _tokens(0, 1)
    with torch.no_grad():
        expected = train["q_head"](train["encoder"](tokens)["z"])
    save_model(model=train["modules"], path=tmp_path)
    loaded = load_model(model=inference["modules"], repo_id_or_path=tmp_path)
    assert loaded is inference["modules"]
    with torch.no_grad():
        actual = inference["q_head"](inference["encoder"](tokens)["z"])
    torch.testing.assert_close(actual, expected)


def test_embedding_requires_prepared_inputs_on_the_model_device():
    _, namespace = _example(NOTEBOOKS[0])
    model = namespace["encoder"]
    tokens = _tokens(0)
    tensors = tokens.to_tensors("cpu")
    kwargs: dict[str, Any] = {name: tensors[name] for name in ("numeric_mask", "numeric_values", "tokenizer_ids")}
    kwargs["tokenizer_names"] = tokens.tokenizer_names
    with pytest.raises(TypeError, match="expects tensor"):
        model.embedding.embeddings["text"](tokens, numeric_values=tensors["numeric_values"], numeric_mask=tensors["numeric_mask"])
    model.to("meta")
    with pytest.raises(ValueError, match="same device"):
        model.embedding(tensors["token_ids"], **kwargs)


def test_embedding_and_backbone_are_independently_constructed():
    embedding = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=2)
    backbone = Backbone(architecture="qwen3", hidden_dim=8, num_layers=1,
                        num_heads=2, train_kernel="reference",
                        decode_kernel="flex", dtype=torch.float32)
    assert not hasattr(backbone, "embedding")
    assert getattr(backbone.model, "embed_tokens") is None
    assert embedding.fourier is not None
    assert not {id(p) for p in embedding.parameters()} & {id(p) for p in backbone.parameters()}


def test_backbone_requires_explicit_packed_metadata():
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2, train_kernel='reference', decode_kernel='flex', dtype=torch.float32)
    embed = torch.randn(5, 8)
    with pytest.raises(ValueError, match="require sequence_ids"):
        backbone(embed)
    with pytest.raises(ValueError, match="sequence_ids"):
        backbone(embed, sequence_ids=torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="attention kwargs"):
        backbone(embed, sequence_ids=torch.zeros(5, dtype=torch.long), attention_mask=torch.ones(5, 5))
    features = backbone(embed, sequence_ids=torch.zeros(5, dtype=torch.long))
    assert isinstance(features, torch.Tensor) and features.shape == embed.shape


def _notebook_functions(notebook, names, namespace):
    """Load selected real example helpers without executing setup or publishing."""
    definitions: list[ast.stmt] = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                   for node in ast.parse(''.join(cell['source'])).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook>', 'exec'), namespace)


def test_ppo_collection_restarts_replay_and_preserves_behavior_prefixes():
    from types import SimpleNamespace
    from mouse_core.data import Datastore, environment_decisions
    from mouse_core.objectives import sample_discrete_action

    notebook, ns = _example(EXAMPLES / '07_train_online_ppo.ipynb', MAX_CONTEXT=6,
                            REPEAT_COUNT=2, MAX_STEPS_PER_EPISODE=2)
    created = []

    class Env:
        def __init__(self, seed):
            self.seed = seed
            self.closed = False
            self.names = ['test']
            self.envs = [self]
            self.output_index = -1
            self.action_space = SimpleNamespace(sample=lambda: (0,))

        def step(self, action):
            self.output_index += 1
            step = self.output_index
            reset = step in (0, 3)
            return ({'observation': (self.seed + step) % 16,
                     'episode_terminated': step in (2, 4), 'episode_truncated': False},
                    0.0 if reset else float(action + 1), False, step == 4, {})

        def close(self):
            self.closed = True

    def make_env(*, seed_offset):
        env = Env(seed_offset)
        created.append(env)
        return env

    from tests._token_batch_helpers import IntIdTokenizer
    tokenizer = Tokenizer(tokenizers={"text": IntIdTokenizer()}, input_fields=[
            dict(type='text', provider='environment', input_field='observation', format='{field}'),
            dict(type='token', provider='environment', format='1', head_output=True)],
        objective_fields=[
            {'input_field': 'action', 'missing_value': None},
            {'input_field': 'reward', 'missing_value': 0.},
            {'input_field': 'terminated', 'missing_value': False},
            {'input_field': 'truncated', 'missing_value': False},
            {'input_field': 'old_log_prob', 'missing_value': 0.},
        ])
    ns.update(env=None, logs=[], trackers=[], contexts=[[]], stores=[Datastore(name='test')],
              make_rollout_env=make_env, tokenizer=tokenizer,
              sample_discrete_action=sample_discrete_action, json=json)
    _notebook_functions(notebook, {'frozenlake_fields', 'behavior_fields', 'run_rollout'}, ns)
    for initial_step in (0, 5):
        assert ns['run_rollout'](env_steps=initial_step, num_episodes=1, max_cache=6) == initial_step + 5
        store = ns['stores'][0]
        rows = list(environment_decisions(store.to_dataset()))
        assert len(rows) == 5
        assert rows[-1].get('terminated', False) or rows[-1].get('truncated', False)
        assert ['action' not in row for row in rows] == [True, False, False, True, False]
        assert rows[0]['observation'] == initial_step
        # No optimizer update occurred: replaying the collection prefix must reproduce
        # every stored action probability, including after an environment-only reset.
        ns['modules'].eval()
        with torch.no_grad():
            tokens = tokenizer.pack_rows(rows=[ns['contexts'][0]], continuing=None)
            logits = ns['policy_head'](ns['encoder'](tokens)['z'])
            for step, row in enumerate(rows[1:], 1):
                if 'action' in row:
                    actual = logits[step - 1].log_softmax(-1)[row['action']]
                    assert float(actual) == pytest.approx(row['old_log_prob'], abs=1e-5)
        # Preprocessing supplies an ignored agent probability on reset outputs. The complete task trains under fault mode.
        loader = DataLoader(stores=[store], events_budget=len(store), batch_size=1,
                            sample_ranges=environment_episode_ranges, num_workers=0,
                            transform=tokenizer.encode)
        try:
            inputs, objective_data, group_id = loader.next_batch()
            out = ns['encoder'](inputs)
            definition = next(node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                              for node in ast.parse(''.join(cell['source'])).body
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(target, ast.Name) and target.id == 'objective'
                                  for target in node.targets))
            exec(compile(ast.Module(body=[definition], type_ignores=[]), '<notebook>', 'exec'), ns)
            objective = ns['objective']
            assert objective.cross_group_backups == 'fault'
            losses, _ = objective(objective_data=objective_data, group_id=group_id,
                                  predictions=ns['policy_head'](out['z']),
                                  value_predictions=ns['value_head'](out['z']))
            assert torch.isfinite(losses['ppo'])
            assert objective_data['old_log_prob'][0].item() == 0.0
            assert objective_data['old_log_prob'][3].item() == 0.0
            with pytest.raises(ValueError, match='cross-group backup'):
                objective(objective_data={name: values[:2] for name, values in objective_data.items()},
                          group_id=group_id[:2], predictions=ns['policy_head'](out['z'][:2]),
                          value_predictions=ns['value_head'](out['z'][:2]))
        finally:
            loader.close()
    assert created[0].closed
    created[-1].close()
    with pytest.raises(ValueError, match='fit a complete outer task'):
        ns['run_rollout'](env_steps=10, num_episodes=1, max_cache=5)


def test_grpo_behavior_probability_uses_the_complete_training_prefix():
    from mouse_core.objectives import sample_discrete_action
    notebook, ns = _example(EXAMPLES / '08_train_online_grpo.ipynb')
    tokenizer = token_tokenizer('observation')
    seen = []

    def encoder(tokens, **kwargs):
        seen.append(tokens.S)
        # Make the logits depend on the first observation, so dropping an old
        # prefix would silently change the action distribution.
        first = float(tokens.token_ids[0])
        return {'z': torch.tensor([[first, 0., -first]]).expand(tokens.P, -1)}

    ns.update(tokenizer=tokenizer, encoder=encoder, policy_head=lambda z: z,
              sample_discrete_action=sample_discrete_action)
    _notebook_functions(notebook, {'_truncate', 'act_once'}, ns)
    context = measurement_events([{'observation': 3}] + [{'observation': 1}] * 299)
    action, log_prob = ns['act_once'](context=context)
    assert seen == [300]
    expected = torch.tensor([3., 0., -3.]).log_softmax(-1)[int(action)]
    assert log_prob == pytest.approx(float(expected))


def test_cartpole_encoder_uses_configured_fourier_features_for_observations():
    notebook, ns = _example(EXAMPLES / '18_train_offline_cartpole_dqn.ipynb',
                            NUM_FREQUENCIES=3)
    encoder = ns['encoder'].eval()
    assert encoder.embedding.embeddings["text"].num_frequencies == 3
    assert encoder.embedding.embeddings["text"].fourier is not None

    # Keep the actual notebook's numeric observation fields and scaling bounds.
    definition = next(node.value for cell in notebook['cells'] if cell['cell_type'] == 'code'
                      for node in ast.parse(''.join(cell['source'])).body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == 'tokenizer'
                          for target in node.targets))
    assert isinstance(definition, ast.Call)
    fields = next(keyword.value for keyword in definition.keywords
                  if keyword.arg == 'input_fields')
    assert isinstance(fields, ast.List)
    input_fields = []
    for field in fields.elts:
        assert isinstance(field, ast.Dict)
        kind = next(value for key, value in zip(field.keys, field.values)
                    if isinstance(key, ast.Constant) and key.value == 'type')
        if isinstance(kind, ast.Constant) and kind.value == 'numeric':
            input_fields.append(ast.literal_eval(field))
    assert [(field["input_field"], field["format"]) for field in input_fields] == [("cart_position", "x"), ("pole_angle", "t")]
    input_fields.append({'type': 'token', 'format': '\n', 'head_output': True})

    class FieldTokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([['xt\n'.index(text)]])}

    tokenizer = Tokenizer(tokenizers={"text": FieldTokenizer()}, input_fields=input_fields,
                          objective_fields=[])
    first = tokenizer.pack_rows(rows=[measurement_events(payloads) for payloads in [[{'cart_position': 0., 'pole_angle': 0.}]]], continuing=None)
    second = tokenizer.pack_rows(rows=[measurement_events(payloads) for payloads in [[{'cart_position': 0.5, 'pole_angle': 0.1}]]], continuing=None)
    np.testing.assert_array_equal(first.token_ids, second.token_ids)
    with torch.no_grad():
        first_readout = encoder(first)['z']
        second_readout = encoder(second)['z']
    assert not torch.allclose(first_readout, second_readout)


def test_pendulum_notebook_logs_only_position_and_keeps_independent_episodes(monkeypatch):
    gym = pytest.importorskip('gymnasium')
    pytest.importorskip('mouse_gym')
    make = gym.make
    physical_observations = []

    def record_pendulum(*args, **kwargs):
        assert args == ('Pendulum-v1',)
        assert 'max_episode_steps' not in kwargs
        env = make(*args, **kwargs)
        reset, step = env.reset, env.step

        def record_reset(**reset_kwargs):
            output = reset(**reset_kwargs)
            physical_observations.append(output[0].copy())
            return output

        def record_step(action):
            output = step(action)
            physical_observations.append(output[0].copy())
            return output

        monkeypatch.setattr(env, 'reset', record_reset)
        monkeypatch.setattr(env, 'step', record_step)
        return env

    monkeypatch.setattr(gym, 'make', record_pendulum)
    path = EXAMPLES / '19_train_online_pendulum_dqn.ipynb'
    notebook = json.loads(path.read_text())
    namespace: dict[str, Any] = {}
    threads = torch.get_num_threads()
    try:
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if index == 1:
                source = source.replace('NUM_CYCLES = 2', 'NUM_CYCLES = 1')
                source = source.replace('TRAIN_STEPS = 2', 'TRAIN_STEPS = 1')
            exec(compile(source, f'{path}:cell {index}', 'exec'), namespace)
        assert namespace['env'].observation_space.shape == (2,)
        assert 'velocity' not in namespace['vocabulary']
        log = namespace['event_log']
        outputs = [json.loads(event['payload']) for event in namespace['store'][:]
                   if event['provider'] == 'environment']
        episode_steps = gym.spec('Pendulum-v1').max_episode_steps
        assert episode_steps is not None
        assert len(outputs) == len(physical_observations) == 2 * (episode_steps + 1)
        for payload, physical in zip(outputs, physical_observations):
            assert physical.shape == (3,)
            assert 'observation' not in payload
            np.testing.assert_array_equal([payload['cos_theta'], payload['sin_theta']], physical[:2])
            assert not any(key.startswith('repeat_episode') for key in payload)
        starts = [event for event in log.events
                  if event['provider'] == 'coordinator' and event['payload'].get('episode_start', False)]
        assert [payload['episode_index'] for payload in starts] == [0, 1]
        assert all('step_index' in event and 'episode_index' not in event['payload'] for event in log)
        decisions = list(environment_decisions(log.to_dataset()))
        for row, encoded in zip(decisions, namespace['tokenizer'].encode(log), strict=True):
            assert encoded.numeric_mask.sum() == (2 + int('torque' in row) + int('reward' in row))
        assert list(environment_decisions(namespace['store'].to_dataset())) == decisions
    finally:
        if 'env' in namespace:
            namespace['env'].close()
        torch.set_num_threads(threads)
