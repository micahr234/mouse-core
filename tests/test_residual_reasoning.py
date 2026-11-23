"""Residual rollout/replay parity, detached feedback and the real notebook loop."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests._reasoning_replay_fixture import configure_notebook_optimizer, recorded_stores, reasoning_loader

import numpy as np
import pytest
import torch
from tests._reasoning_input_reference import input_reference
from torch import nn

from mouse_core.data import (
    Tokenizer, TokenizerModalitySpec, TokenCache, load_tokenizer, make_token_batch,
    pack_token_batch, save_tokenizer,
)
from mouse_core.models import (
    Backbone, BaseBackbone, Embedding, MultiEmbedding, SequentialReasoner, ReasoningBatch,
    load_model, save_model,
)
from mouse_core.objectives import DqnObjective, boundary_discount
from tests.test_autoregressive_reasoning import local_checkpoint

NOTEBOOK = Path(__file__).parents[1] / 'examples/25_train_reasoning_dqn.ipynb'


def example_module():
    nodes = []
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] == 'code':
            nodes.extend(n for n in ast.parse(''.join(cell['source'])).body
                         if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)))
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return SimpleNamespace(**namespace)


class CausalBackbone(BaseBackbone):
    """Small differentiable causal reference for CPU gradient tests, not FlexAttention."""
    def __init__(self, hidden_dim=8):
        super().__init__()
        self.residual_dtype = torch.float32
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self._hidden_dim = hidden_dim

    @property
    def mlp_intermediate_dim(self):
        return 24

    @property
    def rms_norm_eps(self):
        return 1e-6

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def _forward_dense(self, embeds, *, output_hidden_states, **kwargs):
        raise NotImplementedError

    def _forward_packed(self, embeds, *, sequence_ids, output_hidden_states):
        rows = []
        for group in sequence_ids.unique_consecutive():
            row = embeds[sequence_ids == group]
            positions = torch.arange(len(row), device=row.device)[:, None]
            rows.append(torch.tanh(self.linear(row.cumsum(0) / (positions + 1)) + positions * .01))
        return torch.cat(rows)


def model(*, real=False, max_steps=3):
    torch.manual_seed(17)
    embedding = MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8)})
    backbone = (Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                         head_dim=4, max_position_embeddings=128, dtype=torch.float32)
                if real else CausalBackbone())
    value_head = nn.Linear(8, 5)
    with torch.no_grad():
        value_head.weight[-1].zero_()
        value_head.bias[-1] = 100.
    result = example_module().ReasoningDqn(input_mlp_output_scale=100.,
        embedding=embedding, backbone=backbone, value_head=value_head, min_steps=1, max_steps=max_steps,
        residual_connections=True, wrap_latent="step", reasoning_loss="all_tokens",
    ).to(torch.get_default_device())
    return result


def batch(rows=((2, 3, 9), (4, 9), (9,))):
    lengths = [len(row) for row in rows]
    ids = np.concatenate(rows)
    heads = np.isin(np.arange(len(ids)), np.cumsum(lengths) - 1)
    return make_token_batch(token_ids=ids, numeric_values=np.zeros(len(ids)), numeric_mask=np.zeros(len(ids), bool),
                            head_mask=heads, autoregressive_mask=heads.copy(), step_readout_mask=heads.copy(),
                            sequence_ids=np.repeat(np.arange(len(rows)), lengths),
                            num_sequences=len(rows), tokenizer_names=('text',))


def generate(online, tokens, backend='uncached'):
    return online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone, action_values=online.value_head,
                                    backend=backend, max_active_requests=2)


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_real_transformer_rollout_replay_parity_and_request_batches(backend):
    online = model(real=True).train()
    tokens = batch(((2, 3, 9), (4, 9), (9,), (1, 7, 8, 9)))
    expected = generate(online, tokens)
    trace = generate(online, tokens, backend)
    torch.testing.assert_close(trace.states, expected.states, atol=3e-5, rtol=3e-5)
    actual = online(trace)["q"]
    torch.testing.assert_close(actual, online.value_head(trace.states[:, -1])[:, :-1], atol=3e-5, rtol=3e-5)
    assert trace.states.shape == (4, 4, 8) and not trace.states.requires_grad
    assert online.training and online.backbone.training and online.reasoner.training
    assert all(p.grad is None for p in online.parameters())
    # The start token is consumed as z0, not also copied into the context.
    initial = torch.zeros_like(online.reasoner.start_embedding.weight)
    torch.testing.assert_close(trace.states[:, 0], initial.expand(4, -1))
    changed = tokens.clone()
    changed['token_ids'][-2] = 6
    after = generate(online, changed, backend)
    torch.testing.assert_close(after.states[:3], trace.states[:3])
    assert not torch.allclose(after.states[-1, -1], trace.states[-1, -1])


def test_single_replay_gradient_matches_explicit_detached_recurrence():
    online = model()
    reference = deepcopy(online)
    tokens = batch()
    calls = []
    def record(_module, _args, output):
        calls.append(torch.is_grad_enabled())
        if torch.is_grad_enabled():
            output.retain_grad()
            outputs.append(output)
    outputs = []
    hook = online.backbone.register_forward_hook(record)
    trace = generate(online, tokens)
    assert calls and not any(calls)
    # Artificial requires_grad makes accidental feedback through sampled states observable.
    states = trace.tokens["latent_values"].clone().requires_grad_()
    trace.tokens["latent_values"] = states
    actual = online(trace)["q_steps"][:, -1]
    loss = actual.square().sum()
    loss.backward()
    hook.remove()
    assert calls.count(True) == 1
    assert states.grad is None
    assert len(outputs) == 1
    positions = trace.update_positions
    assert (outputs[0].grad[positions].abs().sum(-1) > 0).all()
    # z_N is the readout; running T on the appended z_N would be update N+1.
    assert outputs[0].grad[trace.readout_positions].count_nonzero() == 0

    # Independent sequential construction, preserving every solid residual edge.
    reference_values = []
    offset = 0
    for length in [3, 2, 1]:
        ids = tokens['token_ids'][offset:offset + length]
        embeds = reference.embedding(ids, tokenizer_ids=torch.zeros_like(ids), tokenizer_names=('text',))
        context, state = embeds[:-1], torch.zeros_like(reference.reasoner.start_embedding.weight[0])
        inputs = []
        for depth in range(3):
            inputs.append(input_reference(reference.reasoner, state if depth == 0 else state.detach()))
            row = torch.cat((context, torch.stack(inputs)))
            hidden = reference.backbone(row, sequence_ids=torch.zeros(len(row), dtype=torch.long))
            state = state + hidden[-1].float()
        reference_values.append(reference.value_head(state))
        offset += length
    expected = torch.stack(reference_values)
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in online.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, atol=2e-5, rtol=2e-5)
    grads = online.embedding.embeddings['text'].embed_tokens.weight.grad
    assert grads[9].count_nonzero() == 0
    assert online.reasoner.start_embedding.weight.grad.abs().sum() > 0
    assert (grads[[2, 3, 4]].abs().sum(-1) > 0).all()  # Train context tokens too.
    assert grads[[0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]].count_nonzero() == 0
    assert online.reasoner.norm.weight.grad.abs().sum() > 0


@pytest.mark.flex_training
def test_real_transformer_backward_reaches_all_residual_updates():
    online = model(real=True)
    tokens = batch()
    trace = generate(online, tokens, 'continuous')
    q = online(trace)["q_steps"][:, -1]
    q.square().sum().backward()
    assert online.reasoner.norm.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.backbone.parameters())
    assert online.embedding.embeddings['text'].embed_tokens.weight.grad[9].count_nonzero() == 0
    assert online.reasoner.start_embedding.weight.grad.abs().sum() > 0
    assert (online.embedding.embeddings['text'].embed_tokens.weight.grad[[2, 3, 4]].abs().sum(-1) > 0).all()


def test_context_numeric_and_position_features_receive_gradients():
    online = model()
    online.embedding = MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8)},
                                     numeric_field_names=('reward',), num_frequencies=2, position_num_frequencies=2)
    tokens = batch(((2, 9),))
    tokens.set_non_tensor('numeric_field_names', ('reward',))
    tokens['numeric_mask'][0] = True
    tokens['numeric_field_ids'][0] = 0
    tokens['numeric_values'][0] = .25
    tokens['position_mask'][0] = True
    tokens['position_values'][0] = .5
    trace = generate(online, tokens)
    changed = tokens.clone()
    changed['numeric_values'][0] = -.5
    assert not torch.allclose(generate(online, changed).states[:, -1], trace.states[:, -1])
    online(trace)["q"].square().sum().backward()
    for name, parameter in online.embedding.named_parameters():
        if name.startswith('features.'):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_cache_closed_and_modes_restored_on_generation_failure(monkeypatch):
    online = model(real=True).train()
    online.reasoner.norm.eval()
    sessions = []
    create = online.backbone.decode_session
    def broken(**kwargs):
        session = create(**kwargs)
        sessions.append(session)
        def fail(**kwargs):
            raise RuntimeError('deliberate forward failure')
        session.forward = fail
        return session
    monkeypatch.setattr(online.backbone, 'decode_session', broken)
    with pytest.raises(RuntimeError, match='deliberate'):
        generate(online, batch(), 'continuous')
    assert len(sessions) == 1 and sessions[0]._cache is None
    assert online.training and online.backbone.training and not online.reasoner.norm.training


def test_generation_forwards_decode_budgets(monkeypatch):
    online = model(real=True)
    settings = []
    create = online.backbone.decode_session
    def record(**kwargs):
        settings.append(kwargs)
        return create(**kwargs)
    monkeypatch.setattr(online.backbone, 'decode_session', record)
    online.reasoner.generate(batch(), embedding=online.embedding, backbone=online.backbone, action_values=online.value_head,
                             backend='continuous', max_active_requests=2,
                             max_batch_tokens=8, cache_capacity_tokens=1024)
    assert settings
    assert all(s['max_batch_tokens'] == 8 and s['cache_capacity_tokens'] == 1024
               and s['batch_size'] <= 2 for s in settings)


def test_trace_alignment_and_start_marker_validation():
    online = model()
    tokens = batch()
    trace = generate(online, tokens)
    with pytest.raises(ValueError, match='positions'):
        online(replace(trace, update_positions=trace.update_positions[:-1]))
    with pytest.raises(ValueError, match='lengths'):
        online(replace(trace, lengths=trace.lengths * 0))
    tokens['autoregressive_mask'].zero_()
    with pytest.raises(ValueError, match='autoregressive=True'):
        generate(online, tokens)


def test_passive_batch_skips_generation_and_replay():
    online = model()
    tokens = batch()
    tokens['head_mask'].zero_()
    tokens['step_readout_mask'].zero_()
    tokens['autoregressive_mask'].zero_()
    calls = []
    hook = online.backbone.register_forward_hook(lambda *_: calls.append(True))
    trace = generate(online, tokens)
    assert online(trace)["q"].shape == (0, 4)
    hook.remove()
    assert trace.states.shape == (0, 4, 8) and not calls


@pytest.mark.parametrize('max_steps', [0, -1, True, 1.5])
def test_invalid_reasoning_budget(max_steps):
    with pytest.raises(ValueError, match='positive integer'):
        SequentialReasoner(input_mlp_intermediate_dim=24, input_mlp_output_scale=100., min_steps=1, hidden_dim=8, max_steps=max_steps, residual_connections=True, wrap_latent="step", adaptive_depth=True, rms_norm_eps=1e-6)


class TextBackend:
    def __call__(self, text, **kwargs):
        return {'input_ids': [int(word) for word in text.split()]}


def tokenizer(start='9'):
    return Tokenizer(tokenizers={'text': TextBackend()}, input_fields=[
        dict(type='text', provider='environment', input_field='prompt', format='{prompt}'),
        dict(type='token', provider='environment', input_field='action_requested', format=start,
             head_output=True, autoregressive=True),
    ])


def test_tokenizer_start_field_survives_stream_pack_cache_and_save_load(tmp_path):
    tok = tokenizer()
    events = [dict(provider='environment', step_index=i, payload=payload) for i, payload in enumerate([
        {'prompt': '2 3', 'action_requested': True}, {'prompt': '4'},
        {'prompt': '5 6', 'action_requested': True},
    ])]
    steps = tok.encode(events, include_objectives=False)
    stream = tok.stream(include_objectives=False)
    streamed = [step for event in events if (step := stream.push(event)) is not None] + [stream.flush()]
    for left, right in zip(steps, streamed, strict=True):
        assert right is not None
        np.testing.assert_array_equal(left.autoregressive_mask, right.autoregressive_mask)
    with TokenCache(content_version='starts', max_memory_bytes=10000, directory=tmp_path) as cache:
        cache.put('one', steps)
        cached = cache.get('one')
        assert cached[0].autoregressive_mask is not None
        cached[0].autoregressive_mask[:] = False
        inputs, data = pack_token_batch(steps=cache.get('one'))
    assert inputs['autoregressive_mask'].tolist() == [False, False, True, False, False, False, True]
    assert data['step_ids'].tolist() == [0, 2]
    save_tokenizer(tokenizer=tok, path=tmp_path / 'tokenizer')
    restored = load_tokenizer(repo_id_or_path=str(tmp_path / 'tokenizer'), tokenizers={'text': TextBackend()})
    packed, _ = restored.pack_rows(rows=[events])
    torch.testing.assert_close(packed['autoregressive_mask'], inputs['autoregressive_mask'])
    assert restored.input_fields[-1].format == '9' and restored.input_fields[-1].autoregressive
    # Later observations cannot affect earlier decisions in the episode.
    online = model()
    trace = generate(online, inputs)
    first = trace.states[0].clone()
    inputs['token_ids'][3:] = torch.tensor([7, 8, 8, 9])
    torch.testing.assert_close(generate(online, inputs).states[0], first)
    other, _ = tokenizer('8').pack_rows(rows=[events])
    torch.testing.assert_close(generate(online, other).states[:, 0], trace.states[:, 0])


@pytest.mark.parametrize('spec', [dict(type='text', head_output=True), dict(type='token', head_output=False)])
def test_autoregressive_field_requires_one_token_and_a_readout(spec):
    with pytest.raises(ValueError, match='token field with head_output=True'):
        TokenizerModalitySpec(format='9', autoregressive=True, **spec)
    with pytest.raises(ValueError, match='expected 1'):
        tokenizer('8 9')({'provider': 'environment', 'payload': {'action_requested': True}})


def test_notebook_dqn_update_replays_online_trajectory_through_both_models(tmp_path, monkeypatch):
    example = example_module()
    online = model()
    from mouse_core.models.lora import apply_lora, LoRAConfig
    apply_lora(model=online.backbone, config=LoRAConfig(rank=2))
    target = online.copy().eval()
    with torch.no_grad():
        target.backbone.linear.base.bias.add_(.3)
    original = {n: p.detach().clone() for n, p in online.named_parameters()}
    target_before = {n: p.detach().clone() for n, p in target.named_parameters()}
    loader = reasoning_loader(example)
    reference_loader = reasoning_loader(example)
    try:
        tokens, data = reference_loader.next_batch()
    finally:
        reference_loader.close()
    calls = []
    hook = online.backbone.register_forward_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    objective = DqnObjective(reward=None, value=None, gate=None, temperature=0.,
                            discount=boundary_discount(gamma_step=.9, gamma_terminated=0., gamma_truncated=0.))
    trace = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone, action_values=online.value_head,
        backend='uncached', max_active_requests=2)
    assert (trace.lengths == online.reasoner.max_steps).all()
    with torch.no_grad():
        target_states = target.reasoner(trace,
            embedding=target.embedding, backbone=target.backbone)
        expected_target = target.value_head(target_states[:, 0])
    def forbidden_generation(*args, **kwargs):
        raise AssertionError('The target must replay the online trajectory, never generate')
    monkeypatch.setattr(target.reasoner, 'generate', forbidden_generation)
    replays = []
    for network in (online, target):
        original_replay = network.reasoner.forward
        def capture_replay(trace, _original=original_replay, **kwargs):
            replays.append(trace)
            return _original(trace, **kwargs)
        monkeypatch.setattr(network.reasoner, 'forward', capture_replay)
    original_generation = online.reasoner.generate
    def cpu_generation(*args, **kwargs):
        assert kwargs.pop('backend') == 'continuous'
        return original_generation(*args, backend='uncached', **kwargs)
    monkeypatch.setattr(online.reasoner, 'generate', cpu_generation)
    seen_backups = []
    def check_objective(**kwargs):
        seen_backups.append(kwargs['backup_value_predictions'].clone())
        return objective(**kwargs)
    example.run_train.__globals__.update(
        online=online, target=target, LR_EMBEDDING=1e-3, LR_BACKBONE_LORA=1e-3, LR_REASONING_NORM=1e-3, LR_HEADS=1e-3,
        objective=check_objective, loader=loader, device=torch.device('cpu'),
        MAX_ACTIVE_REQUESTS=2,
        POLYAK_TAU_EMBEDDING=.1, POLYAK_TAU_BACKBONE_LORA=.2, POLYAK_TAU_REASONING_NORM=.2, POLYAK_TAU_VALUE=.3,
        MAX_GRAD_NORM=1., REASONING_REWARD=-.01, GAMMA_THINK=1.)
    configure_notebook_optimizer(example, NOTEBOOK)
    try:
        loss, metrics, next_step = example.run_train(training_step=0, num_steps=1)
    finally:
        loader.close()
    assert torch.isfinite(loss) and not loss.requires_grad
    hook.remove()
    assert all(np.isfinite(value) for value in metrics.values()) and calls.count(True) == 1
    assert len(replays) == 2 and replays[0] is replays[1]
    torch.testing.assert_close(seen_backups[0], expected_target)
    assert not torch.equal(original['backbone.linear.lora_B.weight'], online.backbone.linear.lora_B.weight)
    assert not torch.equal(original['reasoner.norm.weight'], online.reasoner.norm.weight)
    assert all(p.grad is None for p in target.parameters())
    for name, parameter in target.named_parameters():
        rate = (.1 if name.startswith('embedding.') else .3 if name.startswith(('value_head.', 'reasoner.start_embedding.', 'reasoner.input_projection.'))
                else .2)
        if not dict(online.named_parameters())[name].requires_grad:
            rate = 0.
        torch.testing.assert_close(parameter, target_before[name].lerp(dict(online.named_parameters())[name], rate))
    monkeypatch.setattr(online.reasoner, 'generate', original_generation)
    save_model(model=online, path=tmp_path / 'saved')
    restored = model()
    apply_lora(model=restored.backbone, config=LoRAConfig(rank=2))
    load_model(model=restored, repo_id_or_path=tmp_path / 'saved')
    torch.testing.assert_close(restored(generate(restored, tokens))['q'],
                               online(generate(online, tokens))['q'])


@pytest.mark.flex_training(default_device=False)
def test_complete_residual_notebook_trains_and_uploads_checkpoint(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    local_checkpoint(source)
    output = tmp_path / 'trained'
    from tests._hub_example_fixture import capture_upload
    capture_upload(monkeypatch, output)
    namespace: dict[str, Any] = {}
    for index, cell in enumerate(json.loads(NOTEBOOK.read_text())['cells']):
        if cell['cell_type'] != 'code':
            continue
        exec(compile(''.join(cell['source']), f'{NOTEBOOK}:cell{index}', 'exec'), namespace)
        if index == 1:
            namespace.update(PRETRAINED=str(source), load_stores_from_hub=lambda **_: recorded_stores(), TOKEN_BUDGET=64, NUM_WORKERS=0,
                             MAX_EPISODE_STEPS=2, TRAIN_STEPS=1, MAX_REASONING_STEPS=2,
                             GENERATION_START_TOKEN='word2', MAX_ACTIVE_REQUESTS=2)
    assert np.isfinite(namespace['metrics']['dqn_loss'])
    assert json.loads((output / 'reasoning_settings.json').read_text())['generation_start_token'] == 'word2'
    assert (output / 'tokenizer/tokenizer.json').is_file()
    restored = namespace['online'].copy()
    load_model(model=restored, repo_id_or_path=output)
    for name, parameter in namespace['online'].named_parameters():
        torch.testing.assert_close(dict(restored.named_parameters())[name], parameter)


@torch.no_grad()
def test_residual_retains_adaptive_reasoning_in_episode_cache_until_reset(monkeypatch):
    online = model(real=True, max_steps=3).eval()
    session = online.backbone.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=1024)
    histories = [[], []]
    processed = []
    original = session._decode
    def record(h, *args, **kwargs):
        processed.append(len(h))
        return original(h, *args, **kwargs)
    monkeypatch.setattr(session, '_decode', record)
    def selector(tokens):
        heads = tokens['head_mask'].nonzero(as_tuple=True)[0]
        desired = tokens['token_ids'][heads - 1] % 3 + 1
        def choose(values, indices, depths):
            return torch.where(depths < desired.to(depths.device)[indices], 4, 0)
        return choose
    try:
        for updates in [((2, 3, 9), (4, 9)), ((5, 9), (6, 7, 9)), ((8, 9), (3, 9))]:
            if histories[0] and len(histories[0]) > 4:
                session.reset_rows([0])
                histories[0] = []
            before = session.lengths
            full_tokens = batch(tuple(tuple(h) + u for h, u in zip(histories, updates)))
            full_tokens['head_mask'] = full_tokens['token_ids'] == 9
            full_tokens['autoregressive_mask'] = full_tokens['head_mask'].clone()
            full_tokens['step_readout_mask'] = full_tokens['head_mask'].clone()
            expected = online.reasoner.generate(full_tokens, embedding=online.embedding, backbone=online.backbone,
                action_values=online.value_head, select_actions=selector(full_tokens), backend='uncached', max_active_requests=2)
            groups = full_tokens['sequence_ids'][full_tokens['head_mask']]
            latest = torch.stack([(groups == row).nonzero()[-1, 0] for row in range(2)])
            delta = batch(updates)
            deferred_before = sum(len(v) for v in session._deferred.values())
            processed.clear()
            trace = online.reasoner.generate(delta, embedding=online.embedding, backbone=online.backbone, action_values=online.value_head,
                                             backend='continuous', max_active_requests=1, cache=session, select_actions=selector(delta))
            torch.testing.assert_close(trace.states, expected.states[latest], atol=3e-5, rtol=3e-5)
            if any(before):
                with pytest.raises(ValueError, match='cached prefix'):
                    online(trace)
            for row, update in enumerate(updates):
                histories[row].extend(update)
            assert session.lengths == tuple(len(h) + int(expected.lengths[groups == row].sum())
                                            for row, h in enumerate(histories))
            # Count actual transformer queries, including deferred feedback.
            # Old cached tokens must not be processed again.
            deferred_after = sum(len(v) for v in session._deferred.values())
            assert sum(processed) + deferred_after - deferred_before == sum(
                len(update) + int(length) for update, length in zip(updates, trace.lengths))
            if before[1]:
                assert session.lengths[1] > before[1]
    finally:
        session.close()



@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_think_wins_until_cap_and_final_values_match_replay(backend):
    online = model(real=True, max_steps=5)
    trace = generate(online, batch(), backend)
    assert trace.lengths.tolist() == [5, 5, 5]
    assert not hasattr(online.reasoner, 'stop_head')
    replay = online(trace)
    torch.testing.assert_close(replay['states'], trace.states[:, 1:], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(replay['q'], online.value_head(trace.states[:, -1])[:, :-1],
                               atol=3e-5, rtol=3e-5)
    assert trace.tokens['num_sequences'] == 3


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_pending_episode_uses_finished_slot_while_longer_episode_continues(monkeypatch, backend):
    online = model(real=True, max_steps=3)
    tokens = batch(((2, 9, 3, 9), (4, 9), (1, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    expected = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        action_values=online.value_head, backend='uncached', max_active_requests=1)
    resets = []
    create = online.backbone.decode_session
    def record(**kwargs):
        session = create(**kwargs)
        original = session.reset_rows
        def reset(rows):
            resets.append(list(rows))
            original(rows)
        session.reset_rows = reset
        return session
    monkeypatch.setattr(online.backbone, 'decode_session', record)
    actual = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        action_values=online.value_head, backend=backend, max_active_requests=2)
    torch.testing.assert_close(actual.states, expected.states, atol=3e-5, rtol=3e-5)
    if backend == 'continuous':
        assert resets[0] == [1]  # The single-decision episode finishes before row 0.


def test_multiple_actions_retain_every_depth_with_detached_feedback():
    online = model(max_steps=4).eval()
    tokens = batch(((2, 9, 3, 9), (4, 9, 5, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    generated = generate(online, tokens)
    assert generated.lengths.tolist() == [4] * 4
    assert generated.tokens['num_sequences'] == 2
    sampled = generated.tokens['latent_values'].clone().requires_grad_()
    generated.tokens['latent_values'] = sampled
    calls = []
    hook = online.backbone.register_forward_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    states = online.reasoner(generated, embedding=online.embedding, backbone=online.backbone)
    hook.remove()
    assert calls == [True]
    torch.testing.assert_close(states, generated.states[:, 1:])
    with torch.no_grad():
        reference = []
        for row in ((2, 9, 3, 9), (4, 9, 5, 9)):
            prefix = []
            for token in row:
                embed = online.embedding(torch.tensor([token]), tokenizer_ids=torch.tensor([0]), tokenizer_names=('text',))
                if token != 9:
                    prefix.append(embed)
                    continue
                z = torch.zeros_like(online.reasoner.start_embedding.weight)
                prefix.append(input_reference(online.reasoner, z))
                updates = []
                for _ in range(4):
                    inputs = torch.cat(prefix)
                    raw = online.backbone(inputs, sequence_ids=torch.zeros(len(inputs), dtype=torch.long))[-1:]
                    z = z + raw.float()
                    prefix.append(input_reference(online.reasoner, z))
                    updates.append(z[0])
                reference.append(torch.stack(updates))
        torch.testing.assert_close(states, torch.stack(reference))
    online.value_head(states).square().sum().backward()
    assert sampled.grad is None
    assert online.embedding.embeddings['text'].embed_tokens.weight.grad[[2, 3, 4, 5]].abs().sum() > 0


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_adaptive_depths_retain_only_executed_states_and_replay_exactly(backend):
    online = model(real=True, max_steps=4).eval()
    tokens = batch(((2, 9, 3, 9), (4, 9), (1, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    lengths = torch.tensor([1, 3, 2, 4])
    calls = []
    def select(values, indices, depths):
        calls.extend(zip(indices.tolist(), depths.tolist()))
        return torch.where(depths < lengths.to(depths.device)[indices], 4, 1)
    trace = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        action_values=online.value_head, select_actions=select, backend=backend, max_active_requests=2)
    assert trace.lengths.tolist() == lengths.tolist()
    assert sorted(calls) == [(i, k) for i, n in enumerate(lengths.tolist()) for k in range(1, n + 1)]
    assert len(trace.tokens['token_ids']) == len(tokens['token_ids']) + int(lengths.sum())
    torch.testing.assert_close(online(trace)['states'], trace.states[:, 1:], atol=3e-5, rtol=3e-5)
    assert not hasattr(online.reasoner, 'stop_head')


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_greedy_think_or_environment_action_stops_without_looking_ahead(backend):
    online = model(real=True, max_steps=4).eval()
    seen = []
    def values(states):
        seen.append(states.clone())
        scores = [0., 1., 2., 3., 4.] if len(seen) == 1 else [0., 5., 2., 3., 4.]
        return states.new_tensor(scores).expand(len(states), -1)
    trace = online.reasoner.generate(batch(), embedding=online.embedding, backbone=online.backbone,
        action_values=values, backend=backend, max_active_requests=3)
    assert trace.lengths.tolist() == [2, 2, 2] and len(seen) == 2


@pytest.mark.parametrize('max_active_requests', [1, 2])
def test_explicit_positions_and_target_weights_replay_the_same_latent_history(max_active_requests):
    class PositionBackbone(CausalBackbone):
        def __init__(self):
            super().__init__()
            self.position_gain = nn.Parameter(torch.linspace(.1, .8, 8))
            self.context_gain = nn.Parameter(torch.linspace(.02, .09, 8))

        def _forward_packed(self, embeds, *, sequence_ids, output_hidden_states):
            rows = []
            for group in sequence_ids.unique_consecutive():
                row = embeds[sequence_ids == group]
                positions = torch.arange(1, len(row) + 1, device=row.device)[:, None]
                rows.append(positions * self.position_gain + row.cumsum(0) * self.context_gain)
            return torch.cat(rows)

    online = model(max_steps=3).eval()
    online.backbone = PositionBackbone()
    tokens = batch(((2, 9, 3, 4, 9, 5), (6, 7), (8, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    # Logical rows 1 and 3 are empty; row 2 has context but no decision.
    tokens['sequence_ids'] *= 2
    tokens.set_non_tensor('num_sequences', 5)
    lengths = torch.tensor([2, 1, 3])
    trace = online.reasoner.generate(
        tokens, embedding=online.embedding, backbone=online.backbone,
        action_values=online.value_head, backend='uncached',
        max_active_requests=max_active_requests,
        select_actions=lambda values, indices, depths: torch.where(depths < lengths[indices], 4, 1),
    )
    # Explicit positions distinguish the update that produces z_k from the
    # appended z_k token whose transformer output would produce z_(k+1).
    assert trace.initial_positions.tolist() == [1, 6, 12]
    assert trace.update_positions.tolist() == [[1, 2, 0], [6, 0, 0], [12, 13, 14]]
    assert trace.readout_positions.tolist() == [3, 7, 15]
    assert trace.tokens['head_mask'].nonzero(as_tuple=True)[0].tolist() == [3, 7, 15]
    assert trace.tokens['sequence_ids'].tolist() == [0] * 9 + [2] * 2 + [4] * 5
    assert trace.tokens['token_ids'].tolist() == [2, 9, 9, 9, 3, 4, 9, 9, 5, 6, 7, 8, 9, 9, 9, 9]

    target = deepcopy(online)
    with torch.no_grad():
        target.embedding.embeddings['text'].embed_tokens.weight.add_(.37)
        target.reasoner.norm.weight.mul_(1.7)
        target.backbone.position_gain.mul_(-.6)
        target.backbone.context_gain.add_(.11)
        target.value_head.weight.mul_(.4)
        target.value_head.bias.sub_(.2)

    # Spell out the three actual episode inputs independently of the packed
    # metadata. The target consumes ONLINE latents, with its own context
    # embeddings, normalization, transformer weights and value head.
    episode_items = [
        [2, (0, 0), (0, 1), (0, 2), 3, 4, (1, 0), (1, 1), 5],
        [6, 7],
        [8, (2, 0), (2, 1), (2, 2), (2, 3)],
    ]
    for network in (online, target):
        token_table = network.embedding.embeddings['text'].embed_tokens.weight
        expected = []
        for episode in episode_items:
            inputs = torch.stack([
                token_table[item] if isinstance(item, int)
                else input_reference(network.reasoner, torch.zeros_like(network.reasoner.start_embedding.weight[0]) if item[1] == 0 else trace.states[item[0], item[1]].detach())
                for item in episode
            ])
            # Compute each selected update directly from its causal prefix.
            for position, item in enumerate(episode):
                if not isinstance(item, tuple) or item[1] != 0:
                    continue
                decision = item[0]
                state = torch.zeros_like(network.reasoner.start_embedding.weight[0])
                outputs = []
                for depth in range(int(lengths[decision])):
                    update_position = position + depth
                    raw_update = ((update_position + 1) * network.backbone.position_gain
                                  + inputs[:update_position + 1].sum(0) * network.backbone.context_gain)
                    state = state + raw_update.float()
                    outputs.append(state)
                outputs.extend([outputs[-1]] * (3 - len(outputs)))
                expected.append(torch.stack(outputs))
        actual = network(trace)
        expected_states = torch.stack(expected)
        torch.testing.assert_close(actual['states'], expected_states)
        torch.testing.assert_close(actual['q_steps'], network.value_head(expected_states))
        torch.testing.assert_close(actual['q'], network.value_head(expected_states)[torch.arange(3), lengths - 1, :-1])
    torch.testing.assert_close(online(trace)['states'], trace.states[:, 1:])
    assert not torch.allclose(target(trace)['states'], trace.states[:, 1:])


@torch.no_grad()
def test_sparse_cached_rows_preserve_passive_context_and_reset_positions():
    online = model(real=True, max_steps=3).eval()
    session = online.backbone.decode_session(5, max_batch_tokens=3, cache_capacity_tokens=1024)
    histories = [[] for _ in range(5)]

    def tokenize(rows):
        tokens = batch(rows)
        tokens['head_mask'] = tokens['token_ids'] == 9
        tokens['autoregressive_mask'] = tokens['head_mask'].clone()
        tokens['step_readout_mask'] = tokens['head_mask'].clone()
        return tokens

    def selector(tokens):
        heads = tokens['head_mask'].nonzero(as_tuple=True)[0]
        desired = tokens['token_ids'][heads - 1] % 3 + 1
        return lambda values, indices, depths: torch.where(depths < desired[indices], 4, 1)

    updates = [
        ((2, 9), (), (6, 7), (), (4, 9)),
        ((), (), (8, 9), (), (3, 9)),
        ((5, 6), (), (), (), (1, 2)),
        ((10, 9), (), (2, 9), (), (5, 9)),
    ]
    try:
        for iteration, chunks in enumerate(updates):
            if iteration == 3:
                session.reset_rows([2])
                histories[2].clear()
            before = session.lengths
            for row, chunk in zip(histories, chunks, strict=True):
                row.extend(chunk)
            delta = tokenize(chunks)
            actual = online.reasoner.generate(
                delta, embedding=online.embedding, backbone=online.backbone, action_values=online.value_head,
                select_actions=selector(delta), backend='continuous', max_active_requests=1, cache=session,
            )
            full = tokenize(histories)
            expected = online.reasoner.generate(
                full, embedding=online.embedding, backbone=online.backbone, action_values=online.value_head,
                select_actions=selector(full), backend='uncached', max_active_requests=2,
            )
            expected_groups = full['sequence_ids'][full['head_mask']]
            current_groups = delta['sequence_ids'][delta['head_mask']]
            latest = [(expected_groups == group).nonzero()[-1, 0] for group in current_groups]
            expected_states = expected.states[torch.stack(latest)] if latest else expected.states[:0]
            torch.testing.assert_close(actual.states, expected_states, atol=3e-5, rtol=3e-5)
            for group, chunk in enumerate(chunks):
                extra_reasoning = sum((chunk[i - 1] % 3 + 1) for i, token in enumerate(chunk) if token == 9)
                assert session.lengths[group] == before[group] + len(chunk) + extra_reasoning
            if iteration == 2:
                assert not len(actual.lengths)
                assert session.lengths[0] == before[0] + 2
                assert session.lengths[4] == before[4] + 2
    finally:
        session.close()
