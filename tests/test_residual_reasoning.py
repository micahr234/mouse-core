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
from torch import nn

from mouse_core.data import (
    Tokenizer, TokenizerModalitySpec, TokenCache, load_tokenizer, make_token_batch,
    pack_token_batch, save_tokenizer,
)
from mouse_core.models import (
    Backbone, BaseBackbone, Embedding, MultiEmbedding, ResidualReasoner, ResidualBatch,
    load_model, save_model,
)
from mouse_core.objectives import DqnObjective, boundary_discount
from tests.test_autoregressive_reasoning import local_checkpoint

NOTEBOOK = Path(__file__).parents[1] / 'examples/25_train_residual_reasoning_dqn.ipynb'


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
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self._hidden_dim = hidden_dim

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
    return example_module().ResidualDqn(
        embedding=embedding, backbone=backbone, value_head=nn.Linear(8, 4), max_steps=max_steps,
    ).to(torch.get_default_device())


def batch(rows=((2, 3, 9), (4, 9), (9,))):
    lengths = [len(row) for row in rows]
    ids = np.concatenate(rows)
    heads = np.isin(np.arange(len(ids)), np.cumsum(lengths) - 1)
    return make_token_batch(token_ids=ids, numeric_values=np.zeros(len(ids)), numeric_mask=np.zeros(len(ids), bool),
                            head_mask=heads, autoregressive_mask=heads.copy(),
                            sequence_ids=np.repeat(np.arange(len(rows)), lengths),
                            num_sequences=len(rows), tokenizer_names=('text',))


def generate(online, tokens, backend='uncached'):
    return online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
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
    torch.testing.assert_close(actual, online.value_head(trace.states[:, -1]), atol=3e-5, rtol=3e-5)
    assert trace.states.shape == (4, 4, 8) and not trace.states.requires_grad
    assert online.training and online.backbone.training and online.reasoner.training
    assert all(p.grad is None for p in online.parameters())
    # The start token is consumed as z0, not also copied into the context.
    initial = online.embedding(torch.tensor([9]), tokenizer_ids=torch.tensor([0]), tokenizer_names=('text',))
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
    actual = online(trace)["q"]
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
        context, state = embeds[:-1], embeds[-1].float()
        inputs = []
        for _ in range(3):
            inputs.append(reference.reasoner.norm(state.detach()))
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
    assert grads[9].abs().sum() > 0
    assert (grads[[2, 3, 4]].abs().sum(-1) > 0).all()  # Train context tokens too.
    assert grads[[0, 1, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]].count_nonzero() == 0
    assert online.reasoner.norm.weight.grad.abs().sum() > 0


@pytest.mark.flex_training
def test_real_transformer_backward_reaches_all_residual_updates():
    online = model(real=True)
    tokens = batch()
    trace = generate(online, tokens, 'continuous')
    q = online(trace)["q"]
    q.square().sum().backward()
    assert online.reasoner.norm.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.backbone.parameters())
    assert online.embedding.embeddings['text'].embed_tokens.weight.grad[9].abs().sum() > 0
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
    online.reasoner.generate(batch(), embedding=online.embedding, backbone=online.backbone,
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
        ResidualReasoner(hidden_dim=8, max_steps=max_steps)


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
    assert not torch.allclose(generate(online, other).states[:, 0], trace.states[:, 0])


@pytest.mark.parametrize('spec', [dict(type='text', head_output=True), dict(type='token', head_output=False)])
def test_autoregressive_field_requires_one_token_and_a_readout(spec):
    with pytest.raises(ValueError, match='token field with head_output=True'):
        TokenizerModalitySpec(format='9', autoregressive=True, **spec)
    with pytest.raises(ValueError, match='expected 1'):
        tokenizer('8 9')({'provider': 'environment', 'payload': {'action_requested': True}})


def test_notebook_dqn_update_replays_online_trajectory_through_both_models(tmp_path, monkeypatch):
    example = example_module()
    online = model()
    target = online.copy().eval()
    with torch.no_grad():
        target.backbone.linear.bias.add_(.3)
        online.reasoner.stop_head.bias.fill_(1.)
        target.reasoner.stop_head.bias.fill_(-2.)
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
    trace = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        backend='uncached', max_active_requests=2, stop_early=False)
    assert (trace.history_lengths == 1).all() and (trace.lengths == online.reasoner.max_steps).all()
    with torch.no_grad():
        target_states, _ = target.reasoner(trace,
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
    seen_backups = []
    def check_objective(**kwargs):
        seen_backups.append(kwargs['backup_value_predictions'].clone())
        return objective(**kwargs)
    example.run_train.__globals__.update(
        online=online, target=target, LR_EMBEDDING=1e-3, LR_BACKBONE=1e-3, LR_HEADS=1e-3,
        objective=check_objective, loader=loader, device=torch.device('cpu'),
        GENERATION_BACKEND='uncached', MAX_ACTIVE_REQUESTS=2,
        POLYAK_TAU_EMBEDDING=.1, POLYAK_TAU_BACKBONE=.2, POLYAK_TAU_VALUE=.3,
        POLYAK_TAU_REASONING_HEAD=.4, MAX_GRAD_NORM=1., REASONING_STEP_COST=.01)
    configure_notebook_optimizer(example, NOTEBOOK)
    try:
        loss, metrics = example.run_train(num_steps=1)
    finally:
        loader.close()
    assert torch.isfinite(loss) and not loss.requires_grad
    hook.remove()
    assert all(np.isfinite(value) for value in metrics.values()) and calls.count(True) == 1
    assert len(replays) == 2 and replays[0] is replays[1]
    torch.testing.assert_close(seen_backups[0], expected_target)
    assert not torch.equal(original['backbone.linear.weight'], online.backbone.linear.weight)
    assert not torch.equal(original['reasoner.norm.weight'], online.reasoner.norm.weight)
    assert all(p.grad is None for p in target.parameters())
    for name, parameter in target.named_parameters():
        rate = (.1 if name.startswith('embedding.') else .3 if name.startswith('value_head.')
                else .4 if name.startswith('reasoner.stop_head.') else .2)
        torch.testing.assert_close(parameter, target_before[name].lerp(dict(online.named_parameters())[name], rate))
    save_model(model=online, path=tmp_path / 'saved')
    restored = model()
    load_model(model=restored, repo_id_or_path=tmp_path / 'saved')
    torch.testing.assert_close(restored(restored.generate(tokens, backend='uncached', max_active_requests=2))['q'],
                               online(online.generate(tokens, backend='uncached', max_active_requests=2))['q'])


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
    assert json.loads((output / 'residual_settings.json').read_text())['generation_start_token'] == 'word2'
    assert (output / 'tokenizer/tokenizer.json').is_file()
    restored = namespace['online'].copy()
    load_model(model=restored, repo_id_or_path=output)
    for name, parameter in namespace['online'].named_parameters():
        torch.testing.assert_close(dict(restored.named_parameters())[name], parameter)


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_deterministic_halting_handles_mixed_lengths_and_replays_only_valid_updates(monkeypatch, backend):
    online = model(real=True, max_steps=5)
    tokens = batch()
    calls = []
    decisions = iter([[-1., 0., -2.], [0., -1.], [0.]])
    def stops(states):
        if states.ndim == 3:
            return states.new_zeros(states.shape[:2])
        return states.new_tensor(next(decisions))
    monkeypatch.setattr(online.reasoner, 'stop_logits', stops)
    original = online.backbone.forward
    def record(embeds, **kwargs):
        calls.append((embeds.shape, kwargs.get('lengths')))
        return original(embeds, **kwargs)
    monkeypatch.setattr(online.backbone, 'forward', record)
    trace = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
                                     backend=backend, max_active_requests=3)
    assert trace.lengths.tolist() == [2, 1, 3]
    assert len(calls) == 4  # Includes consuming the final stopping token.
    if backend == 'continuous':
        assert [call[1] for call in calls[1:]] == [[1, 1, 1], [1, 0, 1], [0, 0, 1]]
    replay, _ = online.reasoner(trace, embedding=online.embedding, backbone=online.backbone)
    assert calls[-1][0][0] == (2 + 3) + (1 + 2) + (0 + 4)  # Includes stopped states.
    torch.testing.assert_close(replay[:, -1], trace.states[:, -1], atol=3e-5, rtol=3e-5)
    for i, length in enumerate(trace.lengths):
        torch.testing.assert_close(trace.states[i, int(length):],
                                   trace.states[i, int(length)].expand(6 - int(length), -1))


@torch.no_grad()
def test_stop_head_controls_generation_but_training_can_explore_to_cap():
    online = model(max_steps=4)
    online.reasoner.stop_head.bias.fill_(1)
    trace = generate(online, batch())
    assert trace.lengths.tolist() == [1, 1, 1]
    torch.testing.assert_close(trace.states, generate(online, batch()).states)
    full = online.reasoner.generate(batch(), embedding=online.embedding, backbone=online.backbone,
                                    backend='uncached', max_active_requests=2, stop_early=False)
    assert full.lengths.tolist() == [4, 4, 4]
    online.reasoner.stop_head.bias.fill_(-1)
    assert generate(online, batch()).lengths.tolist() == [4, 4, 4]


def test_halting_loss_trains_stop_decisions_from_task_error_and_compute_cost():
    reasoner = ResidualReasoner(hidden_dim=8, max_steps=3)
    logits = torch.zeros(3, 3, requires_grad=True)
    errors = torch.tensor([[9., 4., 1.], [1., 4., 9.], [100., 0., 0.]], requires_grad=True)
    weights = torch.tensor([1., 1., 0.])
    probabilities = reasoner.halting_weights(logits)
    torch.testing.assert_close(probabilities, torch.tensor([[.5, .25, .25]]).expand(3, -1))
    result = reasoner.halting_loss(errors, logits, row_weights=weights, step_cost=0.)
    result['loss'].backward()
    assert logits.grad is not None and errors.grad is not None
    assert logits.grad[0, 0] > 0  # A later, more accurate prediction favors continuing.
    assert logits.grad[1, 0] < 0  # A better first prediction favors stopping.
    assert logits.grad[2].count_nonzero() == 0 and errors.grad[2].count_nonzero() == 0
    assert logits.grad[:, -1].count_nonzero() == 0  # The cap forces remaining mass to stop.
    cost_logits = torch.zeros(1, 3, requires_grad=True)
    cost = reasoner.halting_loss(torch.zeros_like(cost_logits), cost_logits,
                                row_weights=torch.ones(1), step_cost=1.)
    cost['loss'].backward()
    assert cost_logits.grad is not None
    assert (cost_logits.grad[0, :-1] < 0).all()


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_finished_slots_admit_pending_requests_before_other_requests_finish(monkeypatch, backend):
    online = model(real=True, max_steps=3)
    tokens = batch(((2, 3, 9), (4, 9), (1, 7, 8, 9), (6, 9)))
    full = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
                                   backend='uncached', max_active_requests=1, stop_early=False)
    # After consuming its final token, request 1 frees a slot while 0 continues.
    decisions = iter([[-1., 1.], [-1.], [-1., -1.], [1.], [1.]])
    monkeypatch.setattr(online.reasoner, 'stop_logits', lambda states: states.new_tensor(next(decisions)))
    calls, sessions, resets = [], [], []
    original_forward = online.backbone.forward
    original_session = online.backbone.decode_session
    def forward(embeds, **kwargs):
        calls.append(kwargs.get('lengths'))
        return original_forward(embeds, **kwargs)
    def create(**kwargs):
        session = original_session(**kwargs)
        sessions.append(session)
        original_reset = session.reset_rows
        def reset(rows):
            resets.append(list(rows))
            original_reset(rows)
        session.reset_rows = reset
        return session
    monkeypatch.setattr(online.backbone, 'forward', forward)
    monkeypatch.setattr(online.backbone, 'decode_session', create)
    trace = generate(online, tokens, backend)
    assert trace.lengths.tolist() == [3, 1, 2, 1]
    assert len(calls) == 6
    for request, length in enumerate(trace.lengths.tolist()):
        torch.testing.assert_close(trace.states[request, :length + 1], full.states[request, :length + 1],
                                   atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(trace.states[request, length:],
                                   full.states[request, length].expand(4 - length, -1), atol=3e-5, rtol=3e-5)
    if backend == 'continuous':
        assert calls == [[3, 2], [1, 1], [1, 4], [1, 1], [2, 1], [1, 0]]
        assert resets == [[1], [0], [1], [0]]
        assert len(sessions) == 1 and sessions[0]._cache is None


@torch.no_grad()
def test_residual_retains_all_reasoning_in_episode_cache_until_reset(monkeypatch):
    online = model(real=True, max_steps=3).eval()
    session = online.backbone.decode_session(2, max_batch_tokens=8, cache_capacity_tokens=1024)
    histories = [[], []]
    processed = []
    original = online.backbone.forward
    def record(embeds, **kwargs):
        if kwargs.get('cache') is session:
            processed.append(list(kwargs['lengths']))
        return original(embeds, **kwargs)
    monkeypatch.setattr(online.backbone, 'forward', record)
    try:
        for updates in [((2, 3, 9), (4, 9)), ((5, 9), (6, 7, 9)), ((8, 9), (3, 9))]:
            if histories[0] and len(histories[0]) > 4:
                session.reset_rows([0])
                histories[0] = []
            before = session.lengths
            full_tokens = batch(tuple(tuple(h) + u for h, u in zip(histories, updates)))
            full_tokens['head_mask'] = full_tokens['token_ids'] == 9
            full_tokens['autoregressive_mask'] = full_tokens['head_mask'].clone()
            expected = generate(online, full_tokens, 'uncached')
            groups = full_tokens['sequence_ids'][full_tokens['head_mask']]
            latest = torch.stack([(groups == row).nonzero()[-1, 0] for row in range(2)])
            delta = batch(updates)
            processed.clear()
            trace = online.reasoner.generate(delta, embedding=online.embedding, backbone=online.backbone,
                                             backend='continuous', max_active_requests=1, cache=session)
            torch.testing.assert_close(trace.states, expected.states[latest], atol=3e-5, rtol=3e-5)
            if any(before):
                with pytest.raises(ValueError, match='cached prefix'):
                    online(trace)
            for row, update in enumerate(updates):
                histories[row].extend(update)
            assert session.lengths == tuple(len(h) + int(expected.lengths[groups == row].sum())
                                            for row, h in enumerate(histories))
            # Only new events and new reasoning enter; previous tokens are cached.
            assert [sum(call[row] for call in processed) for row in range(2)] == [
                len(update) + 3 for update in updates]
            if before[1]:
                assert session.lengths[1] > before[1]
    finally:
        session.close()


@pytest.mark.parametrize('stop_bias', [-2., 1.])
def test_multi_action_replay_uses_retained_hard_stop_history_and_one_gradient_pass(stop_bias):
    online = model(max_steps=4).eval()
    with torch.no_grad():
        online.reasoner.stop_head.bias.fill_(stop_bias)
    tokens = batch(((2, 9, 3, 9), (4, 9, 5, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    history = generate(online, tokens)
    cap = online.reasoner.max_steps
    expected_length = 1 if stop_bias >= 0 else cap
    assert history.lengths.tolist() == [expected_length] * 4
    candidates = online.reasoner.generate(tokens, embedding=online.embedding, backbone=online.backbone,
        backend='uncached', max_active_requests=2, stop_early=False)
    assert candidates.lengths.tolist() == [cap] * 4
    assert candidates.history_lengths.tolist() == history.lengths.tolist()
    # Exploration after stopping must never change the next decision's context.
    torch.testing.assert_close(candidates.states[:, :expected_length + 1], history.states[:, :expected_length + 1])
    sampled = candidates.tokens["latent_values"].clone().requires_grad_()
    candidates.tokens["latent_values"] = sampled
    calls = []
    hook = online.backbone.register_forward_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    states, logits = online.reasoner(candidates, embedding=online.embedding, backbone=online.backbone)
    hook.remove()
    assert calls == [True]
    torch.testing.assert_close(states, candidates.states[:, 1:])
    q = online.value_head(states[torch.arange(4), online.reasoner.stopping_lengths(logits) - 1])
    torch.testing.assert_close(q, online(online.generate(tokens, backend='uncached'))['q'])
    q.square().sum().backward()
    assert sampled.grad is None
    assert online.embedding.embeddings['text'].embed_tokens.weight.grad[[2, 3, 4, 5]].abs().sum() > 0

    # Independent causal reference explicitly inserts z0...zN between events.
    with torch.no_grad():
        reference = []
        for row in ((2, 9, 3, 9), (4, 9, 5, 9)):
            prefix = []
            for token in row:
                embed = online.embedding(torch.tensor([token]), tokenizer_ids=torch.tensor([0]), tokenizer_names=('text',))
                if token != 9:
                    prefix.append(embed)
                    continue
                z = embed.float()
                prefix.append(online.reasoner.norm(z))
                for _ in range(expected_length):
                    inputs = torch.cat(prefix)
                    z = z + online.backbone(inputs, sequence_ids=torch.zeros(len(inputs), dtype=torch.long))[-1:]
                    prefix.append(online.reasoner.norm(z))
                reference.append(z[0])
        torch.testing.assert_close(history.states[:, -1], torch.stack(reference))


def test_generated_batch_contains_full_episodes_and_isolated_continuations(monkeypatch):
    online = model().eval()
    with torch.no_grad():
        online.reasoner.stop_head.bias.fill_(1.)
    tokens = batch(((2, 9, 3, 9, 7), (4, 9, 5, 9, 6)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    original = tokens.clone()
    generated = online.generate(tokens, backend='uncached', training_candidates=True)
    packed = generated.tokens
    assert packed['num_sequences'] == 6  # Two complete episodes, four continuations.
    assert generated.initial_positions.tolist() == [1, 4, 8, 11]
    assert generated.readout_positions.tolist() == [2, 5, 9, 12]
    assert packed['head_mask'].nonzero().flatten().tolist() == [2, 5, 9, 12]
    assert not packed['autoregressive_mask'].any()
    for group, expected in enumerate(([2, 9, 9, 3, 9, 9, 7], [4, 9, 9, 5, 9, 9, 6])):
        row = packed['sequence_ids'] == group
        assert packed['token_ids'][row].tolist() == expected
        assert packed['latent_mask'][row].tolist() == [False, True, True, False, True, True, False]
    for i in range(4):
        first = generated.update_positions[i, 0]
        extras = generated.update_positions[i, 1:]
        assert packed['sequence_ids'][first] == i // 2
        assert (packed['sequence_ids'][extras] == i + 2).all()
    for key, value in tokens.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, original[key])

    def forbidden(*args, **kwargs):
        raise AssertionError('Forward must never generate')
    monkeypatch.setattr(online.reasoner, 'generate', forbidden)
    before = online(generated)
    # Saved rollout outputs are diagnostics; forward must compute its own outputs.
    different_diagnostics = replace(generated, states=torch.full_like(generated.states, 999.))
    torch.testing.assert_close(online(different_diagnostics)['q_steps'], before['q_steps'])
    # Changing only an exploratory branch cannot enter any later decision's history.
    changed = packed.clone()
    changed['latent_values'][generated.update_positions[0, 1]] += 3.
    after = online(replace(generated, tokens=changed))
    torch.testing.assert_close(after['q'], before['q'])
    torch.testing.assert_close(after['q_steps'][1:], before['q_steps'][1:])
    assert not torch.allclose(after['q_steps'][0, 1:], before['q_steps'][0, 1:])


def test_mixed_stops_align_final_values_actions_rewards_and_notebook_loss(monkeypatch):
    """Independent sequential oracle, then exercise the notebook's real train step."""
    from tensordict import TensorDict

    online = model(max_steps=4).eval()
    target = online.copy().requires_grad_(False).eval()
    with torch.no_grad():
        target.backbone.linear.bias.add_(.3)
        target.reasoner.norm.weight.mul_(1.5)
        target.reasoner.stop_head.bias.fill_(10.)  # Must not choose target stopping positions.
    source_rows = ((2, 9, 3, 9, 7), (4, 9, 5, 9, 6))
    tokens = batch(source_rows)
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    hard_lengths = [1, 2, 3, 4]
    schedule = iter([1., -1., 1., -1., -1., 1., -1., -1., -1., -1.])
    original_stop = online.reasoner.stop_logits
    monkeypatch.setattr(online.reasoner, 'stop_logits', lambda states:
                        original_stop(states) if states.ndim == 3 else states.new_tensor([next(schedule)]))
    generated = online.generate(tokens, backend='uncached', max_active_requests=1, training_candidates=True)
    assert generated.history_lengths.tolist() == hard_lengths
    assert generated.lengths.tolist() == [4] * 4

    def sequential_reference(network):
        results, decision = [], 0
        for row in source_rows:
            history = []
            for token in row:
                embed = network.embedding(torch.tensor([token]), tokenizer_ids=torch.tensor([0]),
                                          tokenizer_names=('text',)).float()
                if token != 9:
                    history.append(embed)
                    continue
                branch, updates = list(history), []
                state = embed
                for step in range(4):
                    # Actual online inputs, evaluated by each network's own norm/backbone.
                    branch.append(network.reasoner.norm(generated.states[decision, step:step + 1].detach()))
                    inputs = torch.cat(branch)
                    state = state + network.backbone(inputs, sequence_ids=torch.zeros(len(inputs), dtype=torch.long))[-1:]
                    updates.append(state[0])
                results.append(torch.stack(updates))
                history.extend(network.reasoner.norm(generated.states[decision, :hard_lengths[decision] + 1].detach()).split(1))
                decision += 1
        return torch.stack(results)

    expected_states = sequential_reference(online)
    with torch.no_grad():
        expected_target_states = sequential_reference(target)
        indices = torch.tensor(hard_lengths) - 1
        expected_target = target.value_head(expected_target_states[torch.arange(4), indices])
    actual = online(generated)
    torch.testing.assert_close(actual['states'], expected_states)
    q_steps = online.value_head(expected_states)
    torch.testing.assert_close(actual['q'], q_steps[torch.arange(4), indices])
    # In particular, the early stops must not use the candidate at the cap.
    assert not torch.allclose(actual['q'][:3], q_steps[:3, -1])
    for i, position in enumerate(generated.readout_positions):
        torch.testing.assert_close(generated.tokens['latent_values'][position], generated.states[i, hard_lengths[i]])

    data = TensorDict(dict(
        action=torch.tensor([float('nan'), 2., 1., float('nan'), 3., 0.]),
        reward=torch.tensor([0., 2., 5., 0., 3., 7.]),
        terminated=torch.tensor([False, False, True, False, False, False]),
        truncated=torch.tensor([False, False, False, False, False, True]),
        sequence_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        step_ids=torch.tensor([0, 1, 3, 4]),
    ), batch_size=[])
    expected_backup = torch.stack((2. + .5 * expected_target[1].max(),
                                  expected_target.new_tensor(5.),
                                  3. + .5 * expected_target[3].max(), expected_target.new_tensor(7.)))
    selected = q_steps[torch.arange(4), :, torch.tensor([2, 1, 3, 0])]
    logits = original_stop(expected_states)
    hazards = logits[:, :-1].sigmoid()
    probability = torch.cat((hazards, torch.ones_like(hazards[:, :1])), 1)
    probability = probability * torch.cat((torch.ones_like(hazards[:, :1]), (1 - hazards).cumprod(1)), 1)
    expected_task = (probability * (selected - expected_backup[:, None]).square()).sum(1).mean()
    expected_cost = .07 * (probability * torch.arange(4)).sum(1).mean()
    objective = DqnObjective(reward=None, value=None, gate=None, temperature=0.,
        discount=boundary_discount(gamma_step=.5, gamma_terminated=0., gamma_truncated=0.))
    seen = []
    def check_objective(**kwargs):
        torch.testing.assert_close(kwargs['predictions'], actual['q'])
        torch.testing.assert_close(kwargs['backup_value_predictions'], expected_target)
        losses, metrics = objective(**kwargs)
        torch.testing.assert_close(metrics['backup'], expected_backup)
        torch.testing.assert_close(metrics['backup_weight'], torch.ones(4))
        seen.append(True)
        return losses, metrics
    def forbidden(*args, **kwargs):
        raise AssertionError('Target must not generate')
    monkeypatch.setattr(target, 'generate', forbidden)
    monkeypatch.setattr(online, 'generate', lambda *args, **kwargs: generated)
    example = example_module()
    example.run_train.__globals__.update(
        online=online, target=target, LR_EMBEDDING=0., LR_BACKBONE=0., LR_HEADS=0.,
        objective=check_objective, loader=SimpleNamespace(next_batch=lambda: (tokens, data)),
        device=torch.device('cpu'), GENERATION_BACKEND='uncached', MAX_ACTIVE_REQUESTS=1,
        POLYAK_TAU_EMBEDDING=0., POLYAK_TAU_BACKBONE=0., POLYAK_TAU_VALUE=0.,
        POLYAK_TAU_REASONING_HEAD=0., MAX_GRAD_NORM=1., REASONING_STEP_COST=.07)
    configure_notebook_optimizer(example, NOTEBOOK)
    loss, metrics = example.run_train(num_steps=1)
    assert seen == [True]
    torch.testing.assert_close(loss, (expected_task + expected_cost).detach())
    assert metrics['dqn_loss'] == pytest.approx(expected_task.item())
    assert metrics['reasoning_cost'] == pytest.approx(expected_cost.item())
    assert all(torch.isfinite(p.grad).all() for p in online.parameters() if p.grad is not None)
    assert all(p.grad is None for p in target.parameters())


@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
@torch.no_grad()
def test_real_transformer_mixed_stops_with_multi_action_training_branches(monkeypatch, backend):
    online = model(real=True, max_steps=4).eval()
    tokens = batch(((2, 9, 3, 9, 7), (4, 9, 5, 9, 6)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    original_stop = online.reasoner.stop_logits
    def rollout(training_candidates):
        schedule = iter([1., -1., 1., -1., -1., 1., -1., -1., -1., -1.])
        monkeypatch.setattr(online.reasoner, 'stop_logits', lambda states:
                            original_stop(states) if states.ndim == 3 else states.new_tensor([next(schedule)]))
        return online.generate(tokens, backend=backend, max_active_requests=1,
                               training_candidates=training_candidates)
    history = rollout(False)
    candidates = rollout(True)
    assert history.lengths.tolist() == candidates.history_lengths.tolist() == [1, 2, 3, 4]
    out = online(candidates)
    torch.testing.assert_close(out['states'], candidates.states[:, 1:], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(out['q'], online.value_head(history.states[:, -1]), atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(out['q'], online(history)['q'], atol=3e-5, rtol=3e-5)
    rows = torch.arange(4)
    torch.testing.assert_close(candidates.tokens['latent_values'][candidates.readout_positions],
                               history.states[rows, history.lengths], atol=3e-5, rtol=3e-5)
