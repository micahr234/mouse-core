"""Independent token, recurrence, cache and gradient checks for reasoning variants."""
from copy import deepcopy

import pytest
import torch
from tests._reasoning_input_reference import input_reference
from torch import nn

from mouse_core.models import Backbone, Embedding, MultiEmbedding, SequentialReasoner
from mouse_core.objectives import final_reasoning_loss, value_reasoning_loss
from tests.test_residual_reasoning import CausalBackbone, batch


VARIANTS = [(True, False), (False, True), (True, True)]


def components(*, residual_connections, adaptive_depth, wrap_latent, cap=3, real=False):
    torch.manual_seed(73)
    embedding = MultiEmbedding(embeddings={'text': Embedding(vocab_size=16, hidden_dim=8)})
    backbone = (Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                         head_dim=4, max_position_embeddings=128, dtype=torch.float32)
                if real else CausalBackbone())
    reasoner = SequentialReasoner(min_steps=1, hidden_dim=8, max_steps=cap,
                                 residual_connections=residual_connections, wrap_latent=wrap_latent, adaptive_depth=adaptive_depth, rms_norm_eps=1e-6)
    with torch.no_grad():
        reasoner.input_projection.weight.copy_(torch.eye(8).roll(1, 0) * .7)
        reasoner.input_projection.bias.copy_(torch.linspace(-.03, .05, 8))
    head = nn.Linear(8, 4 + adaptive_depth)
    return embedding, backbone, reasoner, head


def generate(parts, tokens, *, lengths=None, backend='uncached', cache=None, calls=None):
    embedding, backbone, reasoner, head = parts
    def select(values, indices, depths):
        if calls is not None:
            calls.extend(zip(indices.tolist(), depths.tolist(), strict=True))
        if not reasoner.adaptive_depth:
            assert torch.isfinite(values[:, -1]).all()
            return torch.full_like(depths, 3)
        desired = (torch.full_like(depths, reasoner.max_steps) if lengths is None else lengths[indices])
        return torch.where(depths < desired, 4, 2)
    return reasoner.generate(tokens, embedding=embedding, backbone=backbone, action_values=head,
                             select_actions=select, backend=backend, max_active_requests=2, cache=cache)


def reference_states(parts, tokens, generated):
    """Build literal causal histories, independent of materialized position metadata."""
    embedding, backbone, reasoner, _ = parts
    table = embedding.embeddings['text'].embed_tokens.weight
    outputs = []
    decision = 0
    for group in range(tokens['num_sequences']):
        history = []
        selected = tokens['sequence_ids'] == group
        for token, is_head in zip(tokens['token_ids'][selected].tolist(), tokens['head_mask'][selected].tolist(), strict=True):
            if not is_head:
                history.append(table[token])
                continue
            state = reasoner.start_embedding.weight[0]
            per_depth = []
            for depth in range(int(generated.lengths[decision])):
                latent = (generated.states[decision, depth].detach()
                          if reasoner.wrap_latent and depth > 0 else reasoner.start_embedding.weight[0])
                history.append(input_reference(reasoner, latent))
                packed = torch.stack(history)
                hidden = backbone(packed, sequence_ids=torch.zeros(len(packed), dtype=torch.long))[-1]
                update = hidden.float()
                base = state if reasoner.wrap_latent else reasoner.start_embedding.weight[0]
                state = base + update if reasoner.residual_connections else update
                per_depth.append(state)
            if reasoner.wrap_latent and reasoner.max_steps > 1:
                history.append(input_reference(reasoner, generated.states[decision, int(generated.lengths[decision])].detach()))
            per_depth.extend([per_depth[-1]] * (reasoner.max_steps - len(per_depth)))
            outputs.append(torch.stack(per_depth))
            decision += 1
    return torch.stack(outputs)


@pytest.mark.parametrize('residual_connections,adaptive_depth', [(r, a) for r in (False, True) for a in (False, True)])
@pytest.mark.parametrize('wrap_latent', [False, True])
@pytest.mark.parametrize('cap', [1, 3])
def test_variant_positions_and_outputs_match_literal_token_history(residual_connections, adaptive_depth, cap, wrap_latent):
    parts = components(residual_connections=residual_connections, wrap_latent=wrap_latent, adaptive_depth=adaptive_depth, cap=cap)
    embedding, backbone, reasoner, _ = parts
    tokens = batch(((2, 9, 3, 9, 4), (5, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    lengths = torch.tensor([1, cap, min(2, cap)] if adaptive_depth else [cap] * 3)
    calls = []
    trace = generate(parts, tokens, lengths=lengths, calls=calls)
    assert trace.lengths.tolist() == lengths.tolist()
    assert sorted(calls) == sorted((i, k) for i, length in enumerate(lengths.tolist())
                                  for k in (range(1, length + 1) if adaptive_depth else [cap]))
    assert len(trace.tokens['token_ids']) == len(tokens['token_ids']) + int((lengths - 1 + (wrap_latent and cap > 1)).sum())
    assert trace.readout_positions.tolist() == (trace.initial_positions + lengths - 1 + (wrap_latent and cap > 1)).tolist()
    if not wrap_latent:
        table = embedding.embeddings['text'].embed_tokens.weight
        torch.testing.assert_close(trace.tokens['latent_values'][trace.tokens['latent_mask']],
                                   reasoner.start_embedding.weight.expand(int(lengths.sum()), -1))
    actual = reasoner(trace, embedding=embedding, backbone=backbone)
    torch.testing.assert_close(actual, reference_states(parts, tokens, trace))
    torch.testing.assert_close(actual, trace.states[:, 1:])

    target = deepcopy(parts)
    with torch.no_grad():
        target[0].embeddings['text'].embed_tokens.weight.add_(.4)
        target[1].linear.weight.mul_(.7)
        target[2].norm.weight.mul_(1.5)
        target[2].start_embedding.weight.add_(.1)
        target[2].input_projection.weight.mul_(.8)
    delayed = target[2](trace, embedding=target[0], backbone=target[1])
    torch.testing.assert_close(delayed, reference_states(target, tokens, trace))
    assert not torch.allclose(delayed, actual)


@pytest.mark.parametrize('residual_connections,adaptive_depth', [(r, a) for r in (False, True) for a in (False, True)])
@pytest.mark.parametrize('wrap_latent', [False, True])
@pytest.mark.parametrize('cap', [1, 3])
@torch.no_grad()
def test_real_cache_retains_exact_generated_token_count_and_replays(residual_connections, adaptive_depth, cap, wrap_latent):
    parts = components(residual_connections=residual_connections, wrap_latent=wrap_latent, adaptive_depth=adaptive_depth, cap=cap, real=True)
    embedding, backbone, reasoner, _ = parts
    first = batch(((2, 9), (4, 5, 9)))
    second = batch(((3, 9), (6, 9)))
    full = batch(((2, 9, 3, 9), (4, 5, 9, 6, 9)))
    full['head_mask'] = full['token_ids'] == 9
    full['autoregressive_mask'] = full['head_mask'].clone()
    full['step_readout_mask'] = full['head_mask'].clone()
    session = backbone.decode_session(2, max_batch_tokens=3, cache_capacity_tokens=1024)
    try:
        generated_first = generate(parts, first, backend='continuous', cache=session)
        assert list(session.lengths) == [2 + cap - 1 + (wrap_latent and cap > 1), 3 + cap - 1 + (wrap_latent and cap > 1)]
        generated_second = generate(parts, second, backend='continuous', cache=session)
        assert list(session.lengths) == [4 + 2 * (cap - 1 + (wrap_latent and cap > 1)), 5 + 2 * (cap - 1 + (wrap_latent and cap > 1))]
        expected = generate(parts, full)
        torch.testing.assert_close(generated_first.states, expected.states[[0, 2]], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(generated_second.states, expected.states[[1, 3]], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(reasoner(expected, embedding=embedding, backbone=backbone),
                                   expected.states[:, 1:], atol=3e-5, rtol=3e-5)
        session.reset_rows([0, 1])
        reset = generate(parts, second, backend='continuous', cache=session)
        torch.testing.assert_close(reset.states, generate(parts, second).states, atol=3e-5, rtol=3e-5)
    finally:
        session.close()


def test_fixed_depth_evaluates_environment_head_and_selector_only_at_final_update():
    parts = components(residual_connections=True, wrap_latent=True, adaptive_depth=False, cap=4)
    embedding, backbone, reasoner, head = parts
    evaluations, choices = [], []
    hook = head.register_forward_pre_hook(lambda _, args: evaluations.append(args[0].clone()))
    try:
        trace = generate(parts, batch(), calls=choices)
    finally:
        hook.remove()
    assert sorted(choices) == [(0, 4), (1, 4), (2, 4)]
    assert sum(len(states) for states in evaluations) == 3
    assert trace.lengths.tolist() == [4, 4, 4]


def test_final_only_head_loss_reaches_all_residual_updates_and_no_intermediate_head_outputs():
    embedding, backbone, reasoner, head = parts = components(residual_connections=True, wrap_latent=True, adaptive_depth=False)
    trace = generate(parts, batch(((2, 9),)))
    outputs = []
    def capture(_, args, output):
        output.retain_grad()
        outputs.append(output)
    hook = backbone.register_forward_hook(capture)
    states = reasoner(trace, embedding=embedding, backbone=backbone)
    hook.remove()
    values = head(states)
    values.retain_grad()
    selected = values[:, -1, 2]
    loss, details = final_reasoning_loss(q_values=selected, environment_targets=torch.tensor([3.]),
                                       lengths=trace.lengths, weights=torch.ones(1), reasoning_reward=-.2)
    loss.backward()
    assert values.grad[:, :-1].count_nonzero() == 0
    assert values.grad[0, -1, 2] != 0
    assert (outputs[0].grad[trace.update_positions].abs().sum(-1) > 0).all()
    assert outputs[0].grad[trace.readout_positions].count_nonzero() == 0
    assert details['td_target'].item() == pytest.approx(2.6)


def test_value_reasoning_trains_each_generated_output_and_repeated_initial_embedding():
    embedding, backbone, reasoner, head = parts = components(residual_connections=False, wrap_latent=False, adaptive_depth=True)
    trace = generate(parts, batch(((2, 9),)))
    states = reasoner(trace, embedding=embedding, backbone=backbone)
    values = head(states)
    values.retain_grad()
    loss, _ = value_reasoning_loss(min_steps=1, q_values=values, target_q_values=values.detach(),
                                  lengths=trace.lengths, environment_actions=torch.tensor([2]),
                                  environment_targets=torch.tensor([3.]), weights=torch.ones(1),
                                  reasoning_reward=-.2, gamma_think=1.)
    loss.backward()
    nonzero = values.grad != 0
    assert nonzero[0].nonzero().tolist() == [[0, 4], [1, 4], [2, 2]]
    assert reasoner.start_embedding.weight.grad.abs().sum() > 0
    assert embedding.embeddings['text'].embed_tokens.weight.grad[9].count_nonzero() == 0
    assert reasoner.norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('residual_connections,adaptive_depth', [(0, True), (True, 1), (None, False)])
def test_reasoner_requires_boolean_mechanism_settings(residual_connections, adaptive_depth):
    with pytest.raises(ValueError, match='booleans'):
        SequentialReasoner(min_steps=1, hidden_dim=8, max_steps=2, residual_connections=residual_connections, wrap_latent=residual_connections,
                           adaptive_depth=adaptive_depth, rms_norm_eps=1e-6)


@pytest.mark.parametrize('residual_connections', [False, True])
def test_input_projection_standard_initialization_and_zero_head_gradient_startup(residual_connections):
    torch.manual_seed(19)
    reasoner = SequentialReasoner(hidden_dim=8, min_steps=1, max_steps=1,
                                  residual_connections=residual_connections, wrap_latent=residual_connections, adaptive_depth=True, rms_norm_eps=1e-6)
    assert not hasattr(reasoner, "update_norm")
    for parameter in reasoner.input_projection.parameters():
        assert parameter.count_nonzero() > 0
        assert parameter.abs().max() <= 8 ** -.5
    hidden = torch.randn(3, 8, requires_grad=True)
    head = nn.Linear(8, 5)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.SGD([*reasoner.parameters(), *head.parameters()], lr=.01)
    before = reasoner.input_projection.weight.detach().clone()
    for step in range(2):
        optimizer.zero_grad()
        hidden.grad = None
        updates = reasoner.project_inputs(hidden)
        expected_norm = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6) * reasoner.norm.weight
        expected = torch.nn.functional.linear(expected_norm, reasoner.input_projection.weight,
                                              reasoner.input_projection.bias)
        torch.testing.assert_close(updates, expected)
        states = updates + reasoner.start_embedding.weight if residual_connections else updates
        values = head(states)
        loss, _ = value_reasoning_loss(
            q_values=values[:, None], target_q_values=values.detach()[:, None],
            lengths=torch.ones(3, dtype=torch.long), environment_actions=torch.tensor([0, 1, 2]),
            environment_targets=torch.tensor([1., -2., 3.]), weights=torch.ones(3),
            reasoning_reward=-0., gamma_think=1., min_steps=1)
        loss.backward()
        assert head.weight.grad.abs().sum() > 0
        if step == 1:
            assert reasoner.input_projection.weight.grad.abs().sum() > 0
            assert hidden.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(reasoner.input_projection.weight, before)


@pytest.mark.parametrize('residual_connections', [False, True])
@pytest.mark.parametrize('cap', [1, 3, 5])
@torch.no_grad()
def test_rollout_defers_unused_feedback_without_extra_decoder_call(monkeypatch, residual_connections, cap):
    parts = components(residual_connections=residual_connections, wrap_latent=residual_connections, adaptive_depth=True, cap=cap, real=True)
    session = parts[1].decode_session(2, max_batch_tokens=64, cache_capacity_tokens=1024)
    calls = []
    decode = session._decode
    def counted(*args, **kwargs):
        calls.append(None)
        return decode(*args, **kwargs)
    monkeypatch.setattr(session, '_decode', counted)
    try:
        for _ in range(2):
            before = len(calls)
            generate(parts, batch(((2, 9), (4, 9))), backend='continuous', cache=session)
            assert len(calls) - before == cap
            assert set(session._deferred) == ({0, 1} if residual_connections and cap > 1 else set())
    finally:
        session.close()


@pytest.mark.parametrize('residual,adaptive', VARIANTS)
@torch.no_grad()
def test_one_step_uses_one_packed_forward_and_never_enters_generation(monkeypatch, residual, adaptive):
    import importlib
    module = importlib.import_module('mouse_core.models.sequential_reasoner')
    parts = components(residual_connections=residual, wrap_latent=residual, adaptive_depth=adaptive, cap=1)
    tokens = batch(((2, 9, 3, 9), (4, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['autoregressive_mask'] = tokens['head_mask'].clone()
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    def forbidden(*args, **kwargs):
        raise AssertionError('One-step reasoning must not enter the sequential generator')
    monkeypatch.setattr(module, 'generate', forbidden)
    calls = []
    hook = parts[1].register_forward_hook(lambda *args: calls.append(1))
    trace = generate(parts, tokens, backend='continuous')
    hook.remove()
    assert calls == [1]
    assert trace.tokens['token_ids'].tolist() == tokens['token_ids'].tolist()
    assert trace.initial_positions.tolist() == trace.readout_positions.tolist() == [1, 3, 5]
    torch.testing.assert_close(trace.states[:, 1:], reference_states(parts, tokens, trace))
