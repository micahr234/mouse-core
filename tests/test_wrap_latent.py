"""Independently check feedback, additions and gradient boundaries."""
import pytest
import torch
from mouse_core.models import SequentialReasoner
from tests._reasoning_input_reference import input_reference
from tests.test_residual_reasoning import batch
from tests.test_sequential_reasoning import components, generate


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('wrap', ['none', 'step'])
def test_pointwise_recurrence_and_detached_feedback_gradients(monkeypatch, residual, wrap):
    parts = components(residual_connections=residual, wrap_latent=wrap, adaptive_depth=False, cap=3)
    embedding, backbone, reasoner, _ = parts
    # No attention: any gradient across updates must come from our additive path.
    monkeypatch.setattr(backbone, 'forward', lambda embeds, **kw: backbone.linear(embeds))
    trace = generate(parts, batch(((2, 9),)))
    initial = torch.zeros_like(embedding.embeddings["text"].embed_tokens.weight[9:10][0])
    state = initial
    expected, expected_inputs = [], []
    for depth in range(3):
        latent = state if wrap != "none" else initial
        expected_inputs.append(latent)
        update = backbone.linear(input_reference(reasoner, latent, embedding.embeddings["text"].embed_tokens.weight[9])).float()
        state = latent + update if residual else update
        expected.append(state)
    torch.testing.assert_close(trace.states[0, 1:], torch.stack(expected))
    torch.testing.assert_close(trace.tokens['latent_values'][trace.update_positions[0]], torch.stack(expected_inputs))
    # Feedback can never silently reintroduce serial backpropagation.
    trace.tokens['latent_values'].requires_grad_()
    outputs = []
    def capture(module, args, out):
        out.retain_grad()
        outputs.append(out)
    hook = backbone.linear.register_forward_hook(capture)
    states = reasoner(trace, embedding=embedding, backbone=backbone)
    hook.remove()
    torch.testing.assert_close(states, trace.states[:, 1:])
    states[:, -1].sum().backward()
    assert trace.tokens['latent_values'].grad is None
    gradients = outputs[0].grad[trace.update_positions[0]].abs().sum(-1)
    assert bool((gradients[:-1] > 0).all()) is (residual and wrap != "none")
    assert gradients[-1] > 0
    # Even final-only loss without a residual reaches the injected offset.
    assert embedding.embeddings["text"].embed_tokens.weight.grad[9:10].abs().sum() > 0
    reasoner.zero_grad(set_to_none=True)
    # Initial embeddings remain learnable even with wrapping and no addition.
    reasoner(trace, embedding=embedding, backbone=backbone).sum().backward()
    assert embedding.embeddings["text"].embed_tokens.weight.grad[9:10].abs().sum() > 0


@pytest.mark.parametrize('value', [None, False, True, 0, 1, 'true', 'invalid'])
def test_wrap_mode_requires_supported_string(value):
    with pytest.raises(ValueError, match='wrap_latent'):
        SequentialReasoner(input_mlp_intermediate_dim=24, input_mlp_output_scale=100., hidden_dim=8, min_steps=1, max_steps=3, rms_norm_eps=1e-6,
                          residual_connections=False, wrap_latent=value, adaptive_depth=True)


def test_wrap_flag_is_required():
    with pytest.raises(TypeError, match='wrap_latent'):
        SequentialReasoner(input_mlp_intermediate_dim=24, input_mlp_output_scale=100., hidden_dim=8, min_steps=1, max_steps=3, rms_norm_eps=1e-6,
                          residual_connections=False, adaptive_depth=True)


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('cap', [1, 3])
def test_sequence_carries_only_reasoning_states_and_rebuilds_gradient_path(monkeypatch, residual, cap):
    parts = components(residual_connections=residual, wrap_latent="sequence", adaptive_depth=False, cap=cap)
    embedding, backbone, reasoner, _ = parts
    monkeypatch.setattr(backbone, 'forward', lambda embeds, **kw: backbone.linear(embeds))
    tokens = batch(((2, 9, 3, 4, 9), (5, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    trace = generate(parts, tokens)
    torch.testing.assert_close(trace.states[1, 0], trace.states[0, cap])
    torch.testing.assert_close(trace.states[[0, 2], 0], torch.zeros(2, reasoner.hidden_dim))
    torch.testing.assert_close(trace.tokens['latent_values'][trace.initial_positions[1]], trace.states[0, cap])
    trace.tokens['latent_values'].requires_grad_()
    outputs = []
    def capture(module, args, out):
        out.retain_grad()
        outputs.append(out)
    hook = backbone.linear.register_forward_hook(capture)
    states = reasoner(trace, embedding=embedding, backbone=backbone)
    hook.remove()
    torch.testing.assert_close(states, trace.states[:, 1:])
    states[1, -1].sum().backward()
    assert trace.tokens['latent_values'].grad is None
    gradients = outputs[0].grad[trace.update_positions].abs().sum(-1)
    assert bool((gradients[0] > 0).all()) is residual
    assert gradients[1, -1] > 0
    assert gradients[2].count_nonzero() == 0


@pytest.mark.parametrize('wrap', ['none', 'step', 'sequence'])
def test_replay_injects_current_offset_at_all_positions_without_changing_saved_feedback(monkeypatch, wrap):
    parts = components(residual_connections=False, wrap_latent=wrap, adaptive_depth=False, cap=3)
    embedding, backbone, reasoner, _ = parts
    tokens = batch(((2, 9, 3, 9), (5, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    trace = generate(parts, tokens)
    saved = trace.tokens['latent_values'].clone()
    received = []
    def capture(inputs, **kwargs):
        received.append(inputs.detach().clone())
        return backbone.linear(inputs)
    monkeypatch.setattr(backbone, 'forward', capture)
    reasoner(trace, embedding=embedding, backbone=backbone)
    with torch.no_grad():
        embedding.embeddings["text"].embed_tokens.weight[9:10].add_(.125)
    reasoner(trace, embedding=embedding, backbone=backbone)
    mask = trace.tokens['latent_mask']
    torch.testing.assert_close(received[1][mask] - received[0][mask], torch.full_like(received[0][mask], .125))
    torch.testing.assert_close(received[1][~mask], received[0][~mask])
    torch.testing.assert_close(trace.tokens['latent_values'], saved)
