"""Independently check feedback, additions and gradient boundaries."""
import pytest
import torch
from mouse_core.models import SequentialReasoner
from tests._reasoning_input_reference import input_reference
from tests.test_residual_reasoning import batch
from tests.test_sequential_reasoning import components, generate


@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('wrap', [False, True])
def test_pointwise_recurrence_and_detached_feedback_gradients(monkeypatch, residual, wrap):
    parts = components(residual_connections=residual, wrap_latent=wrap, adaptive_depth=False, cap=3)
    embedding, backbone, reasoner, _ = parts
    # No attention: any gradient across updates must come from our additive path.
    monkeypatch.setattr(backbone, 'forward', lambda embeds, **kw: backbone.linear(embeds))
    trace = generate(parts, batch(((2, 9),)))
    initial = reasoner.start_embedding.weight[0]
    state = initial
    expected, expected_inputs = [], []
    for depth in range(3):
        latent = state if wrap else initial
        expected_inputs.append(latent)
        update = backbone.linear(input_reference(reasoner, latent)).float()
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
    assert bool((gradients[:-1] > 0).all()) is (residual and wrap)
    assert gradients[-1] > 0
    reasoner.zero_grad(set_to_none=True)
    # Initial embeddings remain learnable even with wrapping and no addition.
    reasoner(trace, embedding=embedding, backbone=backbone).sum().backward()
    assert reasoner.start_embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_wrap_flag_requires_boolean(value):
    with pytest.raises(ValueError, match='booleans'):
        SequentialReasoner(hidden_dim=8, min_steps=1, max_steps=3, rms_norm_eps=1e-6,
                          residual_connections=False, wrap_latent=value, adaptive_depth=True)


def test_wrap_flag_is_required():
    with pytest.raises(TypeError, match='wrap_latent'):
        SequentialReasoner(hidden_dim=8, min_steps=1, max_steps=3, rms_norm_eps=1e-6,
                          residual_connections=False, adaptive_depth=True)
