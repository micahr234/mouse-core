"""Only scope starts use token embeddings; even zero continuations use the MLP."""
import pytest
import torch
from tests.test_residual_reasoning import batch
from tests.test_sequential_reasoning import components, generate


@pytest.mark.parametrize('wrap', ['none', 'step', 'sequence'])
@pytest.mark.parametrize('residual', [False, True])
def test_zero_continuation_never_reinjects_embedding(monkeypatch, wrap, residual):
    parts = components(residual_connections=residual, wrap_latent=wrap,
                       adaptive_depth=False, cap=3)
    embedding, backbone, reasoner, _ = parts
    monkeypatch.setattr(backbone, 'forward', lambda inputs, **kwargs: inputs * 0)
    tokens = batch(((2, 9, 3, 9), (4, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    generated_inputs = []
    original = backbone.forward
    def capture(inputs, **kwargs):
        generated_inputs.append(inputs.detach().clone())
        return original(inputs, **kwargs)
    monkeypatch.setattr(backbone, 'forward', capture)
    trace = generate(parts, tokens)
    learned = embedding.embeddings['text'].embed_tokens.weight[9]
    assert len(generated_inputs) == 6
    torch.testing.assert_close(generated_inputs[0][-1], learned)
    torch.testing.assert_close(generated_inputs[1][-1],
                               learned if wrap == 'none' else torch.zeros_like(learned))
    # The second decision restarts only in step/none mode. A zero carried
    # state is still a continuation in sequence mode.
    torch.testing.assert_close(generated_inputs[3][-1],
                               torch.zeros_like(learned) if wrap == 'sequence' else learned)
    generated_inputs.clear()
    reasoner(trace, embedding=embedding, backbone=backbone)
    assert len(generated_inputs) == 1
    inputs = generated_inputs[0]
    learned = embedding.embeddings['text'].embed_tokens.weight[9]
    assert learned.abs().sum() > 0
    positions = trace.tokens['latent_mask'].nonzero().flatten()
    starts = (positions if wrap == 'none' else trace.initial_positions
              if wrap == 'step' else trace.initial_positions[[0, 2]])
    for position in positions.tolist():
        expected = learned if position in starts.tolist() else torch.zeros_like(learned)
        torch.testing.assert_close(inputs[position], expected)
    torch.testing.assert_close(reasoner(trace, embedding=embedding, backbone=backbone), trace.states[:, 1:])


@pytest.mark.parametrize('wrap', ['none', 'step'])
def test_one_step_bypasses_input_block_in_generation_and_replay(monkeypatch, wrap):
    parts = components(residual_connections=True, wrap_latent=wrap,
                       adaptive_depth=False, cap=1)
    embedding, backbone, reasoner, _ = parts
    def forbidden(*args, **kwargs):
        raise AssertionError('The first input must bypass the entire input block')
    monkeypatch.setattr(reasoner, 'project_inputs', forbidden)
    trace = generate(parts, batch(((2, 9), (3, 9))))
    states = reasoner(trace, embedding=embedding, backbone=backbone)
    torch.testing.assert_close(states, trace.states[:, 1:])
    states.square().sum().backward()
    assert embedding.embeddings['text'].embed_tokens.weight.grad[9].abs().sum() > 0
    assert all(parameter.grad is None for parameter in reasoner.parameters())


@pytest.mark.parametrize('wrap', ['step', 'sequence'])
@pytest.mark.parametrize('cap', [1, 3])
@torch.no_grad()
def test_bfloat16_embeddings_match_across_generation_and_replay(wrap, cap):
    parts = components(residual_connections=True, wrap_latent=wrap,
                       adaptive_depth=False, cap=cap, real=True)
    embedding, backbone, reasoner, head = parts
    embedding.to(dtype=torch.bfloat16)
    tokens = batch(((2, 9, 3, 9), (4, 9)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    traces = [reasoner.generate(tokens, embedding=embedding, backbone=backbone,
                               action_values=head, backend=backend, max_active_requests=2)
              for backend in ('uncached', 'continuous')]
    for trace in traces:
        replayed = reasoner(trace, embedding=embedding, backbone=backbone)
        torch.testing.assert_close(replayed, trace.states[:, 1:], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(traces[0].states, traces[1].states, atol=2e-5, rtol=2e-5)
