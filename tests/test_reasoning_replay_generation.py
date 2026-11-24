"""Known-depth replay batches recorded context and computes only required latents."""
from copy import deepcopy
import importlib

import pytest
import torch

from tests.test_sequential_reasoning import components, generate
from tests.test_residual_reasoning import batch


def source():
    tokens = batch(((2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8), (4, 9, 5, 9), (6, 7)))
    tokens['head_mask'] = tokens['token_ids'] == 9
    tokens['step_readout_mask'] = tokens['head_mask'].clone()
    return tokens


@pytest.mark.parametrize('wrap', ['none', 'step', 'sequence'])
@pytest.mark.parametrize('residual', [False, True])
@pytest.mark.parametrize('backend', ['uncached', 'continuous'])
def test_planned_replay_matches_sequential_latents_and_training(wrap, residual, backend):
    parts = components(residual_connections=residual, adaptive_depth=True, wrap_latent=wrap,
                       cap=3, real=backend == 'continuous')
    embedding, backbone, reasoner, _ = parts
    tokens = source()
    lengths = torch.tensor([1, 1, 3, 1, 2, 1, 1, 1])
    expected = generate(parts, tokens, lengths=lengths, backend=backend)
    planned = reasoner.prepare_replay(tokens, lengths=lengths, embedding=embedding, backbone=backbone,
        backend=backend, max_active_requests=2, max_batch_tokens=3, cache_capacity_tokens=1024)
    assert not planned.states.count_nonzero()
    for key in ('initial_positions', 'update_positions', 'readout_positions', 'lengths'):
        torch.testing.assert_close(getattr(planned, key), getattr(expected, key))
    for key, value in expected.tokens.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(planned.tokens[key], value, atol=3e-5, rtol=3e-5)
    for target in (False, True):
        current_embedding, current_backbone, current_reasoner, _ = deepcopy(parts)
        if target:
            with torch.no_grad():
                current_embedding.embeddings['text'].embed_tokens.weight.mul_(.8)
                current_reasoner.norm.weight.mul_(1.2)
        with torch.set_grad_enabled(backend == 'uncached'):
            actual = current_reasoner(planned, embedding=current_embedding, backbone=current_backbone)
            reference = current_reasoner(expected, embedding=current_embedding, backbone=current_backbone)
        torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)
        if backend == 'uncached':
            actual.square().sum().backward()
            assert current_embedding.embeddings['text'].embed_tokens.weight.grad.abs().sum() > 0
            if wrap != 'none':
                assert current_reasoner.input_projection.down_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('wrap,lengths', [('step', [1] * 8), ('none', [1, 2, 3, 1, 2, 1, 1, 3])])
def test_known_embedding_inputs_need_no_model_or_decode(monkeypatch, wrap, lengths):
    embedding, backbone, reasoner, _ = components(residual_connections=True, adaptive_depth=True,
                                                 wrap_latent=wrap, cap=3)
    def forbidden(*args, **kwargs):
        pytest.fail('Known replay inputs must not run the model')
    monkeypatch.setattr(backbone, 'forward', forbidden)
    monkeypatch.setattr(embedding, 'forward', forbidden)
    monkeypatch.setattr(reasoner, 'project_inputs', forbidden)
    monkeypatch.setattr(backbone, 'decode_session', forbidden)
    trace = reasoner.prepare_replay(source(), lengths=torch.tensor(lengths), embedding=embedding,
                                   backbone=backbone, backend='continuous', max_active_requests=2)
    assert trace.lengths.tolist() == lengths
    assert len(trace.tokens['token_ids']) == len(source()['token_ids']) + sum(n - 1 for n in lengths)
    assert not trace.tokens['latent_values'].count_nonzero()


def test_single_token_spans_and_final_inputs_are_batched(monkeypatch):
    module = importlib.import_module('mouse_core.models.sequential_reasoner')
    original = module.generate
    prompts, chunks = [], []
    def record(**kwargs):
        prompts.extend(len(x) for x in kwargs['prompts'])
        advance = kwargs['advance']
        def record_advance(*args):
            result = advance(*args)
            chunks.extend(None if x is None else len(x) for x in result)
            return result
        return original(**{**kwargs, 'advance': record_advance})
    monkeypatch.setattr(module, 'generate', record)
    embedding, backbone, reasoner, _ = components(residual_connections=True, adaptive_depth=True,
                                                 wrap_latent='step', cap=3)
    trace = reasoner.prepare_replay(source(), lengths=torch.tensor([1, 1, 3, 1, 2, 1, 1, 1]),
        embedding=embedding, backbone=backbone, backend='uncached', max_active_requests=2)
    # Second sequence is all one-token; third has no decisions. Neither decodes.
    assert prompts == [6]
    assert chunks == [1, 5, None]
    assert trace.lengths.tolist() == [1, 1, 3, 1, 2, 1, 1, 1]


@pytest.mark.parametrize('lengths', [torch.ones(7, dtype=torch.long), torch.ones(8),
                                    torch.zeros(8, dtype=torch.long), torch.full((8,), 4)])
def test_planned_depths_are_validated(lengths):
    embedding, backbone, reasoner, _ = components(residual_connections=False, adaptive_depth=True,
                                                 wrap_latent='step', cap=3)
    with pytest.raises(ValueError, match='Replay lengths'):
        reasoner.prepare_replay(source(), lengths=lengths, embedding=embedding, backbone=backbone,
                               backend='uncached', max_active_requests=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_planned_continuous_gpu_replay_and_backward():
    from mouse_core.models import Backbone
    embedding, _, reasoner, head = components(residual_connections=True, adaptive_depth=True,
                                               wrap_latent='step', cap=3)
    backbone = Backbone(architecture='qwen3', hidden_dim=8, num_layers=1, num_heads=2,
                        head_dim=16, dtype=torch.float32).cuda()
    parts = embedding.cuda(), backbone, reasoner.cuda(), head.cuda()
    tokens = source().to('cuda')
    lengths = torch.tensor([1, 1, 3, 1, 2, 1, 1, 1], device='cuda')
    expected = generate(parts, tokens, lengths=lengths, backend='continuous')
    planned = reasoner.prepare_replay(tokens, lengths=lengths, embedding=embedding, backbone=backbone,
        backend='continuous', max_active_requests=2, max_batch_tokens=16, cache_capacity_tokens=1024)
    torch.testing.assert_close(planned.tokens['latent_values'], expected.tokens['latent_values'], atol=1e-4, rtol=1e-4)
    backbone.requires_grad_(False)
    actual = reasoner(planned, embedding=embedding, backbone=backbone)
    actual.square().sum().backward()
    assert embedding.embeddings['text'].embed_tokens.weight.grad.abs().sum() > 0
    assert reasoner.input_projection.down_proj.weight.grad.abs().sum() > 0
    assert torch.isfinite(actual).all()


def test_fixed_depth_planning_matches_generation():
    embedding, backbone, reasoner, head = parts = components(residual_connections=True,
        adaptive_depth=False, wrap_latent='step', cap=3)
    tokens = source()
    lengths = torch.full((8,), 3, dtype=torch.long)
    expected = generate(parts, tokens)
    actual = reasoner.prepare_replay(tokens, lengths=lengths, embedding=embedding, backbone=backbone,
                                    backend='uncached', max_active_requests=2)
    torch.testing.assert_close(actual.tokens['latent_values'], expected.tokens['latent_values'])
    with pytest.raises(ValueError, match='Fixed-depth'):
        reasoner.prepare_replay(tokens, lengths=torch.ones_like(lengths), embedding=embedding,
                                backbone=backbone, backend='uncached', max_active_requests=2)
