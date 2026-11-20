"""Core-only storage, packing and live sampling tests; no experiment dependency."""
from dataclasses import replace

import pytest
import torch

from mouse_core.data import (
    ReasoningDataset, ReasoningDataLoader, split_reasoning_episodes, pack_reasoning_episodes,
    start_only_reasoning_episodes,
)
from tests.test_reasoning_segments import segmented_batch
from tests.test_sequential_reasoning import components, generate


def prepared(residual=True):
    inputs, data = segmented_batch()
    # The dataset receives whole nonempty episode samples, not empty decode slots.
    inputs['sequence_ids'][inputs['sequence_ids'] == 2] = 1
    inputs.set_non_tensor('num_sequences', 2)
    data['sequence_ids'][data['sequence_ids'] == 2] = 1
    parts = components(residual_connections=residual, adaptive_depth=True)
    trace = generate(parts, inputs, lengths=torch.tensor([2, 3, 1]))
    return parts, inputs, data, trace


def bank_and_loader(episode, *, sampling='sequential', sample_budget=2, token_budget=None, capacity=4):
    dataset = ReasoningDataset(capacity_episodes=capacity)
    for step in range(capacity):
        dataset.append(replace(episode, generated_step=step))
    loader = ReasoningDataLoader(dataset=dataset, sampling=sampling, sample_budget=sample_budget,
                                token_budget=token_budget, seed=17)
    return dataset, loader


@pytest.mark.parametrize('residual', [False, True])
def test_sparse_split_pack_roundtrip_reorders_duplicates_and_replays_live_weights(residual):
    parts, inputs, data, trace = prepared(residual)
    records = list(split_reasoning_episodes(inputs, trace, data, 7))
    assert all(e.latent_values.device.type == 'cpu' and not e.latent_values.requires_grad for e in records)
    assert all('latent_values' not in e.tokens.keys() for e in records)
    rebuilt, aligned = pack_reasoning_episodes(records)
    for key, value in trace.tokens.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(rebuilt.tokens[key], value)
    assert aligned['step_ids'].tolist() == [0, 2, 4]
    for name in ('initial_positions', 'update_positions', 'readout_positions', 'lengths'):
        torch.testing.assert_close(getattr(rebuilt, name), getattr(trace, name))
    assert rebuilt.head_token_ids == trace.head_token_ids
    embedding, backbone, reasoner, _ = parts
    evaluate = lambda x: reasoner(x, embedding=embedding, backbone=backbone)
    torch.testing.assert_close(evaluate(rebuilt), evaluate(trace))
    reordered, labels = pack_reasoning_episodes([records[1], records[0], records[1]])
    assert labels['step_ids'].tolist() == [0, 1, 3, 5]
    expected = torch.cat([evaluate(pack_reasoning_episodes([record])[0])
                          for record in [records[1], records[0], records[1]]])
    torch.testing.assert_close(evaluate(reordered), expected)
    frozen_vectors = records[0].latent_values.clone()
    with torch.no_grad():
        reasoner.update_projection.weight.mul_(1.2)
    output = evaluate(rebuilt)
    output.square().sum().backward()
    assert reasoner.update_projection.weight.grad.abs().sum() > 0
    assert reasoner.start_embedding.weight.grad.abs().sum() > 0
    torch.testing.assert_close(records[0].latent_values, frozen_vectors)


def test_dataset_add_aligns_source_labels_and_owns_detached_payloads():
    _, inputs, data, trace = prepared()
    dataset = ReasoningDataset(capacity_episodes=1000)
    records = dataset.add(inputs, trace, data, generated_step=4)
    assert len(dataset) == 2
    assert dataset.token_count == len(trace.tokens['token_ids'])
    assert all(r.generated_step == 4 for r in records)
    expected = records[0].latent_values.clone()
    trace.tokens['latent_values'].zero_()
    torch.testing.assert_close(dataset[0].latent_values, expected)
    data['reward'].zero_()
    assert dataset[0].data['reward'].sum() > 0
    assert dataset.memory_bytes() > 0
    dataset.clear()
    assert len(dataset) == dataset.token_count == dataset.memory_bytes() == 0


def test_sequential_cursor_survives_eviction_and_multiple_loaders_are_independent():
    _, inputs, data, trace = prepared()
    episode = next(split_reasoning_episodes(inputs, trace, data, 0))
    dataset, first = bank_and_loader(episode)
    second = ReasoningDataLoader(dataset=dataset, sample_budget=1, token_budget=None,
                                 sampling='sequential', seed=2)
    first.next_batch()
    assert first.sampled_steps == [0, 1]
    second.next_batch()
    assert second.sampled_steps == [0]
    dataset.append(replace(episode, generated_step=4))
    assert [e.generated_step for e in dataset] == [1, 2, 3, 4]
    first.next_batch()
    assert first.sampled_steps == [2, 3]
    second.next_batch()
    assert second.sampled_steps == [1]
    first.next_batch()
    assert first.sampled_steps == [4, 1]
    for step in range(5, 9):
        dataset.append(replace(episode, generated_step=step))
    first.next_batch()
    assert first.sampled_steps == [5, 6]
    first.close()
    assert len(dataset) == 4
    dataset.clear()
    dataset.append(episode)
    first.next_batch()
    assert first.sampled_steps == [0, 0]


@pytest.mark.parametrize('sampling', ['sequential', 'random'])
def test_small_initial_dataset_serves_full_batches_without_consumption(sampling):
    _, inputs, data, trace = prepared()
    episode = next(split_reasoning_episodes(inputs, trace, data, 0))
    dataset, loader = bank_and_loader(episode, sampling=sampling, sample_budget=8, capacity=1)
    for _ in range(5):
        batch, _ = loader.next_batch()
        assert batch.tokens['num_sequences'] == 8
        assert loader.sampled_steps == [0] * 8
    assert len(dataset) == 1
    assert dataset.token_count == episode.token_count


def test_random_sampling_seed_and_fifo_order_are_independent():
    _, inputs, data, trace = prepared()
    episode = next(split_reasoning_episodes(inputs, trace, data, 0))
    dataset, first = bank_and_loader(episode, sampling='random', sample_budget=10)
    _, second = bank_and_loader(episode, sampling='random', sample_budget=10)
    draws = []
    for _ in range(5):
        first.next_batch()
        second.next_batch()
        assert first.sampled_steps == second.sampled_steps
        draws.extend(first.sampled_steps)
    assert set(draws) == {0, 1, 2, 3}
    assert [e.generated_step for e in dataset] == [0, 1, 2, 3]
    dataset.append(replace(episode, generated_step=4))
    assert [e.generated_step for e in dataset] == [1, 2, 3, 4]


def test_token_budget_does_not_split_episodes_and_repack_does_not_mutate_storage():
    _, inputs, data, trace = prepared()
    episode = next(split_reasoning_episodes(inputs, trace, data, 0))
    dataset, loader = bank_and_loader(episode, sample_budget=None,
        token_budget=episode.token_count + 1, capacity=1)
    batch, labels = loader.next_batch()
    assert batch.tokens['num_sequences'] == 2
    assert len(batch.tokens['token_ids']) == 2 * episode.token_count
    before = dataset[0].tokens['token_ids'].clone()
    batch.tokens['token_ids'].zero_()
    labels['reward'].zero_()
    torch.testing.assert_close(dataset[0].tokens['token_ids'], before)
    assert dataset[0].data['reward'].sum() > 0


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_dataset_requires_positive_integer_capacity(capacity):
    with pytest.raises(ValueError, match='capacity_episodes'):
        ReasoningDataset(capacity_episodes=capacity)


@pytest.mark.parametrize('settings', [dict(sample_budget=0), dict(sample_budget=True),
    dict(sample_budget=None), dict(token_budget=1), dict(sampling='other')])
def test_loader_rejects_invalid_budgets_and_sampling(settings):
    kwargs = dict(dataset=ReasoningDataset(capacity_episodes=100), sample_budget=1,
                  token_budget=None, sampling='sequential', seed=0)
    kwargs.update(settings)
    with pytest.raises(ValueError):
        ReasoningDataLoader(**kwargs)


def test_rejects_cached_prefixes_and_incompatible_schemas():
    _, inputs, data, trace = prepared()
    with pytest.raises(ValueError, match='complete episode'):
        list(split_reasoning_episodes(inputs, replace(trace, cached_prefix=True), data, 0))
    episode = next(split_reasoning_episodes(inputs, trace, data, 0))
    dataset, loader = bank_and_loader(episode)
    with pytest.raises(ValueError, match='schemas'):
        dataset.append(replace(episode, latent_values=episode.latent_values.double()))
    assert len(dataset) == 4
    with pytest.raises(ValueError, match='detached CPU'):
        replace(episode, latent_values=episode.latent_values.clone().requires_grad_())
    dataset.clear()
    with pytest.raises(ValueError, match='empty'):
        loader.next_batch()


def test_trace_device_movement_preserves_dtype_gradients_and_metadata():
    _, _, _, trace = prepared()
    states = trace.states.detach().requires_grad_()
    trace = replace(trace, states=states)
    moved = trace.to('cpu')
    assert moved.head_token_ids == trace.head_token_ids
    assert moved.cached_prefix == trace.cached_prefix
    assert moved.states.dtype == states.dtype
    moved.states.sum().backward()
    torch.testing.assert_close(states.grad, torch.ones_like(states))


@pytest.mark.parametrize('residual', [False, True])
def test_start_only_replay_preserves_boundaries_labels_and_matches_literal_forward(residual):
    parts, inputs, data, _ = prepared(residual)
    embedding, backbone, reasoner, _ = parts
    original = inputs.clone()
    episodes = start_only_reasoning_episodes(inputs, data,
        start_embedding=reasoner.start_embedding.weight[0], max_steps=reasoner.max_steps, generated_step=0)
    batch, labels = pack_reasoning_episodes(episodes)
    assert batch.head_token_ids == (3, 9, 14)
    assert batch.lengths.tolist() == [1, 1, 1]
    assert labels['step_ids'].tolist() == [0, 2, 4]
    torch.testing.assert_close(labels['action'], data['action'])
    torch.testing.assert_close(labels['reward'], data['reward'])
    torch.testing.assert_close(batch.tokens['token_ids'], inputs['token_ids'])
    assert batch.initial_positions.tolist() == batch.readout_positions.tolist() == [3, 9, 14]
    assert batch.update_positions[:, 0].tolist() == [3, 9, 14]
    assert not batch.update_positions[:, 1:].any()
    assert int(batch.tokens['latent_mask'].sum()) == 3
    assert all(e.token_count == e.source_token_count for e in episodes)
    assert all('latent_values' not in e.tokens.keys() and not e.latent_values.requires_grad for e in episodes)
    # An independent literal history has exactly the original tokens, with only
    # each final head replaced by Norm(start). No fabricated final feedback state.
    expected = []
    table = embedding.embeddings['text'].embed_tokens.weight
    for group in range(inputs['num_sequences']):
        history = []
        for row in (inputs['sequence_ids'] == group).nonzero().flatten().tolist():
            is_start = bool(inputs['step_readout_mask'][row])
            history.append(reasoner.norm(reasoner.start_embedding.weight[0]) if is_start
                           else table[inputs['token_ids'][row]])
            if is_start:
                hidden = backbone(torch.stack(history), sequence_ids=torch.zeros(len(history), dtype=torch.long))[-1]
                update = reasoner.project_updates(hidden)
                expected.append(update + reasoner.start_embedding.weight[0] if residual else update)
    actual = reasoner(batch, embedding=embedding, backbone=backbone)
    torch.testing.assert_close(actual[:, 0], torch.stack(expected))
    actual[:, 0].square().sum().backward()
    assert reasoner.start_embedding.weight.grad.abs().sum() > 0
    assert reasoner.update_projection.weight.grad.abs().sum() > 0
    for key, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(inputs[key], value)


def test_episode_capacity_ignores_token_length_and_updates_token_metrics():
    _, inputs, data, trace = prepared()
    long, short = split_reasoning_episodes(inputs, trace, data, 0)
    assert long.token_count != short.token_count
    dataset = ReasoningDataset(capacity_episodes=2)
    dataset.append(long)
    dataset.append(short)
    assert len(dataset) == 2
    assert dataset.token_count == long.token_count + short.token_count
    dataset.append(replace(long, generated_step=1))
    assert dataset[0] is short
    assert len(dataset) == 2
    assert dataset.token_count == long.token_count + short.token_count
    dataset.append(replace(long, generated_step=2))
    assert dataset.token_count == 2 * long.token_count
    assert [e.generated_step for e in dataset] == [1, 2]


@pytest.mark.parametrize('invalid', [dict(max_steps=0), dict(max_steps=True),
    dict(start_embedding=torch.zeros(2, 3)), dict(start_embedding=torch.zeros(0)),
    dict(start_embedding=torch.ones(3, dtype=torch.long))])
def test_start_only_replay_rejects_invalid_dimensions(invalid):
    _, inputs, data, _ = prepared()
    kwargs = dict(start_embedding=torch.zeros(8), max_steps=3, generated_step=0)
    kwargs.update(invalid)
    with pytest.raises(ValueError):
        start_only_reasoning_episodes(inputs, data, **kwargs)
