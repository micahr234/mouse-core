from collections import Counter
import numpy as np
import pytest
from mouse_core.data import EventTransform, environment_episode_ranges
from mouse_core.data import Augmenter, EventLog, EpochDataLoader, StepTokens, compose


def transform(row):
    return StepTokens(token_ids=np.array([1]), numeric_values=np.array([0.]),
                      numeric_mask=np.array([False]), head_token_mask=np.array([True]),
                      objective_fields={'branch': row['branch']})


def stores():
    result = []
    for branch, length in enumerate([2, 3, 4, 2, 3]):
        log = EventLog(name=str(branch))
        for step in range(length):
            log.event(provider='coordinator', episode_index=0, payload={
                'episode_start': step == 0,
            })
            log.event(provider='environment', episode_index=0, payload={
                'terminated': step == length - 1,
                'truncated': False,
                'branch': branch,
            })
        result.append(log.to_store())
    return result


@pytest.mark.parametrize('workers', [0, 2])
def test_every_branch_once_per_epoch_and_final_batch_retained(workers):
    loader = EpochDataLoader(stores=stores(), transform=EventTransform(encode=transform), token_budget=5,
                             sample_ranges=environment_episode_ranges, epochs=3, num_workers=workers, seed=9)
    seen = []
    while not loader.exhausted:
        inputs, data, groups = loader.next_batch()
        for group in groups.unique_consecutive():
            branch = data['branch'][groups == group]
            assert len(branch.unique()) == 1
            seen.append(int(branch[0]))
    assert len(seen) == 15
    for epoch in range(3):
        assert Counter(seen[epoch*5:(epoch+1)*5]) == Counter(range(5))
    with pytest.raises(StopIteration):
        loader.next_batch()
    loader.close()


def test_epoch_shuffle_deterministic_across_worker_counts():
    loaders = [EpochDataLoader(stores=stores(), transform=EventTransform(encode=transform), token_budget=5,
                              sample_ranges=environment_episode_ranges, epochs=2, num_workers=w, seed=7)
               for w in [0, 2]]
    assert loaders[0].batches == loaders[1].batches
    for loader in loaders:
        loader.close()


def test_epoch_workers_support_augmentation_and_match_in_process_results():
    loaders = []
    try:
        for workers in (0, 2):
            augment = Augmenter(seed=3, seed_field='episode_index', fields=[
                {'type': 'discrete', 'input_field': 'branch', 'vocab_size': 5,
                 'permute': True, 'mask_prob': .5, 'mask_value': -1}
            ])
            pipeline = EventTransform(encode=compose(stages=(augment, transform)))
            loaders.append(EpochDataLoader(stores=stores(), transform=pipeline,
                token_budget=5, sample_ranges=environment_episode_ranges, num_workers=workers, seed=7))
        while not loaders[0].exhausted:
            left, right = [loader.next_batch() for loader in loaders]
            assert left[1]['branch'].tolist() == right[1]['branch'].tolist()
            assert left[2].tolist() == right[2].tolist()
    finally:
        for loader in loaders:
            loader.close()


def test_oversized_branch_is_rejected_not_truncated():
    with pytest.raises(ValueError, match='complete sample exceeds'):
        EpochDataLoader(stores=stores(), transform=EventTransform(encode=transform), token_budget=3,
                        sample_ranges=environment_episode_ranges)


def test_remaining_batches_stops_at_each_pass():
    loader = EpochDataLoader(stores=stores(), transform=EventTransform(encode=transform), token_budget=5,
                             sample_ranges=environment_episode_ranges, epochs=2)
    for end in loader.epoch_ends:
        while loader.position < end:
            assert loader.remaining_in_epoch == end - loader.position
            loader.next_batch()
    assert loader.remaining_in_epoch == 0


def test_objective_overlays_follow_samples_through_shuffled_passes():
    import torch
    loader = EpochDataLoader(stores=stores(), transform=EventTransform(encode=transform), token_budget=5,
                             sample_ranges=environment_episode_ranges, epochs=3, seed=9)
    loader.set_sample_objective_fields('frozen', [torch.full((len(sample),), float(i))
                                                for i, sample in enumerate(loader.samples)])
    while not loader.exhausted:
        _, data, _ = loader.next_batch()
        torch.testing.assert_close(data['frozen'], data['branch'].float())
