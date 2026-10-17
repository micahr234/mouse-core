from collections import Counter
import numpy as np
import pytest
from mouse_core.data import Datastore, EpochDataLoader, StepTokens


def transform(row):
    return StepTokens(token_ids=np.array([1]), numeric_values=np.array([0.]),
                      numeric_mask=np.array([False]), head_token_mask=np.array([True]),
                      objective_fields={'branch': row['branch']})


def stores():
    result = []
    for branch, length in enumerate([2, 3, 4, 2, 3]):
        store = Datastore(name=str(branch))
        for _ in range(length):
            store.append(data={'episode_index': 0, 'branch': branch})
        result.append(store)
    return result


@pytest.mark.parametrize('workers', [0, 2])
def test_every_branch_once_per_epoch_and_final_batch_retained(workers):
    loader = EpochDataLoader(stores=stores(), transform=transform, token_budget=5,
                             sample_field='episode_index', epochs=3, num_workers=workers, seed=9)
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
    loaders = [EpochDataLoader(stores=stores(), transform=transform, token_budget=5,
                              sample_field='episode_index', epochs=2, num_workers=w, seed=7)
               for w in [0, 2]]
    assert loaders[0].batches == loaders[1].batches
    for loader in loaders:
        loader.close()


def test_oversized_branch_is_rejected_not_truncated():
    with pytest.raises(ValueError, match='complete sample exceeds'):
        EpochDataLoader(stores=stores(), transform=transform, token_budget=3,
                        sample_field='episode_index')


def test_remaining_batches_stops_at_each_pass():
    loader = EpochDataLoader(stores=stores(), transform=transform, token_budget=5,
                             sample_field='episode_index', epochs=2)
    for end in loader.epoch_ends:
        while loader.position < end:
            assert loader.remaining_in_epoch == end - loader.position
            loader.next_batch()
    assert loader.remaining_in_epoch == 0


def test_objective_overlays_follow_samples_through_shuffled_passes():
    import torch
    loader = EpochDataLoader(stores=stores(), transform=transform, token_budget=5,
                             sample_field='episode_index', epochs=3, seed=9)
    loader.set_sample_objective_fields('frozen', [torch.full((len(sample),), float(i))
                                                for i, sample in enumerate(loader.samples)])
    while not loader.exhausted:
        _, data, _ = loader.next_batch()
        torch.testing.assert_close(data['frozen'], data['branch'].float())
