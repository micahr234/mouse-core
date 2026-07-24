"""Tests for DataLoader batch sampling and augmentation."""
from __future__ import annotations
import sys
import sysconfig
import threading
from unittest.mock import patch
import pytest
from datasets import Dataset
from mouse_core.data import DataLoader, Datastore, Augmenter

def _store_with_actions() -> Datastore:
    store = Datastore()
    for action in range(8):
        store.append({'action': action + 1, 'reward': float(action), 'done': 0})
    return store

def test_dataloader_applies_augmenter_before_returning_batch() -> None:
    augmenter = Augmenter(modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 16, 'mask_prob': 1.0}], seed=0)
    loader = DataLoader(sequence_length=3, batch_size=2, num_workers=0, augmenter=augmenter, stores=_store_with_actions())
    batch = loader.next_batch()
    assert all((row['action'] == 0 for sequence in batch for row in sequence))

class _ThreadMarkerAugmenter:

    def __init__(self, marker: str='root') -> None:
        self.marker = marker

    def fork(self, *, seed: int) -> _ThreadMarkerAugmenter:
        return _ThreadMarkerAugmenter(marker=f'worker-{seed}')

    def __call__(self, batch: list[list[dict]]) -> list[list[dict]]:
        thread_name = threading.current_thread().name
        return [[{**row, 'augmenter_marker': self.marker, 'augmenter_thread': thread_name} for row in sequence] for sequence in batch]

def test_dataloader_runs_augmenter_in_worker_thread() -> None:
    loader = DataLoader(sequence_length=3, batch_size=2, num_workers=1, prefetch=1, seed=0, augmenter=_ThreadMarkerAugmenter(), stores=_store_with_actions())
    try:
        batch = loader.next_batch()
    finally:
        loader.close()
    markers = {row['augmenter_marker'] for sequence in batch for row in sequence}
    assert len(markers) == 1
    assert markers.pop().startswith('worker-')
    assert all((row['augmenter_thread'] == 'DataLoader-0' for sequence in batch for row in sequence))

def test_dataloader_num_workers_requires_free_threading() -> None:
    store = _store_with_actions()
    with patch.object(sysconfig, 'get_config_var', return_value=0):
        with pytest.raises(RuntimeError, match='free-threaded'):
            DataLoader(sequence_length=3, batch_size=1, num_workers=1, stores=store)
    if sysconfig.get_config_var('Py_GIL_DISABLED'):
        with patch.object(sys, '_is_gil_enabled', return_value=True):
            with pytest.raises(RuntimeError, match='free-threaded|GIL'):
                DataLoader(sequence_length=3, batch_size=1, num_workers=1, stores=store)

def test_dataloader_snapshots_loaded_source_and_appended_rows() -> None:
    store = Datastore()
    store.from_dataset(Dataset.from_list([{'action': 1, 'reward': 0.0, 'done': 0}, {'action': 2, 'reward': 0.0, 'done': 0}]))
    store.append({'action': 3, 'reward': 0.0, 'done': 0})
    loader = DataLoader(sequence_length=3, batch_size=1, num_workers=0, seed=0, stores=store)
    batch = loader.next_batch()
    actions = [row['action'] for row in batch[0]]
    assert 1 <= len(actions) <= 3
    assert actions == list(range(actions[0], actions[0] + len(actions)))
    assert set(actions) <= {1, 2, 3}

def test_dataloader_seed_is_deterministic() -> None:
    store = _store_with_actions()
    loader_a = DataLoader(sequence_length=3, batch_size=2, num_workers=0, seed=42, stores=store)
    loader_b = DataLoader(sequence_length=3, batch_size=2, num_workers=0, seed=42, stores=store)
    assert loader_a.next_batch() == loader_b.next_batch()

def test_dataloader_seed_is_deterministic_with_workers() -> None:
    store = _store_with_actions()
    loader_a = DataLoader(sequence_length=3, batch_size=2, num_workers=1, seed=42, stores=store)
    loader_b = DataLoader(sequence_length=3, batch_size=2, num_workers=1, seed=42, stores=store)
    try:
        assert loader_a.next_batch() == loader_b.next_batch()
    finally:
        loader_a.close()
        loader_b.close()

class _SeedRecorder:
    """Records augmenter fork seeds in the parent (fork runs before workers start)."""

    def __init__(self, seen: list[int] | None=None) -> None:
        self.seen = seen if seen is not None else []

    def fork(self, *, seed: int) -> _SeedRecorder:
        self.seen.append(seed)
        return _SeedRecorder(seen=self.seen)

    def __call__(self, batch: list[list[dict]]) -> list[list[dict]]:
        return batch

def test_dataloader_sampling_and_augmenter_seed_streams_differ() -> None:
    """Sampling RNG and augmenter fork must not share a seed stream."""
    seen: list[int] = []
    loader = DataLoader(sequence_length=3, batch_size=1, num_workers=2, seed=7, augmenter=_SeedRecorder(seen=seen), stores=_store_with_actions())
    try:
        loader.next_batch()
    finally:
        loader.close()
    assert len(seen) == 2
    assert len(set(seen)) == 2
    assert 7 not in seen and 8 not in seen

def test_dataloader_refresh_picks_up_appended_rows() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append({'action': action, 'reward': 0.0, 'done': 0})
    loader = DataLoader(sequence_length=3, batch_size=1, num_workers=0, stores=store)
    loader.next_batch()
    store.append({'action': 4, 'reward': 0.0, 'done': 0})
    batch_before_refresh = loader.next_batch()
    assert all((row['action'] != 4 for row in batch_before_refresh[0]))
    loader.refresh()
    seen = set()
    for _ in range(40):
        batch = loader.next_batch()
        seen.update((row['action'] for row in batch[0]))
    assert 4 in seen

def test_dataloader_refresh_drains_prefetch_queue_and_updates_store_sizes() -> None:
    store = Datastore()
    for action in range(3):
        store.append({'action': action, 'reward': 0.0, 'done': 0})
    loader = DataLoader(sequence_length=2, batch_size=1, num_workers=1, prefetch=2, stores=store)
    try:
        loader.next_batch()
        assert loader._ns == [3]
        store.append({'action': 99, 'reward': 0.0, 'done': 0})
        loader.refresh()
        assert loader._ns == [4]
    finally:
        loader.close()

def test_dataloader_ragged_windows_up_to_max_length() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append({'action': action, 'reward': 0.0, 'done': 0})
    loader = DataLoader(sequence_length=8, batch_size=1, num_workers=0, seed=0, stores=store)
    batch = loader.next_batch()
    sequence = batch[0]
    assert 1 <= len(sequence) <= 3
    assert all(('is_seam' not in row for row in sequence))
    actions = [row['action'] for row in sequence]
    assert actions == list(range(actions[0], actions[0] + len(actions)))

def test_dataloader_allows_short_stores() -> None:
    store = Datastore()
    store.append({'action': 7, 'reward': 1.0, 'done': 0})
    loader = DataLoader(sequence_length=4, batch_size=1, num_workers=0, stores=store)
    batch = loader.next_batch()
    assert len(batch[0]) == 1
    assert batch[0][0]['action'] == 7

def test_dataloader_allows_empty_stores_until_sampling() -> None:
    store = Datastore()
    loader = DataLoader(sequence_length=2, batch_size=1, num_workers=0, stores=store)
    try:
        with pytest.raises(ValueError, match='all stores are empty'):
            loader.next_batch()
        store.append({'action': 1, 'reward': 0.0, 'done': 0})
        loader.refresh()
        batch = loader.next_batch()
        assert len(batch[0]) == 1
    finally:
        loader.close()

def test_dataloader_preparer_returns_token_batch() -> None:
    from mouse_core.models.embedding import NumericEmbedder
    encoder = NumericEmbedder(hidden_dim=8, modalities=[{'field': 'action', 'type': 'discrete', 'vocab_size': 16}, {'field': 'reward', 'type': 'rff'}])
    loader = DataLoader(sequence_length=3, batch_size=2, num_workers=0, preparer=encoder.make_preparer(), stores=_store_with_actions())
    try:
        tb = loader.next_batch()
        assert tb.B == 2
        assert int(tb.step_counts().sum()) == tb.N
        assert tb.N >= 2
        assert all((1 <= int(n) <= 3 for n in tb.step_counts()))
        embeds, col_values, prediction_indices = encoder(tb)
        assert embeds.shape == (tb.L, 8)
        assert prediction_indices.shape == (tb.N,)
        assert 'sequence_id' in col_values
    finally:
        loader.close()
