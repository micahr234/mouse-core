"""Tests for DataLoader batch sampling and augmentation."""

from __future__ import annotations

import threading

import pytest
from datasets import Dataset

from mouse_core.data import DataLoader, Datastore, Augmenter


def _store_with_actions() -> Datastore:
    store = Datastore()
    for action in range(8):
        store.append({"action": action + 1, "reward": float(action), "done": 0})
    return store


def test_dataloader_applies_augmenter_before_returning_batch() -> None:
    augmenter = Augmenter(
        modalities=[
            {
                "field": "action",
                "type": "discrete",
                "vocab_size": 16,
                "mask_prob": 1.0,
            },
        ],
        seed=0,
    )
    loader = DataLoader(
        _store_with_actions(),
        sequence_length=3,
        batch_size=2,
        num_workers=0,
        augmenter=augmenter,
    )

    batch = loader.next_batch()

    assert all(row["action"] == 0 for sequence in batch for row in sequence)


class _ThreadMarkerAugmenter:
    def __init__(self, marker: str = "root") -> None:
        self.marker = marker

    def fork(self, *, seed: int) -> _ThreadMarkerAugmenter:
        return _ThreadMarkerAugmenter(marker=f"worker-{seed}")

    def __call__(self, batch: list[list[dict]]) -> list[list[dict]]:
        thread_name = threading.current_thread().name
        return [
            [
                {
                    **row,
                    "augmenter_marker": self.marker,
                    "augmenter_thread": thread_name,
                }
                for row in sequence
            ]
            for sequence in batch
        ]


def test_dataloader_runs_augmenter_in_worker_thread() -> None:
    loader = DataLoader(
        _store_with_actions(),
        sequence_length=3,
        batch_size=2,
        num_workers=1,
        prefetch=1,
        seed=0,
        augmenter=_ThreadMarkerAugmenter(),
    )
    try:
        batch = loader.next_batch()
    finally:
        loader.close()

    assert all(row["augmenter_marker"] == "worker-0" for sequence in batch for row in sequence)
    assert all(row["augmenter_thread"] == "DataLoader-0" for sequence in batch for row in sequence)


def test_dataloader_snapshots_loaded_source_and_appended_rows() -> None:
    store = Datastore()
    store.from_dataset(Dataset.from_list([
        {"action": 1, "reward": 0.0, "done": 0},
        {"action": 2, "reward": 0.0, "done": 0},
    ]))
    store.append({"action": 3, "reward": 0.0, "done": 0})

    loader = DataLoader(store, sequence_length=3, batch_size=1, num_workers=0)
    batch = loader.next_batch()

    assert [row["action"] for row in batch[0]] == [1, 2, 3]


def test_dataloader_seed_is_deterministic() -> None:
    store = _store_with_actions()
    loader_a = DataLoader(store, sequence_length=3, batch_size=2, num_workers=0, seed=42)
    loader_b = DataLoader(store, sequence_length=3, batch_size=2, num_workers=0, seed=42)

    assert loader_a.next_batch() == loader_b.next_batch()


def test_dataloader_refresh_picks_up_appended_rows() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append({"action": action, "reward": 0.0, "done": 0})

    loader = DataLoader(store, sequence_length=3, batch_size=1, pack=True, num_workers=0)
    loader.next_batch()

    store.append({"action": 4, "reward": 0.0, "done": 0})
    batch_before_refresh = loader.next_batch()
    assert all(row["action"] != 4 for row in batch_before_refresh[0])

    loader.refresh()
    seen = set()
    for _ in range(20):
        batch = loader.next_batch()
        seen.update(row["action"] for row in batch[0])
    assert 4 in seen


def test_dataloader_refresh_drains_prefetch_queue_and_updates_lengths() -> None:
    store = Datastore()
    for action in range(3):
        store.append({"action": action, "reward": 0.0, "done": 0})

    loader = DataLoader(
        store,
        sequence_length=2,
        batch_size=1,
        pack=True,
        num_workers=1,
        prefetch=2,
    )
    try:
        loader.next_batch()
        assert loader._ns == [3]
        store.append({"action": 99, "reward": 0.0, "done": 0})
        loader.refresh()
        assert loader._ns == [4]
    finally:
        loader.close()


def test_dataloader_pack_allows_empty_stores_until_sampling() -> None:
    store = Datastore()
    loader = DataLoader(store, sequence_length=2, batch_size=1, pack=True, num_workers=0)
    try:
        with pytest.raises(ValueError, match="all stores are empty"):
            loader.next_batch()

        store.append({"action": 1, "reward": 0.0, "done": 0})
        loader.refresh()
        batch = loader.next_batch()
        assert len(batch[0]) == 2
    finally:
        loader.close()
