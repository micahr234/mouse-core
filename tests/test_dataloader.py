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

    markers = {row["augmenter_marker"] for sequence in batch for row in sequence}
    assert len(markers) == 1
    assert markers.pop().startswith("worker-")
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


def test_dataloader_seed_is_deterministic_with_workers() -> None:
    store = _store_with_actions()
    loader_a = DataLoader(store, sequence_length=3, batch_size=2, num_workers=1, seed=42)
    loader_b = DataLoader(store, sequence_length=3, batch_size=2, num_workers=1, seed=42)
    try:
        assert loader_a.next_batch() == loader_b.next_batch()
    finally:
        loader_a.close()
        loader_b.close()


def test_dataloader_sampling_and_augmenter_seed_streams_differ() -> None:
    """Sampling RNG and augmenter fork must not share a seed stream."""
    seen: list[int] = []

    class _SeedRecorder:
        def fork(self, *, seed: int) -> "_SeedRecorder":
            seen.append(seed)
            return self

        def __call__(self, batch: list[list[dict]]) -> list[list[dict]]:
            return batch

    loader = DataLoader(
        _store_with_actions(),
        sequence_length=3,
        batch_size=1,
        num_workers=2,
        seed=7,
        augmenter=_SeedRecorder(),
    )
    try:
        loader.next_batch()
    finally:
        loader.close()

    assert len(seen) == 2
    assert len(set(seen)) == 2  # workers get independent augmenter seeds
    assert 7 not in seen and 8 not in seen  # not naive seed + i arithmetic


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


def test_dataloader_pack_marks_segment_seams() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append({"action": action, "reward": 0.0, "done": 0})

    # sequence_length 8 over a 3-row store forces at least 2 extra segments.
    loader = DataLoader(store, sequence_length=8, batch_size=1, pack=True, num_workers=0, seed=0)
    batch = loader.next_batch()
    sequence = batch[0]

    assert all("is_seam" in row for row in sequence)
    assert sequence[0]["is_seam"] == 0  # first row never starts at a seam
    seam_count = sum(row["is_seam"] for row in sequence)
    assert seam_count >= 2
    # A seam row is exactly a row where a fresh segment begins: the previous
    # row is the end of an independently sampled slice, so consecutive
    # actions need not be contiguous there.
    for prev, row in zip(sequence, sequence[1:]):
        if row["is_seam"] == 0 and prev["action"] < 3:
            assert row["action"] == prev["action"] + 1


def test_dataloader_unpacked_rows_carry_no_seam_flag() -> None:
    loader = DataLoader(_store_with_actions(), sequence_length=3, batch_size=1, num_workers=0)
    batch = loader.next_batch()

    assert all("is_seam" not in row for row in batch[0])


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
