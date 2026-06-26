"""Tests for DataLoader batch sampling and augmentation."""

from __future__ import annotations

import threading

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
