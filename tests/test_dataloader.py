from __future__ import annotations

"""Tests for DataLoader batch sampling and transform pipeline."""

import sys
import sysconfig
import threading
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset

from mouse_core.data import (
    Augmenter,
    DataLoader,
    Datastore,
    NumericTokenizer,
    compose,
)
from mouse_core.data.token_batch import StepTokens, TokenBatch
from mouse_core.models.embedding import NumericEmbedder
from tests._token_batch_helpers import tok_from_encoder


def _store_with_actions() -> Datastore:
    store = Datastore()
    for action in range(8):
        store.append(
            {
                "action": action + 1,
                "reward": float(action),
                "episode_done": 0,
                "task_done": 0,
            }
        )
    return store


def _obj(*names: str) -> list[dict[str, str]]:
    return [{"input_field": name} for name in names]


def _tokenizer(*, objective_fields: list[dict[str, str]] | None = None) -> NumericTokenizer:
    keep = (
        objective_fields
        if objective_fields is not None
        else _obj("action", "reward", "episode_done", "task_done")
    )
    return NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action"},
            {"type": "fourier", "input_field": "reward"},
        ],
        objective_fields=keep,
        grouping_field="grouping_id",
    )


def _stamp_grouping(step: dict) -> dict:
    out = dict(step)
    out.setdefault("grouping_id", 0)
    return out


def _transform(**kwargs):
    return compose(_stamp_grouping, _tokenizer(**kwargs))


def _loader(**kwargs) -> DataLoader:
    kwargs.setdefault("transform", _transform())
    kwargs.setdefault("stores", _store_with_actions())
    return DataLoader(**kwargs)


def _free_threading_ok() -> bool:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return not (callable(is_gil_enabled) and is_gil_enabled())


def test_dataloader_requires_transform() -> None:
    with pytest.raises(TypeError, match="transform"):
        DataLoader(
            sequence_length=3,
            batch_size=1,
            num_workers=0,
            transform=None,  # type: ignore[arg-type]
            stores=_store_with_actions(),
        )


def test_dataloader_applies_augmenter_before_returning_batch() -> None:
    def _stamp_task(step: dict) -> dict:
        out = dict(step)
        out.setdefault("task_index", 0)
        return out

    augmenter = Augmenter(
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "vocab_size": 16,
                "mask_prob": 1.0,
            }
        ],
        seed=0,
    )
    loader = _loader(
        sequence_length=3,
        batch_size=2,
        num_workers=0,
        transform=compose(_stamp_task, augmenter, _stamp_grouping, _tokenizer()),
    )
    tb, obj = loader.next_batch()
    assert isinstance(tb, TokenBatch)
    assert all(int(a) == 0 for a in obj["action"])


def test_dataloader_reseeds_transform_each_batch() -> None:
    def _stamp_task(step: dict) -> dict:
        out = dict(step)
        out.setdefault("task_index", 0)
        return out

    augmenter = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "vocab_size": 16,
                "permute": True,
            }
        ],
    )
    assert augmenter._generation == 0
    loader = _loader(
        sequence_length=3,
        batch_size=1,
        num_workers=0,
        seed=0,
        transform=compose(_stamp_task, augmenter, _stamp_grouping, _tokenizer()),
    )
    loader.next_batch()
    assert augmenter._generation == 1
    loader.next_batch()
    assert augmenter._generation == 2


class _ThreadMarkerTransform:
    """Marks which thread ran the per-step transform."""

    def __init__(self, base: NumericTokenizer) -> None:
        self.base = base
        self.grouper = _stamp_grouping
        self.captured: list[dict] = []

    def __call__(self, step: dict) -> StepTokens:
        thread_name = threading.current_thread().name
        row = {**step, "transform_thread": thread_name}
        self.captured.append(row)
        return self.base(self.grouper(row))


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_runs_transform_in_worker_thread() -> None:
    marker = _ThreadMarkerTransform(_tokenizer(objective_fields=_obj("action", "reward")))
    loader = DataLoader(
        sequence_length=3,
        batch_size=2,
        num_workers=1,
        prefetch=1,
        seed=0,
        transform=marker,
        stores=_store_with_actions(),
    )
    try:
        tb, _ = loader.next_batch()
    finally:
        loader.close()
    assert isinstance(tb, TokenBatch)
    assert marker.captured
    assert all(row["transform_thread"] == "DataLoader-0" for row in marker.captured)


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_worker_error_surfaces_even_with_full_prefetch_queue() -> None:
    """A transform that fails after a few good batches must raise the real error."""
    tokenizer = _tokenizer()
    calls = 0
    lock = threading.Lock()

    def _failing(step: dict) -> StepTokens:
        nonlocal calls
        with lock:
            calls += 1
            n = calls
        if n > 6:
            raise ValueError("boom from worker")
        return tokenizer(_stamp_grouping(step))

    loader = DataLoader(
        sequence_length=1,
        batch_size=1,
        num_workers=1,
        prefetch=2,
        seed=0,
        transform=_failing,
        stores=_store_with_actions(),
    )
    try:
        with pytest.raises(RuntimeError, match="prefetch worker raised") as info:
            for _ in range(20):
                loader.next_batch()
        assert isinstance(info.value.__cause__, ValueError)
        assert "boom from worker" in str(info.value.__cause__)
    finally:
        loader.close()


def test_dataloader_validates_batch_size_and_prefetch() -> None:
    store = _store_with_actions()
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        _loader(sequence_length=3, batch_size=0, num_workers=0, stores=store)
    with pytest.raises(ValueError, match="prefetch must be >= 1"):
        _loader(sequence_length=3, batch_size=1, num_workers=0, prefetch=0, stores=store)


def test_dataloader_num_workers_requires_free_threading() -> None:
    store = _store_with_actions()
    with patch.object(sysconfig, "get_config_var", return_value=0):
        with pytest.raises(RuntimeError, match="free-threaded"):
            _loader(sequence_length=3, batch_size=1, num_workers=1, stores=store)
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        with patch.object(sys, "_is_gil_enabled", return_value=True):
            with pytest.raises(RuntimeError, match="free-threaded|GIL"):
                _loader(sequence_length=3, batch_size=1, num_workers=1, stores=store)


def test_dataloader_snapshots_loaded_source_and_appended_rows() -> None:
    store = Datastore()
    store.from_dataset(
        Dataset.from_list(
            [
                {"action": 1, "reward": 0.0, "episode_done": 0, "task_done": 0},
                {"action": 2, "reward": 0.0, "episode_done": 0, "task_done": 0},
            ]
        )
    )
    store.append({"action": 3, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=3, batch_size=1, num_workers=0, seed=0, stores=store)
    tb, obj = loader.next_batch()
    actions = [int(a) for a in obj["action"]]
    assert 1 <= len(actions) <= 3
    assert actions == list(range(actions[0], actions[0] + len(actions)))
    assert set(actions) <= {1, 2, 3}


def _tb_signature(packed: tuple[TokenBatch, object]) -> tuple:
    tb, obj = packed
    return (
        tb.B,
        tb.L,
        tb.N,
        tuple(tb.modality_ids.tolist()),
        tuple(tb.ids.tolist()),
        tuple(np.asarray(obj["action"].detach().cpu().numpy()).tolist()),
    )


def test_dataloader_seed_is_deterministic() -> None:
    store = _store_with_actions()
    loader_a = _loader(sequence_length=3, batch_size=2, num_workers=0, seed=42, stores=store)
    loader_b = _loader(sequence_length=3, batch_size=2, num_workers=0, seed=42, stores=store)
    assert _tb_signature(loader_a.next_batch()) == _tb_signature(loader_b.next_batch())


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_seed_is_deterministic_with_workers() -> None:
    store = _store_with_actions()
    loader_a = _loader(sequence_length=3, batch_size=2, num_workers=1, seed=42, stores=store)
    loader_b = _loader(sequence_length=3, batch_size=2, num_workers=1, seed=42, stores=store)
    try:
        assert _tb_signature(loader_a.next_batch()) == _tb_signature(loader_b.next_batch())
    finally:
        loader_a.close()
        loader_b.close()


def test_dataloader_index_field_stamps_store_offset() -> None:
    store = _store_with_actions()
    seen: list[int] = []

    def transform(step: dict) -> StepTokens:
        seen.append(int(step["store_index"]))
        return _transform()(step)

    loader = DataLoader(
        stores=store,
        sequence_length=3,
        batch_size=1,
        num_workers=0,
        seed=0,
        index_field="store_index",
        transform=transform,
    )
    loader.next_batch()
    assert seen
    assert all(isinstance(i, int) and i >= 0 for i in seen)


def test_dataloader_refresh_picks_up_appended_rows() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append({"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=3, batch_size=1, num_workers=0, stores=store)
    loader.next_batch()
    store.append({"action": 4, "reward": 0.0, "episode_done": 0, "task_done": 0})
    _, obj_before = loader.next_batch()
    assert all(int(a) != 4 for a in obj_before["action"])
    loader.refresh()
    seen: set[int] = set()
    for _ in range(40):
        _, obj = loader.next_batch()
        seen.update(int(a) for a in obj["action"])
    assert 4 in seen


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_refresh_drains_prefetch_queue_and_updates_store_sizes() -> None:
    store = Datastore()
    for action in range(3):
        store.append({"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=2, batch_size=1, num_workers=1, prefetch=2, stores=store)
    try:
        loader.next_batch()
        assert loader._ns == [3]
        store.append({"action": 99, "reward": 0.0, "episode_done": 0, "task_done": 0})
        loader.refresh()
        assert loader._ns == [4]
    finally:
        loader.close()


def test_dataloader_ragged_windows_up_to_max_length() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append({"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=8, batch_size=1, num_workers=0, seed=0, stores=store)
    tb, obj = loader.next_batch()
    n = int(tb.step_counts()[0])
    assert 1 <= n <= 3
    actions = [int(a) for a in obj["action"]]
    assert actions == list(range(actions[0], actions[0] + len(actions)))


def test_dataloader_allows_short_stores() -> None:
    store = Datastore()
    store.append({"action": 7, "reward": 1.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=4, batch_size=1, num_workers=0, stores=store)
    tb, obj = loader.next_batch()
    assert int(tb.step_counts()[0]) == 1
    assert int(obj["action"][0]) == 7


def test_dataloader_allows_empty_stores_until_sampling() -> None:
    store = Datastore()
    loader = _loader(sequence_length=2, batch_size=1, num_workers=0, stores=store)
    try:
        with pytest.raises(ValueError, match="all stores are empty"):
            loader.next_batch()
        store.append({"action": 1, "reward": 0.0, "episode_done": 0, "task_done": 0})
        loader.refresh()
        tb, _ = loader.next_batch()
        assert int(tb.step_counts()[0]) == 1
    finally:
        loader.close()


def test_dataloader_transform_returns_token_batch() -> None:
    encoder = NumericEmbedder(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 16},
            {"type": 'fourier', "field": "reward"},
        ],
    )
    loader = DataLoader(
        sequence_length=3,
        batch_size=2,
        num_workers=0,
        transform=compose(_stamp_grouping, tok_from_encoder(encoder)),
        stores=_store_with_actions(),
    )
    try:
        tb, obj = loader.next_batch()
        assert tb.B == 2
        assert int(tb.step_counts().sum()) == tb.N
        assert tb.N >= 2
        assert all(1 <= int(n) <= 3 for n in tb.step_counts())
        embeds, prediction_indices = encoder(tb)
        assert embeds.shape == (tb.L, 8)
        assert prediction_indices.shape == (tb.N,)
        assert "sequence_id" in obj.keys()
    finally:
        loader.close()
