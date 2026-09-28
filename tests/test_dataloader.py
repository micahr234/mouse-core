from __future__ import annotations

"""Tests for DataLoader batch sampling and transform pipeline."""

import sys
import sysconfig
import threading
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from datasets import Dataset

from mouse_core.data import (
    Augmenter,
    DataLoader,
    Datastore,
    SampleBoundary,
    SampleMatch,
    Tokenizer,
    compose,
)
from mouse_core.data.augmenter import _stable_hash
from mouse_core.data.dataloader import _sequence_generation
from mouse_core.data.token_batch import StepTokens, TokenBatch
from tests._token_batch_helpers import token_tokenizer


def _store_with_actions() -> Datastore:
    store = Datastore()
    for action in range(8):
        store.append(
            data={
                "action": action + 1,
                "reward": float(action),
                "episode_done": 0,
                "task_done": 0,
            }
        )
    return store


def _obj(*names: str) -> list[dict[str, str]]:
    return [{"input_field": name} for name in names]


def _tokenizer(*, objective_fields: list[dict[str, str]] | None = None) -> Tokenizer:
    keep = (
        objective_fields
        if objective_fields is not None
        else _obj("action", "reward", "episode_done", "task_done")
    )
    return Tokenizer(
        input_fields=[
            {"type": "token", "input_field": "action", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": {"equals": [("step_index", 0)]},
            },
        ],
        objective_fields=keep,
        grouping_field="grouping_id",
    )


def _stamp_grouping(step: dict) -> dict:
    out = dict(step)
    out.setdefault("grouping_id", 0)
    return out


def _transform(**kwargs):
    return compose(stages=(_stamp_grouping, _tokenizer(**kwargs)))


def _loader(**kwargs) -> DataLoader:
    kwargs.setdefault("transform", _transform())
    kwargs.setdefault("stores", _store_with_actions())
    kwargs.setdefault("num_workers", 0)
    return DataLoader(**kwargs)


def _free_threading_ok() -> bool:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return not (callable(is_gil_enabled) and is_gil_enabled())


def test_dataloader_requires_num_workers() -> None:
    with pytest.raises(TypeError, match="num_workers"):
        DataLoader(  # type: ignore[call-arg]
            sequence_length=3,
            batch_size=1,
            transform=_transform(),
            stores=_store_with_actions(),
        )


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
        transform=compose(stages=(_stamp_task, augmenter, _stamp_grouping, _tokenizer())),
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
        transform=compose(stages=(_stamp_task, augmenter, _stamp_grouping, _tokenizer())),
    )
    loader.next_batch()
    assert augmenter._generation_for_call() == 0  # batch k=0, sequence 0
    loader.next_batch()
    assert augmenter._generation_for_call() == 1  # batch k=1, sequence 0 (B=1)
    assert augmenter._generation == 0  # the shared counter is untouched


def test_dataloader_same_seed_field_on_two_sequences_uses_two_seeds() -> None:
    """Same index on two rollouts in one batch must start from different seeds."""
    store = Datastore()
    for _ in range(8):
        store.append(
            data={
                "action": 0,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": 0,
                "task_index": 7,
            }
        )

    augmenter = Augmenter(
        seed=0,
        seed_field="task_index",
        fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "vocab_size": 10,
                "permute": True,
            }
        ],
    )
    loader = DataLoader(
        stores=store,
        sequence_length=3,
        batch_size=2,
        num_workers=0,
        seed=0,
        transform=compose(stages=(augmenter, _stamp_grouping, _tokenizer(objective_fields=_obj("action")))),
    )
    _, obj = loader.next_batch()
    sequence_id = obj["sequence_id"]
    actions = obj["action"]

    def _expected(*, sequence_index: int) -> int:
        generation = _sequence_generation(
            batch_index=0,
            sequence_index=sequence_index,
            batch_size=2,
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([0, generation, _stable_hash("task_index", 7)])
        )
        return int(rng.permutation(10)[0])

    seq0 = [int(actions[i]) for i in range(len(actions)) if int(sequence_id[i]) == 0]
    seq1 = [int(actions[i]) for i in range(len(actions)) if int(sequence_id[i]) == 1]
    assert seq0
    assert seq1
    assert all(a == _expected(sequence_index=0) for a in seq0)
    assert all(a == _expected(sequence_index=1) for a in seq1)
    assert _expected(sequence_index=0) != _expected(sequence_index=1)
    assert augmenter._generation_for_call() == 1


class _ThreadMarkerTransform:
    """Marks which thread ran the per-step transform."""

    def __init__(self, base: Tokenizer) -> None:
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
        ds=Dataset.from_list(
            [
                {"action": 1, "reward": 0.0, "episode_done": 0, "task_done": 0},
                {"action": 2, "reward": 0.0, "episode_done": 0, "task_done": 0},
            ]
        )
    )
    store.append(data={"action": 3, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=3, batch_size=1, num_workers=0, seed=0, stores=store)
    tb, obj = loader.next_batch()
    actions = [int(a) for a in obj["action"]]
    assert 1 <= len(actions) <= 3
    assert actions == list(range(actions[0], actions[0] + len(actions)))
    assert set(actions) <= {1, 2, 3}


def _tb_signature(packed: tuple[TokenBatch, dict[str, torch.Tensor]]) -> tuple:
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


def _augmented_transform() -> Any:
    def _stamp(step: dict) -> dict:
        out = dict(step)
        out.setdefault("task_index", step["action"] % 3)
        out.setdefault("grouping_id", 0)
        return out

    augment = Augmenter(
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
    return compose(stages=(_stamp, augment, _tokenizer(objective_fields=_obj("action"))))


def _signatures(loader: DataLoader, n: int) -> list[tuple]:
    return [_tb_signature(loader.next_batch()) for _ in range(n)]


def test_dataloader_batch_k_is_independent_of_num_workers() -> None:
    """Sync and threaded loaders with the same seed yield the identical ordered stream."""
    store = _store_with_actions()
    sync = _loader(
        sequence_length=3, batch_size=2, num_workers=0, seed=7, stores=store,
        transform=_augmented_transform(),
    )
    expected = _signatures(sync, 12)
    assert len(set(expected)) > 1  # the stream actually varies over k
    if not _free_threading_ok():
        return
    threaded = _loader(
        sequence_length=3, batch_size=2, num_workers=4, prefetch=2, seed=7, stores=store,
        transform=_augmented_transform(),
    )
    try:
        assert _signatures(threaded, 12) == expected
    finally:
        threaded.close()


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_refresh_resumes_numbering_at_next_unseen_batch() -> None:
    store = _store_with_actions()
    reference = _loader(sequence_length=3, batch_size=2, num_workers=0, seed=3, stores=store)
    expected = _signatures(reference, 8)
    loader = _loader(
        sequence_length=3, batch_size=2, num_workers=3, prefetch=4, seed=3, stores=store
    )
    try:
        got = _signatures(loader, 3)
        loader.refresh()  # drops prefetched 3.. and rebuilds them with the same indices
        got += _signatures(loader, 5)
        assert got == expected
        assert loader._next_k == 8
    finally:
        loader.close()


def test_dataloader_unseeded_stream_is_still_ordered_and_fresh() -> None:
    store = _store_with_actions()
    a = _loader(sequence_length=3, batch_size=2, num_workers=0, stores=store)
    b = _loader(sequence_length=3, batch_size=2, num_workers=0, stores=store)
    assert a.seed is None and b.seed is None
    assert a._entropy != b._entropy
    a.next_batch()
    assert a._next_k == 1


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
        store.append(data={"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=3, batch_size=1, num_workers=0, stores=store)
    loader.next_batch()
    store.append(data={"action": 4, "reward": 0.0, "episode_done": 0, "task_done": 0})
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
        store.append(data={"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=2, batch_size=1, num_workers=1, prefetch=2, stores=store)
    try:
        loader.next_batch()
        assert loader._ns == [3]
        store.append(data={"action": 99, "reward": 0.0, "episode_done": 0, "task_done": 0})
        loader.refresh()
        assert loader._ns == [4]
    finally:
        loader.close()


def test_dataloader_ragged_windows_up_to_max_length() -> None:
    store = Datastore()
    for action in (1, 2, 3):
        store.append(data={"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(sequence_length=8, batch_size=1, num_workers=0, seed=0, stores=store)
    tb, obj = loader.next_batch()
    n = int(tb.step_counts()[0])
    assert 1 <= n <= 3
    actions = [int(a) for a in obj["action"]]
    assert actions == list(range(actions[0], actions[0] + len(actions)))


def test_dataloader_allows_short_stores() -> None:
    store = Datastore()
    store.append(data={"action": 7, "reward": 1.0, "episode_done": 0, "task_done": 0})
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
        store.append(data={"action": 1, "reward": 0.0, "episode_done": 0, "task_done": 0})
        loader.refresh()
        tb, _ = loader.next_batch()
        assert int(tb.step_counts()[0]) == 1
    finally:
        loader.close()


def test_dataloader_transform_returns_token_batch() -> None:
    from mouse_core.models.backbone import IdentityBackbone

    backbone = IdentityBackbone(hidden_dim=8, vocab_size=32)
    loader = DataLoader(
        sequence_length=3,
        batch_size=2,
        num_workers=0,
        transform=compose(stages=(_stamp_grouping, token_tokenizer("action"))),
        stores=_store_with_actions(),
    )
    try:
        tb, obj = loader.next_batch()
        assert tb.B == 2
        assert int(tb.step_counts().sum()) == tb.N
        assert tb.N >= 2
        assert all(1 <= int(n) <= 3 for n in tb.step_counts())
        embeds, head_output_indices = backbone.embed(tb)
        assert embeds.shape == (tb.L, 8)
        assert head_output_indices.shape == (tb.N,)
        assert "sequence_id" in obj.keys()
    finally:
        loader.close()


def _packed_episode_store() -> tuple[Datastore, set[int]]:
    """Three episodes packed in one store. Returns start offsets.

    Layout (action, episode_done):
    ``(10, 0), (11, 1) | (12, 1) | (13, 0), (14, 0), (15, 2)``.
    Starts are 0, 2, and 3. Index 1, 4, and 5 are mid-episode.
    """
    store = Datastore()
    rows = (
        (10, 0),
        (11, 1),
        (12, 1),
        (13, 0),
        (14, 0),
        (15, 2),
    )
    for action, episode_done in rows:
        store.append(
            data={
                "action": action,
                "reward": 0.0,
                "episode_done": episode_done,
                "task_done": 0,
            }
        )
    return store, {0, 2, 3}


def _packed_task_store() -> tuple[Datastore, set[int]]:
    """Two tasks packed in one store. Returns task-start offsets.

    Layout (action, episode_done, task_done):
    ``(10, 1, 0), (11, 1, 1) | (12, 0, 0), (13, 2, 2)``.
    Task starts are 0 and 2. Index 1 and 3 are mid-task.
    """
    store = Datastore()
    rows = (
        (10, 1, 0),
        (11, 1, 1),
        (12, 0, 0),
        (13, 2, 2),
    )
    for action, episode_done, task_done in rows:
        store.append(
            data={
                "action": action,
                "reward": 0.0,
                "episode_done": episode_done,
                "task_done": task_done,
            }
        )
    return store, {0, 2}


def _index_transform():
    return compose(
        stages=(
            _stamp_grouping,
            _tokenizer(objective_fields=_obj("action", "store_index", "episode_done", "task_done")),
        )
    )


def _sampled_windows(loader: DataLoader, n: int) -> list[list[tuple[int, int]]]:
    """Per batch, one ``(start, length)`` for each packed sequence."""
    batches: list[list[tuple[int, int]]] = []
    for _ in range(n):
        _, obj = loader.next_batch()
        by_seq: dict[int, list[int]] = {}
        for seq_id, store_index in zip(obj["sequence_id"], obj["store_index"], strict=True):
            by_seq.setdefault(int(seq_id), []).append(int(store_index))
        batches.append([(indices[0], len(indices)) for _, indices in sorted(by_seq.items())])
    return batches


def test_dataloader_sample_start_none_can_begin_mid_segment() -> None:
    store, episode_starts = _packed_episode_store()
    loader = _loader(
        sequence_length=2,
        batch_size=2,
        num_workers=0,
        seed=0,
        sample_start=None,
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    starts = {start for batch in _sampled_windows(loader, 48) for start, _ in batch}
    assert starts - episode_starts


def test_dataloader_sample_start_defaults_to_unrestricted() -> None:
    store, _episode_starts = _packed_episode_store()
    kwargs = dict(
        sequence_length=2,
        batch_size=2,
        num_workers=0,
        seed=0,
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    omitted = _loader(**kwargs)
    explicit = _loader(sample_start=None, sample_end=None, **kwargs)
    assert _sampled_windows(omitted, 4) == _sampled_windows(explicit, 4)


def test_dataloader_sample_start_begins_after_boundary() -> None:
    """Every packed sequence starts on a segment, including a later one."""
    store, episode_starts = _packed_episode_store()
    loader = _loader(
        sequence_length=2,
        batch_size=2,
        num_workers=0,
        seed=1,
        sample_start=SampleBoundary(field="episode_done", value=0),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    windows = _sampled_windows(loader, 64)
    starts = {start for batch in windows for start, _ in batch}
    assert starts <= episode_starts
    assert 0 in starts
    assert starts - {0}


def test_dataloader_sample_start_task_done_uses_task_boundaries() -> None:
    store, task_starts = _packed_task_store()
    loader = _loader(
        sequence_length=2,
        batch_size=2,
        num_workers=0,
        seed=3,
        sample_start=SampleBoundary(field="task_done", value=0),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    starts = {start for batch in _sampled_windows(loader, 64) for start, _ in batch}
    assert starts <= task_starts
    assert starts == task_starts


def test_dataloader_sample_start_short_suffix_stays_ragged() -> None:
    """A short segment at the store end is a shorter window, not a padded one."""
    store, episode_starts = _packed_episode_store()
    sequence_length = 10
    loader = _loader(
        sequence_length=sequence_length,
        batch_size=1,
        num_workers=0,
        seed=2,
        sample_start=SampleBoundary(field="episode_done", value=0),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    windows = [window for batch in _sampled_windows(loader, 32) for window in batch]
    assert windows
    assert {start for start, _ in windows} <= episode_starts
    n = len(store)
    for start, length in windows:
        assert length == min(sequence_length, n - start)
        assert length < sequence_length


def test_dataloader_sample_end_truncates_on_boundary() -> None:
    """Windows include the first boundary row then stop before sequence_length."""
    store, _ = _packed_episode_store()
    loader = _loader(
        sequence_length=10,
        batch_size=1,
        num_workers=0,
        seed=0,
        sample_start=SampleBoundary(field="episode_done", value=0),
        sample_end=SampleBoundary(field="episode_done", value=0),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    windows = [window for batch in _sampled_windows(loader, 48) for window in batch]
    assert windows
    # Episodes lengths: [2, 1, 3]. Each window is one full episode.
    by_start = {start: length for start, length in windows}
    assert by_start.keys() <= {0, 2, 3}
    if 0 in by_start:
        assert by_start[0] == 2
    if 2 in by_start:
        assert by_start[2] == 1
    if 3 in by_start:
        assert by_start[3] == 3


def test_dataloader_sample_end_mid_window_without_start_constraint() -> None:
    store, _ = _packed_episode_store()
    loader = _loader(
        sequence_length=10,
        batch_size=1,
        num_workers=0,
        seed=4,
        sample_start=None,
        sample_end=SampleBoundary(field="episode_done", value=0),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    for _ in range(32):
        _, obj = loader.next_batch()
        indices = [int(x) for x in obj["store_index"]]
        dones = [int(x) for x in obj["episode_done"]]
        assert indices == list(range(indices[0], indices[0] + len(indices)))
        if indices[-1] < len(store) - 1 or dones[-1] != 0:
            assert dones[-1] != 0
            assert all(d == 0 for d in dones[:-1])


def test_dataloader_sample_start_requires_field_column() -> None:
    store = Datastore()
    store.append(data={"action": 1, "reward": 0.0, "task_done": 0})
    with pytest.raises(ValueError, match="episode_done"):
        _loader(
            sequence_length=2,
            batch_size=1,
            num_workers=0,
            sample_start=SampleBoundary(field="episode_done", value=0),
            stores=store,
        )


def test_dataloader_sample_end_requires_field_column() -> None:
    store = Datastore()
    store.append(data={"action": 1, "reward": 0.0, "episode_done": 0})
    with pytest.raises(ValueError, match="task_done"):
        _loader(
            sequence_length=2,
            batch_size=1,
            num_workers=0,
            sample_end=SampleBoundary(field="task_done", value=0),
            stores=store,
        )


def test_dataloader_sample_boundary_type_errors() -> None:
    with pytest.raises(TypeError, match="sample_start"):
        _loader(
            sequence_length=2,
            batch_size=1,
            num_workers=0,
            sample_start=True,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="sample_end"):
        _loader(
            sequence_length=2,
            batch_size=1,
            num_workers=0,
            sample_end="episode_done",  # type: ignore[arg-type]
        )


def _packed_episode_index_store() -> tuple[Datastore, set[int]]:
    """Two tasks; only episode_index==0 and step_index==0 are match starts.

    Layout (action, episode_index, step_index, task_done):
    ep0: (10,0,0), (11,0,1) | ep1: (12,1,0) | task-end (13,1,1,task_done=1)
    | task2 ep0: (14,0,0), (15,0,1,task_done=2)

    Match starts: 0 and 4. Episode-1 start at 2 is not a match start.
    """
    store = Datastore()
    rows = (
        (10, 0, 0, 0),
        (11, 0, 1, 0),
        (12, 1, 0, 0),
        (13, 1, 1, 1),
        (14, 0, 0, 0),
        (15, 0, 1, 2),
    )
    for action, episode_index, step_index, task_done in rows:
        store.append(
            data={
                "action": action,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": task_done,
                "episode_index": episode_index,
                "step_index": step_index,
            }
        )
    return store, {0, 4}


def test_sample_match_rejects_empty_equals_and_not_equals() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SampleMatch(equals=(), not_equals=())


def test_dataloader_sample_start_match_requires_both_fields() -> None:
    """Windows begin only where episode_index==0 and step_index==0."""
    store, match_starts = _packed_episode_index_store()
    loader = _loader(
        sequence_length=2,
        batch_size=2,
        num_workers=0,
        seed=5,
        sample_start=SampleMatch(equals=(("episode_index", 0), ("step_index", 0)), not_equals=()),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    starts = {start for batch in _sampled_windows(loader, 64) for start, _ in batch}
    assert starts == match_starts
    assert 2 not in starts  # episode_index==1, step_index==0


def test_dataloader_sample_start_match_with_task_end() -> None:
    """Each window is exactly one full task (ep0/step0 .. task_done!=0 inclusive)."""
    store, match_starts = _packed_episode_index_store()
    # Task 0: indices 0..3 (task_done at 3); task 1: 4..5 (task_done at 5).
    loader = _loader(
        sequence_length=100,
        batch_size=1,
        num_workers=0,
        seed=6,
        sample_start=SampleMatch(equals=(("episode_index", 0), ("step_index", 0)), not_equals=()),
        sample_end=SampleMatch(equals=(), not_equals=(("task_done", 0),)),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    seen_starts: set[int] = set()
    for _ in range(48):
        _, obj = loader.next_batch()
        indices = [int(x) for x in obj["store_index"]]
        dones = [int(x) for x in obj["task_done"]]
        start = indices[0]
        seen_starts.add(start)
        assert start in match_starts
        assert indices == list(range(start, start + len(indices)))
        assert dones[-1] != 0
        assert all(d == 0 for d in dones[:-1])
        if start == 0:
            assert indices == [0, 1, 2, 3]
        elif start == 4:
            assert indices == [4, 5]
    assert seen_starts == match_starts


def test_dataloader_sample_end_missing_raises() -> None:
    """sample_end set but never met before length/store end → ValueError."""
    store = Datastore()
    for step_index in range(5):
        store.append(
            data={
                "action": step_index,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": 0,
                "episode_index": 0,
                "step_index": step_index,
            }
        )
    loader = _loader(
        sequence_length=10,
        batch_size=1,
        num_workers=0,
        seed=0,
        sample_start=SampleMatch(equals=(("episode_index", 0), ("step_index", 0)), not_equals=()),
        sample_end=SampleMatch(equals=(), not_equals=(("task_done", 0),)),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    with pytest.raises(ValueError, match=r"sample_end was not met"):
        loader.next_batch()


def test_dataloader_sample_end_missing_when_seq_len_too_short_raises() -> None:
    """End exists later in the store but beyond sequence_length → ValueError."""
    store, _ = _packed_episode_index_store()
    # Task 0 needs 4 steps; sequence_length=2 cannot reach task_done at index 3.
    loader = _loader(
        sequence_length=2,
        batch_size=1,
        num_workers=0,
        seed=0,
        sample_start=SampleMatch(equals=(("episode_index", 0), ("step_index", 0)), not_equals=()),
        sample_end=SampleMatch(equals=(), not_equals=(("task_done", 0),)),
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    with pytest.raises(ValueError, match=r"sample_end was not met"):
        loader.next_batch()


def test_dataloader_sample_start_match_requires_field_column() -> None:
    store = Datastore()
    store.append(data={"action": 1, "reward": 0.0, "step_index": 0, "task_done": 0})
    with pytest.raises(ValueError, match="episode_index"):
        _loader(
            sequence_length=2,
            batch_size=1,
            num_workers=0,
            sample_start=SampleMatch(equals=(("episode_index", 0), ("step_index", 0)), not_equals=()),
            stores=store,
        )


def test_dataloader_sample_end_match_requires_field_column() -> None:
    store = Datastore()
    store.append(data={"action": 1, "reward": 0.0, "episode_done": 0})
    with pytest.raises(ValueError, match="task_done"):
        _loader(
            sequence_length=2,
            batch_size=1,
            num_workers=0,
            sample_end=SampleMatch(equals=(), not_equals=(("task_done", 0),)),
            stores=store,
        )
