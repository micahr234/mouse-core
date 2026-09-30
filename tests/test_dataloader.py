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
    Tokenizer,
    compose,
)
from mouse_core.data.augmenter import _stable_hash
from mouse_core.data.dataloader import _sequence_generation
from mouse_core.data.token_batch import StepTokens, TokenBatch
from tests._token_batch_helpers import token_tokenizer


def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0


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
                "when": when_step_index_zero,
            },
        ],
        objective_fields=keep,
    )


def _stamp_grouping(step: dict) -> dict:
    out = dict(step)
    out.setdefault("grouping_id", 0)
    return out


def _transform(**kwargs):
    return compose(stages=(_stamp_grouping, _tokenizer(**kwargs)))


def _loader(**kwargs) -> DataLoader:
    kwargs.setdefault("sample_field", "action")
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
            samples_budget=3,
            batch_size=1,
            sample_field="action",
            transform=_transform(),
            stores=_store_with_actions(),
        )


def test_dataloader_requires_transform() -> None:
    with pytest.raises(TypeError, match="transform"):
        DataLoader(
            samples_budget=3,
            batch_size=1,
            sample_field="action",
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
        samples_budget=3,
        batch_size=2,
        num_workers=0,
        transform=compose(stages=(_stamp_task, augmenter, _stamp_grouping, _tokenizer())),
    )
    tb, obj, _sid = loader.next_batch()
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
        samples_budget=1,
        batch_size=1,
        sample_field="action",
        num_workers=0,
        seed=0,
        transform=compose(stages=(_stamp_task, augmenter, _stamp_grouping, _tokenizer())),
    )
    # One-step samples. The fill keeps the first draw and reseeds the one that does not fit.
    stride = 1 * (1 + 1)
    loader.next_batch()
    assert augmenter._generation_for_call() == _sequence_generation(
        batch_index=0, sequence_index=1, stride=stride
    )
    loader.next_batch()
    assert augmenter._generation_for_call() == _sequence_generation(
        batch_index=1, sequence_index=1, stride=stride
    )
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
    budget = 8
    batch_size = 2
    stride = batch_size * (budget + 1)
    per_fill = budget + 1
    loader = DataLoader(
        stores=store,
        samples_budget=budget,
        batch_size=batch_size,
        sample_field="task_index",
        num_workers=0,
        seed=0,
        transform=compose(stages=(augmenter, _stamp_grouping, _tokenizer(objective_fields=_obj("action")))),
    )
    _, obj, group_id = loader.next_batch()
    actions = obj["action"]

    def _expected(*, generation: int) -> int:
        rng = np.random.default_rng(
            np.random.SeedSequence([0, generation, _stable_hash("task_index", 7)])
        )
        return int(rng.permutation(10)[0])

    gen0 = _sequence_generation(batch_index=0, sequence_index=0, stride=stride)
    gen1 = _sequence_generation(batch_index=0, sequence_index=per_fill, stride=stride)
    seq0 = [int(actions[i]) for i in range(len(actions)) if int(group_id[i]) == 0]
    seq1 = [int(actions[i]) for i in range(len(actions)) if int(group_id[i]) == 1]
    assert seq0
    assert seq1
    assert all(a == _expected(generation=gen0) for a in seq0)
    assert all(a == _expected(generation=gen1) for a in seq1)
    assert _expected(generation=gen0) != _expected(generation=gen1)
    assert augmenter._generation_for_call() == _sequence_generation(
        batch_index=0, sequence_index=per_fill + 1, stride=stride
    )


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
        samples_budget=3,
        batch_size=2,
        sample_field="action",
        num_workers=1,
        prefetch=1,
        seed=0,
        transform=marker,
        stores=_store_with_actions(),
    )
    try:
        tb, _, _sid = loader.next_batch()
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
        samples_budget=1,
        batch_size=1,
        sample_field="action",
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


def test_dataloader_validates_batch_size_length_and_prefetch() -> None:
    store = _store_with_actions()
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        _loader(samples_budget=3, batch_size=0, num_workers=0, stores=store)
    with pytest.raises(ValueError, match="exactly one"):
        _loader(batch_size=1, num_workers=0, stores=store)
    with pytest.raises(ValueError, match="batch_size is required"):
        _loader(samples_budget=3, num_workers=0, stores=store)
    with pytest.raises(ValueError, match="batch_size is required"):
        _loader(token_budget=4, num_workers=0, stores=store)
    with pytest.raises(ValueError, match="exactly one"):
        _loader(
            samples_budget=3,
            token_budget=8,
            batch_size=1,
            num_workers=0,
            stores=store,
        )
    with pytest.raises(TypeError, match="sample_field"):
        DataLoader(  # type: ignore[call-arg] -- intentionally omit required argument
            token_budget=4,
            batch_size=1,
            num_workers=0,
            transform=_transform(),
            stores=store,
        )
    with pytest.raises(ValueError, match="sample_field"):
        _loader(token_budget=4, batch_size=1, num_workers=0, sample_field="", stores=store)
    with pytest.raises(ValueError, match="token_budget must be >= 1"):
        _loader(
            token_budget=0,
            batch_size=1,
            num_workers=0,
            stores=store,
        )
    with pytest.raises(ValueError, match="samples_budget must be >= 1"):
        _loader(samples_budget=0, batch_size=1, num_workers=0, stores=store)
    with pytest.raises(ValueError, match="prefetch must be >= 1"):
        _loader(samples_budget=3, batch_size=1, num_workers=0, prefetch=0, stores=store)


def test_dataloader_num_workers_requires_free_threading() -> None:
    store = _store_with_actions()
    with patch.object(sysconfig, "get_config_var", return_value=0):
        with pytest.raises(RuntimeError, match="free-threaded"):
            _loader(samples_budget=3, batch_size=1, num_workers=1, stores=store)
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        with patch.object(sys, "_is_gil_enabled", return_value=True):
            with pytest.raises(RuntimeError, match="free-threaded|GIL"):
                _loader(samples_budget=3, batch_size=1, num_workers=1, stores=store)


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
    loader = _loader(samples_budget=1, batch_size=1, num_workers=0, seed=0, stores=store)
    seen: set[int] = set()
    for _ in range(24):
        _, obj, _sid = loader.next_batch()
        seen.update(int(a) for a in obj["action"])
    assert seen == {1, 2, 3}


def _tb_signature(
    packed: tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor],
) -> tuple:
    tb, obj, group_id = packed
    return (
        tb.B,
        tb.L,
        tb.N,
        tuple(tb.modality_ids.tolist()),
        tuple(tb.ids.tolist()),
        tuple(np.asarray(obj["action"].detach().cpu().numpy()).tolist()),
        tuple(group_id.tolist()),
    )


def test_dataloader_seed_is_deterministic() -> None:
    store = _store_with_actions()
    loader_a = _loader(samples_budget=3, batch_size=2, num_workers=0, seed=42, stores=store)
    loader_b = _loader(samples_budget=3, batch_size=2, num_workers=0, seed=42, stores=store)
    assert _tb_signature(loader_a.next_batch()) == _tb_signature(loader_b.next_batch())


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_seed_is_deterministic_with_workers() -> None:
    store = _store_with_actions()
    loader_a = _loader(samples_budget=3, batch_size=2, num_workers=1, seed=42, stores=store)
    loader_b = _loader(samples_budget=3, batch_size=2, num_workers=1, seed=42, stores=store)
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
        samples_budget=3, batch_size=2, num_workers=0, seed=7, stores=store,
        transform=_augmented_transform(),
    )
    expected = _signatures(sync, 12)
    assert len(set(expected)) > 1  # the stream actually varies over k
    if not _free_threading_ok():
        return
    threaded = _loader(
        samples_budget=3, batch_size=2, num_workers=4, prefetch=2, seed=7, stores=store,
        transform=_augmented_transform(),
    )
    try:
        assert _signatures(threaded, 12) == expected
    finally:
        threaded.close()


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_refresh_resumes_numbering_at_next_unseen_batch() -> None:
    store = _store_with_actions()
    reference = _loader(samples_budget=3, batch_size=2, num_workers=0, seed=3, stores=store)
    expected = _signatures(reference, 8)
    loader = _loader(
        samples_budget=3, batch_size=2, num_workers=3, prefetch=4, seed=3, stores=store
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
    a = _loader(samples_budget=3, batch_size=2, num_workers=0, stores=store)
    b = _loader(samples_budget=3, batch_size=2, num_workers=0, stores=store)
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
        samples_budget=3,
        batch_size=1,
        sample_field="action",
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
    loader = _loader(samples_budget=3, batch_size=1, num_workers=0, stores=store)
    loader.next_batch()
    store.append(data={"action": 4, "reward": 0.0, "episode_done": 0, "task_done": 0})
    _, obj_before, _sid = loader.next_batch()
    assert all(int(a) != 4 for a in obj_before["action"])
    loader.refresh()
    seen: set[int] = set()
    for _ in range(40):
        _, obj, _sid = loader.next_batch()
        seen.update(int(a) for a in obj["action"])
    assert 4 in seen


@pytest.mark.skipif(not _free_threading_ok(), reason="free-threading (GIL disabled) required")
def test_dataloader_refresh_drains_prefetch_queue_and_updates_store_sizes() -> None:
    store = Datastore()
    for action in range(3):
        store.append(data={"action": action, "reward": 0.0, "episode_done": 0, "task_done": 0})
    loader = _loader(samples_budget=2, batch_size=1, num_workers=1, prefetch=2, stores=store)
    try:
        loader.next_batch()
        assert loader._ns == [3]
        store.append(data={"action": 99, "reward": 0.0, "episode_done": 0, "task_done": 0})
        loader.refresh()
        assert loader._ns == [4]
    finally:
        loader.close()


def test_dataloader_keeps_a_short_sample_and_packs_another_copy() -> None:
    """A sample shorter than the step budget is kept whole. Another copy is added while it fits."""
    store = Datastore()
    for action in (1, 2, 3):
        store.append(
            data={
                "action": action,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": 0,
                "sample_id": 0,
            }
        )
    loader = _loader(
        samples_budget=8,
        sample_field="sample_id",
        batch_size=1,
        num_workers=0,
        seed=0,
        stores=store,
    )
    tb, obj, _sid = loader.next_batch()
    assert list(tb.step_counts()) == [3, 3]
    actions = [int(a) for a in obj["action"]]
    assert actions == [1, 2, 3, 1, 2, 3]


def test_dataloader_allows_short_stores() -> None:
    store = Datastore()
    store.append(data={"action": 7, "reward": 1.0, "episode_done": 0, "task_done": 0})
    loader = _loader(samples_budget=4, batch_size=1, num_workers=0, stores=store)
    tb, obj, _sid = loader.next_batch()
    assert int(tb.step_counts()[0]) == 1
    assert int(obj["action"][0]) == 7


def test_dataloader_allows_empty_stores_until_sampling() -> None:
    store = Datastore()
    loader = _loader(samples_budget=2, batch_size=1, num_workers=0, stores=store)
    try:
        with pytest.raises(ValueError, match="all stores are empty"):
            loader.next_batch()
        store.append(data={"action": 1, "reward": 0.0, "episode_done": 0, "task_done": 0})
        loader.refresh()
        tb, _, _sid = loader.next_batch()
        assert int(tb.step_counts()[0]) == 1
    finally:
        loader.close()


def test_dataloader_transform_returns_token_batch() -> None:
    from mouse_core.models.backbone import IdentityBackbone

    backbone = IdentityBackbone(hidden_dim=8, vocab_size=32)
    loader = DataLoader(
        samples_budget=3,
        batch_size=2,
        sample_field="action",
        num_workers=0,
        transform=compose(stages=(_stamp_grouping, token_tokenizer("action"))),
        stores=_store_with_actions(),
    )
    try:
        tb, obj, _sid = loader.next_batch()
        assert tb.B == 6
        assert int(tb.step_counts().sum()) == tb.N
        assert tb.N == 6
        assert all(int(n) == 1 for n in tb.step_counts())
        embeds, head_output_indices = backbone.embed(tb)
        assert embeds.shape == (tb.L, 8)
        assert head_output_indices.shape == (tb.N,)
        assert "group_id" not in obj.keys()
        assert _sid.tolist() == [0, 1, 2, 3, 4, 5]
    finally:
        loader.close()


def _run_store() -> Datastore:
    """Three contiguous ``sample_id`` runs, lengths 2, 3, and 1."""
    store = Datastore()
    rows = (
        (10, 0),
        (11, 0),
        (12, 1),
        (13, 1),
        (14, 1),
        (15, 2),
    )
    for action, sample_id in rows:
        store.append(
            data={
                "action": action,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": 0,
                "sample_id": sample_id,
            }
        )
    return store


def _index_transform():
    return compose(
        stages=(
            _stamp_grouping,
            _tokenizer(
                objective_fields=_obj("action", "store_index", "sample_id")
            ),
        )
    )


def _sequences(loader: DataLoader, n: int) -> list[list[list[tuple[int, int]]]]:
    """Per batch, each sequence as ``(store_index, sample_id)`` rows."""
    batches: list[list[list[tuple[int, int]]]] = []
    for _ in range(n):
        _, obj, group_id = loader.next_batch()
        by_seq: dict[int, list[tuple[int, int]]] = {}
        for seq_id, store_index, sample_id in zip(
            group_id, obj["store_index"], obj["sample_id"], strict=True
        ):
            by_seq.setdefault(int(seq_id), []).append((int(store_index), int(sample_id)))
        batches.append([rows for _, rows in sorted(by_seq.items())])
    return batches


def test_dataloader_sample_is_a_contiguous_equal_run() -> None:
    """Every packed sequence is one maximal run of equal ``sample_field`` values."""
    store = _run_store()
    loader = _loader(
        samples_budget=3,
        sample_field="sample_id",
        batch_size=1,
        num_workers=0,
        seed=0,
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    runs = {0: (0, 2), 1: (2, 5), 2: (5, 6)}
    for batch in _sequences(loader, 16):
        assert batch
        for rows in batch:
            indices = [index for index, _ in rows]
            ids = {sample_id for _, sample_id in rows}
            assert len(ids) == 1
            sample_id = ids.pop()
            start, end = runs[sample_id]
            assert indices == list(range(start, end))


def test_dataloader_samples_uniformly_over_runs() -> None:
    """The first draw of a fill is uniform over runs, including a one-row run."""
    store = Datastore()
    for i in range(20):
        store.append(
            data={
                "action": i + 1,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": 0,
                "sample_id": 0,
            }
        )
    store.append(
        data={
            "action": 99,
            "reward": 0.0,
            "episode_done": 0,
            "task_done": 0,
            "sample_id": 1,
        }
    )
    loader = _loader(
        samples_budget=20,
        sample_field="sample_id",
        batch_size=1,
        num_workers=0,
        seed=1,
        stores=store,
        index_field="store_index",
        transform=_index_transform(),
    )
    long_first = 0
    short_first = 0
    for _ in range(80):
        _, obj, group_id = loader.next_batch()
        first = [
            int(index)
            for index, seq in zip(obj["store_index"], group_id, strict=True)
            if int(seq) == 0
        ]
        if len(first) == 20:
            long_first += 1
        elif len(first) == 1:
            short_first += 1
        else:
            raise AssertionError(len(first))
    assert long_first > 20
    assert short_first > 20


def test_dataloader_missing_sample_field_raises() -> None:
    store = _store_with_actions()
    with pytest.raises(KeyError, match="missing"):
        _loader(
            samples_budget=2,
            sample_field="missing",
            batch_size=1,
            num_workers=0,
            stores=store,
        )


def test_dataloader_sample_field_must_be_scalar() -> None:
    store = Datastore()
    store.append(
        data={
            "action": 1,
            "reward": 0.0,
            "episode_done": 0,
            "task_done": 0,
            "vec": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="scalar"):
        _loader(
            samples_budget=1,
            sample_field="vec",
            batch_size=1,
            num_workers=0,
            stores=store,
        )


def test_dataloader_samples_budget_leaves_out_a_run_that_does_not_fit() -> None:
    store = Datastore()
    for sample_id in range(4):
        for step in range(3):
            store.append(
                data={
                    "action": sample_id * 3 + step + 1,
                    "reward": 0.0,
                    "episode_done": 0,
                    "task_done": 0,
                    "sample_id": sample_id,
                }
            )
    loader = _loader(
        samples_budget=5,
        sample_field="sample_id",
        batch_size=1,
        num_workers=0,
        seed=0,
        stores=store,
    )
    tb, _, _sid = loader.next_batch()
    assert list(tb.step_counts()) == [3]


def test_dataloader_first_sample_over_budget_raises() -> None:
    store = Datastore()
    for step in range(3):
        store.append(
            data={
                "action": step + 1,
                "reward": 0.0,
                "episode_done": 0,
                "task_done": 0,
                "sample_id": 0,
            }
        )
    loader = _loader(
        samples_budget=2,
        sample_field="sample_id",
        batch_size=1,
        num_workers=0,
        seed=0,
        stores=store,
    )
    with pytest.raises(ValueError, match="does not fit"):
        loader.next_batch()


def when_group_start(ctx):
    return bool(ctx["group_start"])


def _task_transform(*, group_start: bool):
    fields: list[dict] = [
        {"type": "token", "input_field": "action", "head_output": True},
    ]
    if group_start:
        fields.append(
            {"type": "token", "input_field": "mark", "when": when_group_start}
        )
    tokenizer = Tokenizer(
        input_fields=fields,
        objective_fields=_obj("action", "store_index", "task_done"),
    )
    return compose(stages=(_stamp_grouping, tokenizer))


def test_dataloader_samples_budget_keeps_one_run_per_fill() -> None:
    """``batch_size`` fills, each one run when the run length equals the step budget."""
    store = _task_store(n_tasks=8, steps=4)
    loader = _loader(
        samples_budget=4,
        sample_field="task_index",
        batch_size=3,
        num_workers=0,
        seed=0,
        stores=store,
        index_field="store_index",
        transform=_task_transform(group_start=False),
    )
    tb, obj, _sid = loader.next_batch()
    assert tb.B == 3
    assert tb.L == 12
    _assert_whole_segments(tb, obj, _sid, steps_per_segment=4)


def _task_store(*, n_tasks: int, steps: int) -> Datastore:
    """``n_tasks`` tasks of ``steps`` rows. Each task is one ``task_index`` run."""
    store = Datastore()
    for task in range(n_tasks):
        for step in range(steps):
            store.append(
                data={
                    "action": task * steps + step + 1,
                    "reward": 0.0,
                    "episode_done": 0,
                    "task_done": 1 if step == steps - 1 else 0,
                    "episode_index": 0,
                    "step_index": step,
                    "task_index": task,
                    "mark": 7,
                }
            )
    return store


def _assert_whole_segments(
    tb: TokenBatch,
    obj: dict[str, torch.Tensor],
    group_id: torch.Tensor,
    *,
    steps_per_segment: int,
) -> None:
    """Each sequence is one full task and ends on ``task_done``."""
    assert tb.B >= 1
    assert "group_id" not in obj.keys()
    seq = [int(x) for x in group_id]
    expected: list[int] = []
    for b in range(tb.B):
        expected.extend([b] * steps_per_segment)
    assert seq == expected
    assert [int(c) for c in tb.step_counts()] == [steps_per_segment] * tb.B
    indices = [int(x) for x in obj["store_index"]]
    dones = [int(x) for x in obj["task_done"]]
    for b in range(tb.B):
        start = b * steps_per_segment
        win = indices[start : start + steps_per_segment]
        done = dones[start : start + steps_per_segment]
        assert win == list(range(win[0], win[0] + steps_per_segment))
        assert done[-1] != 0
        assert all(d == 0 for d in done[:-1])
    token_seq = [int(x) for x in tb.group_ids]
    assert token_seq == sorted(token_seq)
    assert set(token_seq) == set(range(tb.B))


def test_dataloader_token_budget_fills_the_batch_with_segments() -> None:
    """Whole tasks are packed until the next task would pass the budget."""
    store = _task_store(n_tasks=8, steps=2)
    loader = _loader(
        token_budget=4,
        batch_size=1,
        num_workers=0,
        seed=1,
        sample_field="task_index",
        stores=store,
        index_field="store_index",
        transform=_task_transform(group_start=False),
    )
    tb, obj, _sid = loader.next_batch()
    assert tb.B == 2
    assert tb.L == 4
    _assert_whole_segments(tb, obj, _sid, steps_per_segment=2)

    short = _loader(
        token_budget=5,
        batch_size=1,
        num_workers=0,
        seed=1,
        sample_field="task_index",
        stores=store,
        index_field="store_index",
        transform=_task_transform(group_start=False),
    )
    tb, obj, _sid = short.next_batch()
    # Each task is 2 tokens. Two fit in 5; the third does not, so it is left out.
    assert tb.B == 2
    assert tb.L == 4
    _assert_whole_segments(tb, obj, _sid, steps_per_segment=2)

    two = _loader(
        token_budget=4,
        batch_size=2,
        num_workers=0,
        seed=1,
        sample_field="task_index",
        stores=store,
        index_field="store_index",
        transform=_task_transform(group_start=False),
    )
    tb, obj, _sid = two.next_batch()
    assert tb.B == 4
    assert tb.L == 8
    _assert_whole_segments(tb, obj, _sid, steps_per_segment=2)


def test_dataloader_token_budget_does_not_split_a_segment() -> None:
    """A segment that does not fit is left out, group-start tokens included.

    Each task is two steps. The first step emits one group-start token plus
    one action token; the second emits one action token. A segment costs 3.
    Budget 5 keeps one whole segment and drops the next.
    """
    store = _task_store(n_tasks=6, steps=2)
    loader = _loader(
        token_budget=5,
        batch_size=1,
        num_workers=0,
        seed=2,
        sample_field="task_index",
        stores=store,
        index_field="store_index",
        transform=_task_transform(group_start=True),
    )
    tb, obj, _sid = loader.next_batch()
    assert tb.B == 1
    _assert_whole_segments(tb, obj, _sid, steps_per_segment=2)
    token_seq = [int(x) for x in tb.group_ids]
    assert token_seq == [0, 0, 0]
    assert tb.L == 3
    step_positions = {int(i) for i in tb.head_output_indices}
    prefix_ids = [sid for i, sid in enumerate(token_seq) if i not in step_positions]
    assert prefix_ids == [0]


def test_dataloader_token_budget_rejects_an_oversized_segment() -> None:
    """A segment larger than the budget is not added, and an empty batch raises."""
    store = _task_store(n_tasks=4, steps=2)
    loader = _loader(
        token_budget=1,
        batch_size=1,
        num_workers=0,
        seed=3,
        sample_field="task_index",
        stores=store,
        index_field="store_index",
        transform=_task_transform(group_start=False),
    )
    with pytest.raises(ValueError, match="does not fit"):
        loader.next_batch()
