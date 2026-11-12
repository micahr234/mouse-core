"""Replay payloads stay in mapped files across worker generations."""
import json
from pathlib import Path
import pickle

import cloudpickle
from datasets import Dataset, concatenate_datasets, load_from_disk
import numpy as np
import pytest
import torch

from mouse_core.data import DataLoader, DatasetSnapshot, Datastore, StepTokens


def source(size=8):
    return Dataset.from_list([
        {'episode_index': i, 'provider': 'sensor', 'payload': json.dumps({'value': i, 'padding': 'x' * size})}
        for i in range(4)
    ])


def encode(events):
    value = events[0]['payload']['value']
    return [StepTokens(token_ids=np.array([value]), numeric_values=np.zeros(1),
                       numeric_mask=np.zeros(1, dtype=bool), head_mask=np.ones(1, dtype=bool))]


def loader(store, workers):
    return DataLoader(stores=store, transform=encode, sample_field='episode_index',
                      events_budget=2, num_workers=workers, prefetch=2, seed=42)


def equal_batches(left, right):
    for a, b in zip(left, right, strict=True):
        assert set(a.keys()) == set(b.keys())
        for key in a.keys():
            if isinstance(a[key], torch.Tensor):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
            else:
                assert a[key] == b[key]


def test_snapshot_preserves_selection_duplicates_features_and_format(tmp_path):
    original = source().select([3, 1, 3, 0])
    original.set_format('numpy')
    with DatasetSnapshot(original, directory=tmp_path) as snapshot:
        path = snapshot.path
        assert path is not None and path.exists() and snapshot.bytes_written > 0
        assert snapshot.dataset.to_list() == original.with_format(None).to_list()
        assert snapshot.dataset.features == original.features
        assert snapshot.dataset.format['type'] is None
        assert snapshot.matches(original)
        assert not snapshot.matches(original.with_format(None))
        assert pickle.loads(pickle.dumps(snapshot.dataset)).to_list() == snapshot.dataset.to_list()
        with pytest.raises(TypeError, match='ownership in the parent'):
            pickle.dumps(snapshot)
    assert not path.exists()
    with pytest.raises(RuntimeError, match='closed'):
        _ = snapshot.dataset
    snapshot.close()


def test_borrowed_files_survive_close_and_mixed_sources_are_fully_mapped(tmp_path):
    source().save_to_disk(str(tmp_path / 'source'))
    mapped = load_from_disk(str(tmp_path / 'source'))
    with DatasetSnapshot(mapped) as snapshot:
        assert snapshot.path is None and snapshot.bytes_written == 0
        assert snapshot.dataset.cache_files == mapped.cache_files
    assert all(Path(entry['filename']).exists() for entry in mapped.cache_files)
    # cache_files alone cannot establish that every table is memory-mapped.
    mixed = concatenate_datasets([mapped, source(256 * 1024)])
    assert mixed.cache_files
    with DatasetSnapshot(mixed) as snapshot:
        assert snapshot.path is not None
        assert snapshot.dataset.to_list() == mixed.to_list()
        assert len(cloudpickle.dumps(snapshot.dataset)) < 32 * 1024
    # In-memory selection indices also need persistence, even with mapped data.
    selected = mapped.select([3, 0, 3])
    with DatasetSnapshot(selected) as snapshot:
        assert snapshot.path is not None
        assert snapshot.dataset.to_list() == selected.to_list()


def test_failed_snapshot_write_removes_partial_files(tmp_path, monkeypatch):
    def fail(dataset, path):
        Path(path, 'partial').write_text('partial')
        raise OSError('disk full')
    monkeypatch.setattr(Dataset, 'save_to_disk', fail)
    with pytest.raises(OSError, match='disk full'):
        DatasetSnapshot(source(), directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_payload_size_does_not_scale_with_source_bytes():
    sizes = []
    for size in (256 * 1024, 3 * 1024 * 1024):
        dataset = source(size)
        assert len(cloudpickle.dumps(dataset)) > 4 * size
        with DatasetSnapshot(dataset) as snapshot:
            sizes.append(len(cloudpickle.dumps(snapshot.dataset)))
            assert snapshot.bytes_written > 4 * size
    assert max(sizes) < 32 * 1024
    assert abs(sizes[1] - sizes[0]) < 1024


@pytest.mark.parametrize('workers', [1, 4])
def test_growing_replay_refresh_preserves_batches_and_releases_snapshots(workers):
    store = Datastore()
    store.from_dataset(ds=source(256 * 1024))
    actual, expected = loader(store, workers), loader(store, 0)
    paths, retained = [], []
    payloads = []
    try:
        for generation in range(3):
            paths.extend(snapshot.path for snapshot in actual._snapshots if snapshot.path is not None)
            for _ in range(2):
                batch = actual.next_batch()
                equal_batches(batch, expected.next_batch())
                retained.append((batch, tuple(part.clone() for part in batch)))
            diagnostics = actual.diagnostics
            assert diagnostics['snapshot_bytes'] > 0
            assert diagnostics['snapshot_seconds'] >= 0
            startup = diagnostics['worker_startup_seconds']
            assert startup is None or startup >= 0
            payloads.append(diagnostics['worker_payload_bytes'])
            before = tuple(actual._snapshots)
            actual.refresh()
            expected.refresh()
            assert tuple(actual._snapshots) == before
            if generation != 2:
                store.append(data={'episode_index': 4 + generation, 'provider': 'sensor',
                                   'payload': {'value': 4 + generation, 'padding': 'y' * (2 * 1024 * 1024)}})
                actual.refresh()
                expected.refresh()
                assert all(not path.exists() for path in paths)
        assert max(payloads) < 128 * 1024
        assert max(payloads) - min(payloads) < 4096
    finally:
        actual.close()
        expected.close()
    assert all(not path.exists() for path in paths)
    for before, after in retained:
        equal_batches(before, after)


def test_loader_startup_failure_releases_owned_snapshots(monkeypatch):
    import mouse_core.data.dataloader as module
    store = Datastore()
    store.from_dataset(ds=source())
    observed = []
    original = module.DatasetSnapshot
    def capture(dataset):
        snapshot = original(dataset)
        observed.append(snapshot.path)
        return snapshot
    class Fail:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('worker startup failed')
    monkeypatch.setattr(module, 'DatasetSnapshot', capture)
    monkeypatch.setattr(module, 'ProcessBatches', Fail)
    with pytest.raises(RuntimeError, match='worker startup failed'):
        loader(store, 1)
    assert observed and all(not path.exists() for path in observed)


def test_empty_snapshot_keeps_schema_without_retaining_large_source_buffers():
    from datasets import Features, Value
    for dataset in (Dataset.from_dict({'x': []}, features=Features({'x': Value('int64')})),
                    source(1024 * 1024).select([])):
        with DatasetSnapshot(dataset) as snapshot:
            assert len(snapshot.dataset) == 0
            assert snapshot.dataset.features == dataset.features
            assert snapshot.path is None and snapshot.bytes_written == 0
            assert len(cloudpickle.dumps(snapshot.dataset)) < 16 * 1024


def test_worker_loader_accepts_empty_and_nonempty_stores():
    empty, full = Datastore(), Datastore()
    full.from_dataset(ds=source())
    with DataLoader(stores=[empty, full], transform=encode, sample_field='episode_index',
                    events_budget=2, num_workers=1, seed=42) as batches:
        inputs, _ = batches.next_batch()
        assert inputs['head_mask'].sum() == 2


def test_refresh_startup_failure_closes_loader_and_removes_snapshots(monkeypatch):
    import mouse_core.data.dataloader as module
    store = Datastore()
    store.from_dataset(ds=source())
    actual = loader(store, 1)
    before = actual.next_batch()
    retained = tuple(part.clone() for part in before)
    paths = [snapshot.path for snapshot in actual._snapshots]
    store.append(data={'episode_index': 5, 'provider': 'sensor', 'payload': {'value': 5}})
    original = module.DatasetSnapshot
    def capture(dataset):
        snapshot = original(dataset)
        paths.append(snapshot.path)
        return snapshot
    def fail(*args, **kwargs):
        raise RuntimeError('refresh worker startup failed')
    monkeypatch.setattr(module, 'DatasetSnapshot', capture)
    monkeypatch.setattr(module, 'ProcessBatches', fail)
    with pytest.raises(RuntimeError, match='refresh worker startup failed'):
        actual.refresh()
    assert all(path is not None and not path.exists() for path in paths)
    with pytest.raises(RuntimeError, match='closed'):
        actual.next_batch()
    equal_batches(before, retained)
    actual.close()


def test_clearing_store_releases_previous_snapshot():
    store = Datastore()
    store.from_dataset(ds=source())
    with loader(store, 1) as actual:
        paths = [snapshot.path for snapshot in actual._snapshots]
        store.clear()
        actual.refresh()
        assert all(path is not None and not path.exists() for path in paths)
        assert actual.diagnostics['snapshot_bytes'] == 0
        with pytest.raises(RuntimeError, match='no eligible samples'):
            actual.next_batch()
