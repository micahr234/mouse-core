"""Disk token snapshots stay bounded and independent across spawned readers."""
from concurrent.futures import ThreadPoolExecutor
import gc
import multiprocessing as mp
from pathlib import Path
import pickle
import threading
import weakref

import numpy as np
import pytest
import torch

from mouse_core.data.token_batch import StepTokens, pack_token_batch
from mouse_core.data.token_cache import TokenCache


def sample(value=1, length=12):
    return [StepTokens(
        token_ids=np.full(length, value, dtype=np.int64),
        numeric_values=np.linspace(-.5, .5, length, dtype=np.float32),
        numeric_mask=np.ones(length, dtype=bool),
        numeric_field_ids=np.zeros(length, dtype=np.int64),
        numeric_field_names=('sensor',),
        head_mask=np.arange(length) == length - 1,
        position_values=np.arange(length, dtype=np.float64),
        position_mask=np.ones(length, dtype=bool),
        objective_fields={'reward': 2., 'action': None, 'weights': np.array([.2, .8], dtype=np.float32)},
    )]


def _spawn_reader(cache, connection):
    try:
        assert cache.read_only and cache.cached_samples == 0 and cache.memory_bytes == 0
        steps = cache.get('source/sample')
        assert steps[0].token_ids.tolist() == [7] * 12
        with pytest.raises(RuntimeError, match='owner'):
            cache.put('other', sample())
        path = cache.path
        cache.close()
        connection.send({'value': int(steps[0].token_ids[0]), 'owner_files_survived': path.is_dir()})
    except BaseException as error:
        connection.send({'error': repr(error)})
        raise
    finally:
        connection.close()


def test_roundtrip_detaches_source_returned_samples_and_packed_tensors(tmp_path):
    source = sample()
    with TokenCache(content_version='tokenizer-a:labels-b', max_memory_bytes=1_000_000, directory=tmp_path) as cache:
        cache.put('source/sample', source)
        assert cache.cached_samples == 0
        source[0].token_ids[:] = 9
        source[0].objective_fields['weights'][:] = -1
        loaded = cache.get('source/sample')
        assert loaded[0].token_ids.tolist() == [1] * 12
        np.testing.assert_allclose(loaded[0].objective_fields['weights'], [.2, .8])
        inputs, labels = pack_token_batch(steps=loaded)
        inputs['token_ids'].fill_(99)
        inputs['numeric_values'].fill_(99)
        labels['reward'].fill_(99)
        labels['weights'].fill_(99)
        loaded[0].token_ids[:] = 55
        loaded[0].objective_fields['weights'][:] = 55
        loaded[0].objective_fields['reward'] = 55.
        again = cache.get('source/sample')
        assert again[0].token_ids.tolist() == [1] * 12
        assert again[0].objective_fields['reward'] == 2.
        np.testing.assert_allclose(again[0].objective_fields['weights'], [.2, .8])
        assert again[0].objective_fields['action'] is None
        assert cache.memory_bytes <= cache.max_memory_bytes


def test_snapshot_detaches_tensor_objectives_and_large_numpy_bases(tmp_path):
    large = np.zeros(100_000, dtype=np.int64)
    source = sample(length=12)
    source[0].token_ids = large[:12]
    source[0].objective_fields['reward'] = torch.tensor(4., requires_grad=True)
    ref = weakref.ref(large)
    with TokenCache(content_version='v1', max_memory_bytes=100_000, directory=tmp_path) as cache:
        cache.put('small-view', source)
        del source, large
        gc.collect()
        assert ref() is None
        result = cache.get('small-view')
        base = result[0].token_ids.base
        assert base is None or base.nbytes == result[0].token_ids.nbytes
        reward = result[0].objective_fields['reward']
        assert isinstance(reward, np.ndarray) and reward.item() == 4.
        assert cache.memory_bytes < 20_000


def test_byte_bounded_lru_evicts_least_recently_used_samples(tmp_path, monkeypatch):
    with TokenCache(content_version='v1', max_memory_bytes=1_000_000, directory=tmp_path) as probe:
        probe.put('a', sample())
        probe.get('a')
        sample_bytes = probe.memory_bytes
    import mouse_core.data.token_cache as module
    original = module.pickle.load
    reads = []
    def counted(stream):
        value = original(stream)
        reads.append(value['key'])
        return value
    monkeypatch.setattr(module.pickle, 'load', counted)
    with TokenCache(content_version='v1', max_memory_bytes=sample_bytes * 2, directory=tmp_path) as cache:
        for key in ('a', 'b', 'c'):
            cache.put(key, sample())
        for key in ('a', 'b', 'a', 'c', 'a'):
            cache.get(key)
            assert cache.memory_bytes <= cache.max_memory_bytes
        assert reads == ['a', 'b', 'c']
        assert cache.cached_samples == 2
        cache.get('b')
        assert reads == ['a', 'b', 'c', 'b']
        assert cache.cached_samples == 2


@pytest.mark.parametrize('budget', [0, 1])
def test_oversized_samples_remain_usable_without_retention(tmp_path, budget):
    with TokenCache(content_version='v1', max_memory_bytes=budget, directory=tmp_path) as cache:
        cache.put('large', sample(length=1000))
        first = cache.get('large')
        first[0].token_ids[:] = 4
        assert cache.cached_samples == cache.memory_bytes == 0
        assert cache.get('large')[0].token_ids[0] == 1


def test_byte_budget_includes_numpy_protocol5_backing_buffers(tmp_path):
    source = sample(length=50_000)
    array_bytes = sum(value.nbytes for value in vars(source[0]).values() if isinstance(value, np.ndarray))
    with TokenCache(content_version='v1', max_memory_bytes=array_bytes * 2, directory=tmp_path) as cache:
        cache.put('large', source)
        cache.get('large')
        assert array_bytes <= cache.memory_bytes <= cache.max_memory_bytes
    with TokenCache(content_version='v1', max_memory_bytes=array_bytes // 2, directory=tmp_path) as cache:
        cache.put('large', source)
        assert cache.get('large')[0].T == 50_000
        assert cache.cached_samples == cache.memory_bytes == 0


def test_pickle_contains_only_reader_metadata_not_warm_samples_or_producer_graph(tmp_path):
    with TokenCache(content_version='v1', max_memory_bytes=10_000_000, directory=tmp_path) as owner:
        owner.put('large', sample(length=50_000))
        owner.get('large')
        setattr(owner, 'producer', threading.Lock())  # Deliberately unpicklable source-side state.
        encoded = pickle.dumps(owner)
        assert len(encoded) < 2048
        reader = pickle.loads(encoded)
        assert reader.read_only and not owner.read_only
        assert reader.path == owner.path and reader.content_version == 'v1'
        assert reader.cached_samples == reader.memory_bytes == 0
        assert not hasattr(reader, 'producer')
        assert reader.get('large')[0].T == 50_000
        reader.close()
        assert owner.path.is_dir() and owner.get('large')[0].T == 50_000


def test_spawned_reader_uses_parent_files_without_owning_their_lifetime(tmp_path):
    context = mp.get_context('spawn')
    with TokenCache(content_version='v1', max_memory_bytes=1_000_000, directory=tmp_path) as owner:
        owner.put('source/sample', sample(7))
        owner.get('source/sample')
        parent, child = context.Pipe()
        process = context.Process(target=_spawn_reader, args=(owner, child))
        process.start()
        child.close()
        try:
            assert parent.poll(30), 'Spawned cache reader did not respond'
            assert parent.recv() == {'value': 7, 'owner_files_survived': True}
            process.join(30)
            assert process.exitcode == 0
            assert owner.get('source/sample')[0].token_ids[0] == 7
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
            parent.close()


def test_owner_close_invalidates_warmed_readers_and_removes_only_owned_directory(tmp_path):
    sentinel = tmp_path / 'keep.txt'
    sentinel.write_text('unrelated')
    owner = TokenCache(content_version='v1', max_memory_bytes=1_000_000, directory=tmp_path)
    path = owner.path
    owner.put('sample', sample())
    reader = pickle.loads(pickle.dumps(owner))
    reader.get('sample')
    owner.close()
    owner.close()
    assert not path.exists() and sentinel.read_text() == 'unrelated'
    assert owner.memory_bytes == owner.cached_samples == 0
    with pytest.raises(RuntimeError, match='owner must outlive'):
        reader.get('sample')
    reader.close()
    with pytest.raises(RuntimeError, match='closed'):
        owner.get('sample')
    with pytest.raises(RuntimeError, match='closed'):
        pickle.dumps(owner)


def test_context_exception_and_owner_garbage_collection_remove_files(tmp_path):
    first: Path | None = None
    with pytest.raises(RuntimeError, match='caller failed'):
        with TokenCache(content_version='v1', max_memory_bytes=0, directory=tmp_path) as cache:
            first = cache.path
            raise RuntimeError('caller failed')
    assert first is not None and not first.exists()
    owner = TokenCache(content_version='v1', max_memory_bytes=0, directory=tmp_path)
    second = owner.path
    del owner
    gc.collect()
    assert not second.exists()


def test_duplicate_put_never_replaces_the_committed_sample(tmp_path):
    with TokenCache(content_version='v1', max_memory_bytes=100_000, directory=tmp_path) as cache:
        cache.put('same', sample(1))
        with pytest.raises(KeyError, match='already exists'):
            cache.put('same', sample(2))
        assert cache.get('same')[0].token_ids[0] == 1
        assert len(list(cache.path.iterdir())) == 1


def test_racing_duplicate_put_is_exclusive_and_atomic(tmp_path, monkeypatch):
    import mouse_core.data.token_cache as module
    original_link = module.os.link
    barrier = threading.Barrier(2)
    def commit(source, target):
        barrier.wait(timeout=10)
        return original_link(source, target)
    monkeypatch.setattr(module.os, 'link', commit)
    with TokenCache(content_version='v1', max_memory_bytes=100_000, directory=tmp_path) as cache:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(cache.put, 'same', sample(value)) for value in (1, 2)]
            errors = [job.exception(timeout=15) for job in jobs]
        assert sum(error is None for error in errors) == 1
        assert sum(isinstance(error, KeyError) for error in errors) == 1
        expected = 1 if errors[0] is None else 2
        assert cache.get('same')[0].token_ids.tolist() == [expected] * 12
        assert len(list(cache.path.iterdir())) == 1


def test_failed_write_does_not_publish_partial_data_or_leave_staging_files(tmp_path, monkeypatch):
    import mouse_core.data.token_cache as module
    def fail(_value, stream, **_kwargs):
        stream.write(b'incomplete')
        raise OSError('injected write failure')
    with TokenCache(content_version='v1', max_memory_bytes=0, directory=tmp_path) as cache:
        with monkeypatch.context() as patch:
            patch.setattr(module.pickle, 'dump', fail)
            with pytest.raises(OSError, match='write failure'):
                cache.put('sample', sample())
        assert 'sample' not in cache and list(cache.path.iterdir()) == []
        cache.put('sample', sample())
        assert 'sample' in cache


@pytest.mark.parametrize('value', ['', '   ', None, 4])
def test_invalid_content_versions_are_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='content_version'):
        TokenCache(content_version=value, max_memory_bytes=0, directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('value', ['', '\t', None, 4])
def test_invalid_keys_are_rejected(tmp_path, value):
    with TokenCache(content_version='v1', max_memory_bytes=0, directory=tmp_path) as cache:
        for operation in (lambda: cache.put(value, sample()), lambda: cache.get(value), lambda: value in cache):
            with pytest.raises(ValueError, match='key'):
                operation()


@pytest.mark.parametrize('value', [-1, 1.5, True, None])
def test_memory_budget_must_be_an_explicit_nonnegative_integer(tmp_path, value):
    with pytest.raises(ValueError, match='max_memory_bytes'):
        TokenCache(content_version='v1', max_memory_bytes=value, directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('field,value', [('content_version', 'changed'), ('key', 'other'), ('format', -1)])
def test_disk_identity_mismatches_fail_instead_of_reusing_stale_content(tmp_path, field, value):
    with TokenCache(content_version='v1', max_memory_bytes=0, directory=tmp_path) as cache:
        cache.put('sample', sample())
        path, = cache.path.glob('*.pickle')
        with path.open('rb') as stream:
            envelope = pickle.load(stream)
        envelope[field] = value
        with path.open('wb') as stream:
            pickle.dump(envelope, stream)
        with pytest.raises(ValueError, match='identity or content version'):
            cache.get('sample')


def test_keys_are_hashed_and_cannot_escape_the_owned_directory(tmp_path):
    with TokenCache(content_version='v1', max_memory_bytes=0, directory=tmp_path) as cache:
        key = '../outside/../../sample'
        cache.put(key, sample())
        assert cache.get(key)[0].T == 12
        assert len(list(cache.path.glob('*.pickle'))) == 1
        with pytest.raises(KeyError):
            cache.get('missing')
    assert list(tmp_path.iterdir()) == []


def test_nonnumeric_objective_graphs_are_rejected(tmp_path):
    source = sample()
    source[0].objective_fields['producer'] = object()
    with TokenCache(content_version='v1', max_memory_bytes=100_000, directory=tmp_path) as cache:
        with pytest.raises(TypeError, match='objectives must be numeric'):
            cache.put('sample', source)
        assert list(cache.path.iterdir()) == []


@pytest.mark.parametrize('field', ['tokenizer_names', 'numeric_field_names'])
def test_mutated_names_cannot_retain_arbitrary_producer_objects(tmp_path, field):
    source = sample()
    setattr(source[0], field, (threading.Lock(),))
    with TokenCache(content_version='v1', max_memory_bytes=100_000, directory=tmp_path) as cache:
        with pytest.raises(ValueError, match='name must be a nonempty string'):
            cache.put('sample', source)
        assert list(cache.path.iterdir()) == []
        assert cache.memory_bytes == cache.cached_samples == 0


def test_cyclic_objectives_fail_cleanly_but_repeated_acyclic_values_work(tmp_path):
    source = sample()
    cycle = []
    cycle.append(cycle)
    source[0].objective_fields['cycle'] = cycle
    with TokenCache(content_version='v1', max_memory_bytes=100_000, directory=tmp_path) as cache:
        with pytest.raises(ValueError, match='must not contain cycles'):
            cache.put('sample', source)
        assert list(cache.path.iterdir()) == []
        assert cache.memory_bytes == cache.cached_samples == 0
        shared = [1., 2.]
        source[0].objective_fields['cycle'] = [shared, shared]
        cache.put('sample', source)
        assert cache.get('sample')[0].objective_fields['cycle'] == [[1., 2.], [1., 2.]]


def test_numeric_and_name_subclasses_are_normalized_without_their_producer_graph(tmp_path):
    # Local classes and attached locks cannot be pickled. Only their plain data
    # may survive the snapshot, including caller-supplied identities and names.
    class ProducerString(str):
        pass
    class ProducerFloat(float):
        pass
    class ProducerList(list):
        pass
    class ProducerArray(np.ndarray):
        pass
    def attached(value):
        value.producer = threading.Lock()
        return value
    source = sample()
    source[0].tokenizer_names = (attached(ProducerString('default')),)
    source[0].numeric_field_names = (attached(ProducerString('sensor')),)
    source[0].token_ids = attached(source[0].token_ids.view(ProducerArray))
    source[0].objective_fields = {
        attached(ProducerString('reward')): attached(ProducerFloat(2.)),
        'weights': attached(np.array([.2, .8]).view(ProducerArray)),
        'sequence': attached(ProducerList([1., 2.])),
    }
    version = attached(ProducerString('v1'))
    key = attached(ProducerString('sample'))
    with TokenCache(content_version=version, max_memory_bytes=100_000, directory=tmp_path) as cache:
        cache.put(key, source)
        result = cache.get(key)[0]
        assert type(cache.content_version) is str
        assert type(result.tokenizer_names[0]) is type(result.numeric_field_names[0]) is str
        assert type(result.token_ids) is type(result.objective_fields['weights']) is np.ndarray
        assert type(result.objective_fields['reward']) is float
        assert type(result.objective_fields['sequence']) is list
        assert all(type(name) is str for name in result.objective_fields)
        assert pickle.loads(pickle.dumps(result)).T == 12
        reader = pickle.loads(pickle.dumps(cache))
        assert reader.get('sample')[0].T == 12
        reader.close()
