"""Header indexes, lazy checks, and deterministic process sampling."""
from functools import partial
from multiprocessing.shared_memory import SharedMemory
import gc

import numpy as np
import pytest
import torch

from mouse_core.data import DataLoader, Datastore, EventLog, Tokenizer
from tests._sampling_policies import (
    episode_sample_ranges,
    step_sample_ranges,
    environment_episode_is_complete,
)
from tests._token_batch_helpers import IntIdTokenizer


def store_of(count=6):
    store = Datastore()
    for episode in range(count):
        store.append(data=dict(provider="sensor", episode_index=episode, step_index=0,
                               payload={"value": episode + 1}))
    return store


def tokenizer():
    return Tokenizer(tokenizers={"text": IntIdTokenizer()}, input_fields=[
        dict(type="text", provider="sensor", input_field="value", format="{value}", head_output=True)],
        objective_fields=[dict(input_field="value")])


def loader(store, **kwargs):
    return DataLoader(stores=store, transform=tokenizer().encode, token_budget=4,
                      sample_ranges=episode_sample_ranges, seed=17,
                      num_workers=kwargs.pop("num_workers", 0), **kwargs)


def checks_of(data):
    checks = data._checks[0]
    assert checks is not None
    return checks


def eligible(events):
    return events[0]["payload"]["value"] % 2 == 0


def test_header_index_never_reads_payload_and_handles_sparse_episode_and_step_markers():
    from datasets import Dataset
    data = Dataset.from_list([
        dict(episode_index=0, step_index=0, payload="invalid JSON"),
        dict(episode_index=None, step_index=None, payload="invalid JSON"),
        dict(episode_index=0, step_index=1, payload="invalid JSON"),
        dict(episode_index=1, step_index=0, payload="invalid JSON"),
    ])
    assert list(episode_sample_ranges(data)) == [(0, 3), (3, 4)]
    assert list(step_sample_ranges(data, steps_per_sample=1)) == [(0, 2), (2, 3), (3, 4)]
    with pytest.raises(ValueError, match="contiguous"):
        list(episode_sample_ranges(data.select([0, 3, 0])))
    with pytest.raises(ValueError, match="first event"):
        list(episode_sample_ranges(data.select([1])))


def test_lazy_checks_are_reused_across_refresh_rebuild_and_append():
    store = store_of(1)
    calls = []
    def check(events):
        calls.append(events[0]["payload"]["value"])
        return True
    first = loader(store, sample_eligible=check)
    try:
        assert calls == []
        first.next_batch()
        assert calls == [1]
        ranges, checks = first._runs[0], first._checks[0]
        first.refresh()
        assert first._runs[0] is ranges and first._checks[0] is checks
        first.next_batch()
        assert calls == [1]
        with loader(store, sample_eligible=check) as rebuilt:
            assert rebuilt._runs[0] is ranges and rebuilt._checks[0] is checks
            rebuilt.next_batch()
        store.append(data=dict(provider="sensor", episode_index=1, step_index=0, payload={"value": 2}))
        first.refresh()
        assert checks_of(first).status.tolist() == [1, 0]
        for _ in range(3):
            first.next_batch()
        assert calls == [1, 2]
    finally:
        first.close()


def test_clear_and_replace_same_length_invalidates_checks_and_ranges():
    store = store_of(1)
    with loader(store) as data:
        data.next_batch()
        store.clear()
        store.append(data=dict(provider="sensor", episode_index=0, step_index=0, payload=[]))
        data.refresh()
        with pytest.raises(ValueError, match="JSON object"):
            data.next_batch()


@pytest.mark.parametrize("workers", [0, 4])
def test_validation_errors_are_not_rejections_even_when_ineligible(workers):
    store = Datastore()
    store.append(data=dict(provider="sensor", episode_index=0, step_index=0, payload=[]))
    with loader(store, num_workers=workers, sample_eligible=lambda events: False) as data:
        with pytest.raises((ValueError, RuntimeError), match="JSON object"):
            data.next_batch()
        assert checks_of(data).status.tolist() == [0]


@pytest.mark.parametrize("workers", [0, 4])
def test_all_ineligible_terminates_and_caches_rejections(workers):
    with loader(store_of(3), num_workers=workers, sample_eligible=lambda events: False) as data:
        with pytest.raises((ValueError, RuntimeError), match="no eligible samples"):
            data.next_batch()
        assert checks_of(data).status.tolist() == [2, 2, 2]


def test_workers_and_warm_cache_preserve_sampling_stream():
    # A cold process run and fully warmed parent run must consume the same RNG.
    batches = []
    for workers, warm in ((4, False), (0, True), (4, True)):
        store = store_of()
        with loader(store, num_workers=0, sample_eligible=eligible) as prep:
            if warm:
                for _ in range(10):
                    prep.next_batch()
        with loader(store, num_workers=workers, sample_eligible=eligible) as data:
            run = [data.next_batch() for _ in range(5)]
            data.refresh()
            run.append(data.next_batch())
            assert set(checks_of(data).status) <= {1, 2}
            batches.append(run)
    for other in batches[1:]:
        for expected, actual in zip(batches[0], other):
            for a, b in zip(expected, actual):
                assert set(a.keys()) == set(b.keys())
                for key in a.keys():
                    if isinstance(a[key], torch.Tensor):
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)
                    else:
                        assert a[key] == b[key]
            assert (actual[1]["value"] % 2 == 0).all()


def test_incomplete_tail_is_reconsidered_after_append_and_malformed_tail_raises():
    log = EventLog()
    log.event(provider="coordinator", episode_index=0, step_index=0, payload={"episode_start": True})
    log.event(provider="environment", episode_index=0, step_index=0, payload={"observation": 1})
    store = log.to_store()
    tok = Tokenizer(tokenizers={"text": IntIdTokenizer()}, input_fields=[
        dict(type="text", provider="environment", input_field="observation", format="{observation}", head_output=True)])
    data = DataLoader(stores=store, transform=tok.encode, sample_ranges=episode_sample_ranges,
                      sample_eligible=environment_episode_is_complete,
                      num_workers=0, token_budget=4, seed=5)
    try:
        with pytest.raises(ValueError, match="no eligible samples"):
            data.next_batch()
        store.append(data=dict(provider="environment", episode_index=0, step_index=1,
                               payload={"observation": 2, "terminated": True}))
        data.refresh()
        assert checks_of(data).status.tolist() == [0]
        assert data.next_batch()[0]["token_ids"].tolist() == [1, 2, 1, 2]
        store.clear()
        store.extend([dict(provider="coordinator", episode_index=0, step_index=0, payload={"episode_start": True}),
                      dict(provider="environment", episode_index=0, step_index=0, payload=[])])
        data.refresh()
        with pytest.raises(ValueError, match="JSON object"):
            data.next_batch()
    finally:
        data.close()


def test_shared_status_lifetime_is_owned_by_store_and_snapshot():
    store = store_of(1)
    data = loader(store, sample_eligible=eligible)
    name = checks_of(data)._memory.name
    store.clear()
    # An existing snapshot remains valid even after the store is cleared.
    with pytest.raises(ValueError, match="no eligible samples"):
        data.next_batch()
    data.close()
    del data
    gc.collect()
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name, track=False)


def test_callback_policy_changes_are_not_reused():
    store = store_of(1)
    with loader(store, sample_eligible=lambda events: False) as first:
        with pytest.raises(ValueError, match="no eligible"):
            first.next_batch()
    with loader(store, sample_eligible=lambda events: True) as second:
        second.next_batch()
    with loader(store, sample_eligible=lambda events: 1) as invalid:
        with pytest.raises(TypeError, match="return a bool"):
            invalid.next_batch()


@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_loader_checks_then_excludes_ineligible_samples(workers):
    from mouse_core.data import EpochDataLoader
    store = store_of(4)
    data = EpochDataLoader(stores=[store], transform=tokenizer().encode, token_budget=4,
        sample_ranges=episode_sample_ranges, sample_eligible=eligible,
        num_workers=workers, epochs=2)
    try:
        assert len(data.samples) == 2
        values = []
        while not data.exhausted:
            values.extend(data.next_batch()[1]["value"].tolist())
        assert sorted(values) == [2, 2, 4, 4]
    finally:
        data.close()
    with pytest.raises(ValueError, match="eligible sample"):
        EpochDataLoader(stores=[store], transform=tokenizer().encode, token_budget=4,
            sample_ranges=episode_sample_ranges, sample_eligible=lambda events: False,
            num_workers=workers)


def test_step_windows_have_same_ranges_with_lazy_checks_and_reused_policy():
    from tests.test_event_steps import recorded, make_tokenizer
    store = recorded().to_store()
    with DataLoader(stores=store, transform=make_tokenizer().encode, token_budget=15,
        sample_ranges=partial(step_sample_ranges, steps_per_sample=2), num_workers=0,
        sample_eligible=lambda events: True, seed=3) as data:
        assert data._runs[0].tolist() == [[0, 5], [2, 9], [9, 14], [11, 18]]
        data.next_batch()


def test_store_weights_and_empty_or_rejected_stores_do_not_block_eligible_data():
    empty, rejected, accepted = Datastore(), store_of(1), store_of(2)
    with DataLoader(stores=[empty, rejected, accepted], weights=[1, 3, 2], weight_mode="per_event",
        transform=tokenizer().encode, token_budget=4, num_workers=0, seed=1,
        sample_ranges=episode_sample_ranges, sample_eligible=eligible) as data:
        for _ in range(5):
            assert data.next_batch()[1]["value"].tolist() == [2, 2, 2, 2]


def test_partial_policy_reconstruction_reuses_index_and_checks():
    def accept(events, *, minimum):
        return events[0]["payload"]["value"] >= minimum
    store = store_of(2)
    with loader(store, sample_eligible=partial(accept, minimum=1)) as first:
        first.next_batch()
        with loader(store, sample_eligible=partial(accept, minimum=1)) as second:
            assert second._checks[0] is first._checks[0]
            assert second._runs[0] is first._runs[0]


def test_invalid_bool_step_count_cannot_reuse_integer_policy_cache():
    store = store_of(2)
    with DataLoader(stores=store, transform=tokenizer().encode, token_budget=4, num_workers=0,
                    sample_ranges=partial(step_sample_ranges, steps_per_sample=1)):
        pass
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(stores=store, transform=tokenizer().encode, token_budget=4, num_workers=0,
                   sample_ranges=partial(step_sample_ranges, steps_per_sample=True))


def test_failed_ingestion_preserves_cached_index():
    store = store_of(1)
    with loader(store) as data:
        data.next_batch()
        ranges, checks = data._runs[0], data._checks[0]
        with pytest.raises(ValueError, match="provider"):
            store.extend([dict(provider="sensor", episode_index=1, step_index=0, payload={"value": 2}),
                          dict(provider="", payload={})])
        data.refresh()
        assert data._runs[0] is ranges and data._checks[0] is checks


@pytest.mark.parametrize("workers", [0, 4])
def test_structural_validation_is_automatic_without_eligibility_callback(workers):
    store = Datastore()
    store.append(data=dict(provider="sensor", episode_index=0, step_index=0, payload="text"))
    with loader(store, num_workers=workers) as data:
        with pytest.raises((ValueError, RuntimeError), match="JSON object"):
            data.next_batch()


@pytest.mark.parametrize("payload", ['{broken', '{"x": NaN}', '{"x": 1e999}', 'null', '[]', '1', '"text"'])
def test_loader_decoder_rejects_invalid_or_nonobject_json(payload):
    from datasets import Dataset
    from mouse_core.data.dataloader import _decoded_events
    dataset = Dataset.from_dict(dict(provider=["sensor"], payload=[payload]))
    with pytest.raises(ValueError, match="JSON object"):
        _decoded_events(dataset, 0, 1, None)


def test_sampling_policies_and_validation_hook_are_not_library_api():
    import inspect
    import mouse_core.data as data
    assert "validate_sample" not in inspect.signature(data.DataLoader).parameters
    assert "validate_sample" not in inspect.signature(data.EpochDataLoader).parameters
    for name in ("episode_sample_ranges", "environment_episode_ranges", "step_sample_ranges", "environment_episode_is_complete"):
        assert not hasattr(data, name)


@pytest.mark.parametrize("interrupted", [False, True])
def test_live_checks_are_finalized_before_interpreter_teardown(tmp_path, interrupted):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from mouse_core.data import sample_index as module
    script = tmp_path / "exit_checks.py"
    names = tmp_path / "names.txt"
    script.write_text('''
from pathlib import Path
import sys
from mouse_core.data.sample_index import SampleChecks
# Keep caches reachable until interpreter exit, as a long-lived replay store does.
checks = [SampleChecks(4, None) for _ in range(33)]
Path(sys.argv[1]).write_text("\\n".join(c._memory.name for c in checks))
if sys.argv[2] == "True":
    raise KeyboardInterrupt
''')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(Path(module.__file__).resolve().parents[2])
    result = subprocess.run([sys.executable, str(script), str(names), str(interrupted)],
                            capture_output=True, text=True, env=environment, timeout=30)
    assert (result.returncode != 0) == interrupted
    assert "resource_tracker" not in result.stderr, result.stderr
    assert "Exception ignored" not in result.stderr, result.stderr
    for name in names.read_text().splitlines():
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, track=False)
