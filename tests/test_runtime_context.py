"""Scheduled context follows batches through compose, tokenization, and workers."""

import os
import time
from copy import deepcopy

import numpy as np
import pytest
import torch

from mouse_core.data import (
    DataLoader,
    Datastore,
    Tokenizer,
    compose,
    load_tokenizer,
    save_tokenizer,
)


class Characters:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text))}


def compute_text(payload, *, context):
    return "." * context["count"] if payload.get("request") else ""


def make_tokenizer(format=compute_text):
    return Tokenizer(
        tokenizers={"text": Characters()},
        input_fields=[
            dict(type="text", provider="sensor", format=format),
            dict(
                type="token",
                provider="sensor",
                input_field="request",
                format="V",
                head_output=True,
            ),
        ],
        objective_fields=[dict(input_field="stored", missing_value=99)],
    )


def events():
    return [
        dict(
            provider="sensor",
            episode_index=episode,
            step_index=step,
            payload={"request": True, "stored": 99, "count": 40, "context": "payload"},
        )
        for episode in range(2)
        for step in range(2)
    ]


def store():
    result = Datastore()
    result.extend(events())
    return result


def assert_count(batch, count):
    inputs, labels = batch
    heads = inputs["head_mask"].nonzero(as_tuple=True)[0].tolist()
    assert heads
    starts = [0] + [head + 1 for head in heads[:-1]]
    assert [end - start for start, end in zip(starts, heads)] == [count] * len(heads)
    assert inputs["token_ids"].tolist() == ([46] * count + [ord("V")]) * len(heads)
    assert labels["stored"].tolist() == [99] * len(heads)
    assert "count" not in inputs and "count" not in labels


def test_context_is_separate_from_payload_and_reaches_all_tokenizer_entry_points(
    tmp_path,
):
    seen = []

    def formatter(payload, *, context):
        seen.append((dict(payload), context))
        assert payload["context"] == "payload" and payload["count"] == 40
        return compute_text(payload, context=context)

    source = events()[:2]
    before = deepcopy(source)
    tok = make_tokenizer(formatter)
    context = {"count": 3}
    expected = [46, 46, 46, ord("V")]
    assert tok(source[0], context=context)["token_ids"].tolist() == expected
    assert all(
        step.token_ids.tolist() == expected
        for step in tok.encode(source, context=context)
    )
    packed, labels = tok.pack_rows(rows=[source], context=context)
    assert packed["token_ids"].tolist() == expected * 2
    assert set(labels.keys()) == {"sequence_ids", "step_ids"}
    stream = tok.stream(context=context)
    assert stream.push(source[0]) is None
    completed = stream.push(source[1])
    final = stream.flush()
    assert completed is not None and final is not None
    assert completed.token_ids.tolist() == expected
    assert final.token_ids.tolist() == expected
    assert source == before and context == {"count": 3}
    assert all(runtime is context for _, runtime in seen)

    tok = make_tokenizer()
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(
        repo_id_or_path=str(tmp_path), tokenizers={"text": Characters()}
    )
    assert restored(source[0], context={"count": 2})["token_ids"].tolist() == [
        46,
        46,
        ord("V"),
    ]
    assert '"_format_context"' not in (tmp_path / "tokenizer.json").read_text()


def test_compose_forwards_context_only_to_participating_stages_and_never_retries_errors():
    seen = []

    def first(value):
        seen.append("ordinary")
        return value + 1

    def second(value, *, context):
        return value * context["scale"]

    pipeline = compose(stages=(first, compose(stages=(second,))))
    assert pipeline(2, context={"scale": 4}) == 12
    assert seen == ["ordinary"]
    assert compose(stages=(list,))(iter((1, 2)), context={}) == [1, 2]

    def broken(value, *, context):
        seen.append("broken")
        raise TypeError("inside stage")

    with pytest.raises(TypeError, match="inside stage"):
        compose(stages=(broken,))(0, context={})
    assert seen.count("broken") == 1


def test_context_signature_is_checked_only_at_construction(monkeypatch):
    tok = make_tokenizer()
    pipeline = compose(stages=(list, tok.encode))
    import mouse_core.data.callbacks as callbacks

    def unexpected(*args, **kwargs):
        raise AssertionError("No signature inspection while preparing batches")

    monkeypatch.setattr(callbacks.inspect, "signature", unexpected)
    assert pipeline(events(), context={"count": 2})[0].T == 3


@pytest.mark.parametrize("workers", [0, 4])
def test_loader_schedules_context_in_parent_once_per_batch_and_counts_actual_tokens(
    workers,
):
    parent_pid = os.getpid()
    calls = []
    shared = {"count": 0, "nested": [0]}

    def schedule(index):
        assert os.getpid() == parent_pid
        calls.append(index)
        shared["count"] = index % 3
        shared["nested"][0] = index % 3
        return shared

    def formatter(payload, *, context):
        # Force different completion times while keeping contexts paired with their jobs.
        assert context["count"] == context["nested"][0]
        return compute_text(payload, context=context)

    def delay(events, *, context):
        time.sleep((2 - context["count"]) * 0.002)
        return events

    pipeline = compose(stages=(list, delay, make_tokenizer(formatter).encode))
    with DataLoader(
        stores=store(),
        token_budget=12,
        sample_field="episode_index",
        transform=pipeline,
        context_for_batch=schedule,
        num_workers=workers,
        prefetch=4,
        seed=7,
    ) as loader:
        for index in range(6):
            batch = loader.next_batch()
            assert_count(batch, index % 3)
            assert batch[0]["token_ids"].numel() == 12
        assert calls == list(range(len(calls)))
        if workers:
            assert (
                len(calls) > 6
            )  # Future jobs have already been assigned their contexts.
        else:
            assert len(calls) == 6


def test_worker_count_does_not_change_scheduled_batches():
    def schedule(index):
        return {"count": index % 3}

    batches = []
    for workers in (0, 4):
        with DataLoader(
            stores=store(),
            events_budget=4,
            sample_field="episode_index",
            transform=make_tokenizer().encode,
            context_for_batch=schedule,
            num_workers=workers,
            prefetch=4,
            seed=8,
        ) as loader:
            batches.append([loader.next_batch() for _ in range(5)])
    for serial, parallel in zip(*batches):
        for left, right in zip(serial, parallel):
            assert set(left.keys()) == set(right.keys())
            for key in left.keys():
                if isinstance(left[key], torch.Tensor):
                    torch.testing.assert_close(left[key], right[key])
                else:
                    assert left[key] == right[key]


@pytest.mark.parametrize("workers", [0, 4])
def test_prefetched_context_is_fixed_and_refresh_resumes_at_next_unconsumed_index(
    workers,
):
    calls = []
    state = {"offset": 0}

    def schedule(index):
        calls.append(index)
        return {"count": state["offset"] + index}

    with DataLoader(
        stores=store(),
        events_budget=2,
        sample_field="episode_index",
        transform=make_tokenizer().encode,
        context_for_batch=schedule,
        num_workers=workers,
        prefetch=4,
        seed=7,
    ) as loader:
        assert_count(loader.next_batch(), 0)
        state["offset"] = 10
        assert_count(loader.next_batch(), 1 if workers else 11)
        old_calls = len(calls)
        loader.refresh()
        assert_count(loader.next_batch(), 12)
        assert calls[old_calls] == 2


@pytest.mark.parametrize("workers", [0, 4])
def test_runtime_context_top_level_is_read_only(workers):
    def mutate(payload, *, context):
        context["count"] = 10
        return "."

    with DataLoader(
        stores=store(),
        events_budget=2,
        sample_field="episode_index",
        transform=make_tokenizer(mutate).encode,
        context_for_batch=lambda index: {"count": 0},
        num_workers=workers,
        prefetch=1,
        seed=7,
    ) as loader:
        with pytest.raises(
            (TypeError, RuntimeError), match="does not support item assignment"
        ):
            loader.next_batch()


def test_context_without_schedule_is_none_and_payload_selection_remains_available():
    def formatter(payload, *, context):
        assert context is None
        assert payload["items"] == 1
        return ["unused", "chosen"][payload["items"]]

    tok = Tokenizer(
        tokenizers={"text": Characters()},
        input_fields=[
            dict(
                type="text",
                provider="sensor",
                input_field="items",
                input_index=1,
                format=formatter,
            ),
        ],
    )
    piece = tok(dict(provider="sensor", payload={"items": [0, 1]}))
    np.testing.assert_array_equal(piece["token_ids"], list(map(ord, "chosen")))
