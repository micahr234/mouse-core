"""Text callbacks use ordinary tokenization and preserve labels and readouts."""

from copy import deepcopy
import random

import numpy as np
import pytest

from mouse_core.data import (
    Tokenizer,
    TokenizerModalitySpec,
    load_tokenizer,
    save_tokenizer,
)
from tests.test_tokenizer_functions import CharacterTokenizer


def compute_text(ctx):
    return "." * ctx["count"] if ctx.get("action_requested") else ""


def selected_value(ctx):
    return f"{ctx['prefix']}:{['A', 'B'][ctx['action']]}"


def make_tokenizer(format=compute_text, **options):
    return Tokenizer(
        tokenizers={"text": CharacterTokenizer()},
        input_fields=[
            dict(type="text", provider="environment", format=format, **options),
            dict(
                type="token",
                provider="environment",
                format="V",
                head_output=True,
                input_field="action_requested",
            ),
        ],
        objective_fields=[
            dict(input_field="action", missing_value=None),
            dict(input_field="reward"),
        ],
    )


def events(counts):
    return [
        dict(
            provider="environment",
            step_index=i,
            payload=dict(
                count=n,
                action_requested=True,
                action=None if i == 0 else 1,
                reward=float(i),
            ),
        )
        for i, n in enumerate(counts)
    ]


def test_variable_and_empty_text_preserves_data_and_head_positions():
    rng = random.Random(7)
    counts = [0] + [rng.randint(1, 10) for _ in range(24)]
    source = events(counts)
    original = deepcopy(source)
    steps = make_tokenizer().encode(source)
    assert source == original
    assert len(set(counts)) > 2
    for n, step in zip(counts, steps):
        assert step.token_ids.tolist() == [ord(".")] * n + [ord("V")]
        assert step.head_token_mask.tolist() == [False] * n + [True]
    assert [s.objective_fields["reward"] for s in steps] == list(
        map(float, range(len(counts)))
    )


def test_callback_called_once_only_when_field_emits_and_gets_a_context_copy():
    seen = []

    def formatter(ctx):
        if not ctx.get("action_requested"):
            return ""
        seen.append(dict(ctx))
        ctx["action_requested"] = None
        return ".."

    tok = make_tokenizer(formatter)
    source = events([3])[0]
    original = deepcopy(source)
    assert tok(source)["token_ids"].tolist() == [46, 46, ord("V")]
    assert len(seen) == 1 and source == original
    tok(dict(provider="environment", payload={"action_requested": None}))
    assert len(seen) == 1


def test_input_selection_is_available_by_name():
    tok = Tokenizer(
        tokenizers={"text": CharacterTokenizer()},
        input_fields=[
            dict(
                type="text",
                provider="sensor",
                input_field="action",
                input_index=1,
                format=selected_value,
            )
        ],
    )
    source = dict(provider="sensor", payload={"action": [0, 1], "prefix": "p"})
    assert tok(source)["token_ids"].tolist() == list(map(ord, "p:B"))
    assert tok(dict(provider="sensor", payload={"prefix": "p"}))["token_ids"].size == 0


def test_callback_spec_roundtrips(tmp_path):
    tok = make_tokenizer()
    save_tokenizer(tokenizer=tok, path=tmp_path)
    restored = load_tokenizer(
        repo_id_or_path=str(tmp_path), tokenizers={"text": CharacterTokenizer()}
    )
    for a, b in zip(tok.encode(events([0, 1, 5])), restored.encode(events([0, 1, 5]))):
        np.testing.assert_array_equal(a.token_ids, b.token_ids)
        np.testing.assert_array_equal(a.head_token_mask, b.head_token_mask)


def test_max_tokens_applies_after_callback_tokenization():
    with pytest.raises(ValueError, match="max_tokens"):
        make_tokenizer(max_tokens=3).encode(events([4]))


@pytest.mark.parametrize("value", [None, 3, ["."]])
def test_callback_must_return_text(value):
    with pytest.raises(TypeError, match="return a string"):
        make_tokenizer(lambda ctx: value).encode(events([1]))


@pytest.mark.parametrize("kind", ["image"])
def test_callback_is_for_variable_length_text(kind):
    with pytest.raises(TypeError, match="format"):
        TokenizerModalitySpec(
            type=kind, provider="sensor", input_field="x", format=compute_text
        )


def test_unsaveable_callback_errors(tmp_path):
    with pytest.raises(TypeError, match="named top-level"):
        save_tokenizer(tokenizer=make_tokenizer(lambda ctx: "."), path=tmp_path)


def random_compute_text(ctx):
    return "." * random.randint(1, 10)


def test_random_text_callback_runs_in_four_loader_processes():
    from mouse_core.data import DataLoader, Datastore

    store = Datastore()
    store.extend(events([1] * 16))
    tok = make_tokenizer(random_compute_text)
    with DataLoader(
        stores=store,
        events_budget=16,
        sample_field="payload.count",
        transform=tok.encode,
        num_workers=4,
        prefetch=4,
        seed=7,
    ) as loader:
        lengths = []
        for _ in range(2):
            inputs, data = loader.next_batch()
            ends = inputs["head_token_mask"].nonzero(as_tuple=True)[0].tolist()
            starts = [0] + [end + 1 for end in ends[:-1]]
            lengths.extend(end - start for start, end in zip(starts, ends))
            assert data["head_step_ids"].tolist() == list(range(16))
            assert data["reward"].tolist() == list(map(float, range(16)))
        assert all(1 <= length <= 10 for length in lengths)
        assert len(set(lengths)) > 1


@pytest.mark.parametrize("kind", ["text", "token"])
@pytest.mark.parametrize("format", ["", lambda ctx: ""])
def test_empty_format_never_calls_backend_or_emits_readouts(kind, format):
    def unexpected_backend(*args, **kwargs):
        raise AssertionError(
            "Empty output must bypass the backend, including its special tokens"
        )

    tok = Tokenizer(
        tokenizers={"text": unexpected_backend},
        input_fields=[
            dict(type=kind, provider="environment", format=format, head_output=True),
        ],
    )
    (step,) = tok.encode(events([1]))
    assert step.T == 0 and step.head_token_mask.size == 0


def test_token_format_callback_skips_or_emits_one_token():
    tok = Tokenizer(
        tokenizers={"text": CharacterTokenizer()},
        input_fields=[
            dict(
                type="token",
                provider="environment",
                format=lambda ctx: ctx["text"],
                head_output=True,
            ),
        ],
    )
    for text, expected in [("", []), ("V", [ord("V")])]:
        piece = tok(dict(provider="environment", payload={"text": text}))
        assert piece["token_ids"].tolist() == expected
        assert piece["head_token_mask"].tolist() == [True] * len(expected)
    with pytest.raises(ValueError, match="expected 1"):
        tok(dict(provider="environment", payload={"text": "VV"}))
