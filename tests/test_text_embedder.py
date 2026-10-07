from __future__ import annotations

"""Tests for Tokenizer (fake tokenizer / embeddings, no Hub)."""

from typing import Any

import torch
import torch.nn as nn
from mouse_core.data import Tokenizer
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch
from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.heads import RegressionHead

def when_episode_done_nonzero(ctx):
    return "episode_done" in ctx and ctx["episode_done"] != 0


def when_group_start(ctx):
    return bool(ctx["group_start"])


def when_reward_nonzero(ctx):
    return "reward" in ctx and ctx["reward"] != 0.0


def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0


def when_step_index_zero_or_group_start(ctx):
    return (ctx.get("step_index") == 0) | bool(ctx["group_start"])


def _when_reward_zero(ctx):
    return "reward" in ctx and ctx["reward"] == 0.0


def _when_always(ctx):
    return True


def _when_r_equals_zero(ctx):
    return "r" in ctx and ctx["r"] == 0


def _when_r_nonzero(ctx):
    return "r" in ctx and ctx["r"] != 0.0


def _when_r_zero(ctx):
    return "r" in ctx and ctx["r"] == 0.0



class _FakeTokenizer:

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [ord(c) % 20 + 1 for c in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


_DEFAULT_FIELDS: list[dict[str, Any]] = [
    {"type": "token", "input_field": "action"},
    {"type": "text", "input_field": "observation", "format": "{field}"},
    {"type": "text", "input_field": "reward", "format": "{field}", "when": when_reward_nonzero},
    {"type": "text", "input_field": "episode_done", "format": "{field}", "when": when_episode_done_nonzero},
]


def _tokenizer_fields(fields: list[dict[str, Any]], head_output: str = "action") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in fields:
        data = dict(field)
        if data.get("type") == "text" and "format" not in data:
            data["format"] = "{field}"
        if data.get("input_field") == head_output or data.get("output_field") == head_output:
            data["head_output"] = True
        out.append(data)
    return out


def _obj(*names: str) -> list[dict[str, str]]:
    return [{"input_field": name} for name in names]


def _text_pair(hidden_dim: int = 8, **kwargs):
    vocab = 32
    emb = kwargs.pop("embed_tokens", None)
    hf_tok = kwargs.pop("tokenizer", _FakeTokenizer())
    input_fields = kwargs.pop("input_fields", list(_DEFAULT_FIELDS))
    image_tokenizer = kwargs.pop("image_tokenizer", None)
    objective_fields = kwargs.pop(
        "objective_fields",
        _obj("action", "observation", "reward", "episode_done", "task_done"),
    )
    head_output = kwargs.pop("head_output", "action")
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
    fields = _tokenizer_fields(input_fields, head_output=head_output)
    if not any(
        field.get("input_field") == "episode_index"
        or field.get("output_field") == "episode_index"
        for field in fields
    ):
        head_at = next(
            (i for i, field in enumerate(fields) if field.get("head_output")),
            len(fields),
        )
        fields.insert(
            head_at,
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        )
    tokenizer = Tokenizer(
        input_fields=fields,
        tokenizer=hf_tok,
        image_tokenizer=image_tokenizer,
        objective_fields=objective_fields,
    )
    backbone = IdentityBackbone(hidden_dim=hidden_dim, vocab_size=vocab)
    if emb is not None:
        with torch.no_grad():
            backbone.embed_tokens.weight.copy_(emb.weight)
    return tokenizer, backbone


def test_text_tokenizer_positions_count_per_modality_within_step() -> None:
    tokenizer, _ = _text_pair()
    st = tokenizer(
        {"observation": 1, "action": 0, "reward": 0.5, "episode_done": 0, "task_done": 0, "grouping_id": 0}
    )
    # A text step is a single modality (``__text__``) emitted in several
    # runs; positions keep counting across the runs, 0..T-1.
    assert st.positions.tolist() == list(range(st.T))


def test_text_tokenizer_when_not_equals_omits_value() -> None:
    tokenizer, backbone = _text_pair()
    batch = [
        [
            {"observation": 1, "action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0},
            {"observation": 2, "action": 1, "reward": 1.0, "episode_done": 1, "task_done": 0},
        ]
    ]
    tb, obj, _sid = batch_to_packed(tokenizer, batch)
    embeds, indices = backbone.embed(tb)
    assert obj["reward"].dtype == torch.float32
    assert obj["action"].dtype == torch.int64
    assert obj["reward"].tolist() == [0.0, 1.0]
    assert embeds.ndim == 2 and embeds.shape[1] == 8
    # when not_equals shortens step 0 relative to step 1; head-output indices point at
    # each step's action token (the flagged head-output field).
    import numpy as np

    st0 = tokenizer({**batch[0][0], "grouping_id": 0})
    st1 = tokenizer({**batch[0][1], "grouping_id": 0})
    assert st0.T < st1.T
    assert embeds.shape[0] == st0.T + st1.T
    assert indices.tolist() == [
        int(np.flatnonzero(st0.head_output_mask)[0]),
        st0.T + int(np.flatnonzero(st1.head_output_mask)[0]),
    ]
    seen: list[str] = []

    class _CaptureTok:

        def __call__(
            self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None
        ):
            seen.append(text)
            ids = [ord(c) % 20 + 1 for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    emb = nn.Embedding(32, 8)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[0] = 7.0
    tokenizer2, backbone2 = _text_pair(tokenizer=_CaptureTok(), embed_tokens=emb)
    out, _ = backbone2.embed(
        batch_to_token_batch(tokenizer2, [[{"observation": 1, "action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0}]])
    )
    assert seen == ["1"]
    matches = (out == 7.0).all(dim=-1)
    assert int(matches.sum().item()) == 1




def test_text_tokenizer_field_format_uses_str() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "reward", "format": "{field}", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"reward": 1.0, "grouping_id": 0})
    assert step.ids.tolist() == [ord(c) % 20 + 1 for c in "1.0"]


def test_text_tokenizer_field_format_spec() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "reward",
                "format": "{field:.0f}",
                "head_output": True,
            },
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"reward": 1.0, "grouping_id": 0})
    assert step.ids.tolist() == [ord("1") % 20 + 1]


def test_text_tokenizer_omitted_input_field_is_const() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {
                "type": "text",
                "output_field": "value",
                "format": ",value",
                "head_output": True,
            },
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    spec = next(field for field in tokenizer.input_fields if field.output_field == "value")
    assert spec.input_field is None
    assert spec.output_field == "value"
    assert spec.format == ",value"
    step = tokenizer({"action": 3, "grouping_id": 0})
    assert step.ids.tolist() == [ord(c) % 20 + 1 for c in "3"] + [
        ord(c) % 20 + 1 for c in ",value"
    ]
    assert step.head_output_mask.tolist() == [False] * len("3") + [True] * len(",value")


def test_text_tokenizer_const_format_rejects_placeholder() -> None:
    import pytest

    with pytest.raises(ValueError, match="must not contain placeholders"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "output_field": "value",
                    "format": ",{value}",
                    "head_output": True,
                },
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_text_tokenizer_max_tokens_allows_at_limit() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "output_field": "value",
                "format": "x",
                "max_tokens": 1,
                "head_output": True,
            },
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"grouping_id": 0})
    assert len(step.ids) == 1


def test_text_tokenizer_max_tokens_raises_when_exceeded() -> None:
    import pytest

    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "output_field": "value",
                "format": "xy",
                "max_tokens": 1,
                "head_output": True,
            },
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="tokenized to 2 tokens"):
        tokenizer({"grouping_id": 0})


def test_text_tokenizer_max_tokens_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_tokens must be >= 1"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "output_field": "value",
                    "format": "x",
                    "max_tokens": 0,
                    "head_output": True,
                },
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_text_tokenizer_const_requires_output_field() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires output_field"):
        Tokenizer(
            input_fields=[
                {"type": "text", "format": "value", "head_output": True},
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_text_tokenizer_when_gates_cover_skip_literal_pattern() -> None:
    """Former skip/format_skipped=',' is two when-gated fields."""
    seen: list[str] = []

    class _CaptureTok:

        def __call__(
            self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None
        ):
            seen.append(text)
            ids = [ord(c) % 20 + 1 for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "reward",
                "format": "{field:.0f},",
                "when": when_reward_nonzero,
            },
            {
                "type": "text",
                "output_field": "reward_zero",
                "format": ",",
                "when": _when_reward_zero,
            },
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_CaptureTok(),
    )
    tokenizer({"action": 1, "reward": 0.0, "grouping_id": 0})
    tokenizer({"action": 1, "reward": 2.0, "grouping_id": 0})
    assert seen == ["1", ",", "1", "2,"]


def test_text_tokenizer_when_rejects_non_callable() -> None:
    import pytest

    with pytest.raises(TypeError, match="when= must be a callable"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "input_field": "reward",
                    "format": "{field}",
                    "when": {"skip": True},
                    "head_output": True,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_text_tokenizer_when_or_in_callable() -> None:
    """OR of emit reasons is written inside the callable with | / or."""

    def when_any_reward(ctx):
        if "reward" not in ctx:
            return False
        return (ctx["reward"] == 0.0) | (ctx["reward"] != 0.0)

    tok = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "reward",
                "format": "{field}",
                "when": when_any_reward,
            },
            {"type": "token", "input_field": "action", "head_output": True},
        ],
        tokenizer=_FakeTokenizer(),
    )
    zero = tok({"action": 1, "reward": 0.0, "grouping_id": 0})
    assert zero.ids.tolist() == (
        _FakeTokenizer()("0.0")["input_ids"].view(-1).tolist() + [1]
    )
    nonzero = tok({"action": 1, "reward": 2.0, "grouping_id": 0})
    assert nonzero.ids.tolist() == (
        _FakeTokenizer()("2.0")["input_ids"].view(-1).tolist() + [1]
    )


def test_text_tokenizer_requires_field_format() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires format="):
        Tokenizer(
            input_fields=[
                {"type": "text", "input_field": "observation", "head_output": True},
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_text_tokenizer_field_format_must_use_field_placeholder() -> None:
    import pytest

    with pytest.raises(ValueError, match="exactly one placeholder"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "input_field": "observation",
                    "format": "{observation}",
                    "head_output": True,
                },
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def _enc(text: str) -> list[int]:
    return [ord(c) % 20 + 1 for c in text]


def test_text_tokenizer_const_and_when_gated_unescape_braces() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "a", "format": "{{{field}}}"},
            {
                "type": "text",
                "output_field": "r_zero",
                "format": "{{-}}",
                "when": _when_r_equals_zero,
            },
            {"type": "text", "output_field": "c", "format": "{{c}}", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"a": 1, "r": 0, "grouping_id": 0})
    assert step.ids.tolist() == _enc("{1}") + _enc("{-}") + _enc("{c}")


def test_text_tokenizer_optional_missing_value_emits_nothing() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "a", "format": "{field},"},
            {
                "type": "text",
                "input_field": "r",
                "format": "{field},",
                "required": False,
                "when": _when_r_nonzero,
            },
            {
                "type": "text",
                "output_field": "r_zero",
                "format": ",",
                "when": _when_r_zero,
            },
            {"type": "text", "output_field": "v", "format": "\n", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    assert tokenizer({"a": 1, "r": 0.0, "grouping_id": 0}).ids.tolist() == _enc("1,,\n")
    assert tokenizer({"a": 1, "r": None, "grouping_id": 0}).ids.tolist() == _enc("1,\n")
    assert tokenizer({"a": 1, "grouping_id": 0}).ids.tolist() == _enc("1,\n")


def test_text_tokenizer_required_defaults_true() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    spec = TokenizerModalitySpec(type="text", input_field="a", format="{field}")
    assert spec.required is True
    assert TokenizerModalitySpec(type="text", output_field="c", format="c").required is True
    assert TokenizerModalitySpec(type="token", input_field="a").required is True
    tokenizer = Tokenizer(
        input_fields=[{"type": "text", "input_field": "a", "format": "{field}", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(KeyError, match="Required modality 'a'"):
        tokenizer({"grouping_id": 0})


def test_text_tokenizer_no_input_fields_reject_step_knobs() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    with pytest.raises(TypeError, match="do not accept required=False"):
        TokenizerModalitySpec(type="text", output_field="c", format="c", required=False)


def test_unknown_tokenizer_types_rejected() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    for kind in ("discrete", "fourier", "continuous", "learnable"):
        with pytest.raises(ValueError, match="unknown tokenizer modality type"):
            TokenizerModalitySpec(type=kind, input_field="a")


def test_text_tokenizer_max_tokens_only_on_variable_length_types() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    for kwargs in (
        {"type": "token", "input_field": "a"},
    ):
        with pytest.raises(TypeError, match="does not accept max_tokens="):
            TokenizerModalitySpec(max_tokens=1, **kwargs)  # type: ignore[arg-type]
    assert TokenizerModalitySpec(type="image", input_field="img", max_tokens=4).max_tokens == 4
    assert (
        TokenizerModalitySpec(type="text", input_field="a", format="{field}", max_tokens=4).max_tokens
        == 4
    )


def test_text_tokenizer_rejects_duplicate_field_names_across_types() -> None:
    import pytest

    with pytest.raises(ValueError, match="duplicate tokenizer field name 'value'"):
        Tokenizer(
            input_fields=[
                {"type": "text", "input_field": "a", "output_field": "value", "format": "{field}"},
                {"type": "token", "input_field": "value", "head_output": True},
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )
    with pytest.raises(ValueError, match="duplicate tokenizer field name 'a'"):
        Tokenizer(
            input_fields=[
                {"type": "text", "input_field": "a", "format": "{field}"},
                {"type": "token", "input_field": "a", "head_output": True},
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def _capture_tokenizer(seen: list[str]):
    class _CaptureTok:
        def __call__(
            self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None
        ):
            seen.append(text)
            ids = [ord(c) % 20 + 1 for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    return _CaptureTok()


def _ids_for(text: str) -> list[int]:
    return [ord(c) % 20 + 1 for c in text] or [1]


def test_text_input_index_emits_each_element_in_field_order() -> None:
    import numpy as np

    seen: list[str] = []
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "output_field": "obs_open", "format": "["},
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 0,
                "format": "{field:.2f}",
            },
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 1,
                "format": ",{field:.2f}",
            },
            {
                "type": "text",
                "output_field": "obs_close",
                "format": "]",
                "head_output": True,
            },
        ],
        tokenizer=_capture_tokenizer(seen),
    )
    step = tokenizer({"observation": [1.25, -0.5]})
    assert seen == ["[", "1.25", ",-0.50", "]"]
    assert step.ids.tolist() == (
        _ids_for("[") + _ids_for("1.25") + _ids_for(",-0.50") + _ids_for("]")
    )

    first = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "format": "{field:.2f}",
                "head_output": True,
            }
        ],
        tokenizer=_FakeTokenizer(),
    )
    second = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "format": ",{field:.2f}",
                "head_output": True,
            }
        ],
        tokenizer=_FakeTokenizer(),
    )
    assert first({"observation": 1.25}).ids.tolist() == _ids_for("1.25")
    assert second({"observation": -0.5}).ids.tolist() == _ids_for(",-0.50")

    seen.clear()
    swapped = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 1,
                "format": "{field:.2f}",
            },
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 0,
                "format": ",{field:.2f}",
                "head_output": True,
            },
        ],
        tokenizer=_capture_tokenizer(seen),
    )
    swapped({"observation": [1.25, -0.5]})
    assert seen == ["-0.50", ",1.25"]

    for value in (
        np.array([1.25, -0.5]),
        torch.tensor([1.25, -0.5]),
    ):
        seen.clear()
        tokenizer({"observation": value})
        assert seen == ["[", "1.25", ",-0.50", "]"]


def test_text_scalar_without_input_index_renders_once() -> None:
    seen: list[str] = []
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "format": "{field:.2f}",
                "head_output": True,
            }
        ],
        tokenizer=_capture_tokenizer(seen),
    )
    tokenizer({"observation": 1.25})
    assert seen == ["1.25"]


def test_text_input_index_rejects_bad_shapes() -> None:
    import numpy as np
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    vector = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "format": "{field}",
                "head_output": True,
            }
        ],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="1-D vector"):
        vector({"observation": [1.25, -0.5]})
    with pytest.raises(ValueError, match="rank 2"):
        vector({"observation": np.array([[1.25, -0.5]])})

    indexed = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 1,
                "format": "{field:.2f}",
                "head_output": True,
            }
        ],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="requires a 1-D vector"):
        indexed({"observation": 1.25})
    with pytest.raises(ValueError, match="out of range"):
        indexed({"observation": [1.25]})
    with pytest.raises(ValueError, match="rank 2"):
        indexed({"observation": [[1.25, -0.5], [0.0, 1.0]]})

    with pytest.raises(ValueError, match="input_index must be >= 0"):
        TokenizerModalitySpec(
            type="text", input_field="observation", input_index=-1, format="{field}"
        )


def test_text_input_index_uniqueness_and_rejected_types() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 0,
                "format": "{field}",
            },
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 1,
                "format": "{field}",
                "head_output": True,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="input_index=0"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "input_field": "observation",
                    "input_index": 0,
                    "format": "{field}",
                },
                {
                    "type": "text",
                    "input_field": "observation",
                    "input_index": 0,
                    "format": "{field}",
                    "head_output": True,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )
    with pytest.raises(TypeError, match="does not accept input_index="):
        TokenizerModalitySpec(type="token", input_field="a", input_index=0)
    with pytest.raises(TypeError, match="does not accept input_index="):
        TokenizerModalitySpec(type="text", output_field="c", format="c", input_index=0)
    with pytest.raises(TypeError, match="does not accept input_index="):
        TokenizerModalitySpec(type="image", input_field="img", input_index=0)


def test_text_input_index_roundtrip(tmp_path) -> None:
    from mouse_core.data import load_tokenizer, save_tokenizer

    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "output_field": "obs_open", "format": "["},
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 0,
                "format": "{field:.2f}",
            },
            {
                "type": "text",
                "input_field": "observation",
                "input_index": 1,
                "format": ",{field:.2f}",
                "head_output": True,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    save_tokenizer(tokenizer=tokenizer, path=tmp_path)
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=_FakeTokenizer())
    assert [spec.input_index for spec in loaded.input_fields] == [None, 0, 1]


def test_token_modality_is_single_embed_row() -> None:
    D = 8
    emb = nn.Embedding(32, D)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[16] = 3.0
    tokenizer, backbone = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        input_fields=[{"type": "token", "input_field": "action"},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("action"),
    )
    embeds, indices = backbone.embed(batch_to_token_batch(tokenizer, [[{"action": 16}]]))
    assert embeds.shape[0] == 1
    assert int(indices[0].item()) == 0
    assert torch.equal(embeds[0], emb.weight[16])


def test_text_tokenizer_tokenizes_each_field_separately() -> None:
    seen: list[str] = []

    class _CaptureTok:

        def __call__(
            self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None
        ):
            seen.append(text)
            ids = [ord(c) % 20 + 1 for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    emb = nn.Embedding(32, 8)
    tokenizer, backbone = _text_pair(
        tokenizer=_CaptureTok(),
        embed_tokens=emb,
        input_fields=[
            {"type": "text", "input_field": "observation", "format": "o={field}"},
            {"type": "text", "input_field": "action", "format": "a={field}"},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("action"),
    )
    backbone.embed(batch_to_token_batch(tokenizer, [[{"observation": 3, "action": 2}]]))
    assert seen == ["o=3", "a=2"]


def test_identity_embed_image_token_ids() -> None:
    D = 8

    def fake_image_tok(image):
        return [3, 4]

    emb = nn.Embedding(32, D)
    tokenizer, backbone = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        image_tokenizer=fake_image_tok,
        input_fields=[
            {"type": "text", "input_field": "observation", "format": "{field}"},
            {"type": "image", "input_field": "pixels"},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("observation", "pixels"),
        head_output="pixels",
    )
    batch = [[{"observation": 3, "pixels": [1, 2, 3]}]]
    tb, obj, _sid = batch_to_packed(tokenizer, batch)
    embeds, indices = backbone.embed(tb)
    assert "pixels" in obj.keys()
    assert embeds.ndim == 2 and embeds.shape[1] == D
    # "3" (1 char → 1 token), then the two image tokens (the head-output tokens).
    assert indices.tolist() == [1, 2]
    assert embeds.shape[0] == 3




def test_text_tokenizer_missing_objective_field_raises() -> None:
    import pytest

    tokenizer, _ = _text_pair(objective_fields=_obj("action", "old_log_prob"))
    step = {"action": 1, "observation": "x", "reward": 1.0, "episode_done": 0, "grouping_id": 0}
    with pytest.raises(KeyError, match="old_log_prob"):
        tokenizer(step)


def test_text_tokenizer_keeps_length_one_vector_as_vector() -> None:
    import numpy as np

    tokenizer, _ = _text_pair(objective_fields=_obj("action", "q"))
    step = {"action": 1, "observation": "x", "reward": 1.0, "episode_done": 0, "grouping_id": 0, "q": np.array([0.5])}
    st = tokenizer(step)
    assert isinstance(st.objective_fields["q"], np.ndarray)
    assert st.objective_fields["q"].shape == (1,)


def test_text_model_card_describes_tokenizer(tmp_path) -> None:
    from mouse_core.models.base import _write_model_card

    model = Model(
        backbone=IdentityBackbone(hidden_dim=8, vocab_size=16),
        heads=(head := RegressionHead(
            in_features=8, out_features=4, hidden_dim=8, num_layers=1, use_norm=True,
            propagate_gradient=1.0,
        )),
        action_source="action_value",
        reasoner=None,
    )
    path = tmp_path / "README.md"
    _write_model_card(
        repo_id="user/mouse-text",
        tokenizer_repo_id="user/mouse-text-tokenizer",
        model=model,
        path=path,
    )
    text = path.read_text()
    assert "embed_tokens" in text
    assert "Tokenizer" in text
    assert "from mouse_core.data import load_tokenizer" in text
    assert "load_tokenizer(" in text
    assert "user/mouse-text-tokenizer" in text




def _group_start_tokenizer(**kwargs):
    prefix_format = kwargs.pop("prefix_format", "task={field}\n")
    prefix_field = kwargs.pop("prefix_field", "task_index")
    fields = [
        {
            "type": "text",
            "input_field": prefix_field,
            "output_field": "group_start",
            "format": prefix_format,
            "when": when_group_start,
        },
        {"type": "token", "input_field": "action", "head_output": True},
        {
            "type": "token",
            "input_field": "episode_index",
            "when": when_step_index_zero,
        },
    ]
    return Tokenizer(
        input_fields=fields,
        tokenizer=kwargs.pop("tokenizer", _FakeTokenizer()),
        objective_fields=_obj("action"),
        **kwargs,
    )


def test_text_tokenizer_group_start_carried_on_step() -> None:
    tok = _group_start_tokenizer()
    st = tok({"action": 1, "task_index": 7})
    assert st.group_start_ids is not None
    assert st.group_start_modality_ids is not None
    expected = _FakeTokenizer()("task=7\n")["input_ids"].view(-1).tolist()
    assert st.group_start_ids.tolist() == expected
    assert st.group_start_modality_ids.tolist() == [0] * len(expected)
    assert st.head_output_mask.tolist() == [True]


def test_pack_emits_group_start_once_per_sequence() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_start_tokenizer()
    steps = [
        tok({"action": 1, "task_index": 0}),
        tok({"action": 2, "task_index": 0}),
        tok({"action": 3, "task_index": 1}),
    ]
    inputs, obj, _sid = pack_token_batch(steps=steps, group_ids=[0, 0, 0], batch_size=1, continuing=None)
    prefix = steps[0].group_start_ids
    assert prefix is not None
    p = int(prefix.shape[0])
    assert inputs.L == p + steps[0].T + steps[1].T + steps[2].T
    assert obj["action"].tolist() == [1, 2, 3]
    assert inputs.head_output_indices.tolist() == [
        p + int(steps[0].head_output_mask.nonzero()[0][0]),
        p + steps[0].T + int(steps[1].head_output_mask.nonzero()[0][0]),
        p + steps[0].T + steps[1].T + int(steps[2].head_output_mask.nonzero()[0][0]),
    ]
    # Group-start tokens are __text__ and never head-output.
    assert not any(
        int(i) in set(inputs.head_output_indices.tolist())
        for i in range(p)
    )


def test_pack_group_start_is_per_sequence() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_start_tokenizer()
    steps = [
        tok({"action": 1, "task_index": 0}),
        tok({"action": 2, "task_index": 0}),
    ]
    inputs, _, _sid = pack_token_batch(steps=steps, group_ids=[0, 1], batch_size=2, continuing=None)
    prefix = steps[0].group_start_ids
    assert prefix is not None
    p = int(prefix.shape[0])
    assert inputs.L == (p + steps[0].T) + (p + steps[1].T)
    assert inputs.group_ids.tolist() == (
        [0] * (p + steps[0].T) + [1] * (p + steps[1].T)
    )


def test_pack_continuing_suppresses_group_start() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_start_tokenizer()
    step = tok({"action": 1, "task_index": 5})
    prefix = step.group_start_ids
    assert prefix is not None
    p = int(prefix.shape[0])

    cached, _, _sid = pack_token_batch(
        steps=[step],
        group_ids=[0],
        batch_size=1,
        continuing=[True],
    )
    assert cached.L == step.T

    fresh, _, _sid = pack_token_batch(
        steps=[step],
        group_ids=[0],
        batch_size=1,
        continuing=None,
    )
    assert fresh.L == p + step.T

    restarted, _, _sid = pack_token_batch(
        steps=[step],
        group_ids=[0],
        batch_size=1,
        continuing=[False],
    )
    assert restarted.L == p + step.T


def test_tokenizer_pack_rows_forwards_continuing() -> None:
    tok = _group_start_tokenizer()
    row = {"action": 1, "task_index": 5}
    st = tok(row)
    assert st.group_start_ids is not None
    p = int(st.group_start_ids.shape[0])

    fresh = tok.pack_rows(rows=[[row]], continuing=None)
    assert fresh.L == p + st.T

    cached = tok.pack_rows(rows=[[row]], continuing=[True])
    assert cached.L == st.T


def test_text_tokenizer_group_start_missing_placeholder_raises() -> None:
    import pytest

    tok = _group_start_tokenizer(prefix_format="label={field}\n", prefix_field="label")
    with pytest.raises(KeyError, match="Required modality 'label'"):
        tok({"action": 1, "task_index": 0})


def test_text_tokenizer_group_start_const_without_other_text_adds_text_modality() -> None:
    tok = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "output_field": "group_start",
                "format": "hello\n",
                "when": when_group_start,
            },
            {"type": "token", "input_field": "action", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
        objective_fields=[],
    )
    assert "__text__" in tok.modality_names
    st = tok({"action": 1, "task_index": 3})
    assert st.group_start_ids is not None
    assert st.group_start_modality_ids is not None
    assert st.T == 1
    assert st.modality_names[int(st.group_start_modality_ids[0])] == "__text__"


def test_token_pack_ignores_missing_group_start() -> None:
    from mouse_core.data import Tokenizer, pack_token_batch

    tok = Tokenizer(
        input_fields=[{"type": "token", "input_field": "action", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("action"),
    )
    steps = [
        tok({"action": 1, "task_index": 0}),
        tok({"action": 2, "task_index": 0}),
    ]
    inputs, _, _sid = pack_token_batch(steps=steps, group_ids=[0, 0], batch_size=1, continuing=None)
    assert inputs.L == steps[0].T + steps[1].T
    assert steps[0].group_start_ids is None


def test_episode_index_emits_only_on_step_zero() -> None:
    """``episode_index`` is text only when ``step_index`` is 0, including a later episode."""
    seen: list[str] = []

    class _CaptureTok:

        def __call__(
            self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None
        ):
            seen.append(text)
            ids = [ord(c) % 20 + 1 for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {
                "type": "text",
                "input_field": "episode_index",
                "format": ",e={field}",
                "when": when_step_index_zero,
            },
            {
                "type": "text",
                "output_field": "value",
                "format": "\n",
                "max_tokens": 1,
                "head_output": True,
            },
        ],
        tokenizer=_CaptureTok(),
    )

    def rendered(*, episode_index: int, step_index: Any) -> list[str]:
        seen.clear()
        tok(
            {
                "action": 1,
                "episode_index": episode_index,
                "step_index": step_index,
                "task_index": 3,
            }
        )
        return list(seen)

    assert rendered(episode_index=0, step_index=0) == ["1", ",e=0", "\n"]
    assert rendered(episode_index=0, step_index=torch.tensor(0)) == ["1", ",e=0", "\n"]
    assert rendered(episode_index=0, step_index=1) == ["1", "\n"]
    assert rendered(episode_index=0, step_index=2) == ["1", "\n"]
    # Same task, later episode, its own zero step.
    assert rendered(episode_index=4, step_index=0) == ["1", ",e=4", "\n"]
    assert rendered(episode_index=4, step_index=5) == ["1", "\n"]
    seen.clear()
    tok({"action": 1, "episode_index": 2, "task_index": 3})
    assert seen == ["1", "\n"]


def test_token_episode_index_emits_only_on_step_zero() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "token", "input_field": "action", "head_output": True},
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero,
            },
        ],
    )
    zero = tok({"action": 1, "episode_index": 4, "step_index": 0, "task_index": 0})
    assert zero.ids.tolist() == [1, 4]
    assert zero.head_output_mask.tolist() == [True, False]
    later = tok({"action": 1, "episode_index": 4, "step_index": 2, "task_index": 0})
    assert later.ids.tolist() == [1]
    again = tok({"action": 1, "episode_index": 9, "step_index": 0, "task_index": 0})
    assert again.ids.tolist() == [1, 9]


def test_when_rejects_non_callable_scalar() -> None:
    import pytest

    with pytest.raises(TypeError, match="when= must be a callable"):
        Tokenizer(
            input_fields=[
                {
                    "type": "token",
                    "input_field": "episode_index",
                    "when": 0,
                    "head_output": True,
                }
            ],
        )


def test_head_output_rejects_when() -> None:
    import pytest

    with pytest.raises(ValueError, match="head_output"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "input_field": "episode_index",
                    "format": "{field}",
                    "when": when_step_index_zero,
                    "head_output": True,
                }
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_const_accepts_when() -> None:
    tok = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "output_field": "mark",
                "format": "x",
                "when": when_step_index_zero,
            },
            {"type": "token", "input_field": "action", "head_output": True},
        ],
        tokenizer=_FakeTokenizer(),
    )
    zero = tok({"action": 1, "step_index": 0, "task_index": 0})
    assert zero.ids.tolist() == _FakeTokenizer()("x")["input_ids"].view(-1).tolist() + [1]
    later = tok({"action": 1, "step_index": 2, "task_index": 0})
    assert later.ids.tolist() == [1]


def test_when_equals_or_group_start_routes_value_to_ordinary() -> None:
    """equals match prefers ordinary tokens; group_start alone uses group_start_*."""
    tok = Tokenizer(
        input_fields=[
            {
                "type": "token",
                "input_field": "episode_index",
                "when": when_step_index_zero_or_group_start,
            },
            {"type": "token", "input_field": "action", "head_output": True},
        ],
    )
    # step_index==0 → ordinary emit (not group_start_*), even with group_start set.
    zero = tok({"action": 1, "episode_index": 4, "step_index": 0, "task_index": 0})
    assert zero.ids.tolist() == [4, 1]
    assert zero.group_start_ids is None
    # Mid-task non-zero step → only group_start branch; tokens on group_start_*.
    later = tok({"action": 1, "episode_index": 4, "step_index": 2, "task_index": 0})
    assert later.ids.tolist() == [1]
    assert later.group_start_ids is not None
    assert later.group_start_ids.tolist() == [4]
