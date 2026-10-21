from __future__ import annotations

"""Tests for Tokenizer (fake tokenizer / embeddings, no Hub)."""
from tests._model_helpers import simple_model
from tests._token_batch_helpers import embed_batch

from typing import Any

import torch
import torch.nn as nn
from mouse_core.data import Tokenizer
from tests._token_batch_helpers import (
    IntIdTokenizer,
    batch_to_packed,
    batch_to_token_batch,
)

from mouse_core.models import Embedding
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.mlp import MLP

def when_repeat_episode_done_nonzero(ctx):
    return "repeat_episode_done" in ctx and ctx["repeat_episode_done"] != 0


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
    {"type": "text", "input_field": "action", "format": "{field}"},
    {"type": "text", "input_field": "observation", "format": "{field}"},
    {"type": "text", "input_field": "reward", "format": "{field}", "when": when_reward_nonzero},
    {"type": "text", "input_field": "repeat_episode_done", "format": "{field}", "when": when_repeat_episode_done_nonzero},
]


def _tokenizer_fields(fields: list[dict[str, Any]], head_output: str = "action") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in fields:
        data = dict(field)
        if data.get("type") == "text" and "format" not in data:
            data["format"] = "{field}"
        if data.get("input_field") == head_output:
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
        _obj("action", "observation", "reward", "repeat_episode_done", "episode_done"),
    )
    head_output = kwargs.pop("head_output", "action")
    if kwargs:
        raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
    fields = _tokenizer_fields(input_fields, head_output=head_output)
    if not any(
        field.get("input_field") == "repeat_episode_index"
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
                "input_field": "repeat_episode_index",
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
    backbone = Embedding(hidden_dim=hidden_dim, vocab_size=vocab)
    if emb is not None:
        with torch.no_grad():
            backbone.embed_tokens.weight.copy_(emb.weight)
    return tokenizer, backbone


def test_text_tokenizer_when_not_equals_omits_value() -> None:
    tokenizer, backbone = _text_pair()
    batch = [
        [
            {"observation": 1, "action": 0, "reward": 0.0, "repeat_episode_done": 0, "episode_done": 0},
            {"observation": 2, "action": 1, "reward": 1.0, "repeat_episode_done": 1, "episode_done": 0},
        ]
    ]
    tb, obj, _sid = batch_to_packed(tokenizer, batch)
    embeds = embed_batch(backbone, tb)
    indices = tb.to_tensors(embeds.device)["head_token_ids"]
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
        int(np.flatnonzero(st0.head_token_mask)[0]),
        st0.T + int(np.flatnonzero(st1.head_token_mask)[0]),
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
        emb.weight[ord("0") % 20 + 1] = 7.0
    tokenizer2, backbone2 = _text_pair(tokenizer=_CaptureTok(), embed_tokens=emb)
    out = embed_batch(backbone2, batch_to_token_batch(tokenizer2, [[{'observation': 1, 'action': 0, 'reward': 0.0, 'repeat_episode_done': 0, 'episode_done': 0}]]))
    assert seen == ["0", "1"]
    matches = (out == 7.0).all(dim=-1)
    assert int(matches.sum().item()) == 1


def test_text_tokenizer_field_format_uses_str() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "reward", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"reward": 1.0, "grouping_id": 0})
    assert step.token_ids.tolist() == [ord(c) % 20 + 1 for c in "1.0"]


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
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"reward": 1.0, "grouping_id": 0})
    assert step.token_ids.tolist() == [ord("1") % 20 + 1]


def test_text_tokenizer_omitted_input_field_is_const() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {
                "type": "text",
                "format": ",value",
                "head_output": True,
            },
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    spec = next(field for field in tokenizer.input_fields if field.format == ",value")
    assert spec.input_field is None
    assert spec.format == ",value"
    step = tokenizer({"action": 3, "grouping_id": 0})
    assert step.token_ids.tolist() == [ord(c) % 20 + 1 for c in "3"] + [
        ord(c) % 20 + 1 for c in ",value"
    ]
    assert step.head_token_mask.tolist() == [False] * len("3") + [True] * len(",value")


def test_text_tokenizer_const_format_keeps_the_word_field() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {
                "type": "text",
                "format": "field",
                "head_output": True,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"action": 1})
    assert step.token_ids.tolist() == _enc("1") + _enc("field")


def test_text_tokenizer_const_format_rejects_placeholder() -> None:
    import pytest

    with pytest.raises(ValueError, match="must not contain placeholders"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "format": ",{value}",
                    "head_output": True,
                },
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
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
                "format": "x",
                "max_tokens": 1,
                "head_output": True,
            },
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"grouping_id": 0})
    assert len(step.token_ids) == 1


def test_text_tokenizer_max_tokens_raises_when_exceeded() -> None:
    import pytest

    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "format": "xy",
                "max_tokens": 1,
                "head_output": True,
            },
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
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
                    "format": "x",
                    "max_tokens": 0,
                    "head_output": True,
                },
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
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
                "format": ",",
                "when": _when_reward_zero,
            },
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
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
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
        ],
        tokenizer=_FakeTokenizer(),
    )
    zero = tok({"action": 1, "reward": 0.0, "grouping_id": 0})
    assert zero.token_ids.tolist() == (
        _FakeTokenizer()("0.0")["input_ids"].view(-1).tolist() + _enc("1")
    )
    nonzero = tok({"action": 1, "reward": 2.0, "grouping_id": 0})
    assert nonzero.token_ids.tolist() == (
        _FakeTokenizer()("2.0")["input_ids"].view(-1).tolist() + _enc("1")
    )


def test_text_tokenizer_requires_field_format() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires format="):
        Tokenizer(
            input_fields=[
                {"type": "text", "input_field": "observation", "head_output": True},
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )


def test_text_format_evaluates_expression_on_field() -> None:
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": ",episode {field+1}",
                "head_output": True,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"repeat_episode_index": 0})
    assert step.token_ids.tolist() == _enc(",episode 1")


def test_text_tokenizer_field_format_must_use_field_placeholder() -> None:
    import pytest

    with pytest.raises(ValueError, match="named field"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "input_field": "observation",
                    "format": "{observation}",
                    "head_output": True,
                },
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
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
                "format": "{{-}}",
                "when": _when_r_equals_zero,
            },
            {"type": "text", "format": "{{c}}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"a": 1, "r": 0, "grouping_id": 0})
    assert step.token_ids.tolist() == _enc("{1}") + _enc("{-}") + _enc("{c}")


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
                "format": ",",
                "when": _when_r_zero,
            },
            {"type": "text", "format": "\n", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    assert tokenizer({"a": 1, "r": 0.0, "grouping_id": 0}).token_ids.tolist() == _enc("1,,\n")
    assert tokenizer({"a": 1, "r": None, "grouping_id": 0}).token_ids.tolist() == _enc("1,\n")
    assert tokenizer({"a": 1, "grouping_id": 0}).token_ids.tolist() == _enc("1,\n")


def test_text_tokenizer_required_defaults_true() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    spec = TokenizerModalitySpec(type="text", input_field="a", format="{field}")
    assert spec.required is True
    assert TokenizerModalitySpec(type="text", format="c").required is True
    assert TokenizerModalitySpec(type="token", format="\n").required is True
    tokenizer = Tokenizer(
        input_fields=[{"type": "text", "input_field": "a", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
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
        TokenizerModalitySpec(type="text", format="c", required=False)


def test_unknown_tokenizer_types_rejected() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    for kind in ("discrete", "fourier", "continuous", "learnable", "end"):
        with pytest.raises(ValueError, match="unknown tokenizer modality type"):
            TokenizerModalitySpec(type=kind, input_field="a")


def test_text_tokenizer_max_tokens_only_on_variable_length_types() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    for kwargs in (
        {
            "type": "numeric",
            "input_field": "a",
            "format": "*",
            "fourier_min": 0.0,
            "fourier_max": 1.0,
        },
        {"type": "token", "format": "\n"},
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

    with pytest.raises(ValueError, match="duplicate tokenizer field 'value'"):
        Tokenizer(
            input_fields=[
                {"type": "text", "input_field": "value", "format": "{field}"},
                {"type": "text", "input_field": "value", "format": "{field}", "head_output": True},
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
                    "when": when_step_index_zero,
                },
            ],
            tokenizer=_FakeTokenizer(),
        )
    with pytest.raises(ValueError, match="duplicate tokenizer field 'a'"):
        Tokenizer(
            input_fields=[
                {"type": "text", "input_field": "a", "format": "{field}"},
                {"type": "text", "input_field": "a", "format": "{field}", "head_output": True},
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
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
            {"type": "text", "format": "["},
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
                "format": "]",
                "head_output": True,
            },
        ],
        tokenizer=_capture_tokenizer(seen),
    )
    step = tokenizer({"observation": [1.25, -0.5]})
    assert seen == ["[", "1.25", ",-0.50", "]"]
    assert step.token_ids.tolist() == (
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
    assert first({"observation": 1.25}).token_ids.tolist() == _ids_for("1.25")
    assert second({"observation": -0.5}).token_ids.tolist() == _ids_for(",-0.50")

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
        TokenizerModalitySpec(type="token", format="\n", input_index=0)
    with pytest.raises(TypeError, match="does not accept input_index="):
        TokenizerModalitySpec(type="text", format="c", input_index=0)
    with pytest.raises(TypeError, match="does not accept input_index="):
        TokenizerModalitySpec(type="image", input_field="img", input_index=0)


def test_text_input_index_roundtrip(tmp_path) -> None:
    from mouse_core.data import load_tokenizer, save_tokenizer

    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "format": "["},
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
        tokenizer=IntIdTokenizer(),
        input_fields=[{"type": "text", "input_field": "action", "format": "{field}"},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("action"),
    )
    tb = batch_to_token_batch(tokenizer, [[{"action": 16}]])
    embeds = embed_batch(backbone, tb)
    indices = tb.to_tensors(embeds.device)["head_token_ids"]
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
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("action"),
    )
    embed_batch(backbone, batch_to_token_batch(tokenizer, [[{'observation': 3, 'action': 2}]]))
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
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("observation", "pixels"),
        head_output="pixels",
    )
    batch = [[{"observation": 3, "pixels": [1, 2, 3]}]]
    tb, obj, _sid = batch_to_packed(tokenizer, batch)
    embeds = embed_batch(backbone, tb)
    indices = tb.to_tensors(embeds.device)["head_token_ids"]
    assert "pixels" in obj.keys()
    assert embeds.ndim == 2 and embeds.shape[1] == D
    # "3" (1 char → 1 token), then the two image tokens (the head-output tokens).
    assert indices.tolist() == [1, 2]
    assert embeds.shape[0] == 3


def test_text_tokenizer_missing_objective_field_raises() -> None:
    import pytest

    tokenizer, _ = _text_pair(objective_fields=_obj("action", "old_log_prob"))
    step = {"action": 1, "observation": "x", "reward": 1.0, "repeat_episode_done": 0, "grouping_id": 0}
    with pytest.raises(KeyError, match="old_log_prob"):
        tokenizer(step)


def test_text_tokenizer_keeps_length_one_vector_as_vector() -> None:
    import numpy as np

    tokenizer, _ = _text_pair(objective_fields=_obj("action", "q"))
    step = {"action": 1, "observation": "x", "reward": 1.0, "repeat_episode_done": 0, "grouping_id": 0, "q": np.array([0.5])}
    st = tokenizer(step)
    assert isinstance(st.objective_fields["q"], np.ndarray)
    assert st.objective_fields["q"].shape == (1,)


def test_text_model_card_describes_tokenizer(tmp_path) -> None:
    from mouse_core.models.io import _write_model_card
    model = simple_model(backbone=IdentityBackbone(hidden_dim=8), heads=({'action_value': (head := MLP(in_features=8, out_features=4, hidden_dim=8, num_layers=1, propagate_gradient=1.0))}), reasoner=None)
    path = tmp_path / 'README.md'
    _write_model_card(repo_id='user/mouse-text', model=model, path=path)
    text = path.read_text()
    assert 'embed_tokens' in text
    assert 'tokenizer' in text
    assert 'mouse_core.data.load_tokenizer' in text
    assert 'user/mouse-text' in text and 'stored in this repository' in text


def _group_start_tokenizer(**kwargs):
    prefix_format = kwargs.pop("prefix_format", "episode={field}\n")
    prefix_field = kwargs.pop("prefix_field", "episode_index")
    fields = [
        {
            "type": "text",
            "input_field": prefix_field,
            "format": prefix_format,
            "when": when_group_start,
        },
        {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
        {
            "type": "text",
            "input_field": "repeat_episode_index",
            "format": "{field}",
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
    st = tok({"action": 1, "episode_index": 7})
    assert st.group_start_token_ids is not None
    assert st.group_start_numeric_mask is not None
    expected = _FakeTokenizer()("episode=7\n")["input_ids"].view(-1).tolist()
    assert st.group_start_token_ids.tolist() == expected
    assert st.group_start_numeric_mask.tolist() == [False] * len(expected)
    assert st.head_token_mask.tolist() == [True]


def test_pack_emits_group_start_once_per_sequence() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_start_tokenizer()
    steps = [
        tok({"action": 1, "episode_index": 0}),
        tok({"action": 2, "episode_index": 0}),
        tok({"action": 3, "episode_index": 1}),
    ]
    inputs, obj, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 0, 0], batch_size=1, continuing=None)
    prefix = steps[0].group_start_token_ids
    assert prefix is not None
    p = int(prefix.shape[0])
    assert inputs.L == p + steps[0].T + steps[1].T + steps[2].T
    assert obj["action"].tolist() == [1, 2, 3]
    assert inputs.head_token_ids.tolist() == [
        p + int(steps[0].head_token_mask.nonzero()[0][0]),
        p + steps[0].T + int(steps[1].head_token_mask.nonzero()[0][0]),
        p + steps[0].T + steps[1].T + int(steps[2].head_token_mask.nonzero()[0][0]),
    ]
    # Group-start text tokens are nonnumeric and never head-output.
    assert not any(
        int(i) in set(inputs.head_token_ids.tolist())
        for i in range(p)
    )


def test_pack_group_start_is_per_sequence() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_start_tokenizer()
    steps = [
        tok({"action": 1, "episode_index": 0}),
        tok({"action": 2, "episode_index": 0}),
    ]
    inputs, _, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 1], batch_size=2, continuing=None)
    prefix = steps[0].group_start_token_ids
    assert prefix is not None
    p = int(prefix.shape[0])
    assert inputs.L == (p + steps[0].T) + (p + steps[1].T)
    assert inputs.sequence_ids.tolist() == (
        [0] * (p + steps[0].T) + [1] * (p + steps[1].T)
    )


def test_pack_continuing_suppresses_group_start() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_start_tokenizer()
    step = tok({"action": 1, "episode_index": 5})
    prefix = step.group_start_token_ids
    assert prefix is not None
    p = int(prefix.shape[0])

    cached, _, _sid = pack_token_batch(
        steps=[step],
        sequence_ids=[0],
        batch_size=1,
        continuing=[True],
    )
    assert cached.L == step.T

    fresh, _, _sid = pack_token_batch(
        steps=[step],
        sequence_ids=[0],
        batch_size=1,
        continuing=None,
    )
    assert fresh.L == p + step.T

    restarted, _, _sid = pack_token_batch(
        steps=[step],
        sequence_ids=[0],
        batch_size=1,
        continuing=[False],
    )
    assert restarted.L == p + step.T


def test_tokenizer_pack_rows_forwards_continuing() -> None:
    tok = _group_start_tokenizer()
    row = {"action": 1, "episode_index": 5}
    st = tok(row)
    assert st.group_start_token_ids is not None
    p = int(st.group_start_token_ids.shape[0])

    fresh = tok.pack_rows(rows=[[row]], continuing=None)
    assert fresh.L == p + st.T

    cached = tok.pack_rows(rows=[[row]], continuing=[True])
    assert cached.L == st.T


def test_text_tokenizer_group_start_missing_placeholder_raises() -> None:
    import pytest

    tok = _group_start_tokenizer(prefix_format="label={field}\n", prefix_field="label")
    with pytest.raises(KeyError, match="Required modality 'label'"):
        tok({"action": 1, "episode_index": 0})


def test_text_tokenizer_group_start_const_without_other_text_is_nonnumeric() -> None:
    tok = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "format": "hello\n",
                "when": when_group_start,
            },
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=_FakeTokenizer(),
        objective_fields=[],
    )
    st = tok({"action": 1, "episode_index": 3})
    assert st.group_start_token_ids is not None
    assert st.group_start_numeric_mask is not None
    assert st.T == 1
    assert not st.group_start_numeric_mask.any()


def test_token_pack_ignores_missing_group_start() -> None:
    from mouse_core.data import Tokenizer, pack_token_batch

    tok = Tokenizer(
        input_fields=[{"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        objective_fields=_obj("action"),
        tokenizer=IntIdTokenizer(),
    )
    steps = [
        tok({"action": 1, "episode_index": 0}),
        tok({"action": 2, "episode_index": 0}),
    ]
    inputs, _, _sid = pack_token_batch(steps=steps, sequence_ids=[0, 0], batch_size=1, continuing=None)
    assert inputs.L == steps[0].T + steps[1].T
    assert steps[0].group_start_token_ids is None


def test_repeat_episode_index_emits_only_on_step_zero() -> None:
    """``repeat_episode_index`` is text only when ``step_index`` is 0, including a later episode."""
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
                "input_field": "repeat_episode_index",
                "format": ",e={field}",
                "when": when_step_index_zero,
            },
            {
                "type": "text",
                "format": "\n",
                "max_tokens": 1,
                "head_output": True,
            },
        ],
        tokenizer=_CaptureTok(),
    )

    def rendered(*, repeat_episode_index: int, step_index: Any) -> list[str]:
        seen.clear()
        tok(
            {
                "action": 1,
                "repeat_episode_index": repeat_episode_index,
                "step_index": step_index,
                "episode_index": 3,
            }
        )
        return list(seen)

    assert rendered(repeat_episode_index=0, step_index=0) == ["1", ",e=0", "\n"]
    assert rendered(repeat_episode_index=0, step_index=torch.tensor(0)) == ["1", ",e=0", "\n"]
    assert rendered(repeat_episode_index=0, step_index=1) == ["1", "\n"]
    assert rendered(repeat_episode_index=0, step_index=2) == ["1", "\n"]
    # Same episode, later episode, its own zero step.
    assert rendered(repeat_episode_index=4, step_index=0) == ["1", ",e=4", "\n"]
    assert rendered(repeat_episode_index=4, step_index=5) == ["1", "\n"]
    seen.clear()
    tok({"action": 1, "repeat_episode_index": 2, "episode_index": 3})
    assert seen == ["1", "\n"]


def test_token_repeat_episode_index_emits_only_on_step_zero() -> None:
    tok = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        ],
        tokenizer=IntIdTokenizer(),
    )
    zero = tok({"action": 1, "repeat_episode_index": 4, "step_index": 0, "episode_index": 0})
    assert zero.token_ids.tolist() == [1, 4]
    assert zero.head_token_mask.tolist() == [True, False]
    later = tok({"action": 1, "repeat_episode_index": 4, "step_index": 2, "episode_index": 0})
    assert later.token_ids.tolist() == [1]
    again = tok({"action": 1, "repeat_episode_index": 9, "step_index": 0, "episode_index": 0})
    assert again.token_ids.tolist() == [1, 9]


def test_when_rejects_non_callable_scalar() -> None:
    import pytest

    with pytest.raises(TypeError, match="when= must be a callable"):
        Tokenizer(
            input_fields=[
                {
                    "type": "text",
                    "input_field": "repeat_episode_index",
                    "format": "{field}",
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
                    "input_field": "repeat_episode_index",
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
                "format": "x",
                "when": when_step_index_zero,
            },
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
        ],
        tokenizer=_FakeTokenizer(),
    )
    zero = tok({"action": 1, "step_index": 0, "episode_index": 0})
    assert zero.token_ids.tolist() == _FakeTokenizer()("x")["input_ids"].view(-1).tolist() + _enc("1")
    later = tok({"action": 1, "step_index": 2, "episode_index": 0})
    assert later.token_ids.tolist() == _enc("1")


def test_when_equals_or_group_start_routes_value_to_ordinary() -> None:
    """equals match prefers ordinary tokens; group_start alone uses group_start_*."""
    tok = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "repeat_episode_index",
                "format": "{field}",
                "when": when_step_index_zero_or_group_start,
            },
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
        ],
        tokenizer=IntIdTokenizer(),
    )
    # step_index==0 → ordinary emit (not group_start_*), even with group_start set.
    zero = tok({"action": 1, "repeat_episode_index": 4, "step_index": 0, "episode_index": 0})
    assert zero.token_ids.tolist() == [4, 1]
    assert zero.group_start_token_ids is None
    # Mid-episode non-zero step → only group_start branch; tokens on group_start_*.
    later = tok({"action": 1, "repeat_episode_index": 4, "step_index": 2, "episode_index": 0})
    assert later.token_ids.tolist() == [1]
    assert later.group_start_token_ids is not None
    assert later.group_start_token_ids.tolist() == [4]


def _scaled(value: float, lo: float, hi: float) -> float:
    import numpy as np

    return float(np.float32(2.0 * (value - lo) / (hi - lo) - 1.0))


def _encoded(text: str) -> list[int]:
    return _FakeTokenizer()(text)["input_ids"].view(-1).tolist()


def test_numeric_field_tokenizes_star_and_stores_scaled_value() -> None:
    import pytest


    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "action",
                "format": "{field}",
                "head_output": True,
            },
            {
                "type": "numeric",
                "input_field": "observation",
                "input_index": 0,
                "format": "*",
                "fourier_min": -4.8,
                "fourier_max": 4.8,
            },
            {
                "type": "numeric",
                "input_field": "observation",
                "input_index": 1,
                "format": "*",
                "fourier_min": -10.0,
                "fourier_max": 10.0,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"action": 0, "observation": [1.2, -5.0]})
    assert step.token_ids.tolist() == _encoded("0") + _encoded("*") + _encoded("*")
    mask = step.numeric_mask
    assert step.numeric_values[mask].tolist() == pytest.approx(
        [_scaled(1.2, -4.8, 4.8), _scaled(-5.0, -10.0, 10.0)]
    )
    assert step.numeric_values[~mask].tolist() == [0.0]
    # A value outside the window is stored, not clipped.
    wide = tokenizer({"action": 0, "observation": [20.0, 0.0]})
    assert wide.numeric_values[wide.numeric_mask][0].item() == pytest.approx(
        _scaled(20.0, -4.8, 4.8)
    )
    assert wide.numeric_values[wide.numeric_mask][0].item() > 1.0


def test_zero_numeric_values_keep_their_mask_through_prefix_packing_and_embedding() -> None:
    from mouse_core.data import pack_token_batch

    tokenizer = Tokenizer(
        input_fields=[
            {"type": "numeric", "input_field": "prefix", "format": "*",
             "fourier_min": -1., "fourier_max": 1., "when": when_group_start},
            {"type": "token", "format": "*", "head_output": True},
            {"type": "numeric", "input_field": "observation", "format": "*",
             "fourier_min": -1., "fourier_max": 1.},
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"prefix": 0., "observation": 0.})
    batch, _, _ = pack_token_batch(steps=[step, step], sequence_ids=[0, 0], continuing=None)
    tensors = batch.to_tensors("cpu")
    # Every token has the same ID and value; only the mask distinguishes them.
    assert tensors["token_ids"].unique().numel() == 1
    assert not tensors["numeric_values"].any()
    assert tensors["numeric_mask"].dtype == torch.bool
    assert tensors["numeric_mask"].tolist() == [True, False, True, False, True]
    assert tensors["head_token_ids"].tolist() == [1, 3]
    assert tensors["head_step_ids"].tolist() == [0, 1]

    embedding = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=2)
    assert embedding.fourier is not None
    with torch.no_grad():
        embedding.embed_tokens.weight.zero_()
        embedding.fourier.proj.weight.fill_(1.)
    output = embed_batch(embedding, batch)
    numeric = tensors["numeric_mask"]
    assert torch.all(output[numeric] > 0)
    assert torch.count_nonzero(output[~numeric]) == 0
    output.sum().backward()
    assert embedding.fourier.proj.weight.grad is not None
    assert embedding.fourier.proj.weight.grad.abs().sum() > 0
    assert embedding.embed_tokens.weight.grad is not None
    assert embedding.embed_tokens.weight.grad.abs().sum() > 0

    continued, _, _ = pack_token_batch(steps=[step], sequence_ids=[0], continuing=[True])
    assert continued.numeric_mask.tolist() == [False, True]
    assert continued.head_token_ids.tolist() == [0]


def test_text_only_embedding_accepts_just_token_ids() -> None:
    embedding = Embedding(hidden_dim=8, vocab_size=32)
    token_ids = torch.tensor([2, 7, 2])
    output = embedding(token_ids)
    torch.testing.assert_close(output, embedding.embed_tokens(token_ids))
    torch.testing.assert_close(output, embedding(
        token_ids, numeric_values=torch.zeros(3), numeric_mask=torch.zeros(3, dtype=torch.bool),
    ))


def test_numeric_format_is_literal_text() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    with pytest.raises(ValueError, match="must not contain placeholders"):
        TokenizerModalitySpec(
            type="numeric",
            input_field="observation",
            format="{field}",
            fourier_min=0.0,
            fourier_max=1.0,
        )
    TokenizerModalitySpec(
        type="numeric",
        input_field="x",
        format="field",
        fourier_min=0.0,
        fourier_max=2.0,
    )
    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "numeric",
                "input_field": "x",
                "format": "field",
                "fourier_min": 0.0,
                "fourier_max": 2.0,
                "head_output": True,
            }
        ],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="expected 1"):
        tokenizer({"x": 1.0})


def test_numeric_requires_distinct_finite_bounds() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    with pytest.raises(ValueError, match="fourier_min"):
        TokenizerModalitySpec(type="numeric", input_field="a", format="*")
    with pytest.raises(ValueError, match="must differ"):
        TokenizerModalitySpec(
            type="numeric",
            input_field="a",
            format="*",
            fourier_min=1.0,
            fourier_max=1.0,
        )
    with pytest.raises(TypeError, match="fourier_min"):
        TokenizerModalitySpec(
            type="text",
            input_field="a",
            format="{field}",
            fourier_min=0.0,
            fourier_max=1.0,
        )
    with pytest.raises(TypeError, match="numeric"):
        Tokenizer(
            input_fields=[
                {
                    "type": "numeric",
                    "input_field": "a",
                    "format": "*",
                    "fourier_min": 0.0,
                    "fourier_max": 1.0,
                    "head_output": True,
                }
            ]
        )


def test_numeric_group_start_keeps_the_scaled_value() -> None:
    import pytest

    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "numeric",
                "input_field": "observation",
                "format": "*",
                "fourier_min": 0.0,
                "fourier_max": 1.0,
                "when": when_group_start,
            },
            {"type": "text", "format": "\n", "head_output": True},
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"observation": 0.25})
    assert step.group_start_numeric_values is not None
    assert step.group_start_numeric_values.tolist() == pytest.approx([_scaled(0.25, 0.0, 1.0)])


def test_numeric_config_roundtrip(tmp_path) -> None:
    from mouse_core.data import load_tokenizer, save_tokenizer

    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}", "head_output": True},
            {
                "type": "numeric",
                "input_field": "observation",
                "input_index": 0,
                "format": "*",
                "fourier_min": -4.8,
                "fourier_max": 4.8,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    save_tokenizer(tokenizer=tokenizer, path=tmp_path)
    loaded = load_tokenizer(repo_id_or_path=str(tmp_path), tokenizer=_FakeTokenizer())
    spec = loaded.input_fields[1]
    assert spec.type == "numeric"
    assert spec.format == "*"
    assert spec.input_index == 0
    assert spec.fourier_min == -4.8
    assert spec.fourier_max == 4.8
    row = {"action": 1, "observation": [0.0, 9.0]}
    assert loaded(row).numeric_values.tolist() == tokenizer(row).numeric_values.tolist()


def test_numeric_embed_adds_fourier_of_the_scaled_value() -> None:
    import pytest

    from mouse_core.data import pack_token_batch

    tokenizer = Tokenizer(
        input_fields=[
            {
                "type": "text",
                "input_field": "action",
                "format": "{field}",
                "head_output": True,
            },
            {
                "type": "numeric",
                "input_field": "observation",
                "format": "*",
                "fourier_min": -1.0,
                "fourier_max": 1.0,
            },
        ],
        tokenizer=_FakeTokenizer(),
    )
    fresh = Embedding(hidden_dim=8, vocab_size=32, num_frequencies=4)
    assert fresh.fourier is not None
    assert torch.count_nonzero(fresh.fourier.proj.weight) > 0
    low = tokenizer({"action": 0, "observation": -0.5})
    batch, _, _ = pack_token_batch(
        steps=[low], sequence_ids=[0], batch_size=1, continuing=None
    )
    embeds = embed_batch(fresh, batch)
    mask = torch.tensor(low.numeric_mask)
    star = low.token_ids[low.numeric_mask]
    looked = fresh.embed_tokens.weight[torch.tensor(star.tolist())]
    assert not torch.allclose(embeds[mask], looked)
    high = tokenizer({"action": 0, "observation": 0.5})
    low_batch, _, _ = pack_token_batch(
        steps=[low], sequence_ids=[0], batch_size=1, continuing=None
    )
    high_batch, _, _ = pack_token_batch(
        steps=[high], sequence_ids=[0], batch_size=1, continuing=None
    )
    low_embeds = embed_batch(fresh, low_batch)
    high_embeds = embed_batch(fresh, high_batch)
    assert not torch.allclose(low_embeds[mask], high_embeds[mask])
    low_value = torch.tensor(low.numeric_values[low.numeric_mask])
    high_value = torch.tensor(high.numeric_values[high.numeric_mask])
    delta = high_embeds[mask] - low_embeds[mask]
    fourier_delta = fresh.fourier(high_value) - fresh.fourier(low_value)
    assert torch.allclose(delta, fourier_delta)
    text = ~mask
    assert torch.allclose(high_embeds[text], low_embeds[text])

    plain = Embedding(hidden_dim=8, vocab_size=32)
    with pytest.raises(RuntimeError, match="num_frequencies"):
        embed_batch(plain, low_batch)

    loss = low_embeds.sum()
    loss.backward()
    grad = fresh.fourier.proj.weight.grad
    assert grad is not None
    assert float(grad.abs().sum()) > 0.0


def test_token_field_emits_one_text_token() -> None:
    import pytest

    from mouse_core.data import TokenizerModalitySpec

    tokenizer = Tokenizer(
        input_fields=[
            {"type": "text", "input_field": "action", "format": "{field}"},
            {"type": "token", "format": "\n", "head_output": True},
        ],
        tokenizer=_FakeTokenizer(),
    )
    step = tokenizer({"action": 0})
    assert step.token_ids.tolist() == _encoded("0") + _encoded("\n")
    assert step.head_token_mask.tolist() == [False, True]
    wide = Tokenizer(
        input_fields=[{"type": "token", "format": "ab", "head_output": True}],
        tokenizer=_FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="expected 1"):
        wide({})
    with pytest.raises(TypeError, match="input_field"):
        TokenizerModalitySpec(type="token", format="\n", input_field="a")
    with pytest.raises(TypeError, match="max_tokens"):
        TokenizerModalitySpec(type="token", format="\n", max_tokens=1)
    with pytest.raises(TypeError, match="max_tokens"):
        TokenizerModalitySpec(
            type="numeric",
            input_field="a",
            format="*",
            fourier_min=0.0,
            fourier_max=1.0,
            max_tokens=1,
        )
    with pytest.raises(ValueError, match="placeholders"):
        TokenizerModalitySpec(type="token", format="{field}")


def test_num_frequencies_must_be_a_positive_int() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_frequencies"):
        Embedding(hidden_dim=4, vocab_size=8, num_frequencies=0)
