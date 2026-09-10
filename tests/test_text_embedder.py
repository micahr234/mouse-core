from __future__ import annotations

"""Tests for TextEmbedder / TextTokenizer (fake tokenizer / embeddings, no Hub)."""

from typing import Any

import torch
import torch.nn as nn
from mouse_core.data import TextTokenizer
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch
from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import TextEmbedder
from mouse_core.models.heads import DiscreteActionValueHead


class _FakeTokenizer:

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [ord(c) % 20 + 1 for c in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


_DEFAULT_FIELDS: list[dict[str, Any]] = [
    {"type": "token", "input_field": "action"},
    {"type": "text", "input_field": "observation", "format": "observation={observation}"},
    {"type": "text", "input_field": "reward", "format": "reward={reward}", "skip": 0.0},
    {"type": "text", "input_field": "episode_done", "format": "episode_done={episode_done}", "skip": 0},
]
_DEFAULT_FORMAT = "<action={action},{observation},{reward},{episode_done}>"


def _tokenizer_fields(fields: list[dict[str, Any]], head_output: str = "action") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in fields:
        data = dict(field)
        if data.get("input_field") == head_output:
            data["head_output"] = True
        out.append(data)
    return out


def _obj(*names: str) -> list[dict[str, str]]:
    return [{"input_field": name} for name in names]


def _text_pair(hidden_dim: int = 8, **kwargs):
    vocab = 32
    emb = kwargs.pop("embed_tokens", None)
    if emb is None:
        emb = nn.Embedding(vocab, hidden_dim)
        nn.init.normal_(emb.weight, std=0.02)
    hf_tok = kwargs.pop("tokenizer", _FakeTokenizer())
    format_str = kwargs.pop("format", _DEFAULT_FORMAT)
    group_prefix = kwargs.pop("group_prefix", None)
    input_fields = kwargs.pop("input_fields", list(_DEFAULT_FIELDS))
    image_processor = kwargs.pop("image_processor", None)
    objective_fields = kwargs.pop(
        "objective_fields",
        _obj("action", "observation", "reward", "episode_done", "task_done"),
    )
    head_output = kwargs.pop("head_output", "action")
    tokenizer = TextTokenizer(
        input_fields=_tokenizer_fields(input_fields, head_output=head_output),
        format=format_str,
        group_prefix=group_prefix,
        tokenizer=hf_tok,
        image_processor=image_processor,
        objective_fields=objective_fields,
        grouping_field="grouping_id",
    )
    enc = TextEmbedder(
        hidden_dim=hidden_dim,
        embed_tokens=emb,
        **kwargs,
    )
    return tokenizer, enc


def test_text_tokenizer_positions_count_per_modality_within_step() -> None:
    tokenizer, _ = _text_pair()
    st = tokenizer(
        {"observation": 1, "action": 0, "reward": 0.5, "episode_done": 0, "task_done": 0, "grouping_id": 0}
    )
    # A text step is a single modality (``__text__``) emitted in several
    # runs; positions keep counting across the runs, 0..T-1.
    assert st.positions.tolist() == list(range(st.T))


def test_text_embedder_skip_omits_value_keeps_commas() -> None:
    tokenizer, enc = _text_pair()
    batch = [
        [
            {"observation": 1, "action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0},
            {"observation": 2, "action": 1, "reward": 1.0, "episode_done": 1, "task_done": 0},
        ]
    ]
    tb, obj = batch_to_packed(tokenizer, batch)
    embeds, indices = enc(tb)
    assert obj["reward"].dtype == torch.float32
    assert obj["action"].dtype == torch.int64
    assert obj["reward"].tolist() == [0.0, 1.0]
    assert embeds.ndim == 2 and embeds.shape[1] == 8
    # Skips shorten step 0 relative to step 1; head-output indices point at
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
    tokenizer2, enc2 = _text_pair(tokenizer=_CaptureTok(), embed_tokens=emb)
    out, _ = enc2(
        batch_to_token_batch(tokenizer2, [[{"observation": 1, "action": 0, "reward": 0.0, "episode_done": 0, "task_done": 0}]])
    )
    assert seen == ["<action=", ",observation=1,,>"]
    matches = (out == 7.0).all(dim=-1)
    assert int(matches.sum().item()) == 1


def test_text_tokenizer_appends_learnable() -> None:
    tokenizer, enc = _text_pair(
        format="{action}",
        input_fields=[
            {"type": "token", "input_field": "action"},
            {"type": "learnable", "output_field": "value", "tokens": 1, "head_output": True},
        ],
        objective_fields=_obj("action"),
        head_output="value",
        learnable=[
            {"type": "learnable", "field": "value", "tokens": 1, "std": 0.02, "positions": 1},
        ],
    )
    st = tokenizer({"action": 3, "grouping_id": 0})
    assert st.modality_names[-1] == "value"
    assert st.head_output_mask.tolist() == [False, True]
    embeds, indices = enc(batch_to_token_batch(tokenizer, [[{"action": 3}]]))
    assert embeds.shape[0] == 2
    assert indices.tolist() == [1]


def test_text_tokenizer_learnable_rejects_input_field() -> None:
    import pytest

    with pytest.raises(TypeError, match="no input_field"):
        TextTokenizer(
            input_fields=[
                {
                    "type": "learnable",
                    "input_field": "",
                    "head_output": True,
                },
            ],
            grouping_field="grouping_id",
        )


def test_text_tokenizer_requires_field_format() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires format="):
        TextTokenizer(
            input_fields=[
                {"type": "text", "input_field": "observation", "head_output": True},
            ],
            format="{observation}",
            tokenizer=_FakeTokenizer(),
            grouping_field="grouping_id",
        )


def test_token_modality_is_single_embed_row() -> None:
    D = 8
    emb = nn.Embedding(32, D)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[16] = 3.0
    tokenizer, enc = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        format="{action}",
        input_fields=[{"type": "token", "input_field": "action"}],
        objective_fields=_obj("action"),
    )
    embeds, indices = enc(batch_to_token_batch(tokenizer, [[{"action": 16}]]))
    assert embeds.shape[0] == 1
    assert int(indices[0].item()) == 0
    assert torch.equal(embeds[0], emb.weight[16])


def test_text_tokenizer_field_format_in_step_template() -> None:
    seen: list[str] = []

    class _CaptureTok:

        def __call__(
            self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None
        ):
            seen.append(text)
            ids = [ord(c) % 20 + 1 for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    emb = nn.Embedding(32, 8)
    tokenizer, enc = _text_pair(
        tokenizer=_CaptureTok(),
        embed_tokens=emb,
        format="<{observation}|{action}>",
        input_fields=[
            {"type": "text", "input_field": "observation", "format": "o={observation}"},
            {"type": "text", "input_field": "action", "format": "a={action}"},
        ],
        objective_fields=_obj("action"),
    )
    enc(batch_to_token_batch(tokenizer, [[{"observation": 3, "action": 2}]]))
    # The flagged head-output field ("action") is tokenized as its own run so
    # its token boundaries are exact; the other fields stay merged.
    assert seen == ["<o=3|", "a=2", ">"]


def test_text_embedder_image_token_ids() -> None:
    D = 8

    def fake_image_tok(image):
        return [3, 4]

    emb = nn.Embedding(32, D)
    tokenizer, enc = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        image_processor=fake_image_tok,
        format="<{observation},{pixels}>",
        input_fields=[
            {"type": "text", "input_field": "observation", "format": "{observation}"},
            {"type": "image", "input_field": "pixels"},
        ],
        objective_fields=_obj("observation", "pixels"),
        head_output="pixels",
    )
    batch = [[{"observation": 3, "pixels": [1, 2, 3]}]]
    tb, obj = batch_to_packed(tokenizer, batch)
    embeds, indices = enc(tb)
    assert "pixels" in obj.keys()
    assert embeds.ndim == 2 and embeds.shape[1] == D
    # "<3," (3 chars → 3 tokens), then the two image tokens (the head-output
    # tokens), then the trailing ">" run.
    assert indices.tolist() == [3, 4]
    assert embeds.shape[0] == 6


def test_text_embedder_save_load(tmp_path) -> None:
    D = 8
    emb = nn.Embedding(32, D)
    nn.init.normal_(emb.weight, std=0.02)
    _, enc = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        pretrained=None,
        format="<action={action}>",
        input_fields=[{"type": "token", "input_field": "action"}],
    )
    enc.pretrained = None
    model = Model(
        encoder=enc,
        backbone=IdentityBackbone(hidden_dim=D),
        heads=DiscreteActionValueHead(
            in_features=D, out_features=4, hidden_dim=D, num_layers=1
        ),
    )
    from mouse_core.models.base import _encoder_config

    cfg = _encoder_config(enc)
    assert cfg["type"] == "text"
    assert cfg["kwargs"]["vocab_size"] == 32
    assert "format" not in cfg["kwargs"]
    assert "modalities" not in cfg["kwargs"]
    assert "std" not in cfg["kwargs"]
    assert "separator" not in cfg["kwargs"]

    from mouse_core.models import load_model, save_model

    tokenizer, _ = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        format="<action={action}>",
        input_fields=[{"type": "token", "input_field": "action"}],
        objective_fields=_obj("action"),
    )
    batch = [[{"action": 1}, {"action": 3}]]
    model.eval()
    expected = model(batch_to_token_batch(tokenizer, batch)).predictions
    save_model(model, tmp_path)
    loaded = load_model(tmp_path, train_kernel="varlen", decode_kernel="flex", dtype=torch.float32).eval()
    assert isinstance(loaded.encoder, TextEmbedder)
    assert loaded.encoder.vocab_size == 32
    assert torch.equal(loaded.encoder.embed_tokens.weight, emb.weight)
    actual = loaded(batch_to_token_batch(tokenizer, batch)).predictions
    assert torch.allclose(actual["action_value"], expected["action_value"])


def test_text_embedder_learnable_save_load(tmp_path) -> None:
    D = 8
    emb = nn.Embedding(32, D)
    nn.init.normal_(emb.weight, std=0.02)
    tokenizer, enc = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        format="{action}",
        input_fields=[
            {"type": "token", "input_field": "action"},
            {"type": "learnable", "output_field": "value", "tokens": 1, "head_output": True},
        ],
        objective_fields=_obj("action"),
        head_output="value",
        learnable=[
            {"type": "learnable", "field": "value", "tokens": 1, "std": 0.02, "positions": 1},
        ],
    )
    from mouse_core.models.base import _encoder_config

    cfg = _encoder_config(enc)
    assert cfg["kwargs"]["learnable"][0]["field"] == "value"
    model = Model(
        encoder=enc,
        backbone=IdentityBackbone(hidden_dim=D),
        heads=DiscreteActionValueHead(
            in_features=D, out_features=4, hidden_dim=D, num_layers=1
        ),
    ).eval()
    batch = [[{"action": 1}, {"action": 3}]]
    expected = model(batch_to_token_batch(tokenizer, batch)).predictions
    from mouse_core.models import load_model, save_model

    save_model(model, tmp_path)
    loaded = load_model(tmp_path, train_kernel="varlen", decode_kernel="flex", dtype=torch.float32).eval()
    assert isinstance(loaded.encoder, TextEmbedder)
    assert len(loaded.encoder.learnable) == 1
    actual = loaded(batch_to_token_batch(tokenizer, batch)).predictions
    assert torch.allclose(actual["action_value"], expected["action_value"])


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

    enc = TextEmbedder(hidden_dim=8, vocab_size=16)
    enc.pretrained = "Qwen/Qwen3-0.6B"
    model = Model(
        encoder=enc,
        backbone=IdentityBackbone(hidden_dim=8),
        heads=DiscreteActionValueHead(
            in_features=8, out_features=4, hidden_dim=8, num_layers=1
        ),
    )
    path = tmp_path / "README.md"
    _write_model_card(repo_id="user/mouse-text", model=model, path=path)
    text = path.read_text()
    assert "TextEmbedder" in text
    assert "TextTokenizer" in text
    assert "NumericTokenizer" not in text
    assert 'pretrained="Qwen/Qwen3-0.6B"' in text
    assert '{"input_field": "action"}' in text


def test_text_embedder_requires_exactly_one_table_source() -> None:
    import pytest

    with pytest.raises(TypeError, match="exactly one of"):
        TextEmbedder(hidden_dim=8)
    with pytest.raises(TypeError, match="exactly one of"):
        TextEmbedder(
            hidden_dim=8,
            embed_tokens=nn.Embedding(4, 8),
            vocab_size=4,
        )
    enc = TextEmbedder(hidden_dim=8, vocab_size=4)
    assert enc.embed_tokens.weight.shape == (4, 8)


def _group_prefix_tokenizer(**kwargs):
    return TextTokenizer(
        input_fields=[
            {"type": "token", "input_field": "action", "head_output": True},
        ],
        format="{action}",
        group_prefix=kwargs.pop("group_prefix", "task={task_index}\n"),
        tokenizer=kwargs.pop("tokenizer", _FakeTokenizer()),
        objective_fields=_obj("action"),
        grouping_field="task_index",
        **kwargs,
    )


def test_text_tokenizer_group_prefix_carried_on_step() -> None:
    tok = _group_prefix_tokenizer()
    st = tok({"action": 1, "task_index": 7})
    assert st.group_prefix_ids is not None
    assert st.group_prefix_modality_ids is not None
    expected = _FakeTokenizer()("task=7\n")["input_ids"].view(-1).tolist()
    assert st.group_prefix_ids.tolist() == expected
    assert st.group_prefix_modality_ids.tolist() == [0] * len(expected)
    assert st.head_output_mask.tolist() == [True]


def test_pack_emits_group_prefix_once_per_grouping_segment() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_prefix_tokenizer()
    steps = [
        tok({"action": 1, "task_index": 0}),
        tok({"action": 2, "task_index": 0}),
        tok({"action": 3, "task_index": 1}),
    ]
    inputs, obj = pack_token_batch(steps, sequence_ids=[0, 0, 0], batch_size=1)
    prefix = steps[0].group_prefix_ids
    assert prefix is not None
    p = int(prefix.shape[0])
    assert inputs.L == p + steps[0].T + steps[1].T + p + steps[2].T
    assert obj["action"].tolist() == [1, 2, 3]
    assert inputs.head_output_indices.tolist() == [
        p + int(steps[0].head_output_mask.nonzero()[0][0]),
        p + steps[0].T + int(steps[1].head_output_mask.nonzero()[0][0]),
        p + steps[0].T + steps[1].T + p + int(steps[2].head_output_mask.nonzero()[0][0]),
    ]
    # Group-prefix tokens are __text__ and never head-output.
    assert not any(
        int(i) in set(inputs.head_output_indices.tolist())
        for i in range(p)
    )


def test_pack_group_prefix_is_per_sequence() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_prefix_tokenizer()
    steps = [
        tok({"action": 1, "task_index": 0}),
        tok({"action": 2, "task_index": 0}),
    ]
    inputs, _ = pack_token_batch(steps, sequence_ids=[0, 1], batch_size=2)
    prefix = steps[0].group_prefix_ids
    assert prefix is not None
    p = int(prefix.shape[0])
    assert inputs.L == (p + steps[0].T) + (p + steps[1].T)
    assert inputs.sequence_ids.tolist() == (
        [0] * (p + steps[0].T) + [1] * (p + steps[1].T)
    )


def test_pack_prev_grouping_ids_suppresses_and_reemits_group_prefix() -> None:
    from mouse_core.data import pack_token_batch

    tok = _group_prefix_tokenizer()
    continue_step = tok({"action": 1, "task_index": 5})
    change_step = tok({"action": 2, "task_index": 6})
    prefix = continue_step.group_prefix_ids
    assert prefix is not None
    p = int(prefix.shape[0])

    same, _ = pack_token_batch(
        [continue_step],
        sequence_ids=[0],
        batch_size=1,
        prev_grouping_ids=[5],
    )
    assert same.L == continue_step.T

    changed, _ = pack_token_batch(
        [change_step],
        sequence_ids=[0],
        batch_size=1,
        prev_grouping_ids=[5],
    )
    assert changed.L == p + change_step.T

    fresh, _ = pack_token_batch(
        [continue_step],
        sequence_ids=[0],
        batch_size=1,
        prev_grouping_ids=[None],
    )
    assert fresh.L == p + continue_step.T


def test_text_tokenizer_group_prefix_missing_placeholder_raises() -> None:
    import pytest

    tok = _group_prefix_tokenizer(group_prefix="label={label}\n")
    with pytest.raises(KeyError, match="label"):
        tok({"action": 1, "task_index": 0})


def test_text_tokenizer_empty_group_prefix_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        _group_prefix_tokenizer(group_prefix="")


def test_text_tokenizer_group_prefix_without_text_fields_adds_text_modality() -> None:
    tok = TextTokenizer(
        input_fields=[
            {"type": "learnable", "output_field": "value", "tokens": 1, "head_output": True},
        ],
        group_prefix="task={task_index}\n",
        tokenizer=_FakeTokenizer(),
        objective_fields=[],
        grouping_field="task_index",
    )
    assert "__text__" in tok.modality_names
    st = tok({"task_index": 3})
    assert st.group_prefix_ids is not None
    assert st.group_prefix_modality_ids is not None
    assert st.T == 1
    assert st.modality_names[int(st.group_prefix_modality_ids[0])] == "__text__"


def test_numeric_pack_ignores_missing_group_prefix() -> None:
    from mouse_core.data import NumericTokenizer, pack_token_batch

    tok = NumericTokenizer(
        input_fields=[{"type": "discrete", "input_field": "action", "head_output": True}],
        objective_fields=_obj("action"),
        grouping_field="task_index",
    )
    steps = [
        tok({"action": 1, "task_index": 0}),
        tok({"action": 2, "task_index": 0}),
    ]
    inputs, _ = pack_token_batch(steps, sequence_ids=[0, 0], batch_size=1)
    assert inputs.L == steps[0].T + steps[1].T
    assert steps[0].group_prefix_ids is None
