from __future__ import annotations

"""Tests for TextEmbedder / TextTokenizer (fake tokenizer / embeddings, no Hub)."""

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


_DEFAULT_MODALITIES = [
    {"type": "token", "field": "action"},
    {"type": "text", "field": "observation", "format": "observation={observation}"},
    {"type": "text", "field": "reward", "format": "reward={reward}", "skip": 0.0},
    {"type": "text", "field": "episode_done", "format": "episode_done={episode_done}", "skip": 0},
]
_DEFAULT_FORMAT = "<action={action},{observation},{reward},{episode_done}>"


def _tokenizer_fields(modalities: list[dict]) -> list[dict]:
    out: list[dict] = []
    for modality in modalities:
        data = dict(modality)
        if "field" in data:
            data["input_field"] = data.pop("field")
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
    modalities = kwargs.pop("modalities", list(_DEFAULT_MODALITIES))
    image_processor = kwargs.pop("image_processor", None)
    objective_fields = kwargs.pop(
        "objective_fields",
        _obj("action", "observation", "reward", "episode_done", "task_done"),
    )
    tokenizer = TextTokenizer(
        input_fields=_tokenizer_fields(modalities),
        format=format_str,
        tokenizer=hf_tok,
        image_processor=image_processor,
        objective_fields=objective_fields,
        grouping_field="grouping_id",
    )
    embedder_modalities = [
        {k: v for k, v in m.items() if k not in ("skip", "required")}
        for m in modalities
    ]
    enc = TextEmbedder(
        hidden_dim=hidden_dim,
        embed_tokens=emb,
        format=format_str,
        modalities=embedder_modalities,
        **kwargs,
    )
    return tokenizer, enc


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
    c0 = int(indices[0].item()) + 1
    c1 = int(indices[1].item()) - int(indices[0].item())
    assert c0 < c1
    assert embeds.shape[0] == c0 + c1
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
        modalities=[{"type": 'token', "field": "action"}],
        objective_fields=_obj("action"),
    )
    embeds, indices = enc(batch_to_token_batch(tokenizer, [[{"action": 16}]]))
    assert embeds.shape[0] == 1
    assert int(indices[0].item()) == 0
    assert torch.equal(embeds[0], emb.weight[16])


def test_text_embedder_rejects_learnable() -> None:
    emb = nn.Embedding(32, 8)
    try:
        TextEmbedder(
            hidden_dim=8,
            embed_tokens=emb,
            format="{action}",
            modalities=[
                {"type": 'token', "field": "action"},
                {"type": "learnable"},
            ],
        )
    except ValueError as exc:
        assert "learnable" in str(exc).lower() or "NumericEmbedder" in str(exc)
    else:
        raise AssertionError("expected ValueError for learnable modality")


def test_text_embedder_field_format_in_step_template() -> None:
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
        modalities=[
            {"type": 'text', "field": "observation", "format": 'o={observation}'},
            {"type": 'text', "field": "action", "format": 'a={action}'},
        ],
        objective_fields=_obj("action"),
    )
    enc(batch_to_token_batch(tokenizer, [[{"observation": 3, "action": 2}]]))
    assert seen == ["<o=3|a=2>"]


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
        modalities=[
            {"type": 'text', "field": "observation", "format": '{observation}'},
            {"type": 'image', "field": "pixels"},
        ],
        objective_fields=_obj("observation", "pixels"),
    )
    batch = [[{"observation": 3, "pixels": [1, 2, 3]}]]
    tb, obj = batch_to_packed(tokenizer, batch)
    embeds, indices = enc(tb)
    assert "pixels" in obj.keys()
    assert embeds.ndim == 2 and embeds.shape[1] == D
    assert int(indices[0].item()) + 1 == embeds.shape[0]


def test_text_embedder_save_load(tmp_path) -> None:
    D = 8
    emb = nn.Embedding(32, D)
    nn.init.normal_(emb.weight, std=0.02)
    _, enc = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        pretrained=None,
        format="<action={action}>",
        modalities=[{"type": 'token', "field": "action"}],
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
    assert cfg["kwargs"]["format"] == "<action={action}>"
    assert cfg["kwargs"]["modalities"][0]["type"] == "token"
    assert cfg["kwargs"]["vocab_size"] == 32
    assert "std" not in cfg["kwargs"]
    assert "separator" not in cfg["kwargs"]

    from mouse_core.models import load_model, save_model

    tokenizer, _ = _text_pair(
        hidden_dim=D,
        embed_tokens=emb,
        format="<action={action}>",
        modalities=[{"type": 'token', "field": "action"}],
        objective_fields=_obj("action"),
    )
    batch = [[{"action": 1}, {"action": 3}]]
    model.eval()
    expected, _ = model(batch_to_token_batch(tokenizer, batch))
    save_model(model, tmp_path)
    loaded = load_model(tmp_path).eval()
    assert isinstance(loaded.encoder, TextEmbedder)
    assert loaded.encoder.vocab_size == 32
    assert torch.equal(loaded.encoder.embed_tokens.weight, emb.weight)
    actual, _ = loaded(batch_to_token_batch(tokenizer, batch))
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


def test_text_embedder_requires_exactly_one_table_source() -> None:
    import pytest

    with pytest.raises(TypeError, match="exactly one of"):
        TextEmbedder(hidden_dim=8, modalities=[{"type": "token", "field": "a"}], format="{a}")
    with pytest.raises(TypeError, match="exactly one of"):
        TextEmbedder(
            hidden_dim=8,
            modalities=[{"type": "token", "field": "a"}],
            format="{a}",
            embed_tokens=nn.Embedding(4, 8),
            vocab_size=4,
        )
    enc = TextEmbedder(
        hidden_dim=8, modalities=[{"type": "token", "field": "a"}], format="{a}", vocab_size=4
    )
    assert enc.embed_tokens.weight.shape == (4, 8)
