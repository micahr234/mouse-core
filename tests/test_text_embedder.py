"""Tests for TextEmbedder (fake tokenizer / embeddings, no Hub)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mouse_core.models import Model
from mouse_core.models.backbone import IdentityBackbone
from mouse_core.models.embedding import TextEmbedder
from mouse_core.models.heads import DiscreteActionValueHead


class _FakeTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
        ids = [((ord(c) % 20) + 1) for c in text] or [1]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


_DEFAULT_MODALITIES = [
    {"field": "action", "type": "token"},
    {"field": "observation", "type": "text", "format": "observation={observation}"},
    {"field": "reward", "type": "text", "format": "reward={reward}", "skip": 0.0},
    {"field": "done", "type": "text", "format": "done={done}", "skip": 0},
]


def _text_encoder(hidden_dim: int = 8, **kwargs) -> TextEmbedder:
    vocab = 32
    emb = nn.Embedding(vocab, hidden_dim)
    nn.init.normal_(emb.weight, std=0.02)
    return TextEmbedder(
        hidden_dim=hidden_dim,
        tokenizer=_FakeTokenizer(),
        embed_tokens=emb,
        format=kwargs.pop("format", "<action={action},{observation},{reward},{done}>"),
        modalities=kwargs.pop("modalities", list(_DEFAULT_MODALITIES)),
        **kwargs,
    )


def test_text_embedder_skip_omits_value_keeps_commas() -> None:
    enc = _text_encoder()
    batch = [[
        {"observation": 1, "action": 0, "reward": 0.0, "done": 0},
        {"observation": 2, "action": 1, "reward": 1.0, "done": 1},
    ]]
    embeds, col_values, indices = enc(batch)
    assert col_values["reward"].dtype == torch.float32
    assert col_values["action"].dtype == torch.int64
    assert col_values["reward"].tolist() == [[0.0, 1.0]]
    assert embeds.ndim == 2 and embeds.shape[1] == 8
    c0 = int(indices[0, 0].item()) + 1
    c1 = int(indices[0, 1].item()) - int(indices[0, 0].item())
    assert c0 < c1
    assert embeds.shape[0] == c0 + c1

    seen: list[str] = []

    class _CaptureTok:
        def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
            seen.append(text)
            ids = [((ord(c) % 20) + 1) for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    emb = nn.Embedding(32, 8)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[0] = 7.0
    enc2 = TextEmbedder(
        hidden_dim=8,
        tokenizer=_CaptureTok(),
        embed_tokens=emb,
        format="<action={action},{observation},{reward},{done}>",
        modalities=list(_DEFAULT_MODALITIES),
    )
    out, _, _ = enc2([[{"observation": 1, "action": 0, "reward": 0.0, "done": 0}]])
    assert seen == ["<action=", ",observation=1,,>"]
    matches = (out == 7.0).all(dim=-1)
    assert int(matches.sum().item()) == 1


def test_token_modality_is_single_embed_row() -> None:
    D = 8
    emb = nn.Embedding(32, D)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[16] = 3.0

    enc = TextEmbedder(
        hidden_dim=D,
        tokenizer=_FakeTokenizer(),
        embed_tokens=emb,
        format="{action}",
        modalities=[{"field": "action", "type": "token"}],
    )
    embeds, _, indices = enc([[{"action": 16}]])
    assert embeds.shape[0] == 1
    assert int(indices[0, 0].item()) == 0
    assert torch.equal(embeds[0], emb.weight[16])


def test_text_embedder_rejects_learnable() -> None:
    emb = nn.Embedding(32, 8)
    try:
        TextEmbedder(
            hidden_dim=8,
            tokenizer=_FakeTokenizer(),
            embed_tokens=emb,
            format="{action}",
            modalities=[
                {"field": "action", "type": "token"},
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
        def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str | None = None):
            seen.append(text)
            ids = [((ord(c) % 20) + 1) for c in text] or [1]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    emb = nn.Embedding(32, 8)
    enc = TextEmbedder(
        hidden_dim=8,
        tokenizer=_CaptureTok(),
        embed_tokens=emb,
        format="<{observation}|{action}>",
        modalities=[
            {"field": "observation", "type": "text", "format": "o={observation}"},
            {"field": "action", "type": "text", "format": "a={action}"},
        ],
    )
    enc([[{"observation": 3, "action": 2}]])
    assert seen == ["<o=3|a=2>"]


def test_text_embedder_image_token_ids() -> None:
    D = 8

    def fake_image_tok(image):
        return [3, 4]  # discrete visual token ids

    emb = nn.Embedding(32, D)
    enc = TextEmbedder(
        hidden_dim=D,
        tokenizer=_FakeTokenizer(),
        embed_tokens=emb,
        image_processor=fake_image_tok,
        format="<{observation},{pixels}>",
        modalities=[
            {"field": "observation", "type": "text", "format": "{observation}"},
            {"field": "pixels", "type": "image"},
        ],
    )
    batch = [[{"observation": 3, "pixels": [1, 2, 3]}]]
    embeds, col_values, indices = enc(batch)
    assert "pixels" in col_values
    assert embeds.ndim == 2 and embeds.shape[1] == D
    assert int(indices[0, 0].item()) + 1 == embeds.shape[0]


def test_text_embedder_save_load(tmp_path) -> None:
    D = 8
    emb = nn.Embedding(32, D)
    nn.init.normal_(emb.weight, std=0.02)
    enc = TextEmbedder(
        hidden_dim=D,
        tokenizer=_FakeTokenizer(),
        embed_tokens=emb,
        pretrained=None,
        format="<action={action}>",
        modalities=[{"field": "action", "type": "token"}],
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
    assert "std" not in cfg["kwargs"]
    assert "separator" not in cfg["kwargs"]
    assert model is not None
