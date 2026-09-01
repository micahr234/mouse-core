from __future__ import annotations

import pytest
import torch

from mouse_core.data import NumericTokenizer
from mouse_core.models.embedding import NumericEmbedder
from tests._token_batch_helpers import batch_to_packed, batch_to_token_batch, tok_from_encoder

_tok = tok_from_encoder


def _enc(**kwargs) -> NumericEmbedder:
    return NumericEmbedder(**kwargs)


def _batch(rows: list[dict], S: int = 1) -> list[list[dict]]:
    assert len(rows) == S
    return [rows]


def _tb(encoder, batch):
    return batch_to_token_batch(_tok(encoder), batch)


def test_numeric_embedder_ignores_is_seam_in_row_dicts() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}])
    with_seam = [[{"action": 0, "is_seam": 0}, {"action": 1, "is_seam": 1}]]
    without_seam = [[{"action": 0}, {"action": 1}]]
    embeds, _ = encoder(_tb(encoder, with_seam))
    plain_embeds, _ = encoder(_tb(encoder, without_seam))
    _, with_obj = batch_to_packed(_tok(encoder), with_seam)
    _, without_obj = batch_to_packed(_tok(encoder), without_seam)
    assert "is_seam" not in with_obj.keys()
    assert "is_seam" not in without_obj.keys()
    assert torch.equal(embeds, plain_embeds)


def test_numeric_embedder_faults_on_missing_required_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4},
            {"type": 'fourier', "field": "reward"},
        ],
    )
    batch = _batch([{"reward": 0.5}])
    with pytest.raises(KeyError, match="Required input field 'action' is missing"):
        _tb(encoder, batch)


def test_numeric_embedder_keeps_optional_missing_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4},
            {"type": "fourier", "field": "reward"},
        ],
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            {
                "type": "discrete",
                "input_field": "action",
                "output_field": "action",
                "required": False,
            },
            {"type": "fourier", "input_field": "reward", "output_field": "reward"},
        ],
        grouping_field="grouping_id",
    )
    batch = _batch([{"reward": 0.5}])
    embeds, prediction_indices = encoder(batch_to_token_batch(tokenizer, batch))
    assert embeds.shape == (1, 8)
    assert prediction_indices.shape == (1,)


def test_numeric_embedder_returns_objective_fields() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4},
            {"type": 'fourier', "field": "reward"},
        ],
    )
    batch = _batch([{"action": 2, "reward": 1.5}])
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, prediction_indices = encoder(tb)
    assert embeds.shape == (2, 8)
    assert prediction_indices.tolist() == [1]
    assert obj["action"].item() == 2
    assert obj["reward"].item() == pytest.approx(1.5)


def test_numeric_embedder_expands_multi_field_modality_specs() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": ("action", "prev_action"), "vocab_size": 4},
            {"type": "fourier", "field": ("reward", "value")},
        ],
    )
    batch = _batch(
        [{"action": 2, "prev_action": 1, "reward": 1.5, "value": 0.25}]
    )
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, _ = encoder(tb)
    assert embeds.shape == (4, 8)
    assert [spec.field for spec in encoder.modalities] == [
        "action",
        "prev_action",
        "reward",
        "value",
    ]
    assert obj["action"].item() == 2
    assert obj["prev_action"].item() == 1


def test_numeric_embedder_batch_shape() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4},
            {"type": 'fourier', "field": "reward"},
        ],
    )
    B, S = (3, 5)
    batch = [
        [{"action": (b * S + s) % 4, "reward": float(b * S + s)} for s in range(S)]
        for b in range(B)
    ]
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, prediction_indices = encoder(tb)
    assert embeds.shape == (B * S * 2, 8)
    assert obj["action"].shape == (B * S,)
    assert obj["reward"].shape == (B * S,)
    assert prediction_indices.shape == (B * S,)


def test_numeric_embedder_concat_tokens_in_order() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4},
            {"type": 'fourier', "field": "reward"},
            {"type": "learnable", "tokens": 1},
        ],
    )
    embeds, prediction_indices = encoder(
        _tb(encoder, _batch([{"action": 2, "reward": 1.5}]))
    )
    assert embeds.shape == (3, 8)
    assert prediction_indices.tolist() == [2]


def test_numeric_embedder_rejects_unknown_constructor_kwargs() -> None:
    with pytest.raises(TypeError):
        _enc(
            hidden_dim=8,
            modality_fusion="sum",
            modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}],
        )
    with pytest.raises(TypeError):
        NumericEmbedder(
            hidden_dim=8,
            modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}],
            include_type_token=True,
        )


def test_numeric_embedder_learnable_modality_is_allowed() -> None:
    encoder = _enc(hidden_dim=8, modalities=[{"type": "learnable", "tokens": 1}])
    tb, obj = batch_to_packed(_tok(encoder), [[{}]])
    embeds, _ = encoder(tb)
    assert embeds.shape == (1, 8)
    assert "scratch" not in obj.keys()


def test_numeric_embedder_continuous_one_token_per_scalar() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'continuous', "field": "obs", "dim": 4},
            {"type": 'fourier', "field": "reward"},
        ],
    )
    batch = [[{"obs": [0.1, 0.2, 0.3, 0.4], "reward": 1.0}]]
    tb, obj = batch_to_packed(_tok(encoder), batch)
    embeds, prediction_indices = encoder(tb)
    assert embeds.shape == (5, 8)
    assert obj["obs"].shape == (1, 4)
    assert prediction_indices.tolist() == [4]


def test_numeric_embedder_skip_shortens_step() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4},
            {"type": "fourier", "field": "reward"},
            {"type": "learnable", "tokens": 1},
        ],
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action"},
            {"type": "fourier", "input_field": "reward", "output_field": "reward", "skip": 0.0},
            {"type": "learnable", "tokens": 1},
        ],
        objective_fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "reward", "output_field": "reward"},
        ],
        grouping_field="grouping_id",
    )
    batch = [[{"action": 1, "reward": 0.0}, {"action": 2, "reward": 1.5}]]
    tb, obj = batch_to_packed(tokenizer, batch)
    embeds, indices = encoder(tb)
    assert obj["reward"].tolist() == [0.0, 1.5]
    assert embeds.shape == (5, 8)
    assert indices.tolist() == [1, 4]


def test_numeric_tokenizer_image_requires_callable() -> None:
    with pytest.raises(TypeError, match="image_tokenizer"):
        NumericTokenizer(
            input_fields=[
                {"type": "image", "input_field": "img", "output_field": "img"}
            ],
            grouping_field="grouping_id",
        )
    enc = _enc(hidden_dim=8, modalities=[{"type": 'image', "field": "img", "vocab_size": 32}])
    assert enc.tokens_per_step >= 1


def test_numeric_embedder_prepare_token_batch() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": 'discrete', "field": "action", "vocab_size": 4},
            {"type": 'fourier', "field": "reward"},
        ],
    )
    batch = [[{"action": 1, "reward": 0.5}, {"action": 2, "reward": 1.0}]]
    tb = _tb(encoder, batch)
    assert tb.B == 1 and int(tb.step_counts()[0]) == 2
    assert tb.L == 4
    assert list(tb.sequence_ids) == [0, 0, 0, 0]
    embeds, prediction_indices = encoder(tb)
    assert embeds.shape == (4, 8)
    assert prediction_indices.shape == (2,)


def test_numeric_embedder_fourier_honors_per_modality_std() -> None:
    """Fourier ``std=`` must scale that field's embeddings, not only the global default."""
    encoder = _enc(
        hidden_dim=64,
        std=0.02,
        modalities=[
            {"type": "fourier", "field": "reward", "std": 0.02},
            {"type": "fourier", "field": "bonus", "std": 0.10},
        ],
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "fourier", "input_field": "reward"},
            {"type": "fourier", "input_field": "bonus"},
        ],
        grouping_field="grouping_id",
    )
    rewards = [float(x) for x in range(-4, 5)]
    batch = [[{"reward": r, "bonus": r} for r in rewards]]
    tb = batch_to_token_batch(tokenizer, batch)
    embeds, _ = encoder(tb)
    names = tokenizer.modality_names
    reward_emb = embeds[torch.from_numpy(tb.modality_ids == names.index("reward"))]
    bonus_emb = embeds[torch.from_numpy(tb.modality_ids == names.index("bonus"))]
    reward_rms = float(reward_emb.pow(2).mean().sqrt().item())
    bonus_rms = float(bonus_emb.pow(2).mean().sqrt().item())
    assert reward_rms == pytest.approx(0.02, abs=0.008)
    assert bonus_rms == pytest.approx(0.10, abs=0.03)


def test_static_fourier_no_parameters() -> None:
    from mouse_core.models.embedding import StaticFourierFeatures

    ff = StaticFourierFeatures(num_features=8, in_min=0.01, in_max=10.0)
    assert sum((p.numel() for p in ff.parameters())) == 0
    y = ff(torch.tensor([0.5, -0.5]))
    assert y.shape == (2, 8)


def test_numeric_embedder_extra_fields_in_objective_fields() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[{"type": 'discrete', "field": "action", "vocab_size": 4}],
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action"}
        ],
        objective_fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "old_log_prob", "output_field": "old_log_prob"},
        ],
        grouping_field="grouping_id",
    )
    batch = [
        [{"action": 1, "old_log_prob": 0.25}, {"action": 2, "old_log_prob": -1.5}]
    ]
    tb, obj = batch_to_packed(tokenizer, batch)
    assert tb.L == 2
    assert obj["old_log_prob"].tolist() == pytest.approx([0.25, -1.5])
    embeds, _ = encoder(tb)
    assert embeds.shape == (2, 8)


def test_task_done_is_objective_field_not_input_field() -> None:
    """task_done is an objective column; it is not a transformer input token."""
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "discrete", "field": "action", "vocab_size": 4},
            {"type": "discrete", "field": "episode_done", "vocab_size": 3},
        ],
    )
    tokenizer = NumericTokenizer(
        input_fields=[
            {"type": "discrete", "input_field": "action", "output_field": "action"},
            {
                "type": "discrete",
                "input_field": "episode_done",
                "output_field": "episode_done",
            },
        ],
        objective_fields=[
            {"input_field": "action", "output_field": "action"},
            {"input_field": "episode_done", "output_field": "episode_done"},
            {"input_field": "task_done", "output_field": "task_done"},
        ],
        grouping_field="grouping_id",
    )
    batch = [[{"action": 1, "episode_done": 0, "task_done": 2}]]
    tb, obj = batch_to_packed(tokenizer, batch)
    assert "task_done" not in tb.modality_names
    assert tb.L == 2
    assert obj["task_done"].tolist() == [2]
    embeds, _ = encoder(tb)
    assert embeds.shape == (2, 8)
