from __future__ import annotations

import pytest
import torch

from mouse_core.models.embedding import NumericEmbedder


def _enc(**kwargs) -> NumericEmbedder:
    """Convenience wrapper: all tests disable type tokens (no type_embedding_std needed)."""
    return NumericEmbedder(include_type_token=False, **kwargs)


def _batch(rows: list[dict], S: int = 1) -> list[list[dict]]:
    """Wrap a flat list of S row-dicts into a B=1 batch."""
    assert len(rows) == S
    return [rows]


def test_numeric_embedder_ignores_is_seam_in_row_dicts() -> None:
    """Pack metadata must not live in row dicts; encoder leaves it alone."""
    encoder = _enc(
        hidden_dim=8,
        modalities=[{"field": "action", "type": "discrete", "vocab_size": 4}],
    )
    with_seam = [[
        {"action": 0, "is_seam": 0},
        {"action": 1, "is_seam": 1},
    ]]
    without_seam = [[{"action": 0}, {"action": 1}]]

    embeds, col_values, _ = encoder(with_seam)
    plain_embeds, plain_col_values, _ = encoder(without_seam)

    assert "is_seam" not in col_values
    assert "is_seam" not in plain_col_values
    assert torch.equal(embeds, plain_embeds)


def test_numeric_embedder_faults_on_missing_required_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
        ],
    )
    batch = _batch([{"reward": 0.5}])

    with pytest.raises(KeyError, match="Required modality 'action' is missing"):
        encoder(batch)


def test_numeric_embedder_faults_on_partially_missing_required_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
        ],
    )
    batch = _batch([
        {"action": 1, "reward": 0.0},
        {"reward": 0.5},
    ], S=2)

    with pytest.raises(KeyError, match="missing from 1 of 2 rows"):
        encoder(batch)


def test_numeric_embedder_keeps_default_for_optional_missing_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4, "required": False},
            {"field": "reward", "type": "rff"},
        ],
    )
    batch = _batch([{"reward": 0.5}])

    embeds, _, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)


def test_numeric_embedder_returns_col_values() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
        ],
    )
    batch = _batch([{"action": 2, "reward": 1.5}])

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert col_values["action"].item() == 2
    assert col_values["reward"].item() == pytest.approx(1.5)


def test_numeric_embedder_expands_multi_field_modality_specs() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": ("action", "prev_action"), "type": "discrete", "vocab_size": 4},
            {"field": ("reward", "value"), "type": "rff"},
        ],
    )
    batch = _batch([{"action": 2, "prev_action": 1, "reward": 1.5, "value": 0.25}])

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert [spec.field for spec in encoder.modalities] == ["action", "prev_action", "reward", "value"]
    assert col_values["action"].item() == 2
    assert col_values["prev_action"].item() == 1
    assert col_values["reward"].item() == pytest.approx(1.5)
    assert col_values["value"].item() == pytest.approx(0.25)


def test_numeric_embedder_batch_shape() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
        ],
    )
    B, S = 3, 5
    batch = [
        [{"action": (b * S + s) % 4, "reward": float(b * S + s)} for s in range(S)]
        for b in range(B)
    ]
    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (B, S, 8)
    assert col_values["action"].shape == (B, S)
    assert col_values["reward"].shape == (B, S)


def test_numeric_embedder_sum_fusion_uses_max_modality_tokens() -> None:
    encoder = _enc(
        hidden_dim=8,
        modality_fusion="sum",
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4, "tokens": 1},
            {"field": "reward", "type": "rff", "tokens": 2},
        ],
    )

    embeds, _, step_token_indices = encoder(_batch([{"action": 2, "reward": 1.5}]))

    assert encoder.tokens_per_step == 2
    assert embeds.shape == (1, 2, 8)
    assert step_token_indices.tolist() == [[1]]


def test_numeric_embedder_concat_fusion_uses_sum_modality_tokens() -> None:
    encoder = _enc(
        hidden_dim=8,
        modality_fusion="concat",
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4, "tokens": 1},
            {"field": "reward", "type": "rff", "tokens": 2},
        ],
    )

    embeds, _, step_token_indices = encoder(_batch([{"action": 2, "reward": 1.5}]))

    assert encoder.tokens_per_step == 3
    assert embeds.shape == (1, 3, 8)
    assert step_token_indices.tolist() == [[2]]


def test_numeric_embedder_rejects_unknown_modality_fusion() -> None:
    with pytest.raises(ValueError, match='modality_fusion must be either "sum" or "concat"'):
        _enc(
            hidden_dim=8,
            modality_fusion="average",  # type: ignore[arg-type]
            modalities=[{"field": "action", "type": "discrete", "vocab_size": 4}],
        )


def test_numeric_embedder_learnable_modality_is_allowed() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"type": "learnable", "tokens": 1},
        ],
    )
    batch = [[{}]]

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert "scratch" not in col_values


def test_numeric_embedder_continuous_modality() -> None:
    encoder = _enc(
        hidden_dim=8,
        modalities=[
            {"field": "obs", "type": "continuous", "dim": 4},
            {"field": "reward", "type": "rff"},
        ],
    )
    batch = [[{"obs": [0.1, 0.2, 0.3, 0.4], "reward": 1.0}]]

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert col_values["obs"].shape == (1, 1, 4)


def test_numeric_embedder_requires_type_embedding_std_when_type_token_enabled() -> None:
    with pytest.raises(ValueError, match="type_embedding_std is required"):
        NumericEmbedder(
            hidden_dim=8,
            modalities=[{"field": "action", "type": "discrete", "vocab_size": 4}],
            include_type_token=True,
            # type_embedding_std intentionally omitted
        )


def test_numeric_embedder_type_token_on_with_explicit_std() -> None:
    encoder = NumericEmbedder(
        hidden_dim=8,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
        ],
        include_type_token=True,
        type_embedding_std=0.02,
    )
    batch = _batch([{"action": 1, "reward": 0.5}])
    embeds, _, _ = encoder(batch)
    assert embeds.shape == (1, 1, 8)


def test_numeric_embedder_skip_shortens_concat_step() -> None:
    encoder = _enc(
        hidden_dim=8,
        modality_fusion="concat",
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4, "tokens": 1},
            {"field": "reward", "type": "rff", "tokens": 1, "skip": 0.0},
            {"type": "learnable", "tokens": 1},
        ],
    )
    # reward skipped → 2 tokens; reward present → 3 tokens
    batch = [[
        {"action": 1, "reward": 0.0},
        {"action": 2, "reward": 1.5},
    ]]
    embeds, col_values, indices = encoder(batch)
    assert col_values["reward"].tolist() == [[0.0, 1.5]]
    # step0: action+learnable (2); step1: action+reward+learnable (3) → L=5
    assert embeds.shape == (1, 5, 8)
    assert indices.tolist() == [[1, 4]]


def test_numeric_embedder_skip_sum_uses_present_only() -> None:
    encoder = _enc(
        hidden_dim=8,
        modality_fusion="sum",
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4, "tokens": 2},
            {"field": "reward", "type": "rff", "tokens": 1, "skip": 0.0},
        ],
    )
    batch = [[
        {"action": 1, "reward": 0.0},  # only action → 2 tokens
        {"action": 2, "reward": 1.0},  # both → max(2,1)=2 tokens
    ]]
    embeds, _, indices = encoder(batch)
    assert embeds.shape == (1, 4, 8)
    assert indices.tolist() == [[1, 3]]
