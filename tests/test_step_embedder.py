from __future__ import annotations

import pytest
import torch

from mouse_core.models.embedding import StepEmbedder


def _enc(**kwargs) -> StepEmbedder:
    """Convenience wrapper: all tests disable type tokens (no type_embedding_std needed)."""
    return StepEmbedder(include_type_token=False, **kwargs)


def _batch(rows: list[dict], S: int = 1) -> list[list[dict]]:
    """Wrap a flat list of S row-dicts into a B=1 batch."""
    assert len(rows) == S
    return [rows]


def test_step_embedder_faults_on_missing_required_modality() -> None:
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


def test_step_embedder_faults_on_partially_missing_required_modality() -> None:
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


def test_step_embedder_keeps_default_for_optional_missing_modality() -> None:
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


def test_step_embedder_returns_col_values() -> None:
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


def test_step_embedder_expands_multi_field_modality_specs() -> None:
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


def test_step_embedder_batch_shape() -> None:
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


def test_step_embedder_learnable_modality_is_allowed() -> None:
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


def test_step_embedder_continuous_modality() -> None:
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


def test_step_embedder_requires_type_embedding_std_when_type_token_enabled() -> None:
    with pytest.raises(ValueError, match="type_embedding_std is required"):
        StepEmbedder(
            hidden_dim=8,
            modalities=[{"field": "action", "type": "discrete", "vocab_size": 4}],
            include_type_token=True,
            # type_embedding_std intentionally omitted
        )


def test_step_embedder_type_token_on_with_explicit_std() -> None:
    encoder = StepEmbedder(
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
