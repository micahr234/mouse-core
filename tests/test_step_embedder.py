from __future__ import annotations

import pytest
import torch

from mouse_core.models.embedding import StepEmbedder


def _batch(rows: list[dict], S: int = 1) -> list[list[dict]]:
    """Wrap a flat list of S row-dicts into a B=1 batch."""
    assert len(rows) == S
    return [rows]


def test_step_embedder_faults_on_missing_required_modality() -> None:
    encoder = StepEmbedder(
        hidden_dim=8,
        modalities=[
            {"name": "action", "embed": "discrete", "vocab_size": 4},
            {"name": "reward", "embed": "rff"},
        ],
    )
    batch = _batch([{"reward": 0.5}])

    with pytest.raises(KeyError, match="Required modality 'action' is missing"):
        encoder(batch)


def test_step_embedder_keeps_default_for_optional_missing_modality() -> None:
    encoder = StepEmbedder(
        hidden_dim=8,
        modalities=[
            {"name": "action", "embed": "discrete", "vocab_size": 4, "required": False},
            {"name": "reward", "embed": "rff"},
        ],
    )
    batch = _batch([{"reward": 0.5}])

    embeds, _, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)


def test_step_embedder_returns_col_values() -> None:
    encoder = StepEmbedder(
        hidden_dim=8,
        modalities=[
            {"name": "action", "embed": "discrete", "vocab_size": 4},
            {"name": "reward", "embed": "rff"},
        ],
    )
    batch = _batch([{"action": 2, "reward": 1.5}])

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert col_values["action"].item() == 2
    assert col_values["reward"].item() == pytest.approx(1.5)


def test_step_embedder_batch_shape() -> None:
    encoder = StepEmbedder(
        hidden_dim=8,
        modalities=[
            {"name": "action", "embed": "discrete", "vocab_size": 4},
            {"name": "reward", "embed": "rff"},
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
    encoder = StepEmbedder(
        hidden_dim=8,
        modalities=[
            {"name": "scratch", "embed": "learnable", "tokens": 1},
        ],
    )
    batch = [[{}]]

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert "scratch" not in col_values


def test_step_embedder_continuous_modality() -> None:
    encoder = StepEmbedder(
        hidden_dim=8,
        modalities=[
            {"name": "obs", "embed": "continuous", "dim": 4},
            {"name": "reward", "embed": "rff"},
        ],
    )
    batch = [[{"obs": [0.1, 0.2, 0.3, 0.4], "reward": 1.0}]]

    embeds, col_values, _ = encoder(batch)

    assert embeds.shape == (1, 1, 8)
    assert col_values["obs"].shape == (1, 1, 4)
