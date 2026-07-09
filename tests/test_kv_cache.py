"""Incremental KV-cache decoding must match the full-sequence forward pass.

The transformer backbones pass no explicit positions to the HF model; RoPE
positions for cached decoding are inferred from the cached sequence length.
These tests verify that inference is correct: feeding a sequence in chunks
through the cache yields the same per-step predictions as one full pass.
"""

from __future__ import annotations

import pytest
import torch

from mouse_core.models import Model
from mouse_core.models.backbone import LlamaBackbone, Qwen3Backbone
from mouse_core.models.embedding import StepEmbedder
from mouse_core.models.heads import DiscreteActionValueHead


def _tiny_model(backbone_cls) -> Model:
    hidden_dim = 16
    encoder = StepEmbedder(
        hidden_dim=hidden_dim,
        modalities=[
            {"field": "action", "type": "discrete", "vocab_size": 4},
            {"field": "reward", "type": "rff"},
            {"field": "done", "type": "discrete", "vocab_size": 5},
        ],
        include_type_token=False,
    )
    backbone = backbone_cls(hidden_dim=hidden_dim, num_layers=2, num_heads=2)
    head = DiscreteActionValueHead(
        in_features=hidden_dim,
        out_features=4,
        hidden_dim=hidden_dim,
        num_layers=1,
    )
    return Model(encoder=encoder, backbone=backbone, heads=head).eval()


def _steps(n: int) -> list[dict]:
    return [
        {"action": i % 4, "reward": float(i), "done": int(i % 7 == 6)}
        for i in range(n)
    ]


@pytest.mark.parametrize("backbone_cls", [Qwen3Backbone, LlamaBackbone])
def test_chunked_cached_forward_matches_full_forward(backbone_cls) -> None:
    torch.manual_seed(0)
    model = _tiny_model(backbone_cls)
    steps = _steps(6)

    with torch.no_grad():
        full, _, _ = model([steps])

        cache = None
        chunk_preds = []
        for lo, hi in ((0, 3), (3, 4), (4, 6)):
            preds, _, cache = model([steps[lo:hi]], cache=cache, use_cache=True)
            chunk_preds.append(preds["action_value"])
        incremental = torch.cat(chunk_preds, dim=1)

    assert incremental.shape == full["action_value"].shape
    assert torch.allclose(incremental, full["action_value"], atol=1e-5), (
        "cached incremental decode diverged from full forward — "
        "RoPE cache positions are not being inferred correctly"
    )


def test_step_by_step_cached_rollout_matches_full_forward() -> None:
    """One step at a time, as in the inference notebooks."""
    torch.manual_seed(1)
    model = _tiny_model(Qwen3Backbone)
    steps = _steps(5)

    with torch.no_grad():
        full, _, _ = model([steps])

        cache = None
        last_step_preds = []
        for step in steps:
            preds, _, cache = model([[step]], cache=cache, use_cache=True)
            last_step_preds.append(preds["action_value"][:, -1])
        incremental = torch.stack(last_step_preds, dim=1)

    assert torch.allclose(incremental, full["action_value"], atol=1e-5)
