"""One attention-window convention for training masks and cached inference."""
from __future__ import annotations

from typing import Any


def attention_window_size(config: Any) -> int:
    """Number of visible tokens INCLUDING self; every layer must be windowed.

    Core exposes the maximum *past* distance in tokens. HF's sliding allocator
    counts self too, so its window size is distance + 1. Mixed per-layer policies
    remain unsupported: every Core layer must implement the same mask.
    """
    types = getattr(config, "layer_types", None) or []
    sliding = getattr(config, "use_sliding_window", False) or "sliding_attention" in types
    expected = {"sliding_attention"}
    if any(t not in expected for t in types):
        raise ValueError("Core requires the same sliding attention mask in every layer.")
    if not sliding:
        raise ValueError("Core requires a finite sliding attention window.")
    size = getattr(config, "sliding_window", None)
    if type(size) is not int or size < 1:
        raise ValueError("A sliding attention mask requires a positive integer sliding_window.")
    return size


def configure_attention_window(model: Any, max_distance: int) -> None:
    """Apply a caller-selected mask to a supported, uniformly causal decoder."""
    if type(max_distance) is not int or max_distance < 0:
        raise ValueError("max_distance must be a nonnegative integer.")
    size = max_distance + 1
    layer_type = "sliding_attention"
    model.config.use_sliding_window = True
    model.config.sliding_window = size
    model.config.layer_types = [layer_type] * len(model.layers)
    if hasattr(model, "has_sliding_layers"):
        model.has_sliding_layers = True
    for layer in model.layers:
        layer.attention_type = layer_type
        layer.self_attn.sliding_window = size
