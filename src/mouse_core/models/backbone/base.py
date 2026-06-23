"""Backbone interface for MOUSE models.

A backbone is a sequence processor that takes token embeddings and returns
hidden states of the same shape. It may support KV-caching for incremental
decoding.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class Backbone(nn.Module, ABC):
    """Abstract base for backbones.

    A backbone consumes a token embedding sequence ``[B, T, D]`` and returns
    processed hidden states of shape ``[B, T, D]``. It may receive an optional
    ``attention_mask`` for positions that should be ignored (e.g. padding).

    Implementations may be:
    - a full transformer (Llama, Qwen3, …)
    - a state-space model
    - an identity (no-op) for ablations
    - any custom sequence processor

    The only contract is the calling convention below and the shape of the
    returned hidden states.
    """

    @abstractmethod
    def forward(
        self,
        embeds: torch.Tensor,
        cache: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run the backbone over a token sequence.

        Args:
            embeds: Token embeddings ``[B, T, D]``.
            cache: Opaque cache dict from a prior call. Backbones that support
                KV-caching should read/write under a conventional key (e.g.
                ``"backbone"``) and return an updated dict when ``use_cache``
                is True.
            use_cache: If True, return an updated cache for incremental use.
            cache_position: Optional position indices for incremental decoding.
            attention_mask: Optional mask of shape ``[B, T]`` (or broadcastable)
                where positions with 0 / False are ignored by attention.
                When None, the backbone attends to all provided tokens.
            **kwargs: Implementation-specific options (e.g. ``position_ids``).

        Returns:
            A tuple ``(hidden_states, cache)`` where:

            - ``hidden_states``: ``[B, T, D]``
            - ``cache``: updated cache dict, or ``None`` if ``use_cache=False``.
        """
        ...


def _load_transformer_weights(
    model: nn.Module,
    repo_id_or_path: str | Path,
    *,
    backbone_name: str,
    hub_kwargs: dict[str, Any],
) -> None:
    """Load matching transformer weights into a MOUSE backbone internals.

    MOUSE backbones replace token embeddings with ``StepEmbedder`` and replace
    the final norm with ``Identity``, so those pretrained keys are skipped.
    """
    from transformers import AutoModel

    pretrained = AutoModel.from_pretrained(repo_id_or_path, **hub_kwargs)
    target_state = model.state_dict()
    loadable = {
        key: value
        for key, value in pretrained.state_dict().items()
        if key in target_state
        and target_state[key].shape == value.shape
        and not key.startswith("embed_tokens")
        and key not in ("norm.weight", "norm.bias")
    }
    missing, unexpected = model.load_state_dict(loadable, strict=False)

    ignored_missing = {key for key in missing if key.startswith("embed_tokens") or key.startswith("norm.")}
    real_missing = [key for key in missing if key not in ignored_missing]
    if real_missing:
        warnings.warn(
            f"Loading {backbone_name} from {repo_id_or_path!r} left "
            f"{len(real_missing)} key(s) uninitialised: {real_missing}",
            stacklevel=3,
        )
    if unexpected:
        warnings.warn(
            f"Loading {backbone_name} from {repo_id_or_path!r} had "
            f"{len(unexpected)} unexpected key(s): {unexpected}",
            stacklevel=3,
        )

    del pretrained
