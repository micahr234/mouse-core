"""Backbone interface for MOUSE models.

A backbone is a sequence processor that takes token embeddings and returns
hidden states of the same shape. It may support KV-caching for incremental
decoding.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
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
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run the backbone over a token sequence.

        Args:
            embeds: Token embeddings ``[B, T, D]``.
            cache: Opaque cache dict from a prior call. Backbones that support
                KV-caching should read/write under a conventional key (e.g.
                ``"backbone"``) and return an updated dict when ``use_cache``
                is True. Token positions for incremental decoding are inferred
                from the cached sequence length.
            use_cache: If True, return an updated cache for incremental use.
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


@contextmanager
def _quiet_transformers_load():
    from transformers import logging as transformers_logging

    verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(verbosity)


def _load_transformer_weights(
    model: nn.Module,
    repo_id_or_path: str | Path,
    *,
    hub_kwargs: dict[str, Any],
) -> None:
    """Load matching transformer weights into a MOUSE backbone internals.

    MOUSE backbones replace token embeddings with ``StepEmbedder`` and replace
    the final norm with ``Identity``, so those pretrained keys are skipped.

    Warns with the names of any other backbone tensors that did not receive
    pretrained weights (missing from the checkpoint or shape-mismatched), so a
    config/checkpoint mismatch cannot silently leave the model random-init.
    """
    from transformers import AutoModel

    with _quiet_transformers_load():
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

    intentionally_skipped = ("embed_tokens",)
    not_loaded = [
        key
        for key in target_state
        if key not in loadable and not key.startswith(intentionally_skipped)
    ]
    if not_loaded:
        warnings.warn(
            f"{len(not_loaded)} of {len(target_state)} backbone tensors did not "
            f"receive pretrained weights from {str(repo_id_or_path)!r} and keep "
            f"their random initialization: {not_loaded}",
            stacklevel=2,
        )

    model.load_state_dict(loadable, strict=False)

    del pretrained
