"""Backbone interface for MOUSE models.

A backbone is a sequence processor that takes token embeddings and returns
hidden states of the same shape. Full (uncached) forwards go through
:meth:`Backbone.forward`; incremental decoding goes through a
:class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession` created by
:meth:`Backbone.decode_session`.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from mouse_core.models.backbone.flex_decode import FlexDecodeSession


class Backbone(nn.Module, ABC):
    """Abstract base for backbones.

    A backbone consumes a token embedding sequence ``[B, T, D]`` and returns
    processed hidden states of shape ``[B, T, D]``.

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
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run a full (uncached) forward over a token sequence.

        Args:
            embeds: Token embeddings ``[B, T, D]``.
            output_hidden_states: Also return every layer's hidden states
                (for layerwise heads).
            **kwargs: Implementation-specific options.

        Returns:
            Hidden states ``[B, T, D]``, or ``(hidden_states, layer_hiddens)``
            when ``output_hidden_states=True``.
        """
        ...

    def decode_session(self, batch_size: int, capacity: int) -> FlexDecodeSession:
        """Create a cached-decode session over ``batch_size`` sequences.

        ``Model.forward`` calls this on the first ``use_cache=True`` call and
        carries the session inside its cache dict. Requires the backbone to
        expose a ``transformers`` decoder stack as ``self.model``.
        """
        model = getattr(self, "model", None)
        if model is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support cached decoding."
            )
        return FlexDecodeSession(model, batch_size=batch_size, capacity=capacity)


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
    *,
    model: nn.Module,
    repo_id_or_path: str | Path,
    hub_kwargs: dict[str, Any],
) -> None:
    """Load matching transformer weights into a MOUSE backbone internals.

    MOUSE backbones replace token embeddings with a MOUSE encoder
    (:class:`~mouse_core.models.embedding.NumericEmbedder` or
    :class:`~mouse_core.models.embedding.TextEmbedder`) and replace
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
