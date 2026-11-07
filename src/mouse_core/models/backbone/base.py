"""BaseBackbone interface for MOUSE models.

A backbone is a sequence processor that takes token embeddings and returns
hidden states of the same shape. Pass flat ``[L, D]`` embeddings with explicit
``sequence_ids`` for packed training, or rectangular ``[B, T, D]`` embeddings with
an attention mask. Incremental decoding passes a session created by
``decode_session`` as ``cache=`` together with per-row ``lengths``.

Two ways to train a backbone, both with every trainable parameter in fp32:

- full fine-tuning — no ``lora``; build it with ``dtype=torch.float32`` and
  the base weights train directly;
- fp32 LoRA on a frozen base — ``lora=LoRAConfig(...)``; the base weights
  are frozen and may be built in bf16 (``dtype=preferred_dtype(device=device)``),
  the LoRA adapters are the only trainable backbone parameters.

The base dtype is a constructor argument of the transformer backbones; the
model is placed with ``model.to(device)``, which moves and never casts.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

import torch
import torch.nn as nn

from mouse_core.models.backbone.flex_utils import (
    module_device_dtype,
)
from mouse_core.models.backbone.packed_train import (
    check_autocast_dtype,
)
from mouse_core.models.lora import LoRAConfig, apply_lora
from mouse_core.models.backbone.continuous_decode import ContinuousDecodeSession
from mouse_core.models.backbone.attention_window import attention_window_size


class BaseBackbone(nn.Module, ABC):
    """Abstract base for backbones.

    A backbone consumes plain embedding tensors and returns feature tensors
    of the same shape: flat ``[L, D]`` or rectangular ``[B, T, D]``.

    Implementations may be:
    - a full transformer (:class:`~mouse_core.models.backbone.transformer.Backbone`)
    - a state-space model
    - an identity (no-op) for ablations
    - any custom sequence processor

    The only contract is the calling convention below and the shape of the
    returned hidden states.

    ``lora`` is the backbone's :class:`~mouse_core.models.lora.LoRAConfig`
    (``None`` for a fully trainable fp32 backbone).

    ``gradient_checkpointing`` makes the packed training forward recompute
    each decoder layer in backward instead of storing its activations:
    activation memory drops by roughly the layer count for about 1.4x the
    step time. Set ``backbone.gradient_checkpointing = True`` before
    training; it is not saved with the model.

    Transformer backbones always use block-sparse FlexAttention for uncached
    forwards and paged FlexAttention for cached decode. Training requires CUDA;
    CPU supports inference only.

    - ``train_autocast_dtype`` / ``decode_autocast_dtype`` (default
      ``None``) declare mixed precision per path for a fp32 backbone: the
      packed training forward / cached decode run their matmuls in
      bf16/fp16 under ``torch.autocast`` while weights, norms, RoPE, and
      the residual stream stay fp32; decode also allocates its KV pool in
      the autocast dtype (half the fp32 size). ``None`` runs that path in
      the base dtype, so e.g. ``train_autocast_dtype=torch.bfloat16`` with
      ``decode_autocast_dtype=None`` is mixed-precision training with
      exact fp32 eval decode. Both forwards raise if called inside an
      ambient ``torch.autocast`` region — precision is declared here,
      never inferred.

    Like ``gradient_checkpointing``, autocast settings are execution choices
    for the current machine. Configure them on the model before loading weights.
    """

    lora: LoRAConfig | None = None
    gradient_checkpointing: bool = False
    uses_packed: bool = False
    train_autocast_dtype: torch.dtype | None = None
    decode_autocast_dtype: torch.dtype | None = None

    def _set_autocast(
        self,
        train_autocast_dtype: torch.dtype | None,
        decode_autocast_dtype: torch.dtype | None,
        dtype: torch.dtype,
    ) -> None:
        self.train_autocast_dtype = check_autocast_dtype(train_autocast_dtype, dtype)
        self.decode_autocast_dtype = check_autocast_dtype(decode_autocast_dtype, dtype)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move the backbone; the base dtype is fixed at construction (``dtype=``)."""
        _reject_dtype_cast(type(self).__name__, *args, **kwargs)
        return super().to(*args, **kwargs)

    def half(self) -> "BaseBackbone":
        raise TypeError(f"{type(self).__name__}.half() is not supported; pass dtype= when building it.")

    def bfloat16(self) -> "BaseBackbone":
        raise TypeError(f"{type(self).__name__}.bfloat16() is not supported; pass dtype= when building it.")

    def double(self) -> "BaseBackbone":
        raise TypeError(f"{type(self).__name__}.double() is not supported; pass dtype= when building it.")

    def float(self) -> "BaseBackbone":
        raise TypeError(f"{type(self).__name__}.float() is not supported; pass dtype= when building it.")

    def _attach_lora(self, model: nn.Module, lora: LoRAConfig | None) -> None:
        """Freeze ``model`` and attach fp32 LoRA adapters when ``lora`` is set.

        Without ``lora`` the backbone is left fully trainable. Call once the
        pretrained weights are loaded: wrapping renames every
        ``nn.Linear`` key to ``<name>.base.weight``.
        """
        self.lora = lora
        if lora is not None:
            apply_lora(model=model, config=lora)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the decoder base weights; the backbone casts embeds to it.

        Reads ``self.model`` when present and skips LoRA adapters.
        A parameterless backbone
        (:class:`~mouse_core.models.backbone.none.IdentityBackbone`) reports
        ``float32``.
        """
        stack = getattr(self, "model", None)
        target = stack if isinstance(stack, nn.Module) else self
        try:
            return module_device_dtype(target)[1]
        except ValueError:
            return torch.float32

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Width ``D`` of the hidden states this backbone returns."""

    def forward(
        self, embeds: torch.Tensor, *,
        sequence_ids: torch.Tensor | None = None,
        cache: ContinuousDecodeSession | None = None,
        lengths: list[int] | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Embeddings to features, with masks/group IDs and cache passed separately.

        Flat ``[L, D]`` input requires ``sequence_ids`` and uses the configured
        packed train kernel. Rectangular input uses the supplied attention
        mask, or a decode session together with per-row ``lengths``.
        """
        if not isinstance(embeds, torch.Tensor):
            raise TypeError("backbone input must be an embedding tensor.")
        if cache is not None:
            if sequence_ids is not None or kwargs or lengths is None:
                raise ValueError("cached forward requires lengths and cannot take sequence_ids or attention kwargs.")
            return cache.forward(embeds=embeds, lengths=lengths, output_hidden_states=output_hidden_states)
        if lengths is not None:
            raise ValueError("lengths requires a decode cache.")
        embeds = embeds.to(dtype=self.dtype)
        if sequence_ids is None:
            if embeds.ndim != 3:
                raise ValueError("flat embedding tensors require sequence_ids.")
            return self._forward_dense(embeds, output_hidden_states=output_hidden_states, **kwargs)
        if embeds.ndim != 2 or sequence_ids.ndim != 1 or sequence_ids.shape[0] != embeds.shape[0]:
            raise ValueError("packed input requires embeds [L, D] and sequence_ids [L].")
        if kwargs:
            raise ValueError("sequence_ids supplies packed attention; do not also pass attention kwargs.")
        return self._forward_packed(embeds, sequence_ids=sequence_ids, output_hidden_states=output_hidden_states)

    def _forward_packed(
        self, embeds: torch.Tensor, *, sequence_ids: torch.Tensor, output_hidden_states: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.uses_packed:
            raise NotImplementedError("This backbone must implement _forward_packed for grouped inputs")
        from mouse_core.models.backbone.packed_train import packed_forward
        return packed_forward(
            model=cast(nn.Module, self.model), embeds=embeds, sequence_ids=sequence_ids,
            output_hidden_states=output_hidden_states, checkpoint=self.gradient_checkpointing,
            autocast_dtype=self.train_autocast_dtype,
        )

    @abstractmethod
    def _forward_dense(
        self, embeds: torch.Tensor, *, output_hidden_states: bool, **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Implement a rectangular uncached forward over embedding tensors."""
        ...

    def decode_session(
        self, batch_size: int, *, max_batch_tokens: int = 1024,
        cache_capacity_tokens: int | None = None,
    ) -> ContinuousDecodeSession:
        """HF continuous batching with a fixed circular KV pool and FlexAttention.

        By default reserve one full attention window per row (rounded to
        16-token pages). An explicit shared capacity may be smaller for sparse
        workloads; insufficient capacity raises instead of shortening history.
        ``max_batch_tokens`` bounds queries per scheduled decoder forward.
        Use ``forward`` for a batch or ``submit``/``step`` for independently
        ready rows. Reset a row only at its outer-episode boundary.
        """
        model = getattr(self, "model", None)
        if model is None:
            raise NotImplementedError(f"{type(self).__name__} does not support cached decoding.")
        window_size = attention_window_size(model.config)
        if window_size is None:
            raise ValueError("Core requires a finite attention window.")
        if cache_capacity_tokens is None:
            cache_capacity_tokens = batch_size * ((window_size + 15) // 16) * 16
        return ContinuousDecodeSession(
            model, batch_size=batch_size, max_batch_tokens=max_batch_tokens,
            cache_capacity_tokens=cache_capacity_tokens, autocast_dtype=self.decode_autocast_dtype,
        )


def _reject_dtype_cast(what: str, *args: Any, **kwargs: Any) -> None:
    """``.to()`` on a MOUSE module moves devices only; dtype is a backbone constructor argument."""
    _device, dtype, _non_blocking, _memory_format = cast(Any, torch._C)._nn._parse_to(*args, **kwargs)
    if dtype is not None:
        raise TypeError(
            f"{what}.to() does not cast dtypes. The backbone dtype is fixed when it is built "
            f"(e.g. Backbone(dtype=preferred_dtype(device=device), ...)); use .to(device) to move."
        )


@contextmanager
def _quiet_transformers_load():
    from transformers import logging as transformers_logging

    verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        yield
    finally:
        transformers_logging.set_verbosity(verbosity)


def _rope_parameters_from_config(hf_cfg: Any) -> dict[str, Any] | None:
    """Plain ``rope_parameters`` dict from a HuggingFace config, or ``None``.

    Pretrained Qwen3 / Llama checkpoints store RoPE base frequency here
    (e.g. Qwen3-0.6B uses ``rope_theta=1e6``). The backbone builder must copy
    it: ``Qwen3Config`` / ``LlamaConfig`` default to ``1e4``, and RoPE
    frequencies are computed from config (not loaded as weights).
    """
    rope = getattr(hf_cfg, "rope_parameters", None)
    if rope is None:
        return None
    return dict(rope)


def _load_transformer_weights(
    *,
    model: nn.Module,
    repo_id_or_path: str | Path,
    hub_kwargs: dict[str, Any],
) -> None:
    """Load matching transformer weights into a MOUSE backbone internals.

    Loads decoder weights before removing the pretrained output norm.
    The input token table belongs to a separately constructed Embedding.

    Warns in both directions: backbone tensors that did not receive pretrained
    weights (missing from the checkpoint or shape-mismatched, so they keep
    their random init), and checkpoint tensors the backbone has no slot for
    (beyond the layers dropped by ``num_layers=``). Either means the config
    and checkpoint disagree, and neither may pass silently.
    """
    from transformers import AutoModel

    with _quiet_transformers_load():
        pretrained = AutoModel.from_pretrained(repo_id_or_path, **hub_kwargs)
    pretrained.set_input_embeddings(None)
    target_state = model.state_dict()
    pretrained_state = pretrained.state_dict()
    loadable = {
        key: value
        for key, value in pretrained_state.items()
        if key in target_state
        and target_state[key].shape == value.shape
    }

    not_loaded = [
        key
        for key in target_state
        if key not in loadable
    ]
    if not_loaded:
        warnings.warn(
            f"{len(not_loaded)} of {len(target_state)} backbone tensors did not "
            f"receive pretrained weights from {str(repo_id_or_path)!r} and keep "
            f"their random initialization: {not_loaded}",
            stacklevel=2,
        )

    kept_layers = _layer_count(model)
    unconsumed = [
        key
        for key in pretrained_state
        if key not in loadable
        and not _is_dropped_layer_key(key, kept_layers)
    ]
    if unconsumed:
        warnings.warn(
            f"{len(unconsumed)} pretrained tensors from {str(repo_id_or_path)!r} "
            f"have no matching backbone tensor and were dropped: {unconsumed}",
            stacklevel=2,
        )

    model.load_state_dict(loadable, strict=False)

    del pretrained


def _layer_count(model: nn.Module) -> int | None:
    layers = getattr(model, "layers", None)
    return len(layers) if layers is not None else None


def _is_dropped_layer_key(key: str, kept_layers: int | None) -> bool:
    """True for ``layers.<i>.*`` keys past the truncated depth (``num_layers=``)."""
    if kept_layers is None or not key.startswith("layers."):
        return False
    index = key.split(".", 2)[1]
    return index.isdigit() and int(index) >= kept_layers
