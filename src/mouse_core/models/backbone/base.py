"""Backbone interface for MOUSE models.

A backbone is a sequence processor that takes token embeddings and returns
hidden states of the same shape. Transformer backbones train through the
packed stream forward (:func:`~mouse_core.models.backbone.packed_train.packed_forward`
over ``self.model``); :meth:`Backbone.forward` is the rectangular ``[B, T, D]``
forward used by non-transformer backbones; incremental decoding goes through a
:class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession` created by
:meth:`Backbone.decode_session`.

Two ways to train a backbone, both with every trainable parameter in fp32:

- full fine-tuning — no ``lora``; build it with ``dtype=torch.float32`` and
  the base weights train directly;
- fp32 LoRA on a frozen base — ``lora=LoRAConfig(...)``; the base weights
  are frozen and may be built in bf16 (``dtype=preferred_dtype(device)``),
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

from mouse_core.models.backbone.flex_decode import (
    DecodeKernel,
    FlexDecodeSession,
    check_decode_kernel,
    module_device_dtype,
)
from mouse_core.models.backbone.packed_train import TrainKernel, check_train_kernel
from mouse_core.models.lora import LoRAConfig, apply_lora


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend to avoid driver-specific errors."""
    enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if enable_cudnn_sdp is not None:
        enable_cudnn_sdp(enabled=False)


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

    ``lora`` is the backbone's :class:`~mouse_core.models.lora.LoRAConfig`
    (``None`` for a fully trainable fp32 backbone).

    ``gradient_checkpointing`` makes the packed training forward recompute
    each decoder layer in backward instead of storing its activations:
    activation memory drops by roughly the layer count for about 1.4x the
    step time. Set ``backbone.gradient_checkpointing = True`` before
    training; it is not saved with the model.

    Attention kernels are required constructor arguments of every
    transformer backbone, so the choices are always explicit:

    - ``train_kernel`` runs the uncached packed forward (training, and any
      forward without a cache): ``"varlen"`` (flash varlen on CUDA
      bf16/fp16, masked SDPA otherwise), ``"padded"`` (dense causal SDPA
      on segments padded to ``max_seqlen``), or ``"flex"`` (FlexAttention
      block mask, compiled on CUDA in every dtype). All three give the
      same result.
    - ``decode_kernel`` runs cached decode (``use_cache=True``): ``"flex"``
      (paged FlexAttention), currently the only kernel that reads K/V
      through a page table.

    Like ``gradient_checkpointing`` they are execution choices for the
    current machine, not model properties: not saved with the model
    (``load_model`` takes them as arguments) and reassignable at any time
    (``backbone.train_kernel = "flex"``).
    """

    lora: LoRAConfig | None = None
    gradient_checkpointing: bool = False
    train_kernel: TrainKernel
    decode_kernel: DecodeKernel

    def _set_kernels(self, train_kernel: TrainKernel, decode_kernel: DecodeKernel) -> None:
        self.train_kernel = check_train_kernel(train_kernel)
        self.decode_kernel = check_decode_kernel(decode_kernel)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move the backbone; the base dtype is fixed at construction (``dtype=``)."""
        _reject_dtype_cast(type(self).__name__, *args, **kwargs)
        return super().to(*args, **kwargs)

    def half(self) -> "Backbone":
        raise TypeError(f"{type(self).__name__}.half() is not supported; pass dtype= when building it.")

    def bfloat16(self) -> "Backbone":
        raise TypeError(f"{type(self).__name__}.bfloat16() is not supported; pass dtype= when building it.")

    def double(self) -> "Backbone":
        raise TypeError(f"{type(self).__name__}.double() is not supported; pass dtype= when building it.")

    def float(self) -> "Backbone":
        raise TypeError(f"{type(self).__name__}.float() is not supported; pass dtype= when building it.")

    def _attach_lora(self, model: nn.Module, lora: LoRAConfig | None) -> None:
        """Freeze ``model`` and attach fp32 LoRA adapters when ``lora`` is set.

        Without ``lora`` the backbone is left fully trainable. Call once the
        pretrained weights are loaded: wrapping renames the adapted
        ``nn.Linear`` keys to ``<target>.base.weight``.
        """
        self.lora = lora
        if lora is not None:
            apply_lora(model, lora)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the base weights; ``Model`` casts backbone inputs to it.

        LoRA adapters are fp32 and skipped. A parameterless backbone
        (:class:`~mouse_core.models.backbone.none.IdentityBackbone`) reports
        ``float32``.
        """
        try:
            return module_device_dtype(self)[1]
        except ValueError:
            return torch.float32

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

    def decode_session(self, batch_size: int) -> FlexDecodeSession:
        """Create a cached-decode session over ``batch_size`` sequences.

        ``Model.forward`` calls this on the first ``use_cache=True`` call and
        carries the session inside its ``DecodeCache``. The session runs
        ``self.decode_kernel``; its paged KV pool sizes itself as rows grow.
        Requires the backbone to expose a ``transformers`` decoder stack as
        ``self.model``.
        """
        model = getattr(self, "model", None)
        if model is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support cached decoding."
            )
        check_decode_kernel(self.decode_kernel)  # "flex" is the only kernel; paged FlexAttention.
        return FlexDecodeSession(model, batch_size=batch_size)


def _reject_dtype_cast(what: str, *args: Any, **kwargs: Any) -> None:
    """``.to()`` on a MOUSE module moves devices only; dtype is a backbone constructor argument."""
    _device, dtype, _non_blocking, _memory_format = cast(Any, torch._C)._nn._parse_to(*args, **kwargs)
    if dtype is not None:
        raise TypeError(
            f"{what}.to() does not cast dtypes. The backbone dtype is fixed when it is built "
            f"(e.g. Qwen3Backbone(dtype=preferred_dtype(device), ...)); use .to(device) to move."
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

    MOUSE backbones replace token embeddings with a MOUSE encoder
    (:class:`~mouse_core.models.embedding.NumericEmbedder` or
    :class:`~mouse_core.models.embedding.TextEmbedder`), so the
    ``embed_tokens`` keys are skipped. The final norm is kept and loaded.

    Warns in both directions: backbone tensors that did not receive pretrained
    weights (missing from the checkpoint or shape-mismatched, so they keep
    their random init), and checkpoint tensors the backbone has no slot for
    (beyond the layers dropped by ``num_layers=``). Either means the config
    and checkpoint disagree, and neither may pass silently.
    """
    from transformers import AutoModel

    with _quiet_transformers_load():
        pretrained = AutoModel.from_pretrained(repo_id_or_path, **hub_kwargs)
    target_state = model.state_dict()
    pretrained_state = pretrained.state_dict()
    skipped_prefixes = ("embed_tokens",)
    loadable = {
        key: value
        for key, value in pretrained_state.items()
        if key in target_state
        and target_state[key].shape == value.shape
        and not key.startswith(skipped_prefixes)
    }

    not_loaded = [
        key
        for key in target_state
        if key not in loadable and not key.startswith(skipped_prefixes)
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
        and not key.startswith(skipped_prefixes)
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
