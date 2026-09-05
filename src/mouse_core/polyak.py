"""Polyak averaging of delayed DQN encoder, backbone, and/or heads.

After each ``optimizer.step()``, call :meth:`PolyakAverager.update`.
After the online forward::

    predictions, averager_inputs = model(inputs)
    delayed_predictions = averager(averager_inputs)

Each of ``tau_encoder``, ``tau_backbone``, and ``tau_head`` is independent.
A zero tau means that section is not delayed: if its inputs are also not
from a delayed section, that path is not recomputed.
"""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
import torch.nn as nn
from tensordict import TensorDict

from mouse_core.models.base import AveragerInputs, Model, _run_heads
from mouse_core.models.backbone.base import Backbone
from mouse_core.models.embedding.embedding import Encoder
from mouse_core.models.heads.base import BaseHead

_DQN_HEADS = ("action_value", "action_value_layerwise")


class _PolyakState:
    """Pairs online/delayed parameters and lerps them in fp32.

    A small ``tau`` times the online/delayed gap is far below half a bf16 ULP
    (~``|w| / 512``), so lerping directly in a bf16 parameter rounds every
    update away and the delayed copy never moves. Non-fp32 delayed parameters
    therefore get a persistent fp32 shadow: the interpolation happens on the
    shadow and the result is cast into the delayed parameter each update.
    """

    def __init__(self, online: nn.Module, delayed: nn.Module) -> None:
        self._pairs: list[tuple[nn.Parameter, nn.Parameter, torch.Tensor | None]] = []
        for online_p, delayed_p in zip(
            online.parameters(), delayed.parameters(), strict=True
        ):
            shadow = (
                None
                if delayed_p.dtype == torch.float32
                else delayed_p.detach().to(dtype=torch.float32).clone()
            )
            self._pairs.append((online_p, delayed_p, shadow))

    @torch.no_grad()
    def update(self, tau: float) -> None:
        """θ_delayed ← τ·θ_online + (1−τ)·θ_delayed, accumulated in fp32."""
        if tau <= 0.0:
            return
        for online_p, delayed_p, shadow in self._pairs:
            if shadow is None:
                delayed_p.lerp_(online_p.to(dtype=delayed_p.dtype), tau)
                continue
            shadow.lerp_(online_p.to(dtype=torch.float32), tau)
            delayed_p.copy_(shadow)


_M = TypeVar("_M", bound=nn.Module)


def _copy_delayed(online: _M) -> _M:
    delayed = copy.deepcopy(online)
    delayed.requires_grad_(False)
    delayed.train()
    return delayed


def _head_map(heads: nn.ModuleDict) -> dict[str, BaseHead]:
    return {str(name): heads[name] for name in heads}


class PolyakAverager:
    """Delayed DQN weights for TD bootstrap targets.

    Construct after ``model.to(...)``. After the online forward, delayed Q is
    ``averager(averager_inputs)``. Each section has its own interpolation
    factor; ``0`` means that section is not delayed. Delayed modules run in
    ``train()`` under ``no_grad``. Call :meth:`update` after each
    ``optimizer.step()``.

    A delayed forward starts at the first delayed section and reuses the
    online activations above it:

    - ``tau_encoder > 0`` recomputes embeddings from ``averager_inputs.batch``.
    - else if ``tau_backbone > 0``, the delayed backbone reads
      ``averager_inputs.embeds`` (no second encoder pass).
    - else delayed heads read ``averager_inputs.h`` (no second
      encoder/backbone pass).
    - if every tau is ``0``, delayed Q is ``averager_inputs.predictions``.

    When a later section is delayed but an earlier one is not, the online
    module runs on the delayed inputs (those inputs *are* from a delayed
    net, so the path cannot be skipped).

    Args:
        model: Online model (must have an ``action_value`` or
            ``action_value_layerwise`` head).
        tau_encoder: Interpolation factor for the encoder. ``0`` keeps the
            online encoder (and skips it when backbone inputs are online).
        tau_backbone: Interpolation factor for the backbone. ``0`` keeps the
            online backbone (and skips it when head inputs are online).
        tau_head: Interpolation factor for the heads. ``0`` keeps the online
            heads (and skips them when their inputs are online).
    """

    def __init__(
        self,
        model: Model,
        *,
        tau_encoder: float = 0.0,
        tau_backbone: float = 0.0,
        tau_head: float = 0.01,
    ) -> None:
        if not any(name in model._heads for name in _DQN_HEADS):
            raise ValueError(
                "PolyakAverager requires a DQN-style action-value head."
            )
        self._online = model
        self.tau_encoder = float(tau_encoder)
        self.tau_backbone = float(tau_backbone)
        self.tau_head = float(tau_head)
        self.encoder: Encoder | None = None
        self.backbone: Backbone | None = None
        self.heads: nn.ModuleDict | None = None
        self._encoder_state: _PolyakState | None = None
        self._backbone_state: _PolyakState | None = None
        self._head_state: _PolyakState | None = None
        if self.tau_encoder > 0.0:
            self.encoder = _copy_delayed(model.encoder)
            self._encoder_state = _PolyakState(model.encoder, self.encoder)
        if self.tau_backbone > 0.0:
            self.backbone = _copy_delayed(model.backbone)
            self._backbone_state = _PolyakState(model.backbone, self.backbone)
        if self.tau_head > 0.0:
            self.heads = _copy_delayed(model.heads)
            self._head_state = _PolyakState(model.heads, self.heads)

    def _delayed_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = []
        if self.encoder is not None:
            modules.append(self.encoder)
        if self.backbone is not None:
            modules.append(self.backbone)
        if self.heads is not None:
            modules.append(self.heads)
        return modules

    def __call__(self, averager_inputs: AveragerInputs) -> TensorDict:
        """Delayed head outputs from a :class:`~mouse_core.models.base.AveragerInputs`.

        Sections with tau ``0`` reuse the matching online activation when
        their inputs are not from a delayed section. Delayed modules run in
        ``train()`` under ``torch.no_grad()``.
        """
        if not isinstance(averager_inputs, AveragerInputs):
            raise TypeError(
                "averager(...) expects AveragerInputs from model(inputs), "
                f"got {type(averager_inputs).__name__}."
            )
        delay_encoder = self.encoder is not None
        delay_backbone = self.backbone is not None
        delay_head = self.heads is not None
        if not delay_encoder and not delay_backbone and not delay_head:
            if averager_inputs.predictions is not None:
                return averager_inputs.predictions
            return self._online.head(h=averager_inputs.h)

        for module in self._delayed_modules():
            module.train()
        with torch.no_grad():
            recompute_backbone = delay_encoder or delay_backbone
            if not recompute_backbone:
                h = averager_inputs.h
            else:
                h = self._delayed_h(
                    averager_inputs, delay_encoder=delay_encoder
                )
            if delay_head:
                assert self.heads is not None
                return _run_heads(_head_map(self.heads), h, None)
            return self._online.head(h=h)

    def _delayed_h(
        self,
        averager_inputs: AveragerInputs,
        *,
        delay_encoder: bool,
    ) -> torch.Tensor:
        if delay_encoder:
            assert self.encoder is not None
            embeds, prediction_indices = self.encoder(averager_inputs.batch)
            pool_encoder = self.encoder
        else:
            if (
                averager_inputs.embeds is None
                or averager_inputs.prediction_indices is None
            ):
                raise ValueError(
                    "Delayed backbone with tau_encoder=0 needs "
                    "AveragerInputs.embeds from model(inputs)."
                )
            embeds = averager_inputs.embeds
            prediction_indices = averager_inputs.prediction_indices
            pool_encoder = self._online.encoder
        t = averager_inputs.batch.to_tensors(embeds.device)
        backbone = (
            self.backbone if self.backbone is not None else self._online.backbone
        )
        needs_layerwise = "action_value_layerwise" in self._online._heads
        session_out = self._online._train_backbone_forward(
            backbone,
            embeds,
            t["sequence_ids"],
            t["grouping_ids"],
            needs_layerwise,
        )
        return self._online._pool_backbone_out(
            pool_encoder, session_out, prediction_indices, needs_layerwise
        )

    def update(self) -> None:
        """Move delayed weights toward the online model after an optimizer step.

        Interpolation is accumulated in fp32 even when the delayed modules are
        bf16, so small ``tau`` updates are not rounded away. Sections with
        tau ``0`` have no delayed copy and are skipped.
        """
        if self._encoder_state is not None:
            self._encoder_state.update(self.tau_encoder)
        if self._backbone_state is not None:
            self._backbone_state.update(self.tau_backbone)
        if self._head_state is not None:
            self._head_state.update(self.tau_head)
