"""Polyak averaging of delayed DQN heads (or a delayed full model).

After each ``optimizer.step()``, call :meth:`PolyakAverager.update`.
After the online forward::

    predictions, averager_inputs = model(inputs)
    delayed_predictions = averager(averager_inputs)

``scope="head"`` runs delayed heads on ``averager_inputs.h``.
``scope="model"`` recomputes representations through a delayed copy of the
full stack on ``averager_inputs.batch``.
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from mouse_core.models.base import AveragerInputs, Model, _run_heads
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


class _DelayedHeads(nn.Module):
    """Frozen copy of ``Model.heads`` with the same ``.head(h=)`` entry point."""

    def __init__(self, heads: nn.ModuleDict) -> None:
        super().__init__()
        self.heads = heads
        self._heads: dict[str, BaseHead] = {str(name): heads[name] for name in heads}

    def head(
        self,
        *,
        h: torch.Tensor,
        batch_size: tuple[int, ...] | None = None,
    ) -> TensorDict:
        return _run_heads(self._heads, h, batch_size)


class PolyakAverager:
    """Delayed DQN weights for TD bootstrap targets.

    Construct after ``model.to(...)``. After the online forward, delayed Q is
    ``averager(averager_inputs)``.     ``scope="head"`` copies only the heads;
    ``scope="model"`` copies the full encoder/backbone/heads stack.
    Delayed modules run in ``train()`` under ``no_grad``.
    Call :meth:`update` after each ``optimizer.step()``.

    Args:
        model: Online model (must have an ``action_value`` or
            ``action_value_layerwise`` head).
        scope: ``"head"`` Polyak-averages only the heads on :meth:`update`
            and reads ``averager_inputs.h``. ``"model"`` Polyak-averages
            encoder, backbone, and heads and recomputes representations
            from ``averager_inputs.batch``.
        tau: Interpolation factor in ``[0, 1]`` applied on :meth:`update`.
    """

    def __init__(
        self,
        model: Model,
        *,
        scope: Literal["head", "model"] = "head",
        tau: float = 0.01,
    ) -> None:
        if scope not in ("head", "model"):
            raise ValueError(f"scope must be 'head' or 'model', got {scope!r}.")
        if not any(name in model._heads for name in _DQN_HEADS):
            raise ValueError(
                "PolyakAverager requires a DQN-style action-value head."
            )
        self._online = model
        self.scope: Literal["head", "model"] = scope
        self.tau = float(tau)
        if scope == "head":
            delayed_heads = copy.deepcopy(model.heads)
            delayed_heads.requires_grad_(False)
            self.model: Model | _DelayedHeads = _DelayedHeads(delayed_heads)
            self._state = _PolyakState(model.heads, delayed_heads)
        else:
            self.model = copy.deepcopy(model)
            self.model.requires_grad_(False)
            self._state = _PolyakState(model, self.model)
        self.model.train()

    def __call__(self, averager_inputs: AveragerInputs) -> TensorDict:
        """Delayed head outputs from a :class:`~mouse_core.models.base.AveragerInputs`.

        ``scope="head"`` applies delayed heads to ``averager_inputs.h`` (no
        second encoder/backbone pass). ``scope="model"`` runs the delayed
        stack on ``averager_inputs.batch``. Both run in ``train()`` under
        ``torch.no_grad()``.
        """
        if not isinstance(averager_inputs, AveragerInputs):
            raise TypeError(
                "averager(...) expects AveragerInputs from model(inputs), "
                f"got {type(averager_inputs).__name__}."
            )
        self.model.train()
        with torch.no_grad():
            if self.scope == "model":
                delayed, _ = self.model(averager_inputs.batch)
                return delayed
            return self.model.head(h=averager_inputs.h.detach())

    def update(self) -> None:
        """Move delayed weights toward the online model after an optimizer step.

        Interpolation is accumulated in fp32 even when the delayed modules are
        bf16, so small ``tau`` updates are not rounded away.
        """
        self._state.update(self.tau)
