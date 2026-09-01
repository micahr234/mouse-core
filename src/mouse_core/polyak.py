"""Polyak averaging of a delayed model, used next to the optimizer.

DQN target Q comes from a second ``Model`` owned by :class:`PolyakAverager`.
After each ``optimizer.step()``, call :meth:`PolyakAverager.update`. Before the
loss, call :meth:`PolyakAverager.write_targets` so the objective sees
``action_value_target`` (or ``action_value_layerwise_target``).
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from mouse_core.data.token_batch import TokenBatch
from mouse_core.models.base import Model

_DQN_HEADS = ("action_value", "action_value_layerwise")
_TARGET_KEY = {
    "action_value": "action_value_target",
    "action_value_layerwise": "action_value_layerwise_target",
}


def _polyak_copy(online: nn.Module, delayed: nn.Module, tau: float) -> None:
    """Soft-update ``delayed`` toward ``online``: θ ← τ·θ_online + (1−τ)·θ_delayed."""
    if tau <= 0.0:
        return
    for online_p, delayed_p in zip(online.parameters(), delayed.parameters(), strict=True):
        delayed_p.data.copy_(tau * online_p.data + (1.0 - tau) * delayed_p.data)


class PolyakAverager:
    """Delayed copy of ``model`` for TD bootstrap targets.

    Construct after ``model.to(...)``. The delayed network stays in ``eval``.

    Args:
        model: Online model (must have an ``action_value`` or
            ``action_value_layerwise`` head).
        scope: ``"head"`` snaps encoder/backbone to the current weights on
            every :meth:`write_targets` and Polyak-averages only the heads on
            :meth:`update`. ``"model"`` Polyak-averages encoder, backbone, and
            heads; :meth:`write_targets` recomputes representations through
            those delayed weights.
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
        self.model = model
        self.scope: Literal["head", "model"] = scope
        self.tau = float(tau)
        self.delayed = copy.deepcopy(model)
        self.delayed.requires_grad_(False)
        self.delayed.eval()

    def update(self) -> None:
        """Move delayed weights toward the online model after an optimizer step."""
        if self.scope == "model":
            _polyak_copy(self.model, self.delayed, self.tau)
            return
        _polyak_copy(self.model.heads, self.delayed.heads, self.tau)

    def write_targets(self, batch: TokenBatch, predictions: TensorDict) -> TensorDict:
        """Run the delayed model and write ``*_target`` keys into ``predictions``.

        When ``scope == "head"``, encoder and backbone are copied from the
        online model first so target Q uses current representations and
        delayed heads. When ``scope == "model"``, the delayed encoder and
        backbone are left as last :meth:`update`'d, so representations are
        recomputed through delayed weights.
        """
        if self.scope == "head":
            _polyak_copy(self.model.encoder, self.delayed.encoder, tau=1.0)
            _polyak_copy(self.model.backbone, self.delayed.backbone, tau=1.0)
        self.delayed.eval()
        with torch.no_grad():
            delayed_pred, _, _ = self.delayed(batch)
        for name, target_key in _TARGET_KEY.items():
            if name in delayed_pred.keys():
                predictions[target_key] = delayed_pred[name]
        return predictions
