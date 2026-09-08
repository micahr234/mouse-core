"""Polyak interpolation of delayed heads toward the online heads.

The delayed network is the heads-only :class:`~mouse_core.models.base.Model`
from :meth:`~mouse_core.models.base.Model.delayed_copy`; it reads the online
token states, so only the heads are delayed. After each ``optimizer.step()``
call :meth:`Polyak.update` with that step's ``tau``::

    delayed_model = model.delayed_copy()
    polyak = Polyak(model, delayed_model)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed_model(
            last_hidden_state=out.last_hidden_state,
            head_output_indices=out.head_output_indices,
            hidden_states=out.hidden_states,
        )
    ...
    polyak.update(0.0005)

``θ_delayed ← τ·θ_online + (1−τ)·θ_delayed``. ``τ = 0`` keeps the delayed
heads frozen at their current weights. ``τ = 1`` copies the online heads, so
the delayed output equals what the online heads would produce on the same
states — there is no separate delayed network in effect. Heads are always fp32
(:meth:`~mouse_core.models.base.Model.to` never casts them), so small ``tau``
updates are not rounded away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mouse_core.models.base import Model


class _PolyakState:
    """Pairs online/delayed parameters and lerps them.

    Both sides must be fp32: a small ``tau`` times the online/delayed gap is
    far below half a bf16 ULP (~``|w| / 512``), so a bf16 delayed copy would
    round every update away and never move.
    """

    def __init__(self, online: nn.Module, delayed: nn.Module) -> None:
        online_params = dict(online.named_parameters())
        delayed_params = dict(delayed.named_parameters())
        if set(online_params) != set(delayed_params):
            missing = sorted(set(online_params) ^ set(delayed_params))
            raise ValueError(
                "online and delayed heads must have the same parameter names; "
                f"mismatched: {missing[:8]}{'...' if len(missing) > 8 else ''}."
            )
        self._pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for name, delayed_p in delayed_params.items():
            online_p = online_params[name]
            if online_p.shape != delayed_p.shape:
                raise ValueError(
                    f"parameter {name!r} has shape {tuple(online_p.shape)} online "
                    f"but {tuple(delayed_p.shape)} delayed."
                )
            if online_p.dtype != torch.float32 or delayed_p.dtype != torch.float32:
                raise TypeError(
                    f"heads must be float32, got {name!r} online {online_p.dtype} "
                    f"and delayed {delayed_p.dtype}."
                )
            self._pairs.append((online_p, delayed_p))
        if not self._pairs:
            raise ValueError("Polyak has no parameters to interpolate.")

    @torch.no_grad()
    def update(self, tau: float) -> None:
        """θ_delayed ← τ·θ_online + (1−τ)·θ_delayed."""
        if tau <= 0.0:
            return
        for online_p, delayed_p in self._pairs:
            delayed_p.lerp_(online_p, tau)


class Polyak:
    """Interpolates a heads-only delayed model toward the online model's heads.

    Does not run a forward. Pair with the model from
    :meth:`~mouse_core.models.base.Model.delayed_copy`.

    Args:
        online: Source model (encoder, backbone, heads).
        delayed: Heads-only copy whose parameters are interpolated.
    """

    def __init__(self, online: Model, delayed: Model) -> None:
        from mouse_core.models.base import Model as _Model

        if not isinstance(online, _Model) or not isinstance(delayed, _Model):
            raise TypeError("Polyak interpolates a delayed Model toward an online Model.")
        if delayed.backbone is not None or delayed.encoder is not None:
            raise ValueError(
                "delayed must be the heads-only model from Model.delayed_copy()."
            )
        if online.backbone is None:
            raise ValueError("online must be the full model, not a delayed copy.")
        self._state = _PolyakState(online.heads, delayed.heads)

    def update(self, tau: float) -> None:
        """Move the delayed heads toward the online heads after an optimizer step.

        ``tau`` is this call's interpolation factor: ``0`` leaves the delayed
        heads unchanged (frozen), ``1`` copies the online heads (the delayed
        output then equals the online heads on the same states). Pass a new
        value each step to change it mid-run.
        """
        tau = float(tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}.")
        self._state.update(tau)
