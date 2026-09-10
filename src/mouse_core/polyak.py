"""Polyak interpolation of a delayed model toward the online model.

The delayed model comes from :meth:`~mouse_core.models.base.Model.delayed_copy`:
a copy of the online model in which every trainable parameter has its own
copy and every frozen parameter (the bf16 base weights of a LoRA backbone)
is shared by reference. After each ``optimizer.step()`` call
:meth:`Polyak.update` with this step's ``tau`` for each section — heads,
encoder, and backbone (the reasoner / recurrence section follows the
backbone)::

    delayed_model = model.delayed_copy()
    polyak = Polyak(model, delayed_model)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed_model(inputs)
    ...
    polyak.update(tau_heads=0.0005, tau_encoder=0.0005, tau_backbone=0.0005)

Per section, ``θ_delayed ← τ·θ_online + (1−τ)·θ_delayed``. ``τ = 0`` keeps
that section frozen; ``τ = 1`` copies the online weights (no delay).

Every trainable parameter is fp32 (heads, encoder, reasoner / recurrence,
and either the whole fp32 backbone or the LoRA adapters of a frozen bf16
one), so interpolation runs in place in fp32 and a small ``tau`` never
rounds away. Shared frozen parameters are not interpolated. ``Polyak``
rejects a non-fp32 parameter it would have to interpolate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mouse_core.models.base import Model


class _PolyakState:
    """Pairs online/delayed parameters of one section and lerps them in fp32.

    A parameter that is the same tensor online and delayed is a shared frozen
    weight (the backbone base) and is skipped; every interpolated parameter
    must be fp32.
    """

    def __init__(
        self,
        online: nn.Module,
        delayed: nn.Module,
        *,
        section: str,
    ) -> None:
        if delayed is online:
            raise ValueError(
                f"the delayed {section} is the online {section} itself; "
                "build the delayed model with Model.delayed_copy()."
            )
        online_params = dict(online.named_parameters())
        delayed_params = dict(delayed.named_parameters())
        if set(online_params) != set(delayed_params):
            missing = sorted(set(online_params) ^ set(delayed_params))
            raise ValueError(
                f"online and delayed {section} must have the same parameter names; "
                f"mismatched: {missing[:8]}{'...' if len(missing) > 8 else ''}."
            )
        self._pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        for name, delayed_p in delayed_params.items():
            online_p = online_params[name]
            if online_p is delayed_p:
                if online_p.requires_grad:
                    raise ValueError(
                        f"{section} parameter {name!r} is the same tensor online and "
                        "delayed; a trainable parameter must be copied."
                    )
                continue  # shared frozen weight (backbone base): nothing to interpolate
            if online_p.shape != delayed_p.shape:
                raise ValueError(
                    f"{section} parameter {name!r} has shape {tuple(online_p.shape)} "
                    f"online but {tuple(delayed_p.shape)} delayed."
                )
            if delayed_p.dtype != torch.float32 or online_p.dtype != torch.float32:
                raise TypeError(
                    f"{section} parameter {name!r} is {delayed_p.dtype}; Polyak "
                    "interpolates fp32 parameters only. Build the backbone with "
                    "dtype=torch.float32 to delay a fine-tuned backbone, or freeze it "
                    "with lora= to use a bf16 base."
                )
            self._pairs.append((online_p, delayed_p))

    def __len__(self) -> int:
        return len(self._pairs)

    @torch.no_grad()
    def update(self, tau: float) -> None:
        """θ_delayed ← τ·θ_online + (1−τ)·θ_delayed."""
        if tau <= 0.0:
            return
        for online_p, delayed_p in self._pairs:
            delayed_p.lerp_(online_p, tau)


def _check_tau(name: str, tau: float) -> float:
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {tau}.")
    return tau


class Polyak:
    """Interpolates a delayed model toward the online model, one ``tau`` per section.

    Does not run a forward. Pair with the model from
    :meth:`~mouse_core.models.base.Model.delayed_copy`. The sections are
    heads, encoder, and backbone; the reasoner / recurrence section follows
    the backbone.

    Args:
        online: Source model (encoder, backbone, heads).
        delayed: Model from ``online.delayed_copy()``.
    """

    def __init__(self, online: Model, delayed: Model) -> None:
        from mouse_core.models.base import Model as _Model

        if not isinstance(online, _Model) or not isinstance(delayed, _Model):
            raise TypeError("Polyak interpolates a delayed Model toward an online Model.")
        if delayed is online:
            raise ValueError("delayed must come from Model.delayed_copy(), not be the online model.")
        if not any(p.requires_grad for p in online.parameters()):
            raise ValueError("online must be the trainable model, not a delayed copy.")
        if any(p.requires_grad for p in delayed.parameters()):
            raise ValueError(
                "delayed has trainable parameters; build it with Model.delayed_copy()."
            )
        if (online.reasoner is None) != (delayed.reasoner is None) or (
            online.recurrence is None
        ) != (delayed.recurrence is None):
            raise ValueError(
                "online and delayed models must have the same sections; "
                "build the delayed model with Model.delayed_copy()."
            )

        self._heads = _PolyakState(online.heads, delayed.heads, section="heads")
        self._encoder = _PolyakState(online.encoder, delayed.encoder, section="encoder")
        self._backbone = [_PolyakState(online.backbone, delayed.backbone, section="backbone")]
        if online.reasoner is not None and delayed.reasoner is not None:
            self._backbone.append(
                _PolyakState(online.reasoner, delayed.reasoner, section="reasoner")
            )
        if online.recurrence is not None and delayed.recurrence is not None:
            self._backbone.append(
                _PolyakState(online.recurrence, delayed.recurrence, section="recurrence")
            )

    def update(self, *, tau_heads: float, tau_encoder: float, tau_backbone: float) -> None:
        """Move the delayed model toward the online model after an optimizer step.

        Each ``tau`` is this call's interpolation factor for that section,
        in ``[0, 1]``: ``0`` leaves it unchanged (frozen), ``1`` copies the
        online weights. ``tau_backbone`` also applies to the reasoner /
        recurrence section. Pass new values each step to change them mid-run.
        """
        tau_heads = _check_tau("tau_heads", tau_heads)
        tau_encoder = _check_tau("tau_encoder", tau_encoder)
        tau_backbone = _check_tau("tau_backbone", tau_backbone)
        self._heads.update(tau_heads)
        self._encoder.update(tau_encoder)
        for state in self._backbone:
            state.update(tau_backbone)
