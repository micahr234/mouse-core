"""Polyak interpolation of a delayed model toward the online model.

The delayed model comes from :meth:`~mouse_core.models.base.Model.delayed_copy`:
a copy of the online model in which every trainable parameter has its own
copy and every frozen parameter (the bf16 base weights of a LoRA backbone)
is shared by reference, carrying only the head instances in ``heads=``. After
each ``optimizer.step()`` call :meth:`Polyak.update` with this step's
``tau`` for each section — heads and backbone (the reasoner
section follows the backbone; token embeddings ride with the
backbone)::

    delayed_model = model.delayed_copy(heads=(q_head,))
    polyak = Polyak(online=model, delayed=delayed_model)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed_model(inputs)
    ...
    polyak.update(tau_heads=0.0001, tau_backbone=0.01)

Per section, ``θ_delayed ← τ·θ_online + (1−τ)·θ_delayed``. ``τ = 0`` keeps
that section frozen; ``τ = 1`` copies the online weights (no delay). The
heads section pairs each delayed head with the online head of the same
name; online heads the delayed model does not carry (a behavior or policy
head whose delayed values nothing reads) are not interpolated.

Every trainable parameter is fp32 (heads, reasoner, and
either the whole fp32 backbone — including Identity ``embed_tokens``
— or the LoRA adapters of a frozen bf16 one), so interpolation runs
in place in fp32 and a small ``tau`` never rounds away. Shared frozen
parameters are not interpolated. ``Polyak`` rejects a non-fp32
parameter it would have to interpolate.
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
                "build the delayed model with Model.delayed_copy(heads=...)."
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
    heads and backbone; the reasoner section follows
    the backbone. Token embeddings ride with the backbone. The heads
    section covers the heads the delayed model carries (each paired
    with the online head of the same name); every delayed head must
    exist online.

    Args:
        online: Source model (backbone, heads).
        delayed: Model from ``online.delayed_copy(heads=...)``.
    """

    def __init__(self, *, online: Model, delayed: Model) -> None:
        from mouse_core.models.base import Model as _Model

        if not isinstance(online, _Model) or not isinstance(delayed, _Model):
            raise TypeError("Polyak interpolates a delayed Model toward an online Model.")
        if delayed is online:
            raise ValueError("delayed must come from Model.delayed_copy(heads=...), not be the online model.")
        if not any(p.requires_grad for p in online.parameters()):
            raise ValueError("online must be the trainable model, not a delayed copy.")
        if any(p.requires_grad for p in delayed.parameters()):
            raise ValueError(
                "delayed has trainable parameters; build it with Model.delayed_copy(heads=...)."
            )
        if (online.reasoner is None) != (delayed.reasoner is None):
            raise ValueError(
                "online and delayed models must have the same sections; "
                "build the delayed model with Model.delayed_copy(heads=...)."
            )

        extra = [name for name in delayed.heads if name not in online.heads]
        if extra:
            raise ValueError(
                f"delayed heads {extra} do not exist on the online model "
                f"(online heads are {tuple(online.heads)}); build the delayed "
                "model with Model.delayed_copy(heads=...)."
            )
        self._heads = [
            _PolyakState(online.heads[name], delayed.heads[name], section=f"heads.{name}")
            for name in delayed.heads
        ]
        self._backbone = [_PolyakState(online.backbone, delayed.backbone, section="backbone")]
        if online.reasoner is not None and delayed.reasoner is not None:
            self._backbone.append(
                _PolyakState(online.reasoner, delayed.reasoner, section="reasoner")
            )

    def update(self, *, tau_heads: float, tau_backbone: float) -> None:
        """Move the delayed model toward the online model after an optimizer step.

        Each ``tau`` is this call's interpolation factor for that section,
        in ``[0, 1]``: ``0`` skips that section (no interpolation), ``1``
        copies the online weights. ``tau_backbone`` also applies to the
        reasoner section and to token embeddings on the
        backbone. All-zero ``tau`` returns without touching any delayed
        parameter. Pass new values each step to change them mid-run.
        """
        tau_heads = _check_tau("tau_heads", tau_heads)
        tau_backbone = _check_tau("tau_backbone", tau_backbone)
        if tau_heads == 0.0 and tau_backbone == 0.0:
            return
        if tau_heads > 0.0:
            for state in self._heads:
                state.update(tau_heads)
        if tau_backbone > 0.0:
            for state in self._backbone:
                state.update(tau_backbone)
