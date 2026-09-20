"""Polyak interpolation of a delayed model toward the online model.

The delayed model comes from :meth:`~mouse_core.models.base.Model.copy`,
which names every section (``heads``, ``backbone``, ``reasoner``).
Copied trainable parameters are frozen copies; frozen LoRA-base
weights stay shared by reference. An uncopied backbone or reasoner
**is** the online module. After each ``optimizer.step()`` call
:meth:`Polyak.update` with this step's ``tau`` for each copied
section — heads, and backbone when that section was copied (the
reasoner follows the backbone; token embeddings ride with the
backbone)::

    delayed_model = model.copy(heads=True, backbone=True, reasoner=False)
    polyak = Polyak(online=model, delayed=delayed_model)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed_model(inputs)
    ...
    polyak.update(tau_heads=0.0001, tau_backbone=0.01)

Pass a ``tau`` only for sections the delayed model copied. Heads-only
delay omits ``tau_backbone``::

    delayed_model = model.copy(heads=True, backbone=False, reasoner=False)
    ...
    delayed_predictions = delayed_model.head(h=delayed_model.pool(output=out))
    polyak.update(tau_heads=0.0001)

Per copied section, ``θ_delayed ← τ·θ_online + (1−τ)·θ_delayed``.
``τ = 0`` keeps that section frozen; ``τ = 1`` copies the online
weights (no delay). The heads section pairs each delayed head with
the online head of the same name; online heads the delayed model
does not carry are not interpolated.

Every interpolated parameter is fp32, so a small ``tau`` never
rounds away. Shared frozen parameters are not interpolated.
``Polyak`` rejects a non-fp32 parameter it would have to interpolate.
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
                "copy that section with Model.copy(heads=..., backbone=..., "
                "reasoner=...)."
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


def _check_section_tau(name: str, tau: float | None, *, copied: bool) -> float | None:
    """Require a ``[0, 1]`` float when *copied*; otherwise ``None`` or ``1``."""
    if copied:
        if tau is None:
            raise ValueError(
                f"{name} is required when that section was copied; "
                "pass a float in [0, 1] or copy with that section False."
            )
        return _check_tau(name, tau)
    if tau is None:
        return None
    tau = _check_tau(name, tau)
    if tau != 1.0:
        raise ValueError(
            f"{name} must be None or 1 when that section is shared; "
            "pass 1 for no delay, or None, or copy that section to delay it."
        )
    return tau


class Polyak:
    """Interpolates a delayed model toward the online model, one ``tau`` per copied section.

    Does not run a forward. Pair with the model from
    :meth:`~mouse_core.models.base.Model.copy`. The sections are
    heads and, when copied, backbone; the reasoner section follows
    the backbone. Token embeddings ride with the backbone. The heads
    section covers the heads the delayed model carries (each paired
    with the online head of the same name); every delayed head must
    exist online. A shared section is not interpolated.

    Args:
        online: Source model (backbone, heads).
        delayed: Model from ``online.copy(heads=..., backbone=..., reasoner=...)``.
    """

    def __init__(self, *, online: Model, delayed: Model) -> None:
        from mouse_core.models.base import Model as _Model

        if not isinstance(online, _Model) or not isinstance(delayed, _Model):
            raise TypeError("Polyak interpolates a delayed Model toward an online Model.")
        if delayed is online:
            raise ValueError(
                "delayed must come from Model.copy(heads=..., backbone=..., "
                "reasoner=...), not be the online model."
            )
        if not any(p.requires_grad for p in online.parameters()):
            raise ValueError("online must be the trainable model, not a delayed copy.")
        if (online.reasoner is None) != (delayed.reasoner is None):
            raise ValueError(
                "online and delayed models must have the same sections; "
                "build the delayed model with Model.copy(heads=..., "
                "backbone=..., reasoner=...)."
            )

        extra = [name for name in delayed.heads if name not in online.heads]
        if extra:
            raise ValueError(
                f"delayed heads {extra} do not exist on the online model "
                f"(online heads are {tuple(online.heads)}); build the delayed "
                "model with Model.copy(heads=..., backbone=..., reasoner=...)."
            )

        copied: list[nn.Module] = []
        self._heads: list[_PolyakState] = []
        for name in delayed.heads:
            delayed_head = delayed.heads[name]
            online_head = online.heads[name]
            if delayed_head is online_head:
                continue
            copied.append(delayed_head)
            self._heads.append(
                _PolyakState(online_head, delayed_head, section=f"heads.{name}")
            )
        self._backbone: list[_PolyakState] = []
        if delayed.backbone is not online.backbone:
            copied.append(delayed.backbone)
            self._backbone.append(
                _PolyakState(online.backbone, delayed.backbone, section="backbone")
            )
        if (
            online.reasoner is not None
            and delayed.reasoner is not None
            and delayed.reasoner is not online.reasoner
        ):
            copied.append(delayed.reasoner)
            self._backbone.append(
                _PolyakState(online.reasoner, delayed.reasoner, section="reasoner")
            )
        if any(p.requires_grad for module in copied for p in module.parameters()):
            raise ValueError(
                "a copied delayed section has trainable parameters; build the "
                "delayed model with Model.copy(heads=..., backbone=..., reasoner=...)."
            )

    def update(
        self,
        *,
        tau_heads: float | None = None,
        tau_backbone: float | None = None,
    ) -> None:
        """Move copied delayed sections toward the online model after an optimizer step.

        Pass a ``tau`` only for sections the delayed model copied. Each
        required ``tau`` is a ``float`` in ``[0, 1]``: ``0`` skips that
        section, ``1`` copies the online weights. ``tau_backbone`` also
        applies to a copied reasoner and to token embeddings. Omitting
        a shared section's ``tau`` (or passing ``None`` or ``1``) is a
        no-op; any other float raises. All-zero ``tau`` on the copied
        sections returns without touching any delayed parameter. Pass
        new values each step to change them mid-run.
        """
        copied_heads = bool(self._heads)
        copied_backbone = bool(self._backbone)
        tau_heads = _check_section_tau("tau_heads", tau_heads, copied=copied_heads)
        tau_backbone = _check_section_tau("tau_backbone", tau_backbone, copied=copied_backbone)
        if (not copied_heads or tau_heads == 0.0) and (
            not copied_backbone or tau_backbone == 0.0
        ):
            return
        if copied_heads and tau_heads is not None and tau_heads > 0.0:
            for state in self._heads:
                state.update(tau_heads)
        if copied_backbone and tau_backbone is not None and tau_backbone > 0.0:
            for state in self._backbone:
                state.update(tau_backbone)
