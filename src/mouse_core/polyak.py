"""Bind grouped fp32 Polyak updates to an online/delayed model pair.

Build after moving/copying the models, then reuse after optimizer steps::

    update_delayed = model_polyak(online=model, delayed=delayed_model)
    update_delayed(tau_heads={"q": 0.001}, tau_backbone=0.1)

The callable retains validated parameter pairs, not parameter snapshots.
Each call applies ``delayed.lerp_(online, tau)`` through grouped foreach
operations. Backbone and reasoner use ``tau_backbone``; heads can use one
scalar or a mapping with one rate for every head carried by the delayed
model. Pass ``None`` for shared sections (``1`` is also a no-op). Copied
sections require a rate. Zero skips writes. Frozen shared base weights
are never interpolated. Rebind after replacing parameters, changing their
trainability/dtype/device, or changing model sections. In-place optimizer
steps and ordinary in-place state-dict loads preserve the binding.
"""

from __future__ import annotations

from collections.abc import Mapping
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
                    f"{section} parameter {name!r} is {delayed_p.dtype}; model_polyak "
                    "interpolates fp32 parameters only. Build the backbone with "
                    "dtype=torch.float32 to delay a fine-tuned backbone, or freeze it "
                    "with lora= to use a bf16 base."
                )
            if online_p.device != delayed_p.device:
                raise ValueError(f"{section} parameters must be on the same device")
            self._pairs.append((online_p, delayed_p))
        groups: dict[torch.device, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for online_p, delayed_p in self._pairs:
            sources, targets = groups.setdefault(online_p.device, ([], []))
            sources.append(online_p)
            targets.append(delayed_p)
        self._groups = tuple(groups.values())

    def __len__(self) -> int:
        return len(self._pairs)

    @torch.no_grad()
    def update(self, tau: float) -> None:
        """θ_delayed ← τ·θ_online + (1−τ)·θ_delayed."""
        if tau <= 0.0:
            return
        for sources, targets in self._groups:
            torch._foreach_lerp_(targets, sources, tau)


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


class _ModelPolyak:
    def __init__(self, *, online: Model, delayed: Model) -> None:
        from mouse_core.models.base import Model as _Model

        if not isinstance(online, _Model) or not isinstance(delayed, _Model):
            raise TypeError("model_polyak interpolates a delayed Model toward an online Model.")
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
        heads: dict[str, _PolyakState | None] = {}
        for name in delayed.heads:
            delayed_head = delayed.heads[name]
            online_head = online.heads[name]
            if delayed_head is online_head:
                heads[name] = None
                continue
            copied.append(delayed_head)
            heads[name] = _PolyakState(online_head, delayed_head, section=f"heads.{name}")
        backbone: list[_PolyakState] = []
        if delayed.backbone is not online.backbone:
            copied.append(delayed.backbone)
            backbone.append(_PolyakState(online.backbone, delayed.backbone, section="backbone"))
        if (
            online.reasoner is not None
            and delayed.reasoner is not None
            and delayed.reasoner is not online.reasoner
        ):
            copied.append(delayed.reasoner)
            backbone.append(_PolyakState(online.reasoner, delayed.reasoner, section="reasoner"))
        if any(p.requires_grad for module in copied for p in module.parameters()):
            raise ValueError(
                "a copied delayed section has trainable parameters; build the "
                "delayed model with Model.copy(heads=..., backbone=..., reasoner=...)."
            )

        self._heads = heads
        self._backbone = backbone

    def __call__(
        self,
        *,
        tau_heads: float | Mapping[str, float | None] | None,
        tau_backbone: float | None,
    ) -> None:
        if isinstance(tau_heads, Mapping):
            if set(tau_heads) != set(self._heads):
                raise ValueError("tau_heads must name every delayed head exactly once")
            rates = dict(tau_heads)
        else:
            # Validate the scalar even when the delayed model carries no heads.
            _check_section_tau("tau_heads", tau_heads, copied=any(
                state is not None for state in self._heads.values()
            ))
            rates = dict.fromkeys(self._heads, tau_heads)
        checked = {
            name: _check_section_tau("tau_heads", rates[name], copied=state is not None)
            for name, state in self._heads.items()
        }
        backbone_tau = _check_section_tau(
            "tau_backbone", tau_backbone, copied=bool(self._backbone)
        )
        # Validate every rate before writing any section.
        for name, state in self._heads.items():
            tau = checked[name]
            if state is not None and tau is not None:
                state.update(tau)
        if backbone_tau is not None:
            for state in self._backbone:
                state.update(backbone_tau)


def model_polyak(*, online: Model, delayed: Model) -> _ModelPolyak:
    """Bind and validate an updater once, then call it after optimizer steps.

    ``update = model_polyak(online=model, delayed=delayed_model)`` followed
    by ``update(tau_heads=..., tau_backbone=...)``. Both rates are explicit;
    use ``None`` for shared sections. ``tau_heads`` can be a scalar or a
    mapping whose keys exactly match the delayed model's heads. Each rate
    is in ``[0, 1]``. The copied reasoner follows the backbone rate.

    Bind after model construction/device placement. Rebind after replacing
    parameters or changing model sections, trainability, dtype, or device.
    Validation and parameter pairing happen at binding, not every update.
    """
    return _ModelPolyak(online=online, delayed=delayed)
