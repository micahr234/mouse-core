"""Grouped fp32 Polyak updates with independent rates for named model stages.

Bind once with ``model_polyak(online=..., delayed=...)``. Each update applies
``delayed.lerp_(online, tau)`` to copied parameters; shared frozen base weights
are skipped. Parameter validation and pairing happen only when binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
                "copy that stage with Model.copy(modules=...)."
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
                "pass a float in [0, 1] or leave that section shared."
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
    def __init__(self, *, online: Model, delayed: Model, modules: Sequence[str] | None) -> None:
        from mouse_core.models.base import Model as _Model, _module_groups

        if not isinstance(online, _Model) or type(delayed) is not type(online):
            raise TypeError("online and delayed must have the same Model subclass.")
        if delayed is online:
            raise ValueError("delayed must come from Model.copy(), not be the online model.")
        if not any(p.requires_grad for p in online.parameters()):
            raise ValueError("online must be the trainable model, not a delayed copy.")
        if modules is None:
            modules = tuple(name for name, module in online.named_children() if any(True for _ in module.parameters()))
        sources = _module_groups(online, modules)
        targets = _module_groups(delayed, modules)
        self._states: dict[str, _PolyakState | None] = {}
        for name, source in sources.items():
            target = targets[name]
            if source is target:
                self._states[name] = None
            else:
                if any(p.requires_grad for p in target.parameters()):
                    raise ValueError("a copied delayed section has trainable parameters; use Model.copy() with disjoint copied and shared module groups.")
                self._states[name] = _PolyakState(source, target, section=name)

    def __call__(self, *, tau: Mapping[str, float | None]) -> None:
        """Validate all module rates before updating any parameters."""
        if set(tau) != set(self._states):
            raise ValueError("tau must name every bound module group exactly once.")
        updates: list[tuple[_PolyakState, float]] = []
        for name, state in self._states.items():
            value = _check_section_tau(f"tau[{name!r}]", tau[name], copied=state is not None)
            if state is not None and value is not None:
                updates.append((state, value))
        for state, value in updates:
            state.update(value)


def model_polyak(
    *, online: Model, delayed: Model, modules: Sequence[str] | None = None,
) -> _ModelPolyak:
    """Bind ``update(tau={module_path: rate, ...})`` to registered PyTorch modules.

    Groups default to parameterized direct children; pass dotted module paths
    for finer control. Copied groups require rates in [0, 1]; shared groups
    take None or 1. Bind after device placement and rebind after replacing
    parameters or changing trainability. No state is stored on either model.
    """
    return _ModelPolyak(online=online, delayed=delayed, modules=modules)
