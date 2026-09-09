"""Polyak interpolation of a delayed model toward the online model.

The delayed model comes from :meth:`~mouse_core.models.base.Model.delayed_copy`.
Each section — encoder, backbone (with the reasoner / recurrence section), and
heads — is either a delayed copy (``delayed_copy(encoder=True)`` /
``backbone=True`` / ``heads=True``) or the online module shared by reference.
After each ``optimizer.step()`` call :meth:`Polyak.update` with this step's
per-section ``tau``.

Heads-only delay (the delayed heads read the online token states)::

    delayed_model = model.delayed_copy(heads=True)
    polyak = Polyak(model, delayed_model)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed_model(
            last_hidden_state=out.last_hidden_state,
            head_output_indices=out.head_output_indices,
            hidden_states=out.hidden_states,
        )
    ...
    polyak.update(tau_heads=0.0005)

Full-model delay (the delayed model re-runs the trunk)::

    delayed_model = model.delayed_copy(encoder=True, backbone=True, heads=True)
    polyak = Polyak(model, delayed_model)
    out = model(inputs)
    with torch.no_grad():
        delayed_out = delayed_model(inputs)
    ...
    polyak.update(tau_heads=0.0005, tau_encoder=0.0005, tau_backbone=0.0005)

Per section, ``θ_delayed ← τ·θ_online + (1−τ)·θ_delayed``. ``τ = 0`` keeps
that section frozen; ``τ = 1`` copies the online weights, which is exactly
what a shared (non-copied) section is at every step — so a section whose
``tau`` is not ``1`` must be a delayed copy, and ``update`` rejects a
non-``1`` ``tau`` for a shared section.

Heads are always fp32. Encoder / backbone copies are typically bf16, and a
small ``tau`` times the online/delayed gap is far below half a bf16 ULP.
``Polyak(..., fp32_shadow=True)`` (the default) keeps a persistent fp32
shadow per non-fp32 delayed parameter: interpolation accumulates on the
shadow and is cast into the parameter every update (an extra fp32 copy of
the delayed trunk in memory). ``fp32_shadow=False`` lerps in the parameter
dtype and skips that copy — a small ``tau`` then rounds away until the
online/delayed gap reaches half a ULP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mouse_core.models.base import Model


class _PolyakState:
    """Pairs online/delayed parameters of one section and lerps them.

    fp32 parameters are lerped in place. Non-fp32 parameters get an fp32
    shadow when ``fp32_shadow`` is set (see module docstring).
    """

    def __init__(
        self,
        online: nn.Module,
        delayed: nn.Module,
        *,
        section: str,
        fp32_shadow: bool = True,
    ) -> None:
        online_params = dict(online.named_parameters())
        delayed_params = dict(delayed.named_parameters())
        if set(online_params) != set(delayed_params):
            missing = sorted(set(online_params) ^ set(delayed_params))
            raise ValueError(
                f"online and delayed {section} must have the same parameter names; "
                f"mismatched: {missing[:8]}{'...' if len(missing) > 8 else ''}."
            )
        self._pairs: list[tuple[nn.Parameter, nn.Parameter, torch.Tensor | None]] = []
        for name, delayed_p in delayed_params.items():
            online_p = online_params[name]
            if online_p is delayed_p:
                raise ValueError(
                    f"{section} parameter {name!r} is the same tensor online and "
                    "delayed; a delayed section must be a copy."
                )
            if online_p.shape != delayed_p.shape:
                raise ValueError(
                    f"{section} parameter {name!r} has shape {tuple(online_p.shape)} "
                    f"online but {tuple(delayed_p.shape)} delayed."
                )
            shadow = (
                delayed_p.detach().to(dtype=torch.float32).clone()
                if fp32_shadow and delayed_p.dtype != torch.float32
                else None
            )
            self._pairs.append((online_p, delayed_p, shadow))

    def __len__(self) -> int:
        return len(self._pairs)

    @torch.no_grad()
    def update(self, tau: float) -> None:
        """θ_delayed ← τ·θ_online + (1−τ)·θ_delayed (fp32 when shadowed)."""
        if tau <= 0.0:
            return
        for online_p, delayed_p, shadow in self._pairs:
            if shadow is None:
                delayed_p.lerp_(online_p.to(dtype=delayed_p.dtype), tau)
                continue
            shadow.lerp_(online_p.to(dtype=torch.float32), tau)
            delayed_p.copy_(shadow)


def _check_tau(name: str, tau: float) -> float:
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {tau}.")
    return tau


def _heads_shared(online: Model, delayed: Model) -> bool:
    """True when every delayed head is the online head module itself."""
    if set(delayed.heads.keys()) != set(online.heads.keys()):
        return False
    return all(delayed.heads[name] is online.heads[name] for name in online.heads.keys())


class Polyak:
    """Interpolates the delayed sections of a delayed model toward the online model.

    Does not run a forward. Pair with the model from
    :meth:`~mouse_core.models.base.Model.delayed_copy`. A section is
    interpolated when the delayed model carries its own copy of it and is
    otherwise shared with the online model (``tau = 1`` every step). The
    reasoner / recurrence section follows the backbone.

    Args:
        online: Source model (encoder, backbone, heads).
        delayed: Model from ``online.delayed_copy(...)``.
        fp32_shadow: Accumulate non-fp32 delayed interpolation in fp32
            shadows (default). ``False`` lerps in the parameter dtype.
    """

    def __init__(self, online: Model, delayed: Model, *, fp32_shadow: bool = True) -> None:
        from mouse_core.models.base import Model as _Model

        if not isinstance(online, _Model) or not isinstance(delayed, _Model):
            raise TypeError("Polyak interpolates a delayed Model toward an online Model.")
        if online.backbone is None or online.encoder is None:
            raise ValueError("online must be the full model, not a delayed copy.")
        if delayed is online:
            raise ValueError("delayed must come from Model.delayed_copy(), not be the online model.")

        self._fp32_shadow = bool(fp32_shadow)
        self._heads: _PolyakState | None = None
        self._encoder: _PolyakState | None = None
        self._backbone: list[_PolyakState] = []

        if not _heads_shared(online, delayed):
            self._heads = _PolyakState(
                online.heads, delayed.heads, section="heads", fp32_shadow=self._fp32_shadow
            )
        if delayed.encoder is not None and delayed.encoder is not online.encoder:
            self._encoder = _PolyakState(
                online.encoder, delayed.encoder, section="encoder", fp32_shadow=self._fp32_shadow
            )
        if delayed.backbone is not None and delayed.backbone is not online.backbone:
            self._backbone.append(
                _PolyakState(
                    online.backbone,
                    delayed.backbone,
                    section="backbone",
                    fp32_shadow=self._fp32_shadow,
                )
            )
            if online.reasoner is not None and delayed.reasoner is not None:
                self._backbone.append(
                    _PolyakState(
                        online.reasoner,
                        delayed.reasoner,
                        section="reasoner",
                        fp32_shadow=self._fp32_shadow,
                    )
                )
            if online.recurrence is not None and delayed.recurrence is not None:
                self._backbone.append(
                    _PolyakState(
                        online.recurrence,
                        delayed.recurrence,
                        section="recurrence",
                        fp32_shadow=self._fp32_shadow,
                    )
                )
        if self._heads is None and self._encoder is None and not self._backbone:
            raise ValueError(
                "the delayed model shares every section with the online model; "
                "build it with delayed_copy(encoder=True / backbone=True / heads=True)."
            )

    @property
    def fp32_shadow(self) -> bool:
        """Whether non-fp32 delayed parameters interpolate through fp32 shadows."""
        return self._fp32_shadow

    @property
    def delays_encoder(self) -> bool:
        """Whether the delayed model carries its own encoder copy."""
        return self._encoder is not None

    @property
    def delays_backbone(self) -> bool:
        """Whether the delayed model carries its own backbone copy."""
        return bool(self._backbone)

    @property
    def delays_heads(self) -> bool:
        """Whether the delayed model carries its own head copies."""
        return self._heads is not None

    def update(
        self,
        *,
        tau_heads: float | None = None,
        tau_encoder: float | None = None,
        tau_backbone: float | None = None,
    ) -> None:
        """Move the delayed sections toward the online model after an optimizer step.

        Each ``tau`` is this call's interpolation factor for that section:
        ``0`` leaves it unchanged (frozen), ``1`` copies the online weights.
        Pass new values each step to change them mid-run.

        A section's ``tau`` is required when that section is a delayed copy.
        For a shared section it may be omitted or ``1`` (sharing *is*
        ``tau = 1``); any other value raises, because a section with
        ``tau != 1`` has to be a copy — build it with
        ``delayed_copy(<section>=True)``.
        """
        tau_heads = self._resolve_tau("heads", tau_heads, delayed=self._heads is not None)
        tau_encoder = self._resolve_tau("encoder", tau_encoder, delayed=self._encoder is not None)
        tau_backbone = self._resolve_tau("backbone", tau_backbone, delayed=bool(self._backbone))

        if self._heads is not None:
            self._heads.update(tau_heads)
        if self._encoder is not None:
            self._encoder.update(tau_encoder)
        for state in self._backbone:
            state.update(tau_backbone)

    @staticmethod
    def _resolve_tau(section: str, tau: float | None, *, delayed: bool) -> float:
        if delayed:
            if tau is None:
                raise ValueError(
                    f"the delayed model has its own {section} copy; pass tau_{section}=."
                )
            return _check_tau(f"tau_{section}", tau)
        if tau is None:
            return 1.0
        tau = _check_tau(f"tau_{section}", tau)
        if tau != 1.0:
            raise ValueError(
                f"tau_{section}={tau} but the delayed model shares the online {section} "
                f"(tau = 1 every step); build it with delayed_copy({section}=True) to "
                "interpolate it."
            )
        return tau
