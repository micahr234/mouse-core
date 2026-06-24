"""Training-time augmentations for offline RL batches.

When any augmentation is enabled, :class:`TokenAugmenter` clones the step stream,
applies transforms to the copy, and returns it; the input is left unchanged.
If nothing is enabled, the input is returned as-is (no clone).

**mask_prob** (see :class:`AugmentMaskProbConfig`): per-field Bernoulli mask
with the given probability on each step.  Masked steps have their corresponding
field(s) zeroed (or set to -1 for time).  ``PREDICTION`` and ``COMPUTE`` rows are never masked.
Probabilities ``<= 0`` for a field skip that field; if every entry is zero, masking is
skipped entirely.
Masks are **not** snapshotted: a new random draw runs on every :meth:`TokenAugmenter.__call__`.

**permute_** flags: random choices are **per sequence** (per batch row): the same
action mapping, done flip apply at every step in that row; different batch rows may
use different permutations. **scale_** / **shift_** use :class:`AugmentScalarSpec`:
``low`` / ``high`` for uniform on ``[low, high)`` per batch (configs use this for all
continuous aug), or Gaussian ``mean`` / ``std``; ``low == high`` or ``std: 0`` fixes
the scalar at identity.

Note: ``permute_tokens`` is not applicable without an explicit token stream and is silently
ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentScalarSpec:
    """Scalar augmentation: uniform on ``[low, high)`` per batch, or Gaussian ``mean + std * N(0,1)``.

    YAML configs typically use ``low``/``high`` (``low == high`` fixes at that value = identity).
    Omit ``low``/``high`` for Gaussian; ``std == 0`` fixes at ``mean``.
    """

    mean: float
    std: float = 0.0
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        u_lo, u_hi = self.low, self.high
        if (u_lo is None) != (u_hi is None):
            raise ValueError("AugmentScalarSpec: set both low and high for uniform, or neither for Gaussian.")


def _augment_scalar_active(spec: AugmentScalarSpec, identity_mean: float) -> bool:
    if spec.low is not None and spec.high is not None:
        lo, hi = float(spec.low), float(spec.high)
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            return lo != identity_mean
        return True
    return spec.std != 0.0 or spec.mean != identity_mean


@dataclass(frozen=True)
class AugmentMaskProbConfig:
    """Per-type Bernoulli mask probability (MLM-style): each eligible token row is masked i.i.d.

    Masked rows replace payloads with a neutral value (0 / zero float / black pixel); ``objective_data``
    fields at those positions are aligned. ``PREDICTION`` and ``COMPUTE`` rows are never masked.
    """

    action: float = 0.0
    reward: float = 0.0
    done: float = 0.0
    obs_continuous: float = 0.0
    obs_discrete: float = 0.0
    obs_image: float = 0.0
    time: float = 0.0

    def any_positive(self) -> bool:
        return any(
            p > 0.0
            for p in (
                self.action,
                self.reward,
                self.done,
                self.obs_continuous,
                self.obs_discrete,
                self.obs_image,
                self.time,
            )
        )


@dataclass(frozen=True)
class AugmentTokensConfig:
    """Optional training-time token augmentations (copied streams on train batches).

    PREDICTION and COMPUTE tokens are never modified. See ``augment_tokens.TokenAugmenter``.
    ``mask_prob`` enables MLM-style random zero-masking per token type. Set ``enabled: false`` to disable all augmentations.
    """

    enabled: bool = True  # master switch — false disables all augmentations regardless of other settings
    permute_tokens: bool = False  # random shuffle of token order within each step (excludes PREDICTION and COMPUTE)
    scale_reward: AugmentScalarSpec = field(default_factory=lambda: AugmentScalarSpec(1.0, 0.0))
    shift_reward: AugmentScalarSpec = field(default_factory=lambda: AugmentScalarSpec(0.0, 0.0))
    scale_obs: AugmentScalarSpec = field(default_factory=lambda: AugmentScalarSpec(1.0, 0.0))
    shift_obs: AugmentScalarSpec = field(default_factory=lambda: AugmentScalarSpec(0.0, 0.0))
    scale_obs_image: AugmentScalarSpec = field(default_factory=lambda: AugmentScalarSpec(1.0, 0.0))
    shift_obs_image: AugmentScalarSpec = field(default_factory=lambda: AugmentScalarSpec(0.0, 0.0))
    permute_obs_discrete: bool = False  # remap OBS_DISCRETE token ids only (info_metadata_q_star / actions unchanged — unsafe for semantic categoricals)
    permute_action: Literal[False, "input", "target", "both"] = False  # action permutation mode
    permute_done: bool = False  # random swap of done 0/1
    mask_prob: AugmentMaskProbConfig = field(default_factory=AugmentMaskProbConfig)

    def __post_init__(self) -> None:
        val = self.permute_action
        if val is False:
            return
        if val not in ("input", "target", "both"):
            raise ValueError(
                f"AugmentTokensConfig.permute_action must be false or one of: 'input', 'target', 'both'; got {val!r}."
            )

    def permute_action_enabled(self) -> bool:
        return bool(self.permute_action)

    def permute_action_mode(self) -> Literal["input", "target", "both"]:
        val = self.permute_action
        if val is False:
            return "both"
        return cast(Literal["input", "target", "both"], val)

    def any_enabled(self) -> bool:
        if not self.enabled:
            return False
        if self.permute_tokens or self.permute_obs_discrete or self.permute_action_enabled() or self.permute_done:
            return True
        if _augment_scalar_active(spec=self.scale_reward, identity_mean=1.0) or _augment_scalar_active(spec=self.shift_reward, identity_mean=0.0):
            return True
        if _augment_scalar_active(spec=self.scale_obs, identity_mean=1.0) or _augment_scalar_active(spec=self.shift_obs, identity_mean=0.0):
            return True
        if _augment_scalar_active(spec=self.scale_obs_image, identity_mean=1.0) or _augment_scalar_active(spec=self.shift_obs_image, identity_mean=0.0):
            return True
        if self.mask_prob.any_positive():
            return True
        return False


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------


def _sample_scalar(spec: AugmentScalarSpec, generator: torch.Generator) -> float:
    if spec.low is not None and spec.high is not None:
        lo, hi = float(spec.low), float(spec.high)
        if lo > hi:
            lo, hi = hi, lo
        u = torch.rand((), device=generator.device, generator=generator)
        return float(lo + (hi - lo) * u)
    if spec.std == 0.0:
        return float(spec.mean)
    t = torch.randn((), device=generator.device, generator=generator)
    return float(t * spec.std + spec.mean)


def _inverse_action_perm_rows(perm: torch.Tensor) -> torch.Tensor:
    """``perm[b, old] = new`` → ``out[b, new] = old`` (inverse along the action axis)."""
    B, A = int(perm.shape[0]), int(perm.shape[1])
    out = torch.empty((B, A), device=perm.device, dtype=perm.dtype)
    cols = torch.arange(A, device=perm.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    out.scatter_(1, perm.to(dtype=torch.long), cols)
    return out


@torch.no_grad()
def apply_permute_action_augmentation(
    objective_data: TensorDict,
    perm: torch.Tensor,
    apply_to_input: bool,
    apply_to_target: bool,
) -> None:
    """``a → perm[a]`` with one ``perm`` per batch row; mutates ``objective_data`` in-place.

    ``perm`` shape: ``[B, max_num_actions]``; row ``b`` uses ``perm[b]``.
    """
    if apply_to_input:
        action = objective_data["action"]                          # [B, S]
        objective_data["action"].copy_(torch.gather(perm, dim=1, index=action.long()))

    if apply_to_target and "info_metadata_q_star" in objective_data.keys():
        inv_perm = _inverse_action_perm_rows(perm)
        q = objective_data["info_metadata_q_star"]                 # [B, S, A]
        B, S, A = q.shape
        inv_exp = inv_perm.unsqueeze(1).expand(B, S, A)        # [B, S, A]
        objective_data["info_metadata_q_star"].copy_(torch.gather(q, dim=2, index=inv_exp))


@torch.no_grad()
def apply_permute_done_augmentation(
    objective_data: TensorDict,
    perm: torch.Tensor,
) -> None:
    """``d → perm[d]`` with one ``perm`` over ``{0,1,2}`` per batch row; mutates in-place.

    ``perm`` shape: ``[B, 3]``; row ``b`` uses ``perm[b]``.
    done values: 0=not done, 1=terminal, 2=truncated.
    """
    done = objective_data["done"]                                  # [B, S]
    objective_data["done"].copy_(torch.gather(perm, dim=1, index=done.long()))


@torch.no_grad()
def apply_reward_scale_shift(
    objective_data: TensorDict,
    scale: float,
    shift: float,
) -> None:
    """Scale/shift rewards (in-place)."""
    if scale == 1.0 and shift == 0.0:
        return
    sr = objective_data["reward"]
    objective_data["reward"].copy_(sr * scale + shift)


@torch.no_grad()
def apply_obs_continuous_scale_shift(
    objective_data: TensorDict,
    scale: float,
    shift: float,
) -> None:
    """Scale/shift continuous obs values (in-place)."""
    if scale == 1.0 and shift == 0.0:
        return
    if "obs_continuous" not in objective_data.keys():
        return
    obs = objective_data["obs_continuous"]
    objective_data["obs_continuous"].copy_(obs.double() * scale + shift)


@torch.no_grad()
def apply_obs_image_scale_shift(
    objective_data: TensorDict,
    scale: float,
    shift: float,
) -> None:
    """Scale/shift image pixel values (clamped 0-255, in-place)."""
    if scale == 1.0 and shift == 0.0:
        return
    if "obs_image" not in objective_data.keys():
        return
    obs = objective_data["obs_image"]
    objective_data["obs_image"].copy_((obs.float() * scale + shift).round().clamp(0, 255).to(torch.int64))


@torch.no_grad()
def apply_permute_obs_discrete_augmentation(
    objective_data: TensorDict,
    perm: torch.Tensor,
) -> None:
    """``v → perm[v]`` for OBS_DISCRETE values per batch row (in-place).

    ``perm`` shape: ``[B, max_num_obs_discrete]``; row ``b`` uses ``perm[b]``.
    """
    if "obs_discrete" not in objective_data.keys():
        return
    obs = objective_data["obs_discrete"]                           # [B, S]
    perm_exp = perm.unsqueeze(1).expand_as(obs)                # [B, S]
    objective_data["obs_discrete"].copy_(torch.gather(perm_exp, dim=1, index=obs.long()))


@torch.no_grad()
def apply_field_masks(
    objective_data: TensorDict,
    mask_prob: AugmentMaskProbConfig,
    generator: torch.Generator,
) -> None:
    """Sample Bernoulli masks per step and zero masked fields (in-place)."""
    if not mask_prob.any_positive():
        return

    dev = cast(torch.device, objective_data["action"].device)
    g = generator
    B, S = int(objective_data["action"].shape[0]), int(objective_data["action"].shape[1])

    def _bernoulli_mask(prob: float) -> torch.Tensor | None:
        if prob <= 0.0:
            return None
        rand = torch.rand((B, S), device=dev, generator=g)
        return rand < prob

    mask_action = _bernoulli_mask(mask_prob.action)
    if mask_action is not None and mask_action.any():
        sa = objective_data["action"]
        objective_data["action"].copy_(torch.where(mask_action, torch.zeros_like(sa), sa))

    mask_reward = _bernoulli_mask(mask_prob.reward)
    if mask_reward is not None and mask_reward.any():
        sr = objective_data["reward"]
        objective_data["reward"].copy_(torch.where(mask_reward, torch.zeros_like(sr), sr))

    mask_done = _bernoulli_mask(mask_prob.done)
    if mask_done is not None and mask_done.any():
        sd = objective_data["done"]
        objective_data["done"].copy_(torch.where(mask_done, torch.zeros_like(sd), sd))

    mask_obs_continuous = _bernoulli_mask(mask_prob.obs_continuous)
    if mask_obs_continuous is not None and mask_obs_continuous.any() and "obs_continuous" in objective_data.keys():
        obs = objective_data["obs_continuous"]
        m_exp = mask_obs_continuous.unsqueeze(-1).expand_as(obs)
        objective_data["obs_continuous"].copy_(torch.where(m_exp, torch.zeros_like(obs), obs))

    mask_obs_discrete = _bernoulli_mask(mask_prob.obs_discrete)
    if mask_obs_discrete is not None and mask_obs_discrete.any() and "obs_discrete" in objective_data.keys():
        obs = objective_data["obs_discrete"]                       # [B, S]
        objective_data["obs_discrete"].copy_(torch.where(mask_obs_discrete, torch.zeros_like(obs), obs))

    mask_obs_image = _bernoulli_mask(mask_prob.obs_image)
    if mask_obs_image is not None and mask_obs_image.any() and "obs_image" in objective_data.keys():
        obs = objective_data["obs_image"]
        m_exp = mask_obs_image.unsqueeze(-1).expand_as(obs)
        objective_data["obs_image"].copy_(torch.where(m_exp, torch.zeros_like(obs), obs))

    mask_time = _bernoulli_mask(mask_prob.time)
    if mask_time is not None and mask_time.any() and "time" in objective_data.keys():
        st = objective_data["time"]
        # -1 means "not available"
        objective_data["time"].copy_(torch.where(mask_time, torch.full_like(st, -1), st))


@dataclass
class AugmentSnapshot:
    """Fixed permutations and scalar draws for one batch, reused across multiple ``__call__``."""

    batch_size: int
    device: torch.device
    perm_action: torch.Tensor | None
    perm_done: torch.Tensor | None
    r_scale: float | None
    r_shift: float | None
    o_scale: float | None
    o_shift: float | None
    im_scale: float | None
    im_shift: float | None
    perm_obs_discrete: torch.Tensor | None


class TokenAugmenter:
    """Applies ``AugmentTokensConfig`` to a step TensorDict batch.

    Call with ``objective_data`` to obtain a possibly augmented copy.
    :meth:`__call__` applies permutations/scalars from the stored snapshot; ``mask_prob``
    is sampled anew each call. Call :meth:`update_augmentations` first (required whenever
    any augmentation is enabled).
    """

    def __init__(
        self,
        augment: AugmentTokensConfig,
        max_num_actions: int,
        max_num_obs_discrete: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> None:
        if not isinstance(augment, AugmentTokensConfig):
            raise TypeError(f"augment must be AugmentTokensConfig, got {type(augment).__name__}")
        self._augment = augment
        self._max_num_actions = int(max_num_actions)
        self._max_num_obs_discrete = int(max_num_obs_discrete)
        self._generator = generator if generator is not None else torch.Generator(device=device)
        self._snapshot: AugmentSnapshot | None = None

    @property
    def augment(self) -> AugmentTokensConfig:
        return self._augment

    @property
    def snapshot(self) -> AugmentSnapshot | None:
        return self._snapshot

    def clear_augmentations(self) -> None:
        self._snapshot = None

    @torch.no_grad()
    def update_augmentations(self, objective_data: TensorDict) -> None:
        """Sample permutations and scalar parameters for this batch and store them."""
        augment = self._augment
        if not augment.any_enabled():
            self._snapshot = None
            return

        action = objective_data["action"]
        B = int(action.shape[0])
        dev = cast(torch.device, action.device)
        g = self._generator

        perm_action: torch.Tensor | None = None
        if augment.permute_action_enabled():
            perm_action = torch.stack(
                [torch.randperm(self._max_num_actions, device=dev, generator=g) for _ in range(B)],
                dim=0,
            )

        perm_done: torch.Tensor | None = None
        if augment.permute_done:
            perm_done = torch.stack(
                [torch.randperm(3, device=dev, generator=g) for _ in range(B)],
                dim=0,
            )

        perm_obs_discrete: torch.Tensor | None = None
        if augment.permute_obs_discrete:
            perm_obs_discrete = torch.stack(
                [torch.randperm(self._max_num_obs_discrete, device=dev, generator=g) for _ in range(B)],
                dim=0,
            )

        r_scale: float | None = None
        r_shift: float | None = None
        if _augment_scalar_active(augment.scale_reward, 1.0):
            r_scale = _sample_scalar(augment.scale_reward, g)
        if _augment_scalar_active(augment.shift_reward, 0.0):
            r_shift = _sample_scalar(augment.shift_reward, g)

        o_scale: float | None = None
        o_shift: float | None = None
        if _augment_scalar_active(augment.scale_obs, 1.0):
            o_scale = _sample_scalar(augment.scale_obs, g)
        if _augment_scalar_active(augment.shift_obs, 0.0):
            o_shift = _sample_scalar(augment.shift_obs, g)

        im_scale: float | None = None
        im_shift: float | None = None
        if _augment_scalar_active(augment.scale_obs_image, 1.0):
            im_scale = _sample_scalar(augment.scale_obs_image, g)
        if _augment_scalar_active(augment.shift_obs_image, 0.0):
            im_shift = _sample_scalar(augment.shift_obs_image, g)

        self._snapshot = AugmentSnapshot(
            batch_size=B,
            device=dev,
            perm_action=perm_action,
            perm_done=perm_done,
            r_scale=r_scale,
            r_shift=r_shift,
            o_scale=o_scale,
            o_shift=o_shift,
            im_scale=im_scale,
            im_shift=im_shift,
            perm_obs_discrete=perm_obs_discrete,
        )

    def _assert_snapshot_matches(self, objective_data: TensorDict) -> AugmentSnapshot:
        snap = self._snapshot
        if snap is None:
            raise RuntimeError("TokenAugmenter has no snapshot; call update_augmentations first.")
        B = int(objective_data["action"].shape[0])
        dev = cast(torch.device, objective_data["action"].device)
        if B != snap.batch_size or dev != snap.device:
            raise ValueError(
                f"Batch mismatch: got B={B}, device={dev!r}; snapshot expects "
                f"batch_size={snap.batch_size}, device={snap.device!r}."
            )
        return snap

    @torch.no_grad()
    def __call__(
        self,
        objective_data: TensorDict,
    ) -> TensorDict:
        """Augment a training batch; returns a new TensorDict when augmentation runs.

        Requires ``objective_data`` shape ``[B, S]`` per field.
        Call :meth:`update_augmentations` with the same batch first.
        Permutations/scalars use :attr:`snapshot`; ``mask_prob`` is drawn fresh here.
        """
        augment = self._augment
        if not augment.any_enabled():
            return objective_data

        objective_data = objective_data.clone()
        snap = self._assert_snapshot_matches(objective_data)

        # MLM-style masks first (corrupt inputs before permute/scale)
        if augment.mask_prob.any_positive():
            apply_field_masks(objective_data=objective_data, mask_prob=augment.mask_prob, generator=self._generator)

        if augment.permute_action_enabled():
            assert snap.perm_action is not None
            mode = augment.permute_action_mode()
            apply_permute_action_augmentation(
                objective_data=objective_data,
                perm=snap.perm_action,
                apply_to_input=mode in ("input", "both"),
                apply_to_target=mode in ("target", "both"),
            )

        if augment.permute_done:
            assert snap.perm_done is not None
            apply_permute_done_augmentation(objective_data=objective_data, perm=snap.perm_done)

        if augment.permute_obs_discrete:
            assert snap.perm_obs_discrete is not None
            apply_permute_obs_discrete_augmentation(objective_data=objective_data, perm=snap.perm_obs_discrete)

        if _augment_scalar_active(spec=augment.scale_reward, identity_mean=1.0):
            assert snap.r_scale is not None
            apply_reward_scale_shift(objective_data=objective_data, scale=snap.r_scale, shift=0.0)

        if _augment_scalar_active(spec=augment.shift_reward, identity_mean=0.0):
            assert snap.r_shift is not None
            apply_reward_scale_shift(objective_data=objective_data, scale=1.0, shift=snap.r_shift)

        if _augment_scalar_active(spec=augment.scale_obs, identity_mean=1.0):
            assert snap.o_scale is not None
            apply_obs_continuous_scale_shift(objective_data=objective_data, scale=snap.o_scale, shift=0.0)

        if _augment_scalar_active(spec=augment.shift_obs, identity_mean=0.0):
            assert snap.o_shift is not None
            apply_obs_continuous_scale_shift(objective_data=objective_data, scale=1.0, shift=snap.o_shift)

        if _augment_scalar_active(spec=augment.scale_obs_image, identity_mean=1.0):
            assert snap.im_scale is not None
            apply_obs_image_scale_shift(objective_data=objective_data, scale=snap.im_scale, shift=0.0)

        if _augment_scalar_active(spec=augment.shift_obs_image, identity_mean=0.0):
            assert snap.im_shift is not None
            apply_obs_image_scale_shift(objective_data=objective_data, scale=1.0, shift=snap.im_shift)

        return objective_data
