"""Step augmentation (train or eval).

I/O
---
* **in:** ``dict`` (one step)
* **out:** ``dict`` (values may be transformed)

Field keep/rename is :class:`~mouse_core.data.selector.Selector`, not this class.
Compose in pipeline order::

    transform = compose(augmenter, grouper, selector, tokenizer)

Permute/scale/shift draws are keyed by ``seed_field`` so steps sharing that
id share draws within one :meth:`Augmenter.reseed` generation. Mask decisions
(``mask_prob``) are drawn independently per step.
``DataLoader`` calls ``transform.reseed()`` once per batch. Eval / decode
should pass ``augment=False`` to skip augmentation entirely (raw values reach
the model, and no reseeding is needed).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


def _stable_hash(*parts: Any) -> int:
    """32-bit hash of ``parts`` that is stable across processes.

    Python's builtin ``hash`` is salted per process (``PYTHONHASHSEED``), so it
    cannot seed reproducible draws.
    """
    digest = hashlib.blake2s(repr(parts).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


@dataclass(frozen=True)
class _ScalarDraw:
    """Scalar draw used by direct scale/shift augmentations."""

    mean: float
    std: float = 0.0
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if (self.low is None) != (self.high is None):
            raise ValueError("set both low and high, or neither.")

    def is_active(self, identity: float) -> bool:
        if self.low is not None and self.high is not None:
            lo, hi = sorted((float(self.low), float(self.high)))
            if lo == hi:
                return lo != identity
            return True
        return self.std != 0.0 or self.mean != identity

    def sample(self, rng: np.random.Generator) -> float:
        if self.low is not None and self.high is not None:
            lo, hi = sorted((float(self.low), float(self.high)))
            if lo == hi:
                return lo
            return float(rng.uniform(lo, hi))
        if self.std == 0.0:
            return float(self.mean)
        return float(rng.normal(self.mean, self.std))


@dataclass(frozen=True)
class SequenceAugmentModalitySpec:
    """Specification for augmenting one raw step-record field.

    Each spec has required ``input_field`` / ``output_field`` and a ``type``.
    Same names replace in place; different names write outputs and leave inputs.
    Augmentation ``type`` values describe raw-data behavior, not embedding.
    """

    type: Literal["discrete", "linear", "image"]
    input_field: str | Sequence[str]
    output_field: str | Sequence[str]
    vocab_size: int | None = None
    mask_prob: float = 0.0
    scale_in_low: float | None = None
    scale_out_low: float | None = None
    scale_in_high: float | None = None
    scale_out_high: float | None = None
    scale_mean: float = 1.0
    scale_std: float = 0.0
    scale_low: float | None = None
    scale_high: float | None = None
    shift_mean: float = 0.0
    shift_std: float = 0.0
    shift_low: float | None = None
    shift_high: float | None = None
    permute: bool = False
    mask_value: Any = None

    def __post_init__(self) -> None:
        modality_type = self.type.lower()
        if modality_type not in ("discrete", "linear", "image"):
            raise ValueError(
                f"unknown augment type {self.type!r} for modality {self.input_field!r}; "
                "expected one of ('discrete', 'linear', 'image')."
            )
        object.__setattr__(self, "type", modality_type)
        in_f = (self.input_field,) if isinstance(self.input_field, str) else tuple(self.input_field)
        out_f = (self.output_field,) if isinstance(self.output_field, str) else tuple(self.output_field)
        if len(in_f) != len(out_f):
            raise ValueError(
                f"modality {self.input_field!r}: input_field/output_field arity mismatch"
            )
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError(f"mask_prob for modality {self.input_field!r} must be in [0, 1], got {self.mask_prob}.")
        if self.permute:
            if modality_type != "discrete":
                raise ValueError(f"modality {self.input_field!r}: permute=True requires type='discrete'.")
            if self.vocab_size is None or self.vocab_size <= 0:
                raise ValueError(f"modality {self.input_field!r}: vocab_size must be positive when permute=True.")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError(f"modality {self.input_field!r}: vocab_size must be positive.")
        self.linear_transform()
        self.scale_spec()
        self.shift_spec()
        if self._uses_direct_scale_shift():
            if self.type == "linear":
                raise ValueError(
                    f"modality {self.input_field!r}: linear augmentation uses scale_in_low, "
                    "scale_out_low, scale_in_high, and scale_out_high."
                )
            if self.type == "discrete":
                raise ValueError(
                    f"modality {self.input_field!r}: scale/shift parameters only apply to "
                    "type='image'; discrete modalities support permute, mask_prob, "
                    "and mask_value."
                )

    @property
    def input_fields(self) -> tuple[str, ...]:
        if isinstance(self.input_field, str):
            return (self.input_field,)
        return tuple(self.input_field)

    @property
    def output_fields(self) -> tuple[str, ...]:
        if isinstance(self.output_field, str):
            return (self.output_field,)
        return tuple(self.output_field)

    def scale_spec(self) -> _ScalarDraw:
        return _ScalarDraw(self.scale_mean, self.scale_std, self.scale_low, self.scale_high)

    def shift_spec(self) -> _ScalarDraw:
        return _ScalarDraw(self.shift_mean, self.shift_std, self.shift_low, self.shift_high)

    def _uses_direct_scale_shift(self) -> bool:
        return (
            self.scale_mean != 1.0
            or self.scale_std != 0.0
            or self.scale_low is not None
            or self.scale_high is not None
            or self.shift_mean != 0.0
            or self.shift_std != 0.0
            or self.shift_low is not None
            or self.shift_high is not None
        )

    def linear_transform(self) -> tuple[float, float]:
        endpoints = (
            self.scale_in_low,
            self.scale_out_low,
            self.scale_in_high,
            self.scale_out_high,
        )
        if all(value is None for value in endpoints):
            return (1.0, 0.0)
        if self.type != "linear":
            raise ValueError(
                f"modality {self.input_field!r}: scale_in_*/scale_out_* endpoints require type='linear'."
            )
        if any(value is None for value in endpoints):
            raise ValueError(
                f"modality {self.input_field!r}: set scale_in_low, scale_out_low, scale_in_high, "
                "and scale_out_high together."
            )
        scale_in_low = self.scale_in_low
        scale_out_low = self.scale_out_low
        scale_in_high = self.scale_in_high
        scale_out_high = self.scale_out_high
        assert scale_in_low is not None
        assert scale_out_low is not None
        assert scale_in_high is not None
        assert scale_out_high is not None
        in_low = float(scale_in_low)
        out_low = float(scale_out_low)
        in_high = float(scale_in_high)
        out_high = float(scale_out_high)
        if in_low == in_high:
            raise ValueError(f"modality {self.input_field!r}: scale_in_low and scale_in_high must differ.")
        scale = (out_high - out_low) / (in_high - in_low)
        shift = out_low - in_low * scale
        return (scale, shift)

    def is_active(self) -> bool:
        linear_scale, linear_shift = self.linear_transform()
        return (
            self.permute
            or self.mask_prob > 0.0
            or linear_scale != 1.0
            or linear_shift != 0.0
            or (self.type == "image" and self.scale_spec().is_active(1.0))
            or (self.type == "image" and self.shift_spec().is_active(0.0))
        )


class Augmenter:
    """Callable augmenter for one step dict.

    ``seed_field`` is required: permute/scale/shift draws are keyed by that
    field's value so steps that share an id (e.g. the same task) share draws
    within one :meth:`reseed` generation. Call :meth:`reseed` to advance to a
    new draw set for every key. Mask decisions (``mask_prob``) are drawn
    independently per step (one draw per modality per call), from a stream
    seeded by the base seed and generation.
    """

    def __init__(
        self,
        *,
        modalities: Sequence[Mapping[str, Any] | SequenceAugmentModalitySpec],
        seed_field: str,
        enabled: bool = True,
        seed: int | None = None,
    ) -> None:
        self.enabled = enabled
        self.modalities = tuple(_coerce_modality(spec) for spec in modalities)
        self.seed_field = seed_field
        self._base_seed = seed
        self._generation = 0
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._draw_cache: dict[tuple[int, Any], dict[int, dict[str, Any]]] = {}

    def __call__(self, step: dict, *, augment: bool = True) -> dict:
        """Return an augmented copy of ``step``.

        Pass ``augment=False`` to skip augmentation and return ``step``
        unchanged (the eval / decode path; no reseeding needed).
        """
        if not augment:
            return step
        if not (self.enabled and any(spec.is_active() for spec in self.modalities)):
            return step
        row = dict(step)
        draws = self._draws_for_step(row)
        for index, spec in enumerate(self.modalities):
            draw = draws[index]
            mask_this_step = self._sample_mask(spec)
            for in_f, out_f in zip(spec.input_fields, spec.output_fields):
                if in_f not in row:
                    continue
                value = row[in_f]
                value = self._apply_permutation(spec, draw, value)
                value = self._apply_scale_shift(spec, draw, value)
                row[out_f] = self._mask_or_value(spec, value, mask_this_step)
        return row

    def reseed(self, seed: int | None = None) -> None:
        """Advance to a new draw set for every ``seed_field`` key.

        Within one generation, steps that share the seed-field value share
        permute/scale/shift draws. After ``reseed()``, those keys get a fresh
        set and the cached draws from previous generations are discarded (a
        thread still on an older pinned generation recomputes them on demand).
        Pins the new generation on the calling thread so a batch or eval run
        is not interrupted by another thread's reseed.

        If ``seed`` is given, replaces the base seed, resets the generation
        counter, then advances once.
        """
        with self._lock:
            if seed is not None:
                self._base_seed = int(seed)
                self._generation = 0
            self._generation += 1
            gen = self._generation
            self._draw_cache = {}
        self._tls.generation = gen

    def fork(self, *, seed: int | None = None) -> Augmenter:
        """Create an equivalent augmenter with a different base seed."""
        return Augmenter(
            enabled=self.enabled,
            seed=seed,
            modalities=self.modalities,
            seed_field=self.seed_field,
        )

    def _generation_for_call(self) -> int:
        pinned = getattr(self._tls, "generation", None)
        if pinned is not None:
            return int(pinned)
        return self._generation

    def _draws_for_step(self, step: dict) -> dict[int, dict[str, Any]]:
        if self.seed_field not in step:
            raise KeyError(
                f"Augmenter seed_field {self.seed_field!r} missing from step "
                f"(have {sorted(step)})"
            )
        key = step[self.seed_field]
        if hasattr(key, "item"):
            try:
                key = key.item()
            except Exception:
                pass
        generation = self._generation_for_call()
        cache_key = (generation, key)
        cached = self._draw_cache.get(cache_key)
        if cached is not None:
            return cached
        base = 0 if self._base_seed is None else int(self._base_seed)
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [base, generation, _stable_hash(self.seed_field, key)]
            )
        )
        draws = {
            index: self._draw_modality(spec, rng)
            for index, spec in enumerate(self.modalities)
        }
        self._draw_cache[cache_key] = draws
        return draws

    def _draw_modality(
        self,
        spec: SequenceAugmentModalitySpec,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if spec.permute:
            assert spec.vocab_size is not None
            perm = rng.permutation(spec.vocab_size)
        else:
            perm = None
        scale_spec = spec.scale_spec()
        shift_spec = spec.shift_spec()
        linear_scale, linear_shift = spec.linear_transform()
        return {
            "perm": perm,
            "inverse_perm": _inverse_permutation(perm) if perm is not None else None,
            "scale": (
                linear_scale
                if spec.type == "linear"
                else scale_spec.sample(rng)
                if scale_spec.is_active(1.0)
                else 1.0
            ),
            "shift": (
                linear_shift
                if spec.type == "linear"
                else shift_spec.sample(rng)
                if shift_spec.is_active(0.0)
                else 0.0
            ),
        }

    def _apply_permutation(self, spec: SequenceAugmentModalitySpec, draw: dict[str, Any], value: Any) -> Any:
        perm = draw["perm"]
        if perm is None:
            return value
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[-1] == len(perm):
            return _permute_action_values(value, draw["inverse_perm"])
        idx = int(value)
        if idx < 0 or idx >= len(perm):
            raise ValueError(f"Cannot permute {spec.input_field!r} value {idx}; expected it in [0, {len(perm)}).")
        return int(perm[idx])

    def _apply_scale_shift(self, spec: SequenceAugmentModalitySpec, draw: dict[str, Any], value: Any) -> Any:
        scale = float(draw["scale"])
        shift = float(draw["shift"])
        if scale == 1.0 and shift == 0.0:
            return _copy_value(value)
        if spec.type == "image":
            return _scale_shift_image(value, scale, shift)
        if spec.type == "linear":
            return _scale_shift_value(value, scale, shift)
        return value

    def _mask_rng(self) -> np.random.Generator:
        """Per-thread mask stream, restarted at each generation.

        Mask decisions are per step: each call draws the next value from this
        stream, so steps sharing a ``seed_field`` key still mask independently.
        """
        generation = self._generation_for_call()
        rng = getattr(self._tls, "mask_rng", None)
        if rng is None or getattr(self._tls, "mask_rng_generation", None) != generation:
            base = 0 if self._base_seed is None else int(self._base_seed)
            rng = np.random.default_rng(
                np.random.SeedSequence([base, generation, _stable_hash("mask")])
            )
            self._tls.mask_rng = rng
            self._tls.mask_rng_generation = generation
        return rng

    def _sample_mask(self, spec: SequenceAugmentModalitySpec) -> bool:
        if spec.mask_prob <= 0.0:
            return False
        return bool(self._mask_rng().random() < spec.mask_prob)

    def _mask_or_value(
        self, spec: SequenceAugmentModalitySpec, value: Any, mask: bool
    ) -> Any:
        if not mask:
            return value
        if spec.mask_value is not None:
            return _copy_value(spec.mask_value)
        return _zero_like(value)


def _coerce_modality(spec: Mapping[str, Any] | SequenceAugmentModalitySpec) -> SequenceAugmentModalitySpec:
    if isinstance(spec, SequenceAugmentModalitySpec):
        return spec
    data = dict(spec)
    if "field" in data and "input_field" not in data:
        raise TypeError(
            "Augmenter modalities use input_field=/output_field= (not field=)"
        )
    return SequenceAugmentModalitySpec(**data)


def _scale_shift_value(value: Any, scale: float, shift: float) -> Any:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr) * scale + shift
    return (arr.astype(np.float64) * scale + shift).tolist()


def _scale_shift_image(value: Any, scale: float, shift: float) -> Any:
    arr = np.asarray(value)
    scaled = np.rint(arr.astype(np.float64) * scale + shift).clip(0, 255).astype(np.int64)
    if scaled.ndim == 0:
        return int(scaled)
    return scaled.tolist()


def _zero_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.zeros_like(value).tolist()
    if isinstance(value, Mapping):
        return {key: _zero_like(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_zero_like(item) for item in value]
    if isinstance(value, float):
        return 0.0
    return 0


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy().tolist()
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_copy_value(item) for item in value]
    return value


def _inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    return inverse


def _permute_action_values(value: Any, inverse_perm: np.ndarray) -> Any:
    arr = np.asarray(value)
    if arr.shape[-1] != len(inverse_perm):
        raise ValueError(
            f"Cannot permute action values with width {arr.shape[-1]}; expected {len(inverse_perm)} values."
        )
    out = np.take(arr, inverse_perm, axis=-1)
    if out.ndim == 0:
        return float(out)
    return out.tolist()
