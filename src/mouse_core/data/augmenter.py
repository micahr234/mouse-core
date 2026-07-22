"""Training-time augmentation for raw sequence batches.

``Augmenter`` runs on the raw ``list[list[dict]]`` half of a DataLoader fetch
(before the encoder ``preparer`` builds a ``TokenBatch``). It operates on raw
step dicts and samples augmentation parameters independently for each sequence;
parallel ``segment_ids`` are left unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


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

    Each spec has a ``field`` and a ``type``. Augmentation ``type`` values
    describe the raw-data augmentation behavior, not the model's
    embedding implementation.
    """

    type: Literal["discrete", "linear", "image"]
    field: str | Sequence[str]
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
                f"unknown augment type {self.type!r} for modality {self.field!r}; "
                "expected one of ('discrete', 'linear', 'image')."
            )
        object.__setattr__(self, "type", modality_type)
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError(f"mask_prob for modality {self.field!r} must be in [0, 1], got {self.mask_prob}.")
        if self.permute:
            if modality_type != "discrete":
                raise ValueError(f"modality {self.field!r}: permute=True requires type='discrete'.")
            if self.vocab_size is None or self.vocab_size <= 0:
                raise ValueError(f"modality {self.field!r}: vocab_size must be positive when permute=True.")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError(f"modality {self.field!r}: vocab_size must be positive.")
        self.linear_transform()
        self.scale_spec()
        self.shift_spec()
        if self._uses_direct_scale_shift():
            if self.type == "linear":
                raise ValueError(
                    f"modality {self.field!r}: linear augmentation uses scale_in_low, "
                    "scale_out_low, scale_in_high, and scale_out_high."
                )
            if self.type == "discrete":
                raise ValueError(
                    f"modality {self.field!r}: scale/shift parameters only apply to "
                    "type='image'; discrete modalities support permute, mask_prob, "
                    "and mask_value."
                )

    @property
    def fields(self) -> tuple[str, ...]:
        if isinstance(self.field, str):
            return (self.field,)
        return tuple(self.field)

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
                f"modality {self.field!r}: scale_in_*/scale_out_* endpoints require type='linear'."
            )
        if any(value is None for value in endpoints):
            raise ValueError(
                f"modality {self.field!r}: set scale_in_low, scale_out_low, scale_in_high, "
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
            raise ValueError(f"modality {self.field!r}: scale_in_low and scale_in_high must differ.")
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
    """Callable augmenter for raw ``[batch][sequence]`` step batches."""

    def __init__(
        self,
        modalities: Sequence[Mapping[str, Any] | SequenceAugmentModalitySpec],
        *,
        enabled: bool = True,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
        keep_fields: Sequence[str] | None = None,
    ) -> None:
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both.")
        self.enabled = enabled
        self.modalities = tuple(_coerce_modality(spec) for spec in modalities)
        self._rng = generator if generator is not None else np.random.default_rng(seed)
        self.keep_fields: tuple[str, ...] | None = tuple(keep_fields) if keep_fields is not None else None

    def __call__(self, batch: list[list[dict]]) -> list[list[dict]]:
        """Return an augmented batch without mutating the sampled input batch."""

        augment = self.enabled and any(spec.is_active() for spec in self.modalities)
        keep = self.keep_fields is not None

        if not augment and not keep:
            return batch

        result = [self._augment_sequence(sequence) if augment else [dict(row) for row in sequence] for sequence in batch]

        if keep:
            keep_set = set(self.keep_fields)  # type: ignore[arg-type]
            result = [[{k: v for k, v in row.items() if k in keep_set} for row in seq] for seq in result]

        return result

    def fork(
        self,
        *,
        seed: int | None = None,
        generator: np.random.Generator | None = None,
    ) -> Augmenter:
        """Create an equivalent augmenter with independent RNG state."""

        return Augmenter(
            self.modalities,
            enabled=self.enabled,
            seed=seed,
            generator=generator,
            keep_fields=self.keep_fields,
        )

    def _augment_sequence(self, sequence: list[dict]) -> list[dict]:
        rows = [dict(row) for row in sequence]
        draws = {index: self._draw_modality(spec) for index, spec in enumerate(self.modalities)}

        for row in rows:
            for index, spec in enumerate(self.modalities):
                draw = draws[index]
                mask_this_step = self._sample_mask(spec)
                for field in spec.fields:
                    if field not in row:
                        continue
                    value = row[field]
                    value = self._apply_permutation(spec, draw, value)
                    value = self._apply_scale_shift(spec, draw, value)
                    row[field] = self._mask_or_value(spec, field, value, mask_this_step)

        return rows

    def _draw_modality(self, spec: SequenceAugmentModalitySpec) -> dict[str, Any]:
        if spec.permute:
            assert spec.vocab_size is not None
            perm = self._rng.permutation(spec.vocab_size)
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
                else scale_spec.sample(self._rng)
                if scale_spec.is_active(1.0)
                else 1.0
            ),
            "shift": (
                linear_shift
                if spec.type == "linear"
                else shift_spec.sample(self._rng)
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
            raise ValueError(f"Cannot permute {spec.field!r} value {idx}; expected it in [0, {len(perm)}).")
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

    def _sample_mask(self, spec: SequenceAugmentModalitySpec) -> bool:
        if spec.mask_prob <= 0.0:
            return False
        return bool(self._rng.random() < spec.mask_prob)

    def _mask_or_value(self, spec: SequenceAugmentModalitySpec, field: str, value: Any, mask: bool) -> Any:
        if not mask:
            return value
        if spec.mask_value is not None:
            return _copy_value(spec.mask_value)
        if field == "step_index":
            return -1
        return _zero_like(value)


def _coerce_modality(spec: Mapping[str, Any] | SequenceAugmentModalitySpec) -> SequenceAugmentModalitySpec:
    if isinstance(spec, SequenceAugmentModalitySpec):
        return spec
    return SequenceAugmentModalitySpec(**dict(spec))


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
