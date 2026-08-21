"""Tokenizer modality specs and packing helpers (not used by embedders)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import torch


def _reject_io_fields(data: dict[str, Any], *, who: str) -> None:
    if "input_field" in data or "output_field" in data:
        raise TypeError(
            f"{who} input_fields use field= (not input_field=/output_field=); "
            "rename with Selector before tokenize"
        )


@dataclass
class NumericTokenizerModalitySpec:
    """How the numeric tokenizer packs one modality from a step dict.

    Types:
      * ``discrete`` — integer id → one discrete token
      * ``fourier`` — scalar → one continuous token
      * ``continuous`` — vector → one continuous token per component
      * ``image`` — image tokenizer → discrete visual token ids
      * ``learnable`` — ``tokens`` scratch rows (no step I/O)

    ``field`` is both the step key and the modality name (fixed after Selector).
    """

    type: str
    field: str | Sequence[str] | None = None
    dim: int | None = None
    tokens: int | None = None
    skip: Any = None
    required: bool = True

    _VALID_TYPES: ClassVar[tuple[str, ...]] = (
        "discrete",
        "fourier",
        "continuous",
        "image",
        "learnable",
    )

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown modality type {self.type!r} for modality "
                f"{self.field!r}; expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "learnable":
            object.__setattr__(self, "required", False)
            return
        if self.field is None:
            raise ValueError(f"tokenizer modality type={k!r} requires field=")


@dataclass
class TextTokenizerModalitySpec:
    """Modality for :class:`~mouse_core.data.text_tokenizer.TextTokenizer`."""

    type: str
    field: str | Sequence[str] | None = None
    format: str | None = None
    skip: Any = None
    required: bool = True

    _VALID_TYPES: ClassVar[tuple[str, ...]] = ("text", "token", "image")

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown text tokenizer modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if self.field is None:
            raise ValueError(
                f"text tokenizer modality type={k!r} requires field="
            )
        if k == "text":
            if not self.format:
                raise ValueError(f"text modality {self.field!r} requires format=")
        elif k == "token" and self.format is not None:
            raise ValueError(
                f"token modality {self.field!r} must not set format= "
                "(the integer value selects embed_tokens[id] directly)"
            )


KIND_DISCRETE = "discrete"
KIND_FOURIER = "fourier"
KIND_LEARNABLE = "learnable"
KIND_IMAGE = "image"


def unwrap_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, torch.Tensor) and value.ndim == 0:
        return value.item()
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        try:
            if getattr(value, "ndim", None) == 0:
                return value.item()
        except Exception:
            pass
    return value


def values_equal(a: Any, b: Any) -> bool:
    return unwrap_scalar(a) == unwrap_scalar(b)


def field_names(field: str | Sequence[str] | None) -> tuple[str, ...]:
    if field is None:
        return ()
    if isinstance(field, str):
        return (field,)
    return tuple(field)


def expand_tokenizer_numeric_spec(
    spec: NumericTokenizerModalitySpec, *, learnable_index: int
) -> list[NumericTokenizerModalitySpec]:
    if spec.type == "learnable":
        name = f"__learnable_{learnable_index}"
        return [replace(spec, field=name)]
    names = field_names(spec.field)
    if not names:
        raise ValueError("input-backed tokenizer modalities must set field=")
    return [replace(spec, field=name) for name in names]


def expand_tokenizer_text_spec(
    spec: TextTokenizerModalitySpec,
) -> list[TextTokenizerModalitySpec]:
    names = field_names(spec.field)
    if not names:
        raise ValueError("text/token/image modalities must set field=")
    return [replace(spec, field=name) for name in names]


@dataclass(frozen=True)
class TokenizerModalityMeta:
    """Runtime metadata for one expanded numeric tokenizer modality."""

    spec: NumericTokenizerModalitySpec
    name: str
    kind: str
    dim: int = 0
    n_learnable: int = 0
    freq_sets: int = 1


def resolve_tokenizer_numeric_modalities(
    input_fields: list[dict[str, Any] | NumericTokenizerModalitySpec] | None = None,
) -> tuple[list[NumericTokenizerModalitySpec], list[TokenizerModalityMeta]]:
    """Expand tokenizer input-field specs; each slot is keyed by ``field`` name."""
    raw = input_fields or []
    specs: list[NumericTokenizerModalitySpec] = []
    for i, m in enumerate(raw):
        if isinstance(m, NumericTokenizerModalitySpec):
            spec = m
        else:
            data = dict(m)
            _reject_io_fields(data, who="tokenizer")
            spec = NumericTokenizerModalitySpec(**data)
        specs.extend(expand_tokenizer_numeric_spec(spec, learnable_index=i))

    meta: list[TokenizerModalityMeta] = []
    seen: set[str] = set()
    for spec in specs:
        assert isinstance(spec.field, str)
        name = str(spec.field)
        if name in seen:
            raise ValueError(f"duplicate tokenizer modality name {name!r}")
        seen.add(name)
        k = spec.type
        if k == "discrete":
            meta.append(
                TokenizerModalityMeta(spec=spec, name=name, kind=KIND_DISCRETE)
            )
        elif k in ("fourier", "continuous"):
            dim = 1 if k == "fourier" else int(spec.dim or 0)
            if dim <= 0:
                raise ValueError(
                    f"continuous modality {spec.field!r} requires dim="
                )
            meta.append(
                TokenizerModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_FOURIER,
                    dim=dim,
                    freq_sets=dim,
                )
            )
        elif k == "learnable":
            n = int(spec.tokens or 1)
            if n <= 0:
                raise ValueError("learnable tokens must be >= 1")
            meta.append(
                TokenizerModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_LEARNABLE,
                    n_learnable=n,
                )
            )
        elif k == "image":
            meta.append(
                TokenizerModalityMeta(spec=spec, name=name, kind=KIND_IMAGE)
            )
        else:
            raise ValueError(f"unsupported modality type {k!r}")
    return specs, meta
