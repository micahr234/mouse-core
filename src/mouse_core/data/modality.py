"""Tokenizer modality specs and packing helpers (not used by embedders)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import torch


@dataclass
class NumericTokenizerModalitySpec:
    """How the numeric tokenizer packs one modality from a step dict.

    Types:
      * ``discrete`` — integer id → one discrete token
      * ``fourier`` — scalar → one continuous token
      * ``continuous`` — vector → one continuous token per component
      * ``image`` — image tokenizer → discrete visual token ids
      * ``learnable`` — ``tokens`` scratch rows (no step I/O)

    ``input_field`` is the step key; ``output_field`` is the modality name
    (embedder alignment). Omitted ``output_field`` defaults to ``input_field``.
    Learnable modalities have no ``input_field``; set ``output_field`` to name
    them (e.g. ``"value"``), else they are auto-named ``__learnable_<i>``.

    Exactly one input field must set ``head_output=True``: its tokens are the
    step's **head-output tokens** — the positions the model reads Q / action
    outputs from. A step may emit several (e.g. ``learnable`` with
    ``tokens > 1``); every step must emit at least one, so the head-output
    field must not be skippable on any step.
    """

    type: str
    input_field: str | None = None
    output_field: str | None = None
    dim: int | None = None
    tokens: int | None = None
    skip: Any = None
    required: bool = True
    head_output: bool = False

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
                f"{self.output_field!r}; expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "learnable":
            object.__setattr__(self, "required", False)
            if self.input_field is not None:
                raise TypeError(
                    "learnable tokenizer modalities have no input_field="
                )
            return
        if not self.input_field:
            raise ValueError(
                f"tokenizer modality type={k!r} requires input_field="
            )
        if not self.output_field:
            object.__setattr__(self, "output_field", self.input_field)


@dataclass
class TextTokenizerModalitySpec:
    """Modality for :class:`~mouse_core.data.text_tokenizer.TextTokenizer`.

    Exactly one input field must set ``head_output=True``: the tokens it emits
    are the step's head-output tokens (Q / action readout positions). A
    ``text`` head-output field is tokenized as its own run so its token
    boundaries are exact; every step must emit at least one head-output token,
    so the head-output field must not be skippable on any step.

    A ``text`` field requires ``format=``. ``token`` must not set ``format``.
    ``learnable`` has no step I/O (no ``input_field=`` / ``format=``); set
    ``output_field`` to name it (e.g. ``"value"``), else it is auto-named
    ``__learnable_<i>``. ``tokens`` (default ``1``) is the number of scratch
    rows. Learnable tokens are appended after the rendered step format.
    """

    type: str
    input_field: str | None = None
    output_field: str | None = None
    format: str | None = None
    tokens: int | None = None
    skip: Any = None
    required: bool = True
    head_output: bool = False

    _VALID_TYPES: ClassVar[tuple[str, ...]] = ("text", "token", "image", "learnable")

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown text tokenizer modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "learnable":
            object.__setattr__(self, "required", False)
            if self.input_field is not None:
                raise TypeError(
                    "learnable tokenizer modalities have no input_field="
                )
            if self.format is not None:
                raise ValueError(
                    f"learnable modality {self.output_field!r} must not set format="
                )
            n = int(self.tokens or 1)
            if n <= 0:
                raise ValueError("learnable tokens must be >= 1")
            object.__setattr__(self, "tokens", n)
            return
        if self.tokens is not None:
            raise TypeError(
                f"text tokenizer modality type={k!r} does not accept tokens= "
                "(learnable only)"
            )
        if not self.input_field:
            raise ValueError(
                f"text tokenizer modality type={k!r} requires input_field="
            )
        if not self.output_field:
            object.__setattr__(self, "output_field", self.input_field)
        if k == "text":
            if not self.format:
                raise ValueError(
                    f"text modality {self.output_field!r} requires format="
                )
        elif k == "token" and self.format is not None:
            raise ValueError(
                f"token modality {self.output_field!r} must not set format= "
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
    """Scalar-or-array equality that always yields a Python ``bool``.

    Vectors (``list`` / ``ndarray`` / ``Tensor`` with ``ndim > 0``) compare
    elementwise against ``b`` (which may be a scalar broadcast over every
    element, or a same-shape vector); ``True`` only when every element matches.
    """
    a = unwrap_scalar(a)
    b = unwrap_scalar(b)
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()
    if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
        arr_a = np.asarray(a)
        arr_b = np.asarray(b)
        if arr_b.ndim > 0 and arr_a.shape != arr_b.shape:
            return False
        return bool(np.all(arr_a == arr_b))
    return bool(a == b)


def copy_keep_fields(
    row: dict,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Copy ``input_field`` → ``output_field`` objective columns from a step.

    Shared by :class:`~mouse_core.data.numeric_tokenizer.NumericTokenizer` and
    :class:`~mouse_core.data.text_tokenizer.TextTokenizer`. Every listed input
    must be present (and not ``None``) on the step: there is no silent default,
    since a missing objective column (``old_log_prob``, ``advantage``, …) would
    otherwise train on zeros. Vectors (any length, including 1) stay vectors;
    scalars become ``float`` / ``int``.
    """
    out: dict[str, Any] = {}
    for in_name, out_name in pairs:
        value = row.get(in_name)
        if value is None:
            raise KeyError(
                f"objective_fields input {in_name!r} is missing from step "
                f"(have {sorted(row)}); stamp it on the row before tokenizing"
            )
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)
        if isinstance(value, np.ndarray) and value.ndim > 0:
            if np.issubdtype(value.dtype, np.floating):
                out[out_name] = value.astype(np.float32).ravel()
            elif np.issubdtype(value.dtype, np.integer) or value.dtype == np.bool_:
                out[out_name] = value.astype(np.int64).ravel()
            else:
                raise TypeError(
                    f"objective_fields input {in_name!r} must be numeric, got dtype {value.dtype}"
                )
            continue
        sample = unwrap_scalar(value)
        if isinstance(sample, (float, np.floating)):
            out[out_name] = float(sample)
        elif isinstance(sample, (bool, int, np.integer)):
            out[out_name] = int(sample)
        else:
            raise TypeError(
                f"objective_fields input {in_name!r} must be numeric, got {type(sample).__name__}"
            )
    return out


def expand_tokenizer_numeric_spec(
    spec: NumericTokenizerModalitySpec, *, learnable_index: int
) -> list[NumericTokenizerModalitySpec]:
    """``learnable_index`` names anonymous learnables (ordinal among learnable
    specs, matching the embedder); an explicit ``output_field`` is kept."""
    if spec.type == "learnable":
        if spec.output_field:
            return [spec]
        return [replace(spec, output_field=f"__learnable_{learnable_index}")]
    if not spec.input_field or not spec.output_field:
        raise ValueError(
            "input-backed tokenizer modalities must set input_field="
        )
    return [spec]


def expand_tokenizer_text_spec(
    spec: TextTokenizerModalitySpec, *, learnable_index: int = 0
) -> list[TextTokenizerModalitySpec]:
    if spec.type == "learnable":
        if spec.output_field:
            return [spec]
        return [replace(spec, output_field=f"__learnable_{learnable_index}")]
    if not spec.input_field or not spec.output_field:
        raise ValueError("text/token/image modalities must set input_field=")
    return [spec]


@dataclass(frozen=True)
class TokenizerModalityMeta:
    """Runtime metadata for one expanded numeric tokenizer modality."""

    spec: NumericTokenizerModalitySpec
    name: str
    kind: str
    dim: int = 0
    n_learnable: int = 0


def resolve_tokenizer_numeric_modalities(
    input_fields: Sequence[dict[str, Any] | NumericTokenizerModalitySpec] | None = None,
) -> tuple[list[NumericTokenizerModalitySpec], list[TokenizerModalityMeta]]:
    """Expand tokenizer input-field specs; each slot is keyed by ``output_field``."""
    raw = input_fields or []
    specs: list[NumericTokenizerModalitySpec] = []
    n_learnable = 0
    for m in raw:
        if isinstance(m, NumericTokenizerModalitySpec):
            spec = m
        else:
            data = dict(m)
            spec = NumericTokenizerModalitySpec(**data)
        specs.extend(expand_tokenizer_numeric_spec(spec, learnable_index=n_learnable))
        if spec.type == "learnable":
            n_learnable += 1

    meta: list[TokenizerModalityMeta] = []
    seen: set[str] = set()
    for spec in specs:
        name = str(spec.output_field)
        if not name:
            raise ValueError("tokenizer modality is missing output_field=")
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
                    f"continuous modality {spec.output_field!r} requires dim="
                )
            meta.append(
                TokenizerModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_FOURIER,
                    dim=dim,
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
