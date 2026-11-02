"""Tokenizer modality specs and packing helpers."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, ClassVar, cast

import numpy as np
import torch

# Step-backed ``text`` ``format=`` is an f-string. The step value is the
# name ``field`` (``"{field+1}"``, ``"{field:.0f}"``). Consts name nothing.
TEXT_FORMAT_KEY = "field"
_FSTRING_BUILTINS = {"str": str, "repr": repr, "ascii": ascii}

KIND_TEXT = "text"
KIND_TOKEN = "token"
KIND_IMAGE = "image"

# Per-step emission gate. See ``mouse_core.data.conditions``.
WhenFn = Callable[[Mapping[str, Any]], bool]


@dataclass
class TokenizerModalitySpec:
    """Describe tokens from one concrete provider.

    ``text`` formats an optional input scalar into a variable-length token run.
    ``token`` uses the same formatting and lookup rules but must produce exactly
    one token. ``special=True`` registers a literal token in the vocabulary.
    ``image`` encodes an image into discrete visual tokens.

    ``input_index`` selects an element of a vector before formatting. A missing
    or None input emits nothing. Without input_field, text/token are constants.
    Optional ``when`` gates emission; ``head_output`` marks prediction tokens.

    Any type may set ``numeric_field`` and explicit finite ``fourier_min`` /
    ``fourier_max`` bounds (min < max). ``numeric_index`` optionally selects a
    vector element independently of input_index. The scalar is mapped to [-1,1]
    without clipping, and a projection specific to (provider, numeric_field) is
    added to every emitted token. Vector elements share their field's projection.

    Independently, ``position_field`` selects a scalar with explicit finite
    ``position_fourier_min`` / ``position_fourier_max`` bounds (min < max).
    Its projection is shared across all fields and named token tables. Positions
    also map to [-1,1] without clipping. Missing/None numeric or position values
    add nothing; zero and signed values are valid. Neither addition emits tokens.

    Each declaration requires a nonempty provider name; no wildcard is supported.
    Declarations emit in list order. Content (input_field, input_index) pairs must
    be unique within a provider, while constant declarations may repeat.
    """

    type: str
    provider: str
    tokenizer: str | None = None
    special: bool = False
    input_field: str | None = None
    input_index: int | None = None
    numeric_field: str | None = None
    numeric_index: int | None = None
    position_field: str | None = None
    position_fourier_min: float | None = None
    position_fourier_max: float | None = None
    lookup: Sequence[str] | None = None
    format: str | None = None
    fourier_min: float | None = None
    fourier_max: float | None = None
    _format_code: CodeType | None = field(default=None, init=False, repr=False, compare=False)
    max_tokens: int | None = None
    when: WhenFn | None = None
    head_output: bool = False

    _VALID_TYPES: ClassVar[tuple[str, ...]] = (
        "text",
        "token",
        "image",
    )

    def __post_init__(self) -> None:
        if self.numeric_field is None:
            if self.fourier_min is not None or self.fourier_max is not None or self.numeric_index is not None:
                raise ValueError("Numeric bounds and numeric_index require numeric_field")
        else:
            if not isinstance(self.numeric_field, str) or not self.numeric_field.strip():
                raise ValueError("numeric_field must be a nonempty payload field name")
            if self.numeric_index is not None:
                if isinstance(self.numeric_index, bool) or not isinstance(self.numeric_index, int):
                    raise TypeError("numeric_index must be an int")
                if self.numeric_index < 0:
                    raise ValueError("numeric_index must be >= 0")
            if self.fourier_min is None or self.fourier_max is None:
                raise ValueError("numeric_field requires fourier_min and fourier_max")
            lo = _as_finite_float(self.fourier_min, who=self.numeric_field, name="fourier_min")
            hi = _as_finite_float(self.fourier_max, who=self.numeric_field, name="fourier_max")
            if not lo < hi or not math.isfinite(hi - lo):
                raise ValueError("Numeric bounds require a finite positive range (min < max)")
            self.fourier_min, self.fourier_max = lo, hi
        if self.position_field is not None and (not isinstance(self.position_field, str) or not self.position_field):
            raise ValueError("position_field must be a nonempty payload field name")
        if self.position_field is None:
            if self.position_fourier_min is not None or self.position_fourier_max is not None:
                raise ValueError("Position bounds require position_field")
        else:
            if self.position_fourier_min is None or self.position_fourier_max is None:
                raise ValueError("position_field requires position_fourier_min and position_fourier_max")
            lo = _as_finite_float(self.position_fourier_min, who=self.position_field, name="position_fourier_min")
            hi = _as_finite_float(self.position_fourier_max, who=self.position_field, name="position_fourier_max")
            if not lo < hi or not math.isfinite(hi - lo):
                raise ValueError("Position bounds require a finite positive range (min < max)")
            self.position_fourier_min, self.position_fourier_max = lo, hi
        if self.tokenizer is not None and (not isinstance(self.tokenizer, str) or not self.tokenizer):
            raise ValueError("tokenizer must be a nonempty backend name")
        if type(self.special) is not bool or (self.special and (self.type or "").lower() != "token"):
            raise ValueError("special=True is supported only for token fields")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a specific nonempty string")
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown tokenizer modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if self.lookup is not None:
            if k not in ("text", "token") or not self.input_field:
                raise TypeError("lookup requires a text or token field with input_field=")
            if not isinstance(self.lookup, (list, tuple)) or not self.lookup:
                raise ValueError("lookup must be a nonempty list or tuple of strings")
            if any(not isinstance(value, str) or not value for value in self.lookup):
                raise ValueError("lookup entries must be nonempty strings")
            object.__setattr__(self, "lookup", tuple(self.lookup))
        if k == "text":
            self._init_text()
            return
        if k == "token":
            self._init_token()
            return
        self._reject_input_index(k)
        self._init_named_input()
        self._reject_text_format(k)
        _validate_max_tokens(self)
        _validate_when(self)

    def _init_token(self) -> None:
        self._reject_max_tokens("token")
        self._init_text()
        if self.special:
            self._format_code = _compile_text_format(
                self.format, who="special token", allow_field=False, literal="special token"
            )

    def _init_text(self) -> None:
        if self.input_field is None:
            self._reject_input_index("text const")
            self._reject_no_input_knobs("text const")
            if not self.format:
                raise ValueError(
                    "text const field requires format= "
                    "(a literal string, no placeholders)"
                )
            code = _compile_text_format(
                self.format, who="const", allow_field=False
            )
            object.__setattr__(self, "_format_code", code)
            _validate_max_tokens(self)
            _validate_when(self)
            return
        _validate_input_index(self)
        code = _compile_text_format(
            self.format, who=self.input_field, allow_field=True
        )
        object.__setattr__(self, "_format_code", code)
        _validate_max_tokens(self)
        _validate_when(self)

    def _init_named_input(self) -> None:
        if not self.input_field:
            raise ValueError(
                f"tokenizer modality type={self.type!r} requires input_field="
            )

    def _reject_text_format(self, kind: str) -> None:
        if self.format is not None:
            raise TypeError(
                f"tokenizer modality type={kind!r} does not accept format= "
                "(text and token only)"
            )

    def _reject_max_tokens(self, kind: str) -> None:
        if self.max_tokens is not None:
            raise TypeError(
                f"tokenizer modality type={kind!r} does not accept max_tokens= "
                "(it emits a fixed number of tokens; text / image only)"
            )

    def _reject_input_index(self, kind: str) -> None:
        if self.input_index is not None:
            raise TypeError(
                f"tokenizer modality type={kind!r} does not accept input_index= "
                "(text and token fields with input_field= only)"
            )

    def _reject_no_input_knobs(self, kind: str) -> None:
        """Reject input fields on constant modalities."""
        if self.input_field is not None:
            raise TypeError(f"{kind} tokenizer modalities have no input_field=")


def _compile_text_format(
    format_str: str | None,
    *,
    who: str | None,
    allow_field: bool,
    literal: str | None = None,
) -> CodeType:
    """Compile ``format=`` as an f-string whose only name is ``field``."""
    label = who or "text field"
    if not format_str:
        if literal is not None:
            raise ValueError(
                f"{literal} field {label!r} requires format= "
                "(a literal string, no placeholders)"
            )
        raise ValueError(f"text modality {label!r} requires format=")
    try:
        tree = ast.parse("f" + repr(format_str), mode="eval")
    except SyntaxError as exc:
        kind = f"{literal} field" if literal is not None else "text modality"
        raise ValueError(
            f"{kind} {label!r} format= is not an f-string ({exc.msg})"
        ) from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda)):
            if literal is not None:
                raise ValueError(
                    f"{literal} field {label!r} format= is a literal string "
                    "and must not contain placeholders"
                )
            raise ValueError(
                f"text modality {label!r} format= can use field in an "
                "expression such as {field+1} or {field:.3f}, not a call"
            )
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.append(node.id)
    if allow_field:
        others = sorted({name for name in names if name != TEXT_FORMAT_KEY})
        has_placeholders = any(isinstance(node, ast.FormattedValue) for node in ast.walk(tree))
        if others or has_placeholders and TEXT_FORMAT_KEY not in names:
            extra = f" Got {others}." if others else ""
            raise ValueError(
                f"text modality {label!r} format= is an f-string and the "
                f"step value is named {TEXT_FORMAT_KEY}, as in "
                f"{{{TEXT_FORMAT_KEY}+1}} or {{{TEXT_FORMAT_KEY}:.3f}}."
                f"{extra}"
            )
    elif names:
        if literal is not None:
            raise ValueError(
                f"{literal} field {label!r} format= is a literal string "
                "and must not contain placeholders"
            )
        raise ValueError(
            f"text const field {label!r} format= is "
            "a literal string and must not contain placeholders"
        )
    return compile(tree, "<tokenizer format>", "eval")


def render_text_format(spec: TokenizerModalitySpec, *, field: Any) -> str:
    """Fill ``spec.format`` as an f-string. ``field`` is the step value."""
    code = spec._format_code
    if code is None:
        raise ValueError(
            f"text modality {_field_label(spec)!r} format= was not compiled"
        )
    local = {"field": field} if spec.input_field is not None else {}
    try:
        text = eval(  # noqa: S307 — format= is the caller's f-string
            code,
            {"__builtins__": _FSTRING_BUILTINS},
            local,
        )
    except Exception as exc:
        raise ValueError(
            f"text modality {_field_label(spec)!r} format={spec.format!r} "
            f"failed: {exc}"
        ) from exc
    return text


def _validate_input_index(spec: TokenizerModalitySpec) -> None:
    index = spec.input_index
    if index is None:
        return
    who = _field_label(spec)
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(
            f"tokenizer modality {who!r} input_index must be an int, "
            f"got {type(index).__name__}"
        )
    if index < 0:
        raise ValueError(
            f"tokenizer modality {who!r} input_index must be >= 0, got {index}"
        )


def _validate_max_tokens(spec: TokenizerModalitySpec) -> None:
    if spec.max_tokens is None:
        return
    n = int(spec.max_tokens)
    if n <= 0:
        raise ValueError(
            f"tokenizer modality {_field_label(spec)!r} max_tokens must be >= 1"
        )
    object.__setattr__(spec, "max_tokens", n)


def _normalize_when(when: Any, *, name: str) -> WhenFn | None:
    """Normalize ``when=`` to a callable, or ``None``.

    Accepts a callable, an import-path string (``module:qualname`` from
    JSON), or ``None``. Dict-style equals / not_equals / group_start
    gates are rejected.
    """
    if when is None:
        return None
    if isinstance(when, str):
        from mouse_core.data.conditions import resolve_when_ref

        when = resolve_when_ref(when)
    if not callable(when):
        raise TypeError(
            f"tokenizer modality {name!r} when= must be a callable "
            f"(ctx → bool), an import path string, or None; got "
            f"{type(when).__name__}"
        )
    return cast(WhenFn, when)


def _validate_when(spec: TokenizerModalitySpec) -> None:
    name = _field_label(spec)
    normalized = _normalize_when(spec.when, name=name)
    object.__setattr__(spec, "when", normalized)
    if normalized is None:
        return
    if spec.head_output:
        raise ValueError(
            f"tokenizer modality {name!r} is head_output and cannot set "
            "when= (that field must emit on every step)"
        )


def when_as_bool(value: Any) -> bool:
    """Coerce a ``when`` return value to a Python ``bool``."""
    if isinstance(value, np.ndarray):
        if value.shape != ():
            raise TypeError(
                "tokenizer when= must return a scalar bool, got array "
                f"with shape {value.shape}"
            )
        return bool(value.item())
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError(
                "tokenizer when= must return a scalar bool, got tensor "
                f"with shape {tuple(value.shape)}"
            )
        return bool(value.item())
    return bool(value)


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
    *,
    missing_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy ``input_field`` → ``output_field`` objective columns from a step.

    Used by :class:`~mouse_core.data.tokenizer.Tokenizer`. Each input must be
    present or have an explicitly configured missing value. This prevents an
    absent objective column (``old_log_prob``, ``advantage``, …) from silently
    training on zeros. Vectors (any length, including 1) stay vectors;
    scalars become ``float`` / ``int``. Only the ``action`` output may be
    ``None``, meaning no action ran. Packing derives its validity mask.
    """
    out: dict[str, Any] = {}
    for in_name, out_name in pairs:
        if in_name in row:
            value = row[in_name]
        elif out_name in missing_values:
            value = missing_values[out_name]
        else:
            raise KeyError(f"objective_fields input {in_name!r} is missing from step")
        if value is None and out_name != "action":
            raise KeyError(
                f"objective_fields input {in_name!r} is missing from step "
                f"(have {sorted(row)}); stamp it on the row before tokenizing"
            )
        if value is None:
            out[out_name] = None
            continue
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


def _as_finite_float(value: Any, *, who: str, name: str) -> float:
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"numeric field {who!r} {name} must be a real number, "
            f"got {type(value).__name__}"
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"numeric field {who!r} {name} must be a real number, "
            f"got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"numeric field {who!r} {name} must be finite, got {value!r}")
    return number



def _field_label(spec: TokenizerModalitySpec) -> str:
    """Name used in errors. A const has no column."""
    if spec.input_field is None:
        return "const"
    if spec.input_index is None:
        return spec.input_field
    return f"{spec.input_field}[{spec.input_index}]"


def expand_tokenizer_spec(spec: TokenizerModalitySpec) -> list[TokenizerModalitySpec]:
    if spec.type in ("text", "token") and spec.input_field is None:
        return [spec]
    if not spec.input_field:
        raise ValueError(
            "input-backed tokenizer modalities must set input_field="
        )
    return [spec]


@dataclass(frozen=True)
class TokenizerModalityMeta:
    """Runtime metadata for one expanded tokenizer field."""

    spec: TokenizerModalitySpec
    kind: str


def resolve_tokenizer_modalities(
    input_fields: Sequence[dict[str, Any] | TokenizerModalitySpec] | None = None,
) -> tuple[list[TokenizerModalitySpec], list[TokenizerModalityMeta]]:
    """Validate tokenizer input-field specs and resolve their emission types."""
    raw = input_fields or []
    specs: list[TokenizerModalitySpec] = []
    for m in raw:
        if isinstance(m, TokenizerModalitySpec):
            spec = m
        else:
            spec = TokenizerModalitySpec(**dict(m))
        specs.extend(expand_tokenizer_spec(spec))

    meta: list[TokenizerModalityMeta] = []
    seen_names: set[tuple[str, str]] = set()
    seen_indexed: set[tuple[tuple[str, str], int]] = set()
    indexed_names: set[tuple[str, str]] = set()
    for spec in specs:
        k = spec.type
        if spec.input_field is not None:
            name = (spec.provider, spec.input_field)
            if spec.input_index is None:
                if name in seen_names or name in indexed_names:
                    raise ValueError(f"duplicate tokenizer field {spec.input_field!r} for provider {spec.provider!r}")
                seen_names.add(name)
            else:
                key = (name, spec.input_index)
                if key in seen_indexed:
                    raise ValueError(
                        f"duplicate tokenizer field {spec.input_field!r} with "
                        f"input_index={spec.input_index}; each (input_field, "
                        "input_index) pair must be unique"
                    )
                if name in seen_names:
                    raise ValueError(f"duplicate tokenizer field {spec.input_field!r} for provider {spec.provider!r}")
                seen_indexed.add(key)
                indexed_names.add(name)
        if k == "text":
            meta.append(
                TokenizerModalityMeta(spec=spec, kind=KIND_TEXT)
            )
        elif k == "token":
            meta.append(
                TokenizerModalityMeta(spec=spec, kind=KIND_TOKEN)
            )
        elif k == "image":
            meta.append(
                TokenizerModalityMeta(
                    spec=spec, kind=KIND_IMAGE
                )
            )
        else:
            raise ValueError(f"unsupported modality type {k!r}")
    return specs, meta
