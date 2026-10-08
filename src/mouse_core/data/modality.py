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

# Shared text stream. ``type="text"`` / ``type="token"`` fields emit here.
# Group-start ``when`` fields are tokenized into the same modality.
NAME_TEXT = "__text__"
# Shared numeric stream. ``type="numeric"`` fields emit here. The token
# ids are still vocab rows; ``values`` carries the scaled scalar.
NAME_NUMERIC = "__numeric__"

# Step-backed ``text`` ``format=`` is an f-string. The step value is the
# name ``field`` (``"{field+1}"``, ``"{field:.0f}"``). Consts name nothing.
TEXT_FORMAT_KEY = "field"
_FSTRING_BUILTINS = {"str": str, "repr": repr, "ascii": ascii}

KIND_TEXT = "text"
KIND_TOKEN = "token"
KIND_IMAGE = "image"
KIND_NUMERIC = "numeric"

# Per-step emission gate. See ``mouse_core.data.conditions``.
WhenFn = Callable[[Mapping[str, Any]], bool]


@dataclass
class TokenizerModalitySpec:
    """How the tokenizer packs one field from a step dict.

    Types:
      * ``text`` — format string → HF tokenize → ``__text__``
      * ``token`` — literal ``format`` → one ``__text__`` token. More
        than one token raises
      * ``image`` — image tokenizer → discrete visual token ids
      * ``numeric`` — literal ``format`` → one HF token → ``__numeric__``.
        The number is stored on ``values`` for a Fourier add

    ``text`` and ``token`` share the ``__text__`` stream. Every
    ``numeric`` field shares ``__numeric__``. ``image`` uses
    ``input_field`` as the modality name.

    A ``text`` field requires ``format=``. With ``input_field=``,
    ``format=`` is an f-string and the step value is the name ``field``:
    ``{field}``, ``{field+1}``, ``{field:.0f}``. Expressions may use
    ``field`` with operators and format specs. They may not call
    functions or use any other name. Omit ``input_field=`` and the
    field is a const: ``format=`` is the literal string to tokenize
    (the word field is text; ``{field}`` is still a placeholder and is
    rejected). Write ``{{`` / ``}}`` for a literal brace. ``image``
    does not accept ``format=``.

    A ``token`` field has no ``input_field``. ``format=`` is the
    literal text to tokenize, and it must be one token or the
    tokenizer raises. It rejects ``max_tokens``, ``input_index``,
    and Fourier bounds.

    A ``numeric`` field requires ``input_field=``, ``format=``,
    ``fourier_min=``, and ``fourier_max=``. ``format=`` is the literal
    text to tokenize. The word field is text. ``{field}`` is still a
    placeholder and is rejected. The number is not written into that
    text. It is mapped from ``[fourier_min, fourier_max]`` onto
    ``[-1, 1]`` (no clipping) and stored on every token the format
    produced. That format must tokenize to one id; zero or several
    raise. ``fourier_min`` and ``fourier_max`` must be finite and
    must differ. Other types reject those two arguments. The backbone
    adds a Fourier projection of the stored value when it is built
    with ``num_frequencies``. ``numeric`` rejects ``max_tokens``
    because the limit is one token.


    Omit ``input_index`` and the text or numeric field is that whole
    scalar. Set ``input_index`` (an int ``>= 0``) and the field reads
    that element of a 1-D vector (``list``, ``tuple``, ``ndarray``, or
    ``Tensor``). Repeat ``input_field`` with a different
    ``input_index`` for each element; fields still emit in
    ``input_fields`` order. A 1-D vector without ``input_index``
    raises, a scalar with ``input_index`` raises, and rank 2 or
    higher raises. ``token``, ``image``, and const text reject
    ``input_index``.

    Optional ``when=`` is a callable ``ctx → bool``. The tokenizer builds
    ``ctx`` from the step dict and injects boolean ``group_start``. It
    calls the predicate twice: with ``group_start=False`` for ordinary
    step tokens, and with ``group_start=True`` for pack-time
    ``group_start_*`` tokens. Write OR in the callable
    (``|`` / ``or``), e.g.
    ``lambda ctx: (ctx.get("step_index") == 0) | ctx["group_start"]``.
    Named callables round-trip via ``module:qualname``; define them
    inline in the notebook or caller. Omit ``when``
    (``None``) and the field always emits on the ordinary run.
    ``head_output`` fields cannot set ``when`` (they must emit on every
    step).

    ``required`` (default ``True``) means the step must carry
    ``input_field``; a missing / ``None`` value raises. With
    ``required=False`` a missing value emits nothing. Const text fields
    have no ``input_field``; they reject ``required=False``. Consts may
    still use ``when=``.

    Exactly one input field must set ``head_output=True``: its tokens are
    the step's **head-output tokens** — the positions the model reads Q /
    action outputs from. A step may emit several (e.g. a ``text`` run
    with more than one id); every step must emit at least one, so the
    head-output field must not be gated off on every step.

    Optional ``max_tokens=`` raises if that field emits more ids than
    the limit. Only ``text`` and ``image`` accept it. ``token`` and
    ``numeric`` tokenize to one id and raise otherwise; they reject
    ``max_tokens``.
    Every ``input_field`` must be unique, except text or numeric
    fields that share a column with a different ``input_index``. That pair
    ``(input_field, input_index)`` is the unique key, so the same
    pair twice raises. A const has no column, so several consts are
    fine. Fields emit in ``input_fields`` order.
    """

    type: str
    input_field: str | None = None
    input_index: int | None = None
    format: str | None = None
    fourier_min: float | None = None
    fourier_max: float | None = None
    _format_code: CodeType | None = field(default=None, init=False, repr=False, compare=False)
    max_tokens: int | None = None
    when: WhenFn | None = None
    required: bool = True
    head_output: bool = False

    _VALID_TYPES: ClassVar[tuple[str, ...]] = (
        "text",
        "token",
        "image",
        "numeric",
    )

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown tokenizer modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "text":
            self._reject_fourier_bounds(k)
            self._init_text()
            return
        if k == "numeric":
            self._init_numeric()
            return
        if k == "token":
            self._init_token()
            return
        self._reject_fourier_bounds(k)
        self._reject_input_index(k)
        self._init_named_input()
        self._reject_text_format(k)
        _validate_max_tokens(self)
        _validate_when(self)

    def _init_numeric(self) -> None:
        if not self.input_field:
            raise ValueError(
                "tokenizer modality type='numeric' requires input_field="
            )
        _validate_input_index(self)
        _validate_fourier_bounds(self)
        if not self.format:
            raise ValueError(
                f"numeric field {_field_label(self)!r} requires format= "
                "(a literal string, no placeholders)"
            )
        code = _compile_text_format(
            self.format, who=self.input_field, allow_field=False, literal="numeric"
        )
        object.__setattr__(self, "_format_code", code)
        self._reject_max_tokens("numeric")
        _validate_when(self)

    def _init_token(self) -> None:
        self._reject_fourier_bounds("token")
        self._reject_input_index("token")
        self._reject_max_tokens("token")
        self._reject_no_input_knobs("token")
        if not self.format:
            raise ValueError(
                "token field requires format= (a literal string, no placeholders)"
            )
        code = _compile_text_format(
            self.format, who="token", allow_field=False, literal="token"
        )
        object.__setattr__(self, "_format_code", code)
        _validate_when(self)

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
                "(text, token, and numeric only)"
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
                "(text and numeric fields with input_field= only)"
            )

    def _reject_fourier_bounds(self, kind: str) -> None:
        if self.fourier_min is not None or self.fourier_max is not None:
            raise TypeError(
                f"tokenizer modality type={kind!r} does not accept "
                "fourier_min= or fourier_max= (numeric only)"
            )

    def _reject_no_input_knobs(self, kind: str) -> None:
        """Fields with no ``input_field`` have no step value to require."""
        if self.input_field is not None:
            raise TypeError(f"{kind} tokenizer modalities have no input_field=")
        if self.required is not True:
            raise TypeError(
                f"{kind} tokenizer modalities do not accept required=False "
                "(no input_field)"
            )


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
        if others or TEXT_FORMAT_KEY not in names:
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
    local = {"field": field} if spec.input_field is not None else {}
    try:
        text = eval(  # noqa: S307 — format= is the caller's f-string
            spec._format_code,
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
) -> dict[str, Any]:
    """Copy ``input_field`` → ``output_field`` objective columns from a step.

    Used by :class:`~mouse_core.data.tokenizer.Tokenizer`. Every listed input
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


def _validate_fourier_bounds(spec: TokenizerModalitySpec) -> None:
    who = _field_label(spec)
    if spec.fourier_min is None or spec.fourier_max is None:
        raise ValueError(
            f"numeric field {who!r} requires fourier_min= and fourier_max="
        )
    lo = _as_finite_float(spec.fourier_min, who=who, name="fourier_min")
    hi = _as_finite_float(spec.fourier_max, who=who, name="fourier_max")
    if lo == hi:
        raise ValueError(
            f"numeric field {who!r} fourier_min and fourier_max must differ, "
            f"got {lo}"
        )
    object.__setattr__(spec, "fourier_min", lo)
    object.__setattr__(spec, "fourier_max", hi)


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
    name: str
    kind: str


def resolve_tokenizer_modalities(
    input_fields: Sequence[dict[str, Any] | TokenizerModalitySpec] | None = None,
) -> tuple[list[TokenizerModalitySpec], list[TokenizerModalityMeta]]:
    """Expand tokenizer input-field specs.

    ``text`` / ``token`` meta ``name`` is :data:`NAME_TEXT`.
    ``numeric`` meta ``name`` is :data:`NAME_NUMERIC`. ``image`` is
    keyed by ``input_field``.
    """
    raw = input_fields or []
    specs: list[TokenizerModalitySpec] = []
    for m in raw:
        if isinstance(m, TokenizerModalitySpec):
            spec = m
        else:
            spec = TokenizerModalitySpec(**dict(m))
        specs.extend(expand_tokenizer_spec(spec))

    meta: list[TokenizerModalityMeta] = []
    seen_names: set[str] = set()
    seen_indexed: set[tuple[str, int]] = set()
    indexed_names: set[str] = set()
    for spec in specs:
        k = spec.type
        if spec.input_field is not None:
            name = spec.input_field
            if spec.input_index is None:
                if name in seen_names or name in indexed_names:
                    raise ValueError(f"duplicate tokenizer field {name!r}")
                seen_names.add(name)
            else:
                key = (name, spec.input_index)
                if key in seen_indexed:
                    raise ValueError(
                        f"duplicate tokenizer field {name!r} with "
                        f"input_index={spec.input_index}; each (input_field, "
                        "input_index) pair must be unique"
                    )
                if name in seen_names:
                    raise ValueError(f"duplicate tokenizer field {name!r}")
                seen_indexed.add(key)
                indexed_names.add(name)
        if k == "text":
            meta.append(
                TokenizerModalityMeta(spec=spec, name=NAME_TEXT, kind=KIND_TEXT)
            )
        elif k == "token":
            meta.append(
                TokenizerModalityMeta(spec=spec, name=NAME_TEXT, kind=KIND_TOKEN)
            )
        elif k == "image":
            meta.append(
                TokenizerModalityMeta(
                    spec=spec, name=str(spec.input_field), kind=KIND_IMAGE
                )
            )
        elif k == "numeric":
            meta.append(
                TokenizerModalityMeta(
                    spec=spec, name=NAME_NUMERIC, kind=KIND_NUMERIC
                )
            )
        else:
            raise ValueError(f"unsupported modality type {k!r}")
    return specs, meta
