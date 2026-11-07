"""Tokenizer modality specs and packing helpers."""

from __future__ import annotations

import ast
import builtins
import keyword
import math
import symtable
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import CodeType
from typing import Any, ClassVar

import numpy as np
import torch

from mouse_core.data.callbacks import accepts_context

KIND_TEXT = "text"
KIND_TOKEN = "token"
KIND_IMAGE = "image"


@dataclass
class TokenizerModalitySpec:
    """An ordered token emission with optional provider and payload filters.

    Omitted provider matches every provider. ``input_field`` may be a name or
    a list of names; the declaration runs when any selected value is present
    and non-None. Omit it to expose the entire payload and run on every event.
    Missing selected names are None. False, zero and empty strings are present.

    Text/token formats are f-string expressions using payload names directly.
    ``format_functions={"action_name": action_name}`` enables explicit calls
    such as ``{action_name(action)}``. Expressions use normal Python builtins,
    attribute access, comprehensions and lexical scoping.
    Callable formats receive a payload copy, with selected values under their
    original names, and may opt into separate runtime settings with ``format(payload, *, context)``.
    ``input_index`` selects a scalar for a single named input. Images encode
    each present selected input in selection order (payload order if omitted).
    Token fields emit exactly one token unless their formatted text is empty.
    ``special=True`` registers a literal token. ``head_output`` marks outputs.

    Numeric and position additions use independent payload fields and explicit
    finite Fourier bounds (min < max), scaled to [-1,1] without clipping.
    Missing additions contribute nothing. Numeric projections are specific to
    (provider filter, numeric_field); positions share a projection.
    Filters may overlap or repeat; declarations always emit in list order.
    """

    type: str
    provider: str | None = None
    tokenizer: str | None = None
    special: bool = False
    input_field: str | Sequence[str] | None = None
    input_index: int | None = None
    numeric_field: str | None = None
    numeric_index: int | None = None
    position_field: str | None = None
    position_fourier_min: float | None = None
    position_fourier_max: float | None = None
    format_functions: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    format: str | Callable[..., str] | None = None
    fourier_min: float | None = None
    fourier_max: float | None = None
    _format_code: CodeType | None = field(default=None, init=False, repr=False, compare=False)
    _format_context: bool = field(default=False, init=False, repr=False, compare=False)
    max_tokens: int | None = None
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
        if self.provider is not None and (not isinstance(self.provider, str) or not self.provider.strip()):
            raise ValueError("provider must be a nonempty string or None")
        if self.input_field is not None:
            names = (self.input_field,) if isinstance(self.input_field, str) else self.input_field
            if not isinstance(names, (list, tuple)) or not names:
                raise ValueError("input_field must be a nonempty name or list of names, or None")
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError("input_field names must be nonempty strings")
            if len(set(names)) != len(names):
                raise ValueError("input_field names must be unique")
            if not isinstance(self.input_field, str):
                self.input_field = tuple(names)
        if self.input_index is not None and not isinstance(self.input_field, str):
            raise ValueError("input_index requires a single input_field name")
        if not isinstance(self.format_functions, Mapping):
            raise TypeError("format_functions must map names to callables")
        for name, function in self.format_functions.items():
            if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name) or name.startswith("__"):
                raise ValueError("format_functions names must be public Python identifiers")
            if not callable(function):
                raise TypeError("format_functions values must be callable")
            if name in self.input_names:
                raise ValueError(f"Format function {name!r} conflicts with an input field")
        self.format_functions = dict(self.format_functions)
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown tokenizer modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "text":
            self._init_text()
            return
        if k == "token":
            self._init_token()
            return
        self._reject_input_index(k)
        self._reject_text_format(k)
        _validate_max_tokens(self)

    @property
    def input_names(self) -> tuple[str, ...]:
        if self.input_field is None:
            return ()
        return (self.input_field,) if isinstance(self.input_field, str) else tuple(self.input_field)

    def _init_token(self) -> None:
        self._reject_max_tokens("token")
        self._init_text()

    def _init_text(self) -> None:
        _validate_input_index(self)
        _validate_max_tokens(self)
        if callable(self.format):
            if self.special or self.format_functions:
                raise TypeError("Callable formats cannot use special or format_functions")
            self._format_context = accepts_context(self.format)
            return
        self._format_code = _compile_text_format(self)

    def _reject_text_format(self, kind: str) -> None:
        if self.format is not None or self.format_functions:
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


def _compile_text_format(spec: TokenizerModalitySpec) -> CodeType:
    """Compile a Python f-string over event fields, functions and builtins."""
    if not isinstance(spec.format, str):
        raise ValueError(f"text modality {_field_label(spec)!r} requires format= string or callable")
    # Raw triple quoting preserves both literal backslashes and escapes inside
    # replacement expressions. repr(format) would double escapes inside them.
    delimiter = next((quote * 3 for quote in ('"', "'") if quote * 3 not in spec.format), None)
    if delimiter is None:
        raise ValueError("format cannot contain both triple-quote delimiters")
    try:
        source = "rf" + delimiter + "\n" + spec.format + "\n" + delimiter
        tree = ast.parse(source, mode="eval")
        assert isinstance(tree.body, ast.JoinedStr)
        first, last = tree.body.values[0], tree.body.values[-1]
        assert isinstance(first, ast.Constant) and isinstance(last, ast.Constant)
        assert isinstance(first.value, str) and isinstance(last.value, str)
        first.value = first.value[1:]
        last.value = last.value[:-1]
    except SyntaxError as exc:
        raise ValueError(f"format= is not an f-string ({exc.msg})") from exc
    if spec.special:
        if not spec.format or any(isinstance(node, ast.FormattedValue) for node in ast.walk(tree)):
            raise ValueError("special token requires a nonempty literal string without placeholders")
        if spec.format_functions:
            raise ValueError("special token cannot use format_functions")
    if spec.input_field is not None:
        # Let Python distinguish external names from comprehension variables,
        # lambda parameters, closures and assignment-expression bindings.
        symbols = symtable.symtable(source, "<tokenizer format>", "eval")
        assigned = {symbol.get_name() for symbol in symbols.get_symbols() if symbol.is_assigned()}
        referenced = set()
        scopes = [symbols]
        while scopes:
            scope = scopes.pop()
            referenced.update(symbol.get_name() for symbol in scope.get_symbols()
                              if symbol.is_global() and symbol.is_referenced())
            scopes.extend(scope.get_children())
        unknown = referenced - assigned - set(spec.input_names) - set(spec.format_functions) - set(vars(builtins))
        if unknown:
            raise ValueError(f"Format names {sorted(unknown)} are not selected by input_field or registered as functions")
    return compile(tree, "<tokenizer format>", "eval")


def render_text_format(spec: TokenizerModalitySpec, *, payload: Mapping[str, Any]) -> str:
    """Render the selected payload under its original field names."""
    code = spec._format_code
    if code is None:
        raise ValueError(f"text modality {_field_label(spec)!r} format= was not compiled")
    collisions = payload.keys() & spec.format_functions.keys()
    if collisions:
        raise ValueError(f"Payload fields conflict with format_functions: {sorted(collisions)}")
    # One fresh namespace lets nested Python scopes (generators, lambdas, etc.)
    # resolve event fields and functions, without importing this module's globals.
    namespace = {**payload, **spec.format_functions}
    namespace["__builtins__"] = vars(builtins)
    try:
        return eval(code, namespace)
    except Exception as exc:
        raise ValueError(f"text modality {_field_label(spec)!r} format={spec.format!r} failed: {exc}") from exc


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
        return "all fields"
    if spec.input_index is None:
        return ", ".join(spec.input_names)
    return f"{spec.input_field}[{spec.input_index}]"


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
        specs.append(spec)

    meta: list[TokenizerModalityMeta] = []
    for spec in specs:
        k = spec.type
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
