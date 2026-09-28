"""Tokenizer modality specs and packing helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from string import Formatter
from typing import Any, ClassVar

import numpy as np
import torch

# Shared text stream. ``type="text"`` / ``type="token"`` fields emit here;
# ``when={"group_start": True}`` fields are tokenized into the same modality.
NAME_TEXT = "__text__"

# Keys allowed on ``TokenizerModalitySpec.when``.
WHEN_KEYS = frozenset({"equals", "not_equals", "group_start"})

# Step-backed ``text`` ``format=`` interpolates the field value here.
# A format spec is allowed (``"{field:.0f}"``). Consts have no placeholders.
TEXT_FORMAT_KEY = "field"

KIND_TEXT = "text"
KIND_TOKEN = "token"
KIND_IMAGE = "image"


@dataclass
class TokenizerModalitySpec:
    """How the tokenizer packs one field from a step dict.

    Types:
      * ``text`` — format string → HF tokenize → ``__text__``
      * ``token`` — integer id → one ``__text__`` token (pretrained vocab)
      * ``image`` — image tokenizer → discrete visual token ids

    ``text`` and ``token`` share the ``__text__`` stream. ``image`` uses
    ``output_field`` as the modality name. Omitted ``output_field``
    defaults to ``input_field``.

    A ``text`` field requires ``format=``. ``input_field=`` reads the
    step value into exactly one placeholder ``{field}``; a format spec
    such as ``"{field:.0f}"`` is allowed. Omit ``input_field=`` and the
    field is a const: ``output_field=`` names it and ``format=`` is the
    literal string to tokenize (no placeholders). ``format=`` / const
    ``format=`` are ``str.format`` strings: write ``{{`` / ``}}`` for a
    literal brace. ``token`` / ``image`` do not accept ``format=``.

    Optional ``when=`` is a dict of emission conditions. Every listed
    condition is **OR**ed — the field emits if any matches. Omit
    ``when`` (or pass ``None`` / an empty dict) and the field always
    emits (subject to ``required``). Allowed keys:

    * ``equals`` — list of ``(field, value)`` pairs; matches when that
      step value equals ``value`` (a missing key does not match).
    * ``not_equals`` — list of ``(field, value)`` pairs; matches when
      that step value does not equal ``value`` (a missing key does not
      match).
    * ``group_start`` — ``True`` marks tokens for insertion at the start
      of each grouping-field segment (and each packed sequence) by
      :func:`~mouse_core.data.token_batch.pack_token_batch`. Incremental
      decode passes ``prev_grouping_ids`` so a cached segment does not
      re-emit them.

    When a value condition and ``group_start`` are both set, a matching
    value emits on the step's ordinary token run; otherwise the tokens
    are carried as ``group_start_*`` for pack-time insertion.
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
    the limit. Only the variable-length types accept it (``text`` /
    ``image``); ``token`` emits one id and rejects it. Every
    ``output_field`` (including ``text`` / ``token`` names) must be
    unique. Fields emit in ``input_fields`` order.
    """

    type: str
    input_field: str | None = None
    output_field: str | None = None
    format: str | None = None
    max_tokens: int | None = None
    when: dict[str, Any] | None = None
    required: bool = True
    head_output: bool = False

    _VALID_TYPES: ClassVar[tuple[str, ...]] = ("text", "token", "image")

    def __post_init__(self) -> None:
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
        self._init_named_input()
        self._reject_text_format(k)
        if k == "image":
            _validate_max_tokens(self)
        else:
            self._reject_max_tokens(k)
        _validate_when(self)

    def _init_text(self) -> None:
        if self.input_field is None:
            self._reject_no_input_knobs("text const")
            if not self.output_field:
                raise ValueError("text const field requires output_field=")
            if not self.format:
                raise ValueError(
                    f"text const field {self.output_field!r} requires "
                    "format= (a literal string, no placeholders)"
                )
            names = _text_format_placeholders(self.format, who=self.output_field)
            if names:
                raise ValueError(
                    f"text const field {self.output_field!r} format= is "
                    "a literal string and must not contain placeholders"
                )
            _validate_max_tokens(self)
            _validate_when(self)
            return
        if not self.output_field:
            object.__setattr__(self, "output_field", self.input_field)
        names = _text_format_placeholders(self.format, who=self.output_field)
        if len(names) != 1 or names[0] != TEXT_FORMAT_KEY:
            raise ValueError(
                f"text modality {self.output_field!r} format= must contain "
                f"exactly one placeholder {{{TEXT_FORMAT_KEY}}}"
            )
        _validate_max_tokens(self)
        _validate_when(self)

    def _init_named_input(self) -> None:
        if not self.input_field:
            raise ValueError(
                f"tokenizer modality type={self.type!r} requires input_field="
            )
        if not self.output_field:
            object.__setattr__(self, "output_field", self.input_field)

    def _reject_text_format(self, kind: str) -> None:
        if self.format is not None:
            raise TypeError(
                f"tokenizer modality type={kind!r} does not accept format= "
                "(text only)"
            )

    def _reject_max_tokens(self, kind: str) -> None:
        if self.max_tokens is not None:
            raise TypeError(
                f"tokenizer modality type={kind!r} does not accept max_tokens= "
                "(it emits a fixed number of tokens; text / image only)"
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


def _text_format_placeholders(format_str: str | None, *, who: str | None) -> list[str]:
    label = who or "text field"
    if not format_str:
        raise ValueError(f"text modality {label!r} requires format=")
    return _literal_placeholders(format_str, who=label)


def _literal_placeholders(text: str, *, who: str) -> list[str]:
    names: list[str] = []
    for _, name, _, _ in Formatter().parse(text):
        if name is None:
            continue
        if name == "":
            raise ValueError(
                f"text modality {who!r} format does not accept an empty "
                "placeholder"
            )
        names.append(name)
    return names


def _validate_max_tokens(spec: TokenizerModalitySpec) -> None:
    if spec.max_tokens is None:
        return
    n = int(spec.max_tokens)
    if n <= 0:
        raise ValueError(
            f"tokenizer modality {spec.output_field!r} max_tokens must be >= 1"
        )
    object.__setattr__(spec, "max_tokens", n)


def _normalize_when_pair(
    value: Any, *, name: str, knob: str
) -> tuple[str, Any]:
    """Accept ``(field, value)`` or a length-2 list (JSON round-trip)."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(
            f"tokenizer modality {name!r} when[{knob!r}] entries must be "
            "(field, value) pairs"
        )
    field, expected = value[0], value[1]
    if not isinstance(field, str) or field == "":
        raise TypeError(
            f"tokenizer modality {name!r} when[{knob!r}] field must be a "
            "non-empty string"
        )
    return (field, expected)


def _normalize_when(
    when: Any, *, name: str
) -> dict[str, Any] | None:
    """Normalize ``when=`` to a dict with only active keys, or ``None``."""
    if when is None:
        return None
    if not isinstance(when, dict):
        raise TypeError(
            f"tokenizer modality {name!r} when= must be a dict or None"
        )
    unknown = set(when) - WHEN_KEYS
    if unknown:
        raise TypeError(
            f"tokenizer modality {name!r} when= unknown keys "
            f"{sorted(unknown)}; expected subset of {sorted(WHEN_KEYS)}"
        )
    out: dict[str, Any] = {}
    for knob in ("equals", "not_equals"):
        if knob not in when or when[knob] is None:
            continue
        raw = when[knob]
        if not isinstance(raw, list):
            raise TypeError(
                f"tokenizer modality {name!r} when[{knob!r}] must be a "
                "list of (field, value) pairs"
            )
        pairs = [
            _normalize_when_pair(item, name=name, knob=knob) for item in raw
        ]
        if pairs:
            out[knob] = pairs
    if "group_start" in when and when["group_start"] is not None:
        gs = when["group_start"]
        if not isinstance(gs, bool):
            raise TypeError(
                f"tokenizer modality {name!r} when['group_start'] must be a bool"
            )
        if gs:
            out["group_start"] = True
    return out or None


def _validate_when(spec: TokenizerModalitySpec) -> None:
    name = spec.output_field or spec.input_field or "field"
    normalized = _normalize_when(spec.when, name=name)
    object.__setattr__(spec, "when", normalized)
    if normalized is None:
        return
    if spec.head_output:
        raise ValueError(
            f"tokenizer modality {name!r} is head_output and cannot set "
            "when= (that field must emit on every step)"
        )


def when_has_group_start(spec: TokenizerModalitySpec) -> bool:
    """True when ``when`` requests pack-time group-start insertion."""
    return bool(spec.when and spec.when.get("group_start"))


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


def expand_tokenizer_spec(spec: TokenizerModalitySpec) -> list[TokenizerModalitySpec]:
    if spec.type == "text" and spec.input_field is None:
        if not spec.output_field:
            raise ValueError("text const field requires output_field=")
        return [spec]
    if not spec.input_field or not spec.output_field:
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

    ``text`` / ``token`` meta ``name`` is :data:`NAME_TEXT`. ``image``
    is keyed by ``output_field``.
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
    seen: set[str] = set()
    for spec in specs:
        k = spec.type
        name = str(spec.output_field)
        if not name:
            raise ValueError("tokenizer modality is missing output_field=")
        if name in seen:
            raise ValueError(
                f"duplicate tokenizer field name {name!r}; set a distinct "
                "output_field= on each field"
            )
        seen.add(name)
        if k in ("text", "token"):
            meta.append(
                TokenizerModalityMeta(
                    spec=spec,
                    name=NAME_TEXT,
                    kind=KIND_TEXT if k == "text" else KIND_TOKEN,
                )
            )
        elif k == "image":
            meta.append(
                TokenizerModalityMeta(spec=spec, name=name, kind=KIND_IMAGE)
            )
        else:
            raise ValueError(f"unsupported modality type {k!r}")
    return specs, meta
