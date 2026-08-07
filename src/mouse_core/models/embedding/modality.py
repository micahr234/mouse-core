"""Embedder modality specs (separate from tokenizer packing specs)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar


def _field_names(field: str | Sequence[str] | None) -> tuple[str, ...]:
    if field is None:
        return ()
    if isinstance(field, str):
        return (field,)
    return tuple(field)


def _reject_io_fields(data: dict[str, Any], *, who: str) -> None:
    if "input_field" in data or "output_field" in data:
        raise TypeError(
            f"{who} modalities use field= (not input_field=/output_field=); "
            "rename with Selector before tokenize"
        )


@dataclass
class NumericEmbedderModalitySpec:
    """How the numeric embedder embeds one named modality.

    Alignment with the tokenizer is by **name** (``field``).
    """

    type: str
    field: str | Sequence[str] | None = None
    vocab_size: int | None = None
    dim: int | None = None
    tokens: int | None = None
    std: float | None = None

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
                f"unknown embedder modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "learnable":
            return
        if self.field is None:
            raise ValueError(
                f"embedder modality type={k!r} requires field= (modality name)"
            )


@dataclass
class TextEmbedderModalitySpec:
    """Modality for :class:`~mouse_core.models.embedding.text.TextEmbedder`."""

    type: str
    field: str | Sequence[str] | None = None
    format: str | None = None

    _VALID_TYPES: ClassVar[tuple[str, ...]] = ("text", "token", "image")

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown text embedder modality type {self.type!r}; "
                f"expected one of {self._VALID_TYPES} "
                "(learnable scratch tokens are NumericEmbedder-only)"
            )
        object.__setattr__(self, "type", k)
        if self.field is None:
            raise ValueError(
                f"text embedder modality type={k!r} requires field="
            )
        if k == "text" and not self.format:
            raise ValueError(f"text modality {self.field!r} requires format=")
        if k == "token" and self.format is not None:
            raise ValueError(
                f"token modality {self.field!r} must not set format="
            )


KIND_DISCRETE = "discrete"
KIND_FOURIER = "fourier"
KIND_LEARNABLE = "learnable"
KIND_IMAGE = "image"


@dataclass(frozen=True)
class EmbedderModalityMeta:
    """Runtime metadata for one named numeric embedder modality."""

    spec: NumericEmbedderModalitySpec
    name: str
    kind: str
    vocab_size: int = 0
    dim: int = 0
    n_learnable: int = 0
    freq_sets: int = 1


def expand_embedder_numeric_spec(
    spec: NumericEmbedderModalitySpec, *, learnable_index: int
) -> list[NumericEmbedderModalitySpec]:
    if spec.type == "learnable":
        name = f"__learnable_{learnable_index}"
        return [replace(spec, field=name)]
    names = _field_names(spec.field)
    if not names:
        raise ValueError("embedder modalities must set field= (modality name)")
    return [replace(spec, field=name) for name in names]


def expand_embedder_text_spec(
    spec: TextEmbedderModalitySpec,
) -> list[TextEmbedderModalitySpec]:
    names = _field_names(spec.field)
    if not names:
        raise ValueError("text/token/image modalities must set field=")
    return [replace(spec, field=name) for name in names]


def _coerce_numeric_modalities(
    modalities: list[dict[str, Any] | NumericEmbedderModalitySpec]
    | Mapping[str, dict[str, Any]]
    | None,
) -> list[dict[str, Any] | NumericEmbedderModalitySpec]:
    if modalities is None:
        return []
    if isinstance(modalities, Mapping):
        out: list[dict[str, Any]] = []
        for name, cfg in modalities.items():
            data = dict(cfg)
            data.setdefault("field", name)
            out.append(data)
        return out
    return list(modalities)


def resolve_embedder_numeric_modalities(
    modalities: list[dict[str, Any] | NumericEmbedderModalitySpec]
    | Mapping[str, dict[str, Any]]
    | None = None,
) -> tuple[list[NumericEmbedderModalitySpec], list[EmbedderModalityMeta]]:
    """Expand embedder modality specs keyed by ``field`` name."""
    raw = _coerce_numeric_modalities(modalities)
    specs: list[NumericEmbedderModalitySpec] = []
    for i, m in enumerate(raw):
        if isinstance(m, NumericEmbedderModalitySpec):
            spec = m
        else:
            data = dict(m)
            _reject_io_fields(data, who="embedder")
            for banned in ("skip", "required"):
                if banned in data:
                    raise TypeError(
                        f"embedder modalities do not accept {banned}= "
                        "(tokenizer packing knob)"
                    )
            spec = NumericEmbedderModalitySpec(**data)
        specs.extend(expand_embedder_numeric_spec(spec, learnable_index=i))

    meta: list[EmbedderModalityMeta] = []
    seen: set[str] = set()
    for spec in specs:
        assert isinstance(spec.field, str)
        name = str(spec.field)
        if name in seen:
            raise ValueError(f"duplicate embedder modality name {name!r}")
        seen.add(name)
        k = spec.type
        if k == "discrete":
            vs = int(spec.vocab_size or 0)
            meta.append(
                EmbedderModalityMeta(
                    spec=spec, name=name, kind=KIND_DISCRETE, vocab_size=vs
                )
            )
        elif k in ("fourier", "continuous"):
            dim = 1 if k == "fourier" else int(spec.dim or 0)
            if dim <= 0:
                raise ValueError(
                    f"continuous modality {name!r} requires dim="
                )
            meta.append(
                EmbedderModalityMeta(
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
                EmbedderModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_LEARNABLE,
                    n_learnable=n,
                )
            )
        elif k == "image":
            vs = int(spec.vocab_size or 0)
            meta.append(
                EmbedderModalityMeta(
                    spec=spec, name=name, kind=KIND_IMAGE, vocab_size=vs
                )
            )
        else:
            raise ValueError(f"unsupported modality type {k!r}")
    return specs, meta
