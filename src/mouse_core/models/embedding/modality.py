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
            "rename with the tokenizer input_field=/output_field="
        )


@dataclass
class NumericEmbedderModalitySpec:
    """How the numeric embedder embeds one named modality.

    Alignment with the tokenizer is by **name** (``field``).

    ``std`` is required: it is the init scale of this modality's content
    embeddings (table rows / Fourier features) and of its type vectors.

    ``positions`` is required: the maximum number of tokens this modality
    emits in one step. The modality owns one type vector per position
    (``[positions, D]``); token ``t`` receives row ``TokenBatch.positions[t]``.
    Must be ``>= dim`` for ``continuous`` and ``>= tokens`` for ``learnable``.

    ``fourier`` / ``continuous`` must set ``fourier_min`` and ``fourier_max``
    (the static Fourier input range for this field; no default). Other
    types must not set them.
    """

    type: str
    field: str | Sequence[str] | None = None
    vocab_size: int | None = None
    dim: int | None = None
    tokens: int | None = None
    std: float | None = None
    positions: int | None = None
    fourier_min: float | None = None
    fourier_max: float | None = None

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
        if self.std is None:
            raise ValueError(
                f"embedder modality type={k!r} field={self.field!r} requires std= "
                "(embedding init scale; no default)"
            )
        std = float(self.std)
        if std < 0.0:
            raise ValueError(f"embedder modality std must be >= 0, got {self.std!r}")
        object.__setattr__(self, "std", std)
        if self.positions is None:
            raise ValueError(
                f"embedder modality type={k!r} field={self.field!r} requires positions= "
                "(max tokens per step; one type vector each; no default)"
            )
        positions = int(self.positions)
        if positions < 1:
            raise ValueError(
                f"embedder modality positions must be >= 1, got {self.positions!r}"
            )
        object.__setattr__(self, "positions", positions)
        if k in ("fourier", "continuous"):
            if self.fourier_min is None or self.fourier_max is None:
                raise ValueError(
                    f"embedder modality type={k!r} field={self.field!r} requires "
                    "fourier_min= and fourier_max= (Fourier input range; no default)"
                )
            fmin = float(self.fourier_min)
            fmax = float(self.fourier_max)
            if fmin <= 0.0 or fmax <= 0.0:
                raise ValueError(
                    "embedder modality fourier_min and fourier_max must be > 0, "
                    f"got fourier_min={self.fourier_min!r} fourier_max={self.fourier_max!r}"
                )
            if fmin >= fmax:
                raise ValueError(
                    "embedder modality fourier_min must be < fourier_max, "
                    f"got fourier_min={self.fourier_min!r} fourier_max={self.fourier_max!r}"
                )
            object.__setattr__(self, "fourier_min", fmin)
            object.__setattr__(self, "fourier_max", fmax)
        elif self.fourier_min is not None or self.fourier_max is not None:
            raise TypeError(
                f"embedder modality type={k!r} field={self.field!r} does not accept "
                "fourier_min=/fourier_max= (fourier/continuous only)"
            )
        if k == "learnable":
            return
        if self.field is None:
            raise ValueError(
                f"embedder modality type={k!r} requires field= (modality name)"
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
    n_positions: int = 1
    """Max tokens this modality emits per step (``spec.positions``); one type vector each."""


def expand_embedder_numeric_spec(
    spec: NumericEmbedderModalitySpec, *, learnable_index: int
) -> list[NumericEmbedderModalitySpec]:
    """Expand one spec into one spec per field name.

    Learnable specs without a ``field`` are auto-named ``__learnable_<i>``;
    an explicit ``field`` (e.g. ``"value"``) is kept and must match the
    tokenizer's ``output_field``. ``learnable_index`` is the ordinal among
    *learnable* specs (0 for the first learnable, 1 for the second, …). It
    must not depend on the position in the full list: the saved config stores
    the expanded list, where a multi-field spec occupies several slots, so a
    raw index would shift on reload and the table's ``state_dict`` key would
    no longer match.
    """
    if spec.type == "learnable" and spec.field is None:
        name = f"__learnable_{learnable_index}"
        return [replace(spec, field=name)]
    names = _field_names(spec.field)
    if not names:
        raise ValueError("embedder modalities must set field= (modality name)")
    return [replace(spec, field=name) for name in names]


def _coerce_numeric_modalities(
    modalities: Sequence[dict[str, Any] | NumericEmbedderModalitySpec]
    | Mapping[str, dict[str, Any]]
    | None,
) -> list[dict[str, Any] | NumericEmbedderModalitySpec]:
    if modalities is None:
        return []
    if isinstance(modalities, Mapping):
        out: list[dict[str, Any] | NumericEmbedderModalitySpec] = []
        for name, cfg in modalities.items():
            data = dict(cfg)
            data.setdefault("field", name)
            out.append(data)
        return out
    return list(modalities)


def resolve_embedder_numeric_modalities(
    modalities: Sequence[dict[str, Any] | NumericEmbedderModalitySpec]
    | Mapping[str, dict[str, Any]]
    | None = None,
) -> tuple[list[NumericEmbedderModalitySpec], list[EmbedderModalityMeta]]:
    """Expand embedder modality specs keyed by ``field`` name."""
    raw = _coerce_numeric_modalities(modalities)
    specs: list[NumericEmbedderModalitySpec] = []
    n_learnable = 0
    for m in raw:
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
        specs.extend(expand_embedder_numeric_spec(spec, learnable_index=n_learnable))
        if spec.type == "learnable":
            n_learnable += 1

    meta: list[EmbedderModalityMeta] = []
    seen: set[str] = set()
    for spec in specs:
        assert isinstance(spec.field, str)
        name = str(spec.field)
        if name in seen:
            raise ValueError(f"duplicate embedder modality name {name!r}")
        seen.add(name)
        k = spec.type
        assert spec.positions is not None
        n_pos = int(spec.positions)
        if k == "discrete":
            vs = int(spec.vocab_size or 0)
            meta.append(
                EmbedderModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_DISCRETE,
                    vocab_size=vs,
                    n_positions=n_pos,
                )
            )
        elif k in ("fourier", "continuous"):
            dim = 1 if k == "fourier" else int(spec.dim or 0)
            if dim <= 0:
                raise ValueError(
                    f"continuous modality {name!r} requires dim="
                )
            if n_pos < dim:
                raise ValueError(
                    f"{k} modality {name!r} emits dim={dim} tokens per step but "
                    f"declares positions={n_pos}; positions must be >= dim"
                )
            meta.append(
                EmbedderModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_FOURIER,
                    dim=dim,
                    freq_sets=dim,
                    n_positions=n_pos,
                )
            )
        elif k == "learnable":
            n = int(spec.tokens or 1)
            if n <= 0:
                raise ValueError("learnable tokens must be >= 1")
            if n_pos < n:
                raise ValueError(
                    f"learnable modality {name!r} emits tokens={n} per step but "
                    f"declares positions={n_pos}; positions must be >= tokens"
                )
            meta.append(
                EmbedderModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_LEARNABLE,
                    n_learnable=n,
                    n_positions=n_pos,
                )
            )
        elif k == "image":
            vs = int(spec.vocab_size or 0)
            meta.append(
                EmbedderModalityMeta(
                    spec=spec,
                    name=name,
                    kind=KIND_IMAGE,
                    vocab_size=vs,
                    n_positions=n_pos,
                )
            )
        else:
            raise ValueError(f"unsupported modality type {k!r}")
    return specs, meta
