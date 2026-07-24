"""NumericEmbedder — typed GPU maps over a flat TokenBatch.

The DataLoader (via :meth:`NumericEmbedder.prepare` /
:meth:`make_preparer`) builds a concatenated token stream with parallel
type/id/scalar arrays. This module only applies embedding tables and static
Fourier features — no Python walks over nested step dicts on the GPU path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

from mouse_core.models.embedding.encoding import StaticFourierFeatures
from mouse_core.models.embedding.linear import ScaledEmbedding
from mouse_core.models.embedding.token_batch import TokenBatch, empty_token_batch

# Token-type kind tags stored in preparer metadata (not the runtime type id).
_KIND_DISCRETE = "discrete"
_KIND_FOURIER = "fourier"
_KIND_LEARNABLE = "learnable"
_KIND_IMAGE = "image"


class Encoder(nn.Module, ABC):
    """Abstract base for encoders over :class:`TokenBatch`."""

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def tokens_per_step(self) -> int:
        """Capacity hint; real layout comes from ``prediction_indices``."""
        ...

    @abstractmethod
    def prepare(self, batch: list[list[dict]]) -> TokenBatch:
        """CPU: ragged ``[B][len_b]`` steps → flat :class:`TokenBatch` (no padding)."""
        ...

    def make_preparer(self) -> Callable[..., TokenBatch]:
        """Return a ``(batch) -> TokenBatch`` callable for workers."""
        return self.prepare

    @abstractmethod
    def forward(
        self, token_batch: TokenBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Embed ``TokenBatch`` → ``(embeds [L, D], col_values, prediction_indices [N])``."""
        ...

    @abstractmethod
    def pool_step_reprs(self, h: torch.Tensor, prediction_indices: torch.Tensor) -> torch.Tensor:
        """Gather prediction tokens → ``[N, D]`` (train) or ``[B, S, D]`` (decode).

        ``h`` is ``[L, D]`` (flat packed) or ``[B, L, D]`` (decode).
        Train: ``prediction_indices`` is ``[N]`` absolute indices into ``0 .. L-1``.
        Decode: ``prediction_indices`` is ``[B, S]`` into the token axis of ``h``.
        """
        ...


@dataclass
class ModalitySpec:
    """How to turn one modality into tokens in the preparer.

    Types:
      * ``discrete`` — integer id → embedding table (one token)
      * ``rff`` — scalar → one static-Fourier token
      * ``continuous`` — vector → one static-Fourier token per component
      * ``image`` — requires an image tokenizer → discrete visual token ids
      * ``learnable`` — ``tokens`` scratch embedding rows
    """

    type: str
    field: str | Sequence[str] | None = None
    vocab_size: int | None = None
    dim: int | None = None
    tokens: int | None = None
    std: float | None = None
    skip: Any = None
    required: bool = True

    _VALID_TYPES: ClassVar[tuple[str, ...]] = (
        "discrete",
        "rff",
        "continuous",
        "image",
        "learnable",
    )

    def __post_init__(self) -> None:
        k = (self.type or "").lower()
        if k not in self._VALID_TYPES:
            raise ValueError(
                f"unknown modality type {self.type!r} for modality {self.field!r}; "
                f"expected one of {self._VALID_TYPES}"
            )
        object.__setattr__(self, "type", k)
        if k == "learnable":
            object.__setattr__(self, "required", False)


def _unwrap_scalar(value: Any) -> Any:
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


def _values_equal(a: Any, b: Any) -> bool:
    return _unwrap_scalar(a) == _unwrap_scalar(b)


def _field_names(field: str | Sequence[str] | None) -> tuple[str, ...]:
    if field is None:
        return ()
    if isinstance(field, str):
        return (field,)
    return tuple(field)


def _expand_modality_spec(spec: ModalitySpec, fallback_field: str) -> list[ModalitySpec]:
    fields = _field_names(spec.field)
    if not fields and spec.type == "learnable":
        return [replace(spec, field=fallback_field)]
    if not fields:
        raise ValueError("input-backed modalities must set a field")
    return [replace(spec, field=field) for field in fields]


@dataclass(frozen=True)
class _ModalityMeta:
    """Runtime metadata for one expanded modality."""

    spec: ModalitySpec
    type_id: int
    kind: str
    vocab_size: int = 0
    dim: int = 0
    n_learnable: int = 0
    freq_sets: int = 1


class NumericEmbedder(Encoder):
    """Typed embedding tables + static Fourier over a :class:`TokenBatch`."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        modalities: list[dict[str, Any] | ModalitySpec] | None = None,
        fourier_min: float = 0.01,
        fourier_max: float = 10.0,
        std: float = 0.02,
        image_tokenizer: Callable[[Any], Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self._hidden_dim = int(hidden_dim)
        self.fourier_min = float(fourier_min)
        self.fourier_max = float(fourier_max)
        self.std = float(std)
        self.image_tokenizer = image_tokenizer

        raw = modalities or []
        self.modalities: list[ModalitySpec] = []
        for i, m in enumerate(raw):
            spec = m if isinstance(m, ModalitySpec) else ModalitySpec(**m)
            self.modalities.extend(_expand_modality_spec(spec, fallback_field=f"__learnable_{i}"))

        if any(s.type == "image" for s in self.modalities) and image_tokenizer is None:
            raise TypeError(
                "NumericEmbedder with type='image' modalities requires image_tokenizer="
            )

        self._meta: list[_ModalityMeta] = []
        self._tables = nn.ModuleDict()
        max_freq_sets = 1
        type_id = 0
        for spec in self.modalities:
            assert isinstance(spec.field, str)
            k = spec.type
            if k == "discrete":
                vs = int(spec.vocab_size or 0)
                if vs <= 0:
                    raise ValueError(f"discrete modality {spec.field!r} requires vocab_size=")
                meta = _ModalityMeta(spec=spec, type_id=type_id, kind=_KIND_DISCRETE, vocab_size=vs)
                self._tables[str(type_id)] = ScaledEmbedding(vs, hidden_dim, scale=spec.std or std)
                self._meta.append(meta)
                type_id += 1
            elif k in ("rff", "continuous"):
                dim = 1 if k == "rff" else int(spec.dim or 0)
                if dim <= 0:
                    raise ValueError(f"continuous modality {spec.field!r} requires dim=")
                max_freq_sets = max(max_freq_sets, dim)
                meta = _ModalityMeta(
                    spec=spec, type_id=type_id, kind=_KIND_FOURIER, dim=dim, freq_sets=dim
                )
                self._meta.append(meta)
                type_id += 1
            elif k == "learnable":
                n = int(spec.tokens or 1)
                if n <= 0:
                    raise ValueError("learnable tokens must be >= 1")
                meta = _ModalityMeta(
                    spec=spec, type_id=type_id, kind=_KIND_LEARNABLE, n_learnable=n
                )
                self._tables[str(type_id)] = ScaledEmbedding(n, hidden_dim, scale=spec.std or std)
                self._meta.append(meta)
                type_id += 1
            elif k == "image":
                # Tokenizer produces ids; table sized lazily is awkward — require vocab on spec.
                vs = int(spec.vocab_size or spec.size or 0)
                if vs <= 0:
                    raise ValueError(
                        f"image modality {spec.field!r} requires vocab_size= "
                        "(size of the image tokenizer vocabulary)"
                    )
                meta = _ModalityMeta(spec=spec, type_id=type_id, kind=_KIND_IMAGE, vocab_size=vs)
                self._tables[str(type_id)] = ScaledEmbedding(vs, hidden_dim, scale=spec.std or std)
                self._meta.append(meta)
                type_id += 1
            else:
                raise ValueError(f"unsupported modality type {k!r}")

        # Shared Fourier type id for all fourier modalities (dispatch by kind in forward).
        # Each fourier modality still has its own type_id for masking/debug; forward
        # routes kind==fourier through self.fourier.
        rff_scale = float(std) / (0.5 ** 0.5)
        self.fourier = StaticFourierFeatures(
            num_features=hidden_dim,
            in_min=fourier_min,
            in_max=fourier_max,
            num_freq_sets=max_freq_sets,
            output_scale=rff_scale,
        )
        self._n_types = type_id
        self._learnable_fields: set[str] = set()
        for m in self._meta:
            if m.kind == _KIND_LEARNABLE and isinstance(m.spec.field, str):
                self._learnable_fields.add(m.spec.field)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def tokens_per_step(self) -> int:
        """Max tokens if nothing is skipped (capacity hint)."""
        total = 0
        for m in self._meta:
            if m.kind == _KIND_DISCRETE:
                total += 1
            elif m.kind == _KIND_FOURIER:
                total += m.dim
            elif m.kind == _KIND_LEARNABLE:
                total += m.n_learnable
            elif m.kind == _KIND_IMAGE:
                total += 1  # unknown a priori; hint only
        return total

    def make_preparer(self) -> Callable[..., TokenBatch]:
        """Preparer capturing modality metadata (no nn modules) for worker threads."""
        meta = self._meta
        image_tokenizer = self.image_tokenizer
        learnable_fields = set(self._learnable_fields)

        def _prepare(batch: list[list[dict]]) -> TokenBatch:
            return _prepare_numeric(batch, meta, image_tokenizer, learnable_fields)

        return _prepare

    def prepare(self, batch: list[list[dict]]) -> TokenBatch:
        return _prepare_numeric(
            batch, self._meta, self.image_tokenizer, self._learnable_fields
        )

    def forward(
        self, token_batch: TokenBatch | list[list[dict]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if not isinstance(token_batch, TokenBatch):
            token_batch = self.prepare(token_batch)
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            device = self.fourier.get_buffer("freqs").device
            dtype = self.fourier.get_buffer("freqs").dtype
        t = token_batch.to_tensors(device)
        types = t["token_types"]
        ids = t["token_ids"]
        scalars = t["scalars"]
        L = types.shape[0]
        D = self._hidden_dim
        embeds = torch.zeros(L, D, device=device, dtype=dtype)

        if L > 0:
            for type_id_str, table in self._tables.items():
                tid = int(type_id_str)
                mask = types == tid
                # Empty mask → empty gather/assign (no host .any() sync).
                embeds[mask] = table(ids[mask]).to(dtype=dtype)

            fourier_type_ids = {m.type_id for m in self._meta if m.kind == _KIND_FOURIER}
            for tid in fourier_type_ids:
                mask = types == tid
                embeds[mask] = self.fourier(scalars[mask], ids[mask]).to(dtype=dtype)

        col_values = t["col_values"]
        prediction_indices = t["prediction_indices"]
        return embeds, col_values, prediction_indices

    def pool_step_reprs(self, h: torch.Tensor, prediction_indices: torch.Tensor) -> torch.Tensor:
        D = self._hidden_dim
        if h.ndim == 2:
            # Flat packed [L, D]; prediction_indices [N]
            return h[prediction_indices.reshape(-1)]
        # [B, L, D] decode layout; prediction_indices [B, S]
        B, S = prediction_indices.shape
        idx = prediction_indices.unsqueeze(-1).expand(B, S, D)
        return h.gather(1, idx)


def _prepare_numeric(
    batch: list[list[dict]],
    meta: list[_ModalityMeta],
    image_tokenizer: Callable[[Any], Sequence[int]] | None,
    learnable_fields: set[str],
) -> TokenBatch:
    B = len(batch)
    step_counts = np.asarray([len(rows) for rows in batch], dtype=np.int64)
    N = int(step_counts.sum()) if B > 0 else 0
    if B > 0 and (step_counts < 0).any():
        raise ValueError("sequence step counts must be non-negative")

    # col_values accumulation
    non_learnable = [m for m in meta if m.kind != _KIND_LEARNABLE]
    raw_cols: dict[str, list[Any]] = {str(m.spec.field): [] for m in non_learnable}

    tok_types: list[int] = []
    tok_ids: list[int] = []
    tok_scalars: list[float] = []
    seq_ids: list[int] = []
    step_ids: list[int] = []
    prediction_indices = np.zeros(N, dtype=np.int64)
    flat_step = 0

    for b in range(B):
        for s in range(int(step_counts[b])):
            row = batch[b][s]
            step_start = len(tok_types)

            for m in meta:
                field = str(m.spec.field)
                spec = m.spec
                if m.kind == _KIND_LEARNABLE:
                    for i in range(m.n_learnable):
                        tok_types.append(m.type_id)
                        tok_ids.append(i)
                        tok_scalars.append(0.0)
                        seq_ids.append(b)
                        step_ids.append(s)
                    continue

                value = row.get(field)
                if field in raw_cols:
                    raw_cols[field].append(value)

                if value is None:
                    if spec.required:
                        raise KeyError(
                            f"Required modality {field!r} is missing from "
                            f"batch[{b}][{s}]"
                        )
                    continue
                if spec.skip is not None and _values_equal(value, spec.skip):
                    continue

                if m.kind == _KIND_DISCRETE:
                    tok_types.append(m.type_id)
                    tok_ids.append(int(_unwrap_scalar(value)))
                    tok_scalars.append(0.0)
                    seq_ids.append(b)
                    step_ids.append(s)
                elif m.kind == _KIND_FOURIER:
                    if m.dim == 1:
                        vals = [float(_unwrap_scalar(value))]
                    else:
                        arr = np.asarray(value, dtype=np.float32).ravel()
                        if arr.size < m.dim:
                            pad = np.zeros(m.dim - arr.size, dtype=np.float32)
                            arr = np.concatenate([arr, pad])
                        vals = [float(arr[i]) for i in range(m.dim)]
                    for i, v in enumerate(vals):
                        tok_types.append(m.type_id)
                        tok_ids.append(i)  # freq bank index
                        tok_scalars.append(v)
                        seq_ids.append(b)
                        step_ids.append(s)
                elif m.kind == _KIND_IMAGE:
                    if image_tokenizer is None:
                        raise RuntimeError("image_tokenizer is not configured")
                    ids = list(image_tokenizer(value))
                    if not ids:
                        raise ValueError(f"image tokenizer returned no tokens for {field!r}")
                    for tid in ids:
                        tok_types.append(m.type_id)
                        tok_ids.append(int(tid))
                        tok_scalars.append(0.0)
                        seq_ids.append(b)
                        step_ids.append(s)

            step_end = len(tok_types)
            if step_end == step_start:
                raise ValueError(
                    "step has no tokens after skips; ensure at least one modality "
                    "is present (e.g. add a learnable modality)"
                )
            prediction_indices[flat_step] = step_end - 1
            flat_step += 1

    # Build col_values [N] / [N, dim]
    col_values: dict[str, np.ndarray] = {}
    for m in non_learnable:
        field = str(m.spec.field)
        values = raw_cols[field]
        if m.kind == _KIND_DISCRETE or (m.kind == _KIND_FOURIER and m.dim == 1):
            fill = 0 if m.kind == _KIND_DISCRETE else 0.0
            out = []
            for v in values:
                if v is None:
                    out.append(fill)
                else:
                    out.append(
                        int(_unwrap_scalar(v))
                        if m.kind == _KIND_DISCRETE
                        else float(_unwrap_scalar(v))
                    )
            dtype = np.int64 if m.kind == _KIND_DISCRETE else np.float32
            col_values[field] = np.asarray(out, dtype=dtype)
        elif m.kind == _KIND_FOURIER:
            buf = np.zeros((N, m.dim), dtype=np.float32)
            for i, v in enumerate(values):
                if v is None:
                    continue
                a = np.asarray(v, dtype=np.float32).ravel()
                d = min(a.size, m.dim)
                buf[i, :d] = a[:d]
            col_values[field] = buf
        elif m.kind == _KIND_IMAGE:
            # Store a placeholder int grid; objectives rarely need raw images.
            col_values[field] = np.zeros(N, dtype=np.int64)

    # Per-step sequence ownership for flat objectives.
    if N > 0:
        col_values["sequence_id"] = np.repeat(
            np.arange(B, dtype=np.int64), step_counts
        )

    if B == 0 or N == 0:
        return empty_token_batch(B)

    return TokenBatch(
        token_types=np.asarray(tok_types, dtype=np.int64),
        token_ids=np.asarray(tok_ids, dtype=np.int64),
        scalars=np.asarray(tok_scalars, dtype=np.float32),
        sequence_ids=np.asarray(seq_ids, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
        prediction_indices=prediction_indices,
        col_values=col_values,
        B=B,
    )
