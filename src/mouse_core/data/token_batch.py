"""StepTokens (one step) and TokenBatch (packed multi-sequence batch)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ModalityInfo:
    """Name-keyed modality descriptor carried on StepTokens / TokenBatch."""

    type: str
    dim: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", str(self.type).lower())


def step_counts_from_sequence_id(
    sequence_id: np.ndarray | None,
    B: int,
) -> np.ndarray:
    """Per-sequence step counts ``[B]`` from flat ``sequence_id`` ``[N]``.

    Missing IDs (empty decode rows) become zeros when ``minlength=B``.
    """
    if B <= 0:
        return np.zeros(0, dtype=np.int64)
    if sequence_id is None:
        return np.zeros(B, dtype=np.int64)
    sid = np.asarray(sequence_id, dtype=np.int64).reshape(-1)
    if sid.size == 0:
        return np.zeros(B, dtype=np.int64)
    return np.bincount(sid, minlength=B).astype(np.int64)[:B]


def _validate_modality_table(
    modality_names: Sequence[str],
    modality_map: Mapping[str, ModalityInfo],
) -> tuple[tuple[str, ...], dict[str, ModalityInfo]]:
    names = tuple(str(n) for n in modality_names)
    if len(names) != len(set(names)):
        raise ValueError(f"modality_names must be unique, got {names}")
    mmap = {str(k): v for k, v in modality_map.items()}
    for n in names:
        if n not in mmap:
            raise ValueError(f"modality_map missing entry for name {n!r}")
    for k in mmap:
        if k not in names:
            raise ValueError(
                f"modality_map has extra name {k!r} not in modality_names {names}"
            )
    return names, mmap


@dataclass
class StepTokens:
    """Tokens and step-level fields for a single environment / dataset step.

    Produced by :class:`~mouse_core.data.numeric_tokenizer.NumericTokenizer` /
    :class:`~mouse_core.data.text_tokenizer.TextTokenizer`. Pack many steps into
    a :class:`TokenBatch` with :func:`pack_token_batch`.

    ``modality_ids[t]`` indexes ``modality_names``; type/kind comes from
    ``modality_map[modality_names[modality_ids[t]]]``.
    """

    modality_ids: np.ndarray  # [T] index into modality_names
    ids: np.ndarray  # [T]
    values: np.ndarray  # [T]
    modality_names: tuple[str, ...]
    modality_map: dict[str, ModalityInfo]
    grouping_id: int
    grouping_field: str
    objective_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.grouping_field:
            raise ValueError("StepTokens requires a non-empty grouping_field")
        names, mmap = _validate_modality_table(self.modality_names, self.modality_map)
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "modality_map", mmap)
        t = int(np.asarray(self.modality_ids).shape[0])
        if t == 0:
            raise ValueError("StepTokens must contain at least one token")
        for name in ("modality_ids", "ids", "values"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (t,):
                raise ValueError(f"{name} must have shape [{t}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        mids = np.asarray(self.modality_ids, dtype=np.int64)
        if mids.min(initial=0) < 0 or mids.max(initial=0) >= len(names):
            raise ValueError(
                f"modality_ids must be in [0, {len(names)}), got "
                f"min={int(mids.min())} max={int(mids.max())}"
            )
        object.__setattr__(self, "modality_ids", mids)
        object.__setattr__(self, "grouping_id", int(self.grouping_id))

    @property
    def T(self) -> int:
        return int(self.modality_ids.shape[0])


@dataclass
class TokenBatch:
    """Flat concatenated token stream (no padding) plus parallel payload arrays.

    Built by :func:`pack_token_batch` from many :class:`StepTokens`. Length ``L``
    is the total number of tokens across all sequences and steps. Step-level
    fields use ``N = len(prediction_indices)`` (ragged windows allowed).
    Per-sequence step counts are derived from ``objective_fields["sequence_id"]`` +
    ``B`` (see :meth:`step_counts`); they are not stored separately.

    Token type/kind is looked up via ``modality_map[modality_names[modality_ids[i]]]``:

    * discrete / learnable / image — ``ids[i]`` is a table/vocab row; ``values[i]`` is 0
    * fourier — ``values[i]`` is the scalar; ``ids[i]`` is the Fourier freq-bank index

    Attributes:
        modality_ids: ``[L]`` int64 — index into ``modality_names``.
        modality_names: interned modality names for this batch.
        modality_map: name → :class:`ModalityInfo` (type/kind lookup).
        ids: ``[L]`` int64 — discrete row id, or continuous freq-bank index.
        values: ``[L]`` float32 — continuous scalar (0 when discrete).
        sequence_ids: ``[L]`` int64 — which of the ``B`` sequences each token belongs to.
        grouping_ids: ``[L]`` int64 — attention group within the sequence.
        prediction_indices: ``[N]`` int64 — index of each step's prediction token.
        objective_fields: step-level arrays for objectives.
        B: Number of sequences.
        grouping_field: Name of the grouping column.
    """

    modality_ids: np.ndarray
    ids: np.ndarray
    values: np.ndarray
    modality_names: tuple[str, ...]
    modality_map: dict[str, ModalityInfo]
    sequence_ids: np.ndarray
    grouping_ids: np.ndarray
    prediction_indices: np.ndarray
    grouping_field: str
    objective_fields: dict[str, np.ndarray] = field(default_factory=dict)
    B: int = 0

    def __post_init__(self) -> None:
        if not self.grouping_field:
            raise ValueError("TokenBatch requires a non-empty grouping_field")
        names, mmap = _validate_modality_table(self.modality_names, self.modality_map)
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "modality_map", mmap)
        L = int(np.asarray(self.modality_ids).shape[0])
        for name in (
            "modality_ids",
            "ids",
            "values",
            "sequence_ids",
            "grouping_ids",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (L,):
                raise ValueError(f"{name} must have shape [{L}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        if L > 0:
            mids = np.asarray(self.modality_ids, dtype=np.int64)
            if mids.min() < 0 or mids.max() >= len(names):
                raise ValueError(
                    f"modality_ids must be in [0, {len(names)}), got "
                    f"min={int(mids.min())} max={int(mids.max())}"
                )
        pred = np.asarray(self.prediction_indices, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "prediction_indices", pred)
        n = int(pred.shape[0])
        counts = self.step_counts()
        if int(counts.sum()) != n:
            raise ValueError(
                f"prediction_indices length [{n}] must equal sum of step counts "
                f"from sequence_id [{int(counts.sum())}] (B={self.B})"
            )
        sid = self.objective_fields.get("sequence_id")
        if n > 0:
            if sid is None:
                raise ValueError("objective_fields must include sequence_id when N > 0")
            sid_arr = np.asarray(sid, dtype=np.int64).reshape(-1)
            if sid_arr.shape != (n,):
                raise ValueError(
                    f"sequence_id must have shape [{n}], got {sid_arr.shape}"
                )
            if self.grouping_field not in self.objective_fields:
                raise ValueError(
                    f"objective_fields must include grouping_field {self.grouping_field!r} "
                    "when N > 0"
                )

    @property
    def L(self) -> int:
        return int(self.modality_ids.shape[0])

    @property
    def N(self) -> int:
        return int(self.prediction_indices.shape[0])

    @property
    def S(self) -> int:
        """Max steps in the batch (convenience; rows may be shorter)."""
        if self.B <= 0:
            return 0
        counts = self.step_counts()
        return int(counts.max()) if counts.size else 0

    def step_counts(self) -> np.ndarray:
        """Steps per sequence ``[B]``, derived from ``objective_fields["sequence_id"]``."""
        return step_counts_from_sequence_id(
            self.objective_fields.get("sequence_id"), self.B
        )

    def to_tensors(self, device: torch.device | str | None = None) -> dict[str, Any]:
        """Move arrays to torch tensors on ``device`` (CPU if None)."""
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _long(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)

        def _float(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)

        fields: dict[str, torch.Tensor] = {}
        for k, v in self.objective_fields.items():
            arr = np.asarray(v)
            if np.issubdtype(arr.dtype, np.floating):
                fields[k] = torch.from_numpy(arr.astype(np.float32)).to(dev)
            else:
                fields[k] = torch.from_numpy(arr.astype(np.int64)).to(dev)

        return {
            "modality_ids": _long(self.modality_ids),
            "ids": _long(self.ids),
            "values": _float(self.values),
            "modality_names": self.modality_names,
            "modality_map": self.modality_map,
            "sequence_ids": _long(self.sequence_ids),
            "grouping_ids": _long(self.grouping_ids),
            "prediction_indices": _long(self.prediction_indices),
            "objective_fields": fields,
            "B": self.B,
            "grouping_field": self.grouping_field,
        }


def empty_token_batch(
    B: int = 0,
    *,
    grouping_field: str,
    modality_names: Sequence[str] = (),
    modality_map: Mapping[str, ModalityInfo] | None = None,
) -> TokenBatch:
    """Empty batch (L=0, N=0); all ``B`` sequences have zero step count."""
    names, mmap = _validate_modality_table(
        modality_names, dict(modality_map or {})
    )
    return TokenBatch(
        modality_ids=np.zeros(0, dtype=np.int64),
        ids=np.zeros(0, dtype=np.int64),
        values=np.zeros(0, dtype=np.float32),
        modality_names=names,
        modality_map=mmap,
        sequence_ids=np.zeros(0, dtype=np.int64),
        grouping_ids=np.zeros(0, dtype=np.int64),
        prediction_indices=np.zeros(0, dtype=np.int64),
        grouping_field=grouping_field,
        objective_fields={},
        B=B,
    )


def _as_field_array(value: Any) -> np.ndarray:
    """Normalize one step field value to a numpy array (scalar → 0-d)."""
    if value is None:
        return np.asarray(0, dtype=np.int64)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.dtype == object:
        raise TypeError(f"objective_fields values must be numeric, got {type(value)}")
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == np.bool_:
        return arr.astype(np.int64, copy=False)
    return arr


def _stack_objective_fields(
    steps: Sequence[StepTokens],
    *,
    sequence_ids: Sequence[int],
    grouping_field: str,
) -> dict[str, np.ndarray]:
    """Stack per-step ``objective_fields`` into ``[N]`` / ``[N, ...]`` arrays."""
    n = len(steps)
    keys: set[str] = set()
    for st in steps:
        keys.update(st.objective_fields)
    keys.discard("sequence_id")
    keys.discard(grouping_field)

    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        raw = [st.objective_fields.get(key) for st in steps]
        arrays = [_as_field_array(v) if v is not None else None for v in raw]
        proto = next((a for a in arrays if a is not None), np.asarray(0, dtype=np.int64))
        if proto.ndim == 0:
            fill = 0.0 if np.issubdtype(proto.dtype, np.floating) else 0
            dtype = np.float32 if np.issubdtype(proto.dtype, np.floating) else np.int64
            buf = np.empty(n, dtype=dtype)
            for i, a in enumerate(arrays):
                buf[i] = fill if a is None else a.reshape(())
            out[key] = buf
        else:
            shapes = [a.shape for a in arrays if a is not None]
            max_shape = tuple(max(s[d] for s in shapes) for d in range(proto.ndim))
            dtype = np.float32 if np.issubdtype(proto.dtype, np.floating) else proto.dtype
            buf = np.zeros((n, *max_shape), dtype=dtype)
            for i, a in enumerate(arrays):
                if a is None:
                    continue
                slicer = tuple(slice(0, a.shape[d]) for d in range(a.ndim))
                buf[i][slicer] = a
            out[key] = buf

    out["sequence_id"] = np.asarray(list(sequence_ids), dtype=np.int64)
    out[grouping_field] = np.asarray([st.grouping_id for st in steps], dtype=np.int64)
    return out


def pack_token_batch(
    steps: Sequence[StepTokens],
    *,
    sequence_ids: Sequence[int] | None = None,
    batch_size: int | None = None,
    grouping_field: str | None = None,
) -> TokenBatch:
    """Pack per-step :class:`StepTokens` into a flat :class:`TokenBatch`.

    All steps must share the same ``modality_names``, ``modality_map``, and
    ``grouping_field``.
    """
    if not steps:
        if grouping_field is None:
            raise ValueError(
                "pack_token_batch of empty steps requires grouping_field="
            )
        if batch_size is None:
            return empty_token_batch(0, grouping_field=grouping_field)
        return empty_token_batch(batch_size, grouping_field=grouping_field)

    gf = steps[0].grouping_field
    names = steps[0].modality_names
    mmap = steps[0].modality_map
    if grouping_field is not None and grouping_field != gf:
        raise ValueError(
            f"grouping_field mismatch: arg {grouping_field!r} vs step {gf!r}"
        )
    for i, st in enumerate(steps):
        if st.grouping_field != gf:
            raise ValueError(
                f"steps[{i}].grouping_field {st.grouping_field!r} != {gf!r}"
            )
        if st.modality_names != names:
            raise ValueError(
                f"steps[{i}].modality_names {st.modality_names!r} != {names!r}"
            )
        if st.modality_map != mmap:
            raise ValueError(f"steps[{i}].modality_map does not match steps[0]")

    if sequence_ids is None:
        seq_per_step = [0] * len(steps)
    else:
        if len(sequence_ids) != len(steps):
            raise ValueError(
                f"sequence_ids length ({len(sequence_ids)}) must match "
                f"steps ({len(steps)})"
            )
        seq_per_step = [int(s) for s in sequence_ids]

    modality_ids: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    values: list[np.ndarray] = []
    seq_ids: list[np.ndarray] = []
    grouping_ids: list[np.ndarray] = []
    prediction_indices: list[int] = []

    offset = 0
    for st, sid in zip(steps, seq_per_step):
        t = st.T
        modality_ids.append(st.modality_ids)
        ids.append(st.ids)
        values.append(st.values)
        seq_ids.append(np.full(t, sid, dtype=np.int64))
        grouping_ids.append(np.full(t, st.grouping_id, dtype=np.int64))
        prediction_indices.append(offset + t - 1)
        offset += t

    inferred_B = (max(seq_per_step) + 1) if seq_per_step else 0
    if batch_size is None:
        B = inferred_B
    else:
        if batch_size < inferred_B:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= inferred B ({inferred_B})"
            )
        B = int(batch_size)

    if offset == 0:
        return empty_token_batch(
            B, grouping_field=gf, modality_names=names, modality_map=mmap
        )

    return TokenBatch(
        modality_ids=np.concatenate(modality_ids),
        ids=np.concatenate(ids),
        values=np.concatenate(values),
        modality_names=names,
        modality_map=dict(mmap),
        sequence_ids=np.concatenate(seq_ids),
        grouping_ids=np.concatenate(grouping_ids),
        prediction_indices=np.asarray(prediction_indices, dtype=np.int64),
        grouping_field=gf,
        objective_fields=_stack_objective_fields(
            steps, sequence_ids=seq_per_step, grouping_field=gf
        ),
        B=B,
    )
