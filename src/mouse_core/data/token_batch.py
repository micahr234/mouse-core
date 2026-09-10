"""StepTokens (one step) and TokenBatch (packed multi-sequence batch)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict


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

    ``positions[t]`` is the token's 0-based index among the tokens of the
    same modality in this step (coordinate of a continuous vector, learnable
    slot, image patch, text run offset). Embedders use it to give each token
    of a multi-token modality its own type vector.

    ``head_output_mask[t]`` marks the step's head-output tokens (the positions
    the model reads Q / action outputs from). Tokenizers set it from the
    input field flagged ``head_output=True``; every step must have at least
    one head-output token, and may have several.
    """

    modality_ids: np.ndarray  # [T] index into modality_names
    ids: np.ndarray  # [T]
    values: np.ndarray  # [T]
    positions: np.ndarray  # [T] index within modality within the step
    modality_names: tuple[str, ...]
    modality_map: dict[str, ModalityInfo]
    grouping_id: int
    grouping_field: str
    head_output_mask: np.ndarray  # [T] bool
    objective_fields: dict[str, Any] = field(default_factory=dict)
    group_prefix_modality_ids: np.ndarray | None = None
    group_prefix_ids: np.ndarray | None = None
    group_prefix_values: np.ndarray | None = None
    group_prefix_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.grouping_field:
            raise ValueError("StepTokens requires a non-empty grouping_field")
        names, mmap = _validate_modality_table(self.modality_names, self.modality_map)
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "modality_map", mmap)
        t = int(np.asarray(self.modality_ids).shape[0])
        if t == 0:
            raise ValueError("StepTokens must contain at least one token")
        for name in ("modality_ids", "ids", "values", "positions"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (t,):
                raise ValueError(f"{name} must have shape [{t}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        pos = np.asarray(self.positions, dtype=np.int64)
        if pos.min(initial=0) < 0:
            raise ValueError(f"positions must be >= 0, got min={int(pos.min())}")
        object.__setattr__(self, "positions", pos)
        mids = np.asarray(self.modality_ids, dtype=np.int64)
        if mids.min(initial=0) < 0 or mids.max(initial=0) >= len(names):
            raise ValueError(
                f"modality_ids must be in [0, {len(names)}), got "
                f"min={int(mids.min())} max={int(mids.max())}"
            )
        object.__setattr__(self, "modality_ids", mids)
        object.__setattr__(self, "grouping_id", int(self.grouping_id))
        mask = np.asarray(self.head_output_mask, dtype=bool)
        if mask.shape != (t,):
            raise ValueError(
                f"head_output_mask must have shape [{t}], got {mask.shape}"
            )
        if not bool(mask.any()):
            raise ValueError(
                "step has no head-output tokens; the tokenizer input field "
                "flagged head_output=True must emit at least one token on "
                "every step (it must never be skipped)"
            )
        object.__setattr__(self, "head_output_mask", mask)
        group_prefix_arrays = (
            self.group_prefix_modality_ids,
            self.group_prefix_ids,
            self.group_prefix_values,
            self.group_prefix_positions,
        )
        if all(a is None for a in group_prefix_arrays):
            return
        if any(a is None for a in group_prefix_arrays):
            raise ValueError(
                "group_prefix_modality_ids / group_prefix_ids / "
                "group_prefix_values / group_prefix_positions must all be "
                "set or all None"
            )
        pt = int(np.asarray(self.group_prefix_ids).shape[0])
        if pt == 0:
            raise ValueError("group_prefix token arrays must be non-empty when set")
        for name in (
            "group_prefix_modality_ids",
            "group_prefix_ids",
            "group_prefix_values",
            "group_prefix_positions",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (pt,):
                raise ValueError(f"{name} must have shape [{pt}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        pmids = np.asarray(self.group_prefix_modality_ids, dtype=np.int64)
        if pmids.min(initial=0) < 0 or pmids.max(initial=0) >= len(names):
            raise ValueError(
                f"group_prefix_modality_ids must be in [0, {len(names)}), got "
                f"min={int(pmids.min())} max={int(pmids.max())}"
            )
        object.__setattr__(self, "group_prefix_modality_ids", pmids)
        object.__setattr__(
            self, "group_prefix_ids", np.asarray(self.group_prefix_ids, dtype=np.int64)
        )
        object.__setattr__(
            self,
            "group_prefix_values",
            np.asarray(self.group_prefix_values, dtype=np.float32),
        )
        ppos = np.asarray(self.group_prefix_positions, dtype=np.int64)
        if ppos.min(initial=0) < 0:
            raise ValueError(
                f"group_prefix_positions must be >= 0, got min={int(ppos.min())}"
            )
        object.__setattr__(self, "group_prefix_positions", ppos)

    @property
    def T(self) -> int:
        return int(self.modality_ids.shape[0])


@dataclass
class TokenBatch:
    """Flat concatenated token stream (no padding) plus parallel payload arrays.

    Built by :func:`pack_token_batch` from many :class:`StepTokens`. Length ``L``
    is the total number of tokens across all sequences and steps. ``N`` is the
    number of steps (ragged windows allowed); ``P = len(head_output_indices)``
    is the number of head-output tokens, ``P >= N`` (every step has at least
    one, and may have several). ``head_output_steps[p]`` maps head-output token
    ``p`` to its step ``0..N-1``. Per-sequence step counts are derived from
    the first head-output token of each step (see :meth:`step_counts`); they
    are not stored separately.

    Token type/kind is looked up via ``modality_map[modality_names[modality_ids[i]]]``:

    * discrete / learnable / image — ``ids[i]`` is a table/vocab row; ``values[i]`` is 0
    * fourier — ``values[i]`` is the scalar; ``ids[i]`` is the Fourier freq-bank index

    Attributes:
        modality_ids: ``[L]`` int64 — index into ``modality_names``.
        modality_names: interned modality names for this batch.
        modality_map: name → :class:`ModalityInfo` (type/kind lookup).
        ids: ``[L]`` int64 — discrete row id, or continuous freq-bank index.
        values: ``[L]`` float32 — continuous scalar (0 when discrete).
        positions: ``[L]`` int64 — index of the token among its modality's
            tokens within its step (see :class:`StepTokens`).
        sequence_ids: ``[L]`` int64 — which of the ``B`` sequences each token belongs to.
        grouping_ids: ``[L]`` int64 — attention group within the sequence.
        head_output_indices: ``[P]`` int64 — token index of every head-output
            token, strictly increasing.
        head_output_steps: ``[P]`` int64 — step id ``0..N-1`` of each
            head-output token (dense, non-decreasing).
        B: Number of sequences.
        grouping_field: Name of the grouping column.
    """

    modality_ids: np.ndarray
    ids: np.ndarray
    values: np.ndarray
    positions: np.ndarray
    modality_names: tuple[str, ...]
    modality_map: dict[str, ModalityInfo]
    sequence_ids: np.ndarray
    grouping_ids: np.ndarray
    head_output_indices: np.ndarray
    head_output_steps: np.ndarray
    grouping_field: str
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
            "positions",
            "sequence_ids",
            "grouping_ids",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (L,):
                raise ValueError(f"{name} must have shape [{L}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        if L > 0:
            pos = np.asarray(self.positions, dtype=np.int64)
            if int(pos.min()) < 0:
                raise ValueError(f"positions must be >= 0, got min={int(pos.min())}")
            mids = np.asarray(self.modality_ids, dtype=np.int64)
            if mids.min() < 0 or mids.max() >= len(names):
                raise ValueError(
                    f"modality_ids must be in [0, {len(names)}), got "
                    f"min={int(mids.min())} max={int(mids.max())}"
                )
        if L > 0:
            sids = np.asarray(self.sequence_ids, dtype=np.int64)
            if int(sids.min()) < 0 or int(sids.max()) >= self.B:
                raise ValueError(
                    f"sequence_ids must be in [0, {self.B}), got "
                    f"min={int(sids.min())} max={int(sids.max())}"
                )
            if bool(np.any(sids[1:] < sids[:-1])):
                raise ValueError(
                    "sequence_ids must be non-decreasing: every sequence's tokens "
                    "must form one contiguous block, in sequence order (the model "
                    "walks head_output_indices row by row and pads per block)."
                )
        pred = np.asarray(self.head_output_indices, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "head_output_indices", pred)
        psteps = np.asarray(self.head_output_steps, dtype=np.int64).reshape(-1)
        object.__setattr__(self, "head_output_steps", psteps)
        p = int(pred.shape[0])
        if psteps.shape != (p,):
            raise ValueError(
                f"head_output_steps must have shape [{p}] (one step id per "
                f"head-output token), got {psteps.shape}"
            )
        if p > 0:
            if L == 0:
                raise ValueError("head_output_indices require L > 0 when P > 0")
            if int(pred.min()) < 0 or int(pred.max()) >= L:
                raise ValueError(
                    f"head_output_indices must be in [0, {L}), got "
                    f"min={int(pred.min())} max={int(pred.max())}"
                )
            if bool(np.any(pred[1:] <= pred[:-1])):
                raise ValueError("head_output_indices must be strictly increasing")
            diffs = np.diff(psteps)
            if int(psteps[0]) != 0 or bool(np.any((diffs < 0) | (diffs > 1))):
                raise ValueError(
                    "head_output_steps must be dense non-decreasing step ids "
                    f"starting at 0, got {psteps.tolist()}"
                )
        counts = self.step_counts()
        if int(counts.sum()) != self.N:
            raise ValueError(
                f"step count [{self.N}] must equal sum of per-sequence step "
                f"counts from sequence_ids [{int(counts.sum())}] (B={self.B})"
            )

    @property
    def L(self) -> int:
        return int(self.modality_ids.shape[0])

    @property
    def N(self) -> int:
        """Number of steps."""
        if self.head_output_steps.shape[0] == 0:
            return 0
        return int(self.head_output_steps[-1]) + 1

    @property
    def P(self) -> int:
        """Number of head-output tokens (``>= N``)."""
        return int(self.head_output_indices.shape[0])

    @property
    def S(self) -> int:
        """Max steps in the batch (convenience; rows may be shorter)."""
        if self.B <= 0:
            return 0
        counts = self.step_counts()
        return int(counts.max()) if counts.size else 0

    def step_counts(self) -> np.ndarray:
        """Steps per sequence ``[B]``, from each step's first head-output token."""
        if self.P == 0:
            return np.zeros(self.B, dtype=np.int64)
        first = np.ones(self.P, dtype=bool)
        first[1:] = self.head_output_steps[1:] != self.head_output_steps[:-1]
        return step_counts_from_sequence_id(
            np.asarray(self.sequence_ids, dtype=np.int64)[
                self.head_output_indices[first]
            ],
            self.B,
        )

    def to_tensors(self, device: torch.device | str | None = None) -> dict[str, Any]:
        """Move arrays to torch tensors on ``device`` (CPU if None)."""
        dev = torch.device(device) if device is not None else torch.device("cpu")

        def _long(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)

        def _float(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)

        return {
            "modality_ids": _long(self.modality_ids),
            "ids": _long(self.ids),
            "values": _float(self.values),
            "positions": _long(self.positions),
            "modality_names": self.modality_names,
            "modality_map": self.modality_map,
            "sequence_ids": _long(self.sequence_ids),
            "grouping_ids": _long(self.grouping_ids),
            "head_output_indices": _long(self.head_output_indices),
            "head_output_steps": _long(self.head_output_steps),
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
        positions=np.zeros(0, dtype=np.int64),
        modality_names=names,
        modality_map=mmap,
        sequence_ids=np.zeros(0, dtype=np.int64),
        grouping_ids=np.zeros(0, dtype=np.int64),
        head_output_indices=np.zeros(0, dtype=np.int64),
        head_output_steps=np.zeros(0, dtype=np.int64),
        grouping_field=grouping_field,
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
    keys.discard("head_output_count")
    keys.discard(grouping_field)

    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        raw = [st.objective_fields.get(key) for st in steps]
        arrays = [_as_field_array(v) if v is not None else None for v in raw]
        present = [a for a in arrays if a is not None]
        if not present:
            out[key] = np.zeros(n, dtype=np.int64)
            continue
        # Column dtype is decided by *every* step, not the first one: a single
        # float anywhere makes the column float32, so int-typed steps can
        # never truncate later float values.
        dtype = (
            np.float32
            if any(np.issubdtype(a.dtype, np.floating) for a in present)
            else np.int64
        )
        ndims = {a.ndim for a in present}
        if len(ndims) != 1:
            raise ValueError(
                f"objective field {key!r} mixes array ranks {sorted(ndims)} across "
                "steps; every step must provide the same rank."
            )
        ndim = ndims.pop()
        if ndim == 0:
            buf = np.zeros(n, dtype=dtype)
            for i, a in enumerate(arrays):
                if a is not None:
                    buf[i] = a.reshape(())
            out[key] = buf
        else:
            shapes = [a.shape for a in present]
            max_shape = tuple(max(s[d] for s in shapes) for d in range(ndim))
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


def _fields_to_tensordict(fields: dict[str, np.ndarray], n: int) -> TensorDict:
    """CPU ``TensorDict[N]`` from stacked numpy objective columns."""
    tensors: dict[str, torch.Tensor] = {}
    for k, v in fields.items():
        arr = np.asarray(v)
        if np.issubdtype(arr.dtype, np.floating):
            tensors[k] = torch.from_numpy(arr.astype(np.float32, copy=False))
        else:
            tensors[k] = torch.from_numpy(arr.astype(np.int64, copy=False))
    return TensorDict(tensors, batch_size=[n])


def pack_token_batch(
    steps: Sequence[StepTokens],
    *,
    sequence_ids: Sequence[int] | None = None,
    batch_size: int | None = None,
    grouping_field: str | None = None,
    prev_grouping_ids: Sequence[int | None] | None = None,
) -> tuple[TokenBatch, TensorDict]:
    """Pack per-step :class:`StepTokens` into model and objective inputs.

    All steps must share the same ``modality_names``, ``modality_map``, and
    ``grouping_field``. Returns ``(inputs, objective_data)``.

    When a step carries ``group_prefix_*`` tokens (from
    :class:`~mouse_core.data.text_tokenizer.TextTokenizer` ``group_prefix=``),
    they are inserted at the start of each grouping-field segment: the first
    step of a sequence, or a step whose ``grouping_id`` differs from the
    previous step in that sequence. ``prev_grouping_ids`` is length ``B``
    (optional ``None`` entries); pass the last grouping already in a cached
    sequence so incremental decode does not emit the group prefix again.
    """
    empty_objective = TensorDict({}, batch_size=[0])
    if not steps:
        if grouping_field is None:
            raise ValueError(
                "pack_token_batch of empty steps requires grouping_field="
            )
        if batch_size is None:
            return empty_token_batch(0, grouping_field=grouping_field), empty_objective
        return empty_token_batch(batch_size, grouping_field=grouping_field), empty_objective

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
    positions: list[np.ndarray] = []
    seq_ids: list[np.ndarray] = []
    grouping_ids: list[np.ndarray] = []
    head_output_indices: list[int] = []
    head_output_steps: list[int] = []
    head_output_counts: list[int] = []

    inferred_B = (max(seq_per_step) + 1) if seq_per_step else 0
    if batch_size is None:
        B = inferred_B
    else:
        if batch_size < inferred_B:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= inferred B ({inferred_B})"
            )
        B = int(batch_size)

    last_gid: list[int | None]
    if prev_grouping_ids is None:
        last_gid = [None] * B
    else:
        if len(prev_grouping_ids) != B:
            raise ValueError(
                f"prev_grouping_ids length ({len(prev_grouping_ids)}) must "
                f"match batch_size ({B})"
            )
        last_gid = [
            None if g is None else int(g) for g in prev_grouping_ids
        ]

    offset = 0
    for step_idx, (st, sid) in enumerate(zip(steps, seq_per_step)):
        emit_group_prefix = (
            st.group_prefix_ids is not None
            and last_gid[sid] != st.grouping_id
        )
        if emit_group_prefix:
            assert st.group_prefix_modality_ids is not None
            assert st.group_prefix_ids is not None
            assert st.group_prefix_values is not None
            assert st.group_prefix_positions is not None
            pt = int(st.group_prefix_ids.shape[0])
            modality_ids.append(st.group_prefix_modality_ids)
            ids.append(st.group_prefix_ids)
            values.append(st.group_prefix_values)
            positions.append(st.group_prefix_positions)
            seq_ids.append(np.full(pt, sid, dtype=np.int64))
            grouping_ids.append(np.full(pt, st.grouping_id, dtype=np.int64))
            offset += pt
        t = st.T
        modality_ids.append(st.modality_ids)
        ids.append(st.ids)
        values.append(st.values)
        positions.append(st.positions)
        seq_ids.append(np.full(t, sid, dtype=np.int64))
        grouping_ids.append(np.full(t, st.grouping_id, dtype=np.int64))
        ho = np.flatnonzero(st.head_output_mask)
        head_output_indices.extend((offset + ho).tolist())
        head_output_steps.extend([step_idx] * int(ho.size))
        head_output_counts.append(int(ho.size))
        offset += t
        last_gid[sid] = st.grouping_id

    if offset == 0:
        return (
            empty_token_batch(
                B, grouping_field=gf, modality_names=names, modality_map=mmap
            ),
            empty_objective,
        )

    fields = _stack_objective_fields(
        steps, sequence_ids=seq_per_step, grouping_field=gf
    )
    fields["head_output_count"] = np.asarray(head_output_counts, dtype=np.int64)
    inputs = TokenBatch(
        modality_ids=np.concatenate(modality_ids),
        ids=np.concatenate(ids),
        values=np.concatenate(values),
        positions=np.concatenate(positions),
        modality_names=names,
        modality_map=dict(mmap),
        sequence_ids=np.concatenate(seq_ids),
        grouping_ids=np.concatenate(grouping_ids),
        head_output_indices=np.asarray(head_output_indices, dtype=np.int64),
        head_output_steps=np.asarray(head_output_steps, dtype=np.int64),
        grouping_field=gf,
        B=B,
    )
    return inputs, _fields_to_tensordict(fields, inputs.N)
