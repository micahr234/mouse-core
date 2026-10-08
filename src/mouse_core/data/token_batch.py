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


def step_counts_from_group_id(
    *,
    group_id: np.ndarray | None,
    B: int,
) -> np.ndarray:
    """Per-group step counts ``[B]`` from flat ``group_id`` ``[N]``.

    Missing IDs (empty decode rows) become zeros when ``minlength=B``.
    """
    if B <= 0:
        return np.zeros(0, dtype=np.int64)
    if group_id is None:
        return np.zeros(B, dtype=np.int64)
    sid = np.asarray(group_id, dtype=np.int64).reshape(-1)
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

    Produced by :class:`~mouse_core.data.tokenizer.Tokenizer`. Pack many steps into
    a :class:`TokenBatch` with :func:`pack_token_batch`.

    ``modality_ids[t]`` indexes ``modality_names``; type/kind comes from
    ``modality_map[modality_names[modality_ids[t]]]``.

    ``positions[t]`` is the token's 0-based index among the tokens of the
    same modality in this step (image patch, text run offset).

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
    head_output_mask: np.ndarray  # [T] bool
    objective_fields: dict[str, Any] = field(default_factory=dict)
    group_start_modality_ids: np.ndarray | None = None
    group_start_ids: np.ndarray | None = None
    group_start_values: np.ndarray | None = None
    group_start_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
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
        group_start_arrays = (
            self.group_start_modality_ids,
            self.group_start_ids,
            self.group_start_values,
            self.group_start_positions,
        )
        if all(a is None for a in group_start_arrays):
            return
        if any(a is None for a in group_start_arrays):
            raise ValueError(
                "group_start_modality_ids / group_start_ids / "
                "group_start_values / group_start_positions must all be "
                "set or all None"
            )
        pt = int(np.asarray(self.group_start_ids).shape[0])
        if pt == 0:
            raise ValueError("group_start token arrays must be non-empty when set")
        for name in (
            "group_start_modality_ids",
            "group_start_ids",
            "group_start_values",
            "group_start_positions",
        ):
            arr = np.asarray(getattr(self, name))
            if arr.shape != (pt,):
                raise ValueError(f"{name} must have shape [{pt}], got {arr.shape}")
            object.__setattr__(self, name, arr)
        pmids = np.asarray(self.group_start_modality_ids, dtype=np.int64)
        if pmids.min(initial=0) < 0 or pmids.max(initial=0) >= len(names):
            raise ValueError(
                f"group_start_modality_ids must be in [0, {len(names)}), got "
                f"min={int(pmids.min())} max={int(pmids.max())}"
            )
        object.__setattr__(self, "group_start_modality_ids", pmids)
        object.__setattr__(
            self, "group_start_ids", np.asarray(self.group_start_ids, dtype=np.int64)
        )
        object.__setattr__(
            self,
            "group_start_values",
            np.asarray(self.group_start_values, dtype=np.float32),
        )
        ppos = np.asarray(self.group_start_positions, dtype=np.int64)
        if ppos.min(initial=0) < 0:
            raise ValueError(
                f"group_start_positions must be >= 0, got min={int(ppos.min())}"
            )
        object.__setattr__(self, "group_start_positions", ppos)

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

    * text / token / image — ``ids[i]`` is a vocab row; ``values[i]`` is 0
    * numeric — ``ids[i]`` is a vocab row of the literal ``format``;
      ``values[i]`` is the scalar mapped from ``[fourier_min, fourier_max]``
      onto ``[-1, 1]``

    Attributes:
        modality_ids: ``[L]`` int64 — index into ``modality_names``.
        modality_names: interned modality names for this batch.
        modality_map: name → :class:`ModalityInfo` (type/kind lookup).
        ids: ``[L]`` int64 — vocab / table row id.
        values: ``[L]`` float32 — 0 for text / token / image. For numeric,
            the scalar mapped onto ``[-1, 1]``.
        positions: ``[L]`` int64 — index of the token among its modality's
            tokens within its step (see :class:`StepTokens`).
        group_ids: ``[L]`` int64 — which of the ``B`` groups each token belongs to.
            One group is one sample. Attention and loss do not cross it.
        head_output_indices: ``[P]`` int64 — token index of every head-output
            token, strictly increasing.
        head_output_steps: ``[P]`` int64 — step id ``0..N-1`` of each
            head-output token (dense, non-decreasing).
        B: Number of groups.
    """

    modality_ids: np.ndarray
    ids: np.ndarray
    values: np.ndarray
    positions: np.ndarray
    modality_names: tuple[str, ...]
    modality_map: dict[str, ModalityInfo]
    group_ids: np.ndarray
    head_output_indices: np.ndarray
    head_output_steps: np.ndarray
    B: int = 0

    def __post_init__(self) -> None:
        names, mmap = _validate_modality_table(self.modality_names, self.modality_map)
        object.__setattr__(self, "modality_names", names)
        object.__setattr__(self, "modality_map", mmap)
        L = int(np.asarray(self.modality_ids).shape[0])
        for name in (
            "modality_ids",
            "ids",
            "values",
            "positions",
            "group_ids",
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
            sids = np.asarray(self.group_ids, dtype=np.int64)
            if int(sids.min()) < 0 or int(sids.max()) >= self.B:
                raise ValueError(
                    f"group_ids must be in [0, {self.B}), got "
                    f"min={int(sids.min())} max={int(sids.max())}"
                )
            if bool(np.any(sids[1:] < sids[:-1])):
                raise ValueError(
                    "group_ids must be non-decreasing: every group's tokens "
                    "must form one contiguous block, in group order (the model "
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
                f"counts from group_ids [{int(counts.sum())}] (B={self.B})"
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
        """Steps per group ``[B]``, from each step's first head-output token."""
        if self.P == 0:
            return np.zeros(self.B, dtype=np.int64)
        first = np.ones(self.P, dtype=bool)
        first[1:] = self.head_output_steps[1:] != self.head_output_steps[:-1]
        return step_counts_from_group_id(
            group_id=np.asarray(self.group_ids, dtype=np.int64)[
                self.head_output_indices[first]
            ],
            B=self.B,
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
            "group_ids": _long(self.group_ids),
            "head_output_indices": _long(self.head_output_indices),
            "head_output_steps": _long(self.head_output_steps),
            "B": self.B,
        }


def empty_token_batch(
    *,
    B: int = 0,
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
        group_ids=np.zeros(0, dtype=np.int64),
        head_output_indices=np.zeros(0, dtype=np.int64),
        head_output_steps=np.zeros(0, dtype=np.int64),
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


def _stack_objective_fields(steps: Sequence[StepTokens]) -> dict[str, np.ndarray]:
    """Stack per-step ``objective_fields`` into ``[N]`` / ``[N, ...]`` arrays.

    Ragged float vector columns are right-padded with ``-inf`` — the sentinel
    for actions that do not exist, which objectives exclude. Ragged integer
    columns raise: there is no integer sentinel, so the caller must pad them
    explicitly or store the field as float. ``group_id`` is not a column:
    packing returns that tensor beside this dict.
    """
    n = len(steps)
    keys: set[str] = set()
    for st in steps:
        keys.update(st.objective_fields)
    if "group_id" in keys:
        raise ValueError(
            "group_id is not an objective column; pack_token_batch returns "
            "it beside objective_data."
        )
    keys.discard("head_output_count")

    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        raw = [st.objective_fields.get(key) for st in steps]
        missing = [i for i, v in enumerate(raw) if v is None]
        if missing:
            raise KeyError(
                f"objective field {key!r} is missing on steps {missing}; "
                "every step must provide every stacked key"
            )
        arrays = [_as_field_array(v) for v in raw]
        present = arrays
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
                buf[i] = a.reshape(())
            out[key] = buf
        else:
            shapes = [a.shape for a in present]
            max_shape = tuple(max(s[d] for s in shapes) for d in range(ndim))
            ragged = any(s != max_shape for s in shapes)
            if ragged and dtype is not np.float32:
                raise ValueError(
                    f"objective field {key!r} has ragged shapes "
                    f"{sorted(set(shapes))} and an integer dtype; only float "
                    "columns can be padded (with -inf). Pad the values "
                    "explicitly or store the field as float."
                )
            # Pad ragged float vectors with -inf, the library-wide sentinel for
            # actions that do not exist (SP / SV exclude non-finite entries).
            # Finite zero padding would silently train toward fabricated values.
            fill = -np.inf if ragged else 0
            buf = np.full((n, *max_shape), fill, dtype=dtype)
            for i, a in enumerate(arrays):
                slicer = tuple(slice(0, a.shape[d]) for d in range(a.ndim))
                buf[i][slicer] = a
            out[key] = buf

    return out


def to_device(
    *,
    data: Mapping[str, torch.Tensor],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Move every tensor in ``data`` to ``device``."""
    return {key: value.to(device) for key, value in data.items()}


def _fields_to_tensors(fields: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """CPU tensors from stacked numpy objective columns."""
    tensors: dict[str, torch.Tensor] = {}
    for k, v in fields.items():
        arr = np.asarray(v)
        if np.issubdtype(arr.dtype, np.floating):
            tensors[k] = torch.from_numpy(arr.astype(np.float32, copy=False))
        else:
            tensors[k] = torch.from_numpy(arr.astype(np.int64, copy=False))
    return tensors


def pack_token_batch(
    *,
    steps: Sequence[StepTokens],
    group_ids: Sequence[int] | None = None,
    batch_size: int | None = None,
    continuing: Sequence[bool] | None,
) -> tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor]:
    """Pack per-step :class:`StepTokens` into model and objective inputs.

    All steps must share the same ``modality_names`` and ``modality_map``.
    Returns ``(inputs, objective_data, group_id)``. ``group_id`` is
    int64 ``[N]``, one id per step. It is not a key in ``objective_data``.
    Move ``objective_data`` with :func:`to_device` and ``group_id`` with
    ``Tensor.to``.

    Ragged float vector columns in ``objective_data`` are right-padded with
    ``-inf``, the sentinel objectives exclude as "action does not exist".
    Ragged integer columns raise — there is no integer sentinel.

    When a step carries ``group_start_*`` tokens (from input fields whose
    ``when`` callable is true with ``group_start=True`` and false with
    ``group_start=False``), they are inserted once, on the first step of
    each sequence. ``continuing`` is length ``B``: ``True`` means that
    sequence already has cached tokens, so this pack does not emit the
    sample-start prefix again. Pass ``None`` when no sequence is a
    continuation (a fresh pack, including an empty batch).
    """
    empty_objective: dict[str, torch.Tensor] = {}
    empty_group_id = torch.zeros(0, dtype=torch.int64)
    if not steps:
        if batch_size is None:
            return empty_token_batch(B=0), empty_objective, empty_group_id
        return empty_token_batch(B=batch_size), empty_objective, empty_group_id

    names = steps[0].modality_names
    mmap = steps[0].modality_map
    for i, st in enumerate(steps):
        if st.modality_names != names:
            raise ValueError(
                f"steps[{i}].modality_names {st.modality_names!r} != {names!r}"
            )
        if st.modality_map != mmap:
            raise ValueError(f"steps[{i}].modality_map does not match steps[0]")

    if group_ids is None:
        seq_per_step = [0] * len(steps)
    else:
        if len(group_ids) != len(steps):
            raise ValueError(
                f"group_ids length ({len(group_ids)}) must match "
                f"steps ({len(steps)})"
            )
        seq_per_step = [int(s) for s in group_ids]
        if any(s < 0 for s in seq_per_step):
            raise ValueError(
                f"group_ids must be >= 0, got min={min(seq_per_step)}"
            )

    modality_ids: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    values: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    seq_ids: list[np.ndarray] = []
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

    if continuing is None:
        started = [False] * B
    else:
        if len(continuing) != B:
            raise ValueError(
                f"continuing length ({len(continuing)}) must match batch_size ({B})"
            )
        started = [bool(flag) for flag in continuing]

    offset = 0
    for step_idx, (st, sid) in enumerate(zip(steps, seq_per_step)):
        emit_group_start = st.group_start_ids is not None and not started[sid]
        if emit_group_start:
            assert st.group_start_modality_ids is not None
            assert st.group_start_ids is not None
            assert st.group_start_values is not None
            assert st.group_start_positions is not None
            pt = int(st.group_start_ids.shape[0])
            modality_ids.append(st.group_start_modality_ids)
            ids.append(st.group_start_ids)
            values.append(st.group_start_values)
            positions.append(st.group_start_positions)
            seq_ids.append(np.full(pt, sid, dtype=np.int64))
            offset += pt
        t = st.T
        modality_ids.append(st.modality_ids)
        ids.append(st.ids)
        values.append(st.values)
        positions.append(st.positions)
        seq_ids.append(np.full(t, sid, dtype=np.int64))
        ho = np.flatnonzero(st.head_output_mask)
        head_output_indices.extend((offset + ho).tolist())
        head_output_steps.extend([step_idx] * int(ho.size))
        head_output_counts.append(int(ho.size))
        offset += t
        started[sid] = True

    fields = _stack_objective_fields(steps)
    fields["head_output_count"] = np.asarray(head_output_counts, dtype=np.int64)
    group_id = torch.tensor(seq_per_step, dtype=torch.int64)
    inputs = TokenBatch(
        modality_ids=np.concatenate(modality_ids),
        ids=np.concatenate(ids),
        values=np.concatenate(values),
        positions=np.concatenate(positions),
        modality_names=names,
        modality_map=dict(mmap),
        group_ids=np.concatenate(seq_ids),
        head_output_indices=np.asarray(head_output_indices, dtype=np.int64),
        head_output_steps=np.asarray(head_output_steps, dtype=np.int64),
        B=B,
    )
    return inputs, _fields_to_tensors(fields), group_id
