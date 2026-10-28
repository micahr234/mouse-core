"""StepTokens (one step) and TokenBatch (packed multi-sequence batch)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


_TensorCache = dict[tuple[int, torch.device, bool], tuple[object, dict[str, torch.Tensor]]]
_tensor_cache: ContextVar[_TensorCache | None] = ContextVar("token_batch_tensors", default=None)


def _tokenizer_ids(value, shape, names) -> np.ndarray:
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("tokenizer_names must contain distinct nonempty names")
    if value is None:
        if len(names) != 1:
            raise ValueError("tokenizer_ids are required for multiple tokenizers")
        return np.zeros(shape, dtype=np.int64)
    array = _sequence_id_array(value)
    if array.shape != shape or (array.size and (array.min() < 0 or array.max() >= len(names))):
        raise ValueError("tokenizer_ids must match token_ids and identify a tokenizer name")
    return array


def _sequence_id_array(value) -> np.ndarray:
    """Preserve sequence identity; coercing floats can merge separate groups."""
    array = np.asarray(value)
    if array.size and array.dtype.kind not in "iu":
        raise ValueError("sequence_ids must contain integer ids")
    if array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise ValueError("sequence_ids exceed the int64 range")
    return array.astype(np.int64, copy=False)


@contextmanager
def reuse_token_tensors() -> Iterator[None]:
    """Reuse batch conversions within one online/delayed forward pair.

    Each TokenBatch/device/inference-mode combination is converted once.
    Batches and returned tensors must be treated as read-only in this scope.
    Exiting (including on error) drops the cache; later scopes see fresh data.
    Model activations and parameters are never cached. Nested scopes have
    independent caches. The cache is local to the execution context.
    """
    cache: _TensorCache = {}
    token = _tensor_cache.set(cache)
    try:
        yield
    finally:
        _tensor_cache.reset(token)
        cache.clear()


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


@dataclass
class StepTokens:
    """One step's token IDs, numeric payloads, and tokenizer-marked head tokens.

    ``tokenizer_ids`` selects a name in ``tokenizer_names`` for each token;
    IDs remain local to that tokenizer's vocabulary and embedding table.
    ``numeric_mask`` marks numeric tokens even when their scaled value is zero.
    ``group_start_*`` arrays hold an optional prefix emitted once per sequence.
    A passive step can contain no head tokens, or even no input tokens, while
    retaining its objective labels.
    """

    token_ids: np.ndarray  # [T] vocabulary rows
    numeric_values: np.ndarray  # [T] scaled values; zero for nonnumeric tokens
    numeric_mask: np.ndarray  # [T] bool
    head_token_mask: np.ndarray  # [T] bool
    objective_fields: dict[str, Any] = field(default_factory=dict)
    group_start_token_ids: np.ndarray | None = None
    group_start_numeric_values: np.ndarray | None = None
    group_start_numeric_mask: np.ndarray | None = None
    tokenizer_ids: np.ndarray | None = None
    group_start_tokenizer_ids: np.ndarray | None = None
    tokenizer_names: tuple[str, ...] = ("default",)

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids, dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError("StepTokens requires a 1-D token_ids array")
        self.tokenizer_names = tuple(self.tokenizer_names)
        self.tokenizer_ids = _tokenizer_ids(self.tokenizer_ids, token_ids.shape, self.tokenizer_names)
        for name, dtype in (("token_ids", np.int64), ("numeric_values", np.float32),
                            ("numeric_mask", np.bool_), ("head_token_mask", np.bool_)):
            array = np.asarray(getattr(self, name), dtype=dtype)
            if array.shape != token_ids.shape:
                raise ValueError(f"{name} must have shape [{token_ids.size}], got {array.shape}")
            setattr(self, name, array)
        prefix = (self.group_start_token_ids, self.group_start_numeric_values,
                  self.group_start_numeric_mask)
        if all(array is None for array in prefix):
            if self.group_start_tokenizer_ids is not None:
                raise ValueError("group_start_tokenizer_ids requires group_start token arrays")
            return
        if any(array is None for array in prefix):
            raise ValueError("group_start_token_ids / group_start_numeric_values / group_start_numeric_mask must all be set or all None")
        prefix_ids = np.asarray(self.group_start_token_ids, dtype=np.int64)
        if prefix_ids.ndim != 1 or prefix_ids.size == 0:
            raise ValueError("group_start token arrays must be non-empty and 1-D")
        self.group_start_tokenizer_ids = _tokenizer_ids(
            self.group_start_tokenizer_ids, prefix_ids.shape, self.tokenizer_names)
        for name, dtype in (("group_start_token_ids", np.int64),
                            ("group_start_numeric_values", np.float32),
                            ("group_start_numeric_mask", np.bool_)):
            array = np.asarray(getattr(self, name), dtype=dtype)
            if array.shape != prefix_ids.shape:
                raise ValueError(f"{name} must have shape [{prefix_ids.size}], got {array.shape}")
            setattr(self, name, array)

    @property
    def T(self) -> int:
        return int(self.token_ids.shape[0])


@dataclass
class TokenBatch:
    """Packed token data for independent sequences, with no padding.

    ``token_ids``, ``numeric_values``, ``numeric_mask``, and ``sequence_ids``
    have shape ``[L]``. ``tokenizer_ids`` also has shape ``[L]`` and selects
    the embedding table by its index in ``tokenizer_names``. Token IDs are
    local to that table; numeric values
    are scaled by the tokenizer's Fourier bounds and used only where the mask
    is true. Sequence IDs define attention boundaries.

    ``head_token_ids`` contains the ``[P]`` positions marked by the
    tokenizer. ``head_step_ids[p]`` identifies the corresponding step in
    ``0..N-1``; a step may have multiple head tokens. ``B`` is the number of
    sequences, including empty decode rows. ``step_sequence_ids`` records the
    sequence of every step, including steps without readouts. Omitting it
    retains the legacy dense-head layout.
    """

    token_ids: np.ndarray
    numeric_values: np.ndarray
    numeric_mask: np.ndarray
    sequence_ids: np.ndarray
    head_token_ids: np.ndarray
    head_step_ids: np.ndarray
    B: int = 0
    step_sequence_ids: np.ndarray | None = None
    tokenizer_ids: np.ndarray | None = None
    tokenizer_names: tuple[str, ...] = ("default",)

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids, dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a 1-D array")
        L = int(token_ids.size)
        self.tokenizer_names = tuple(self.tokenizer_names)
        self.tokenizer_ids = _tokenizer_ids(self.tokenizer_ids, token_ids.shape, self.tokenizer_names)
        for name, dtype in (("token_ids", np.int64), ("numeric_values", np.float32),
                            ("numeric_mask", np.bool_), ("sequence_ids", np.int64)):
            array = (_sequence_id_array(self.sequence_ids) if name == "sequence_ids"
                     else np.asarray(getattr(self, name), dtype=dtype))
            if array.shape != (L,):
                raise ValueError(f"{name} must have shape [{L}], got {array.shape}")
            setattr(self, name, array)
        if L > 0:
            sequence_ids = self.sequence_ids
            if int(sequence_ids.min()) < 0 or int(sequence_ids.max()) >= self.B:
                raise ValueError(
                    f"sequence_ids must be in [0, {self.B}), got "
                    f"min={int(sequence_ids.min())} max={int(sequence_ids.max())}"
                )
            if bool(np.any(sequence_ids[1:] < sequence_ids[:-1])):
                raise ValueError(
                    "sequence_ids must be non-decreasing: every sequence's tokens "
                    "must form one contiguous block, in sequence order (the model "
                    "walks head_token_ids row by row and pads per block)."
                )
        head_token_ids = np.asarray(self.head_token_ids, dtype=np.int64).reshape(-1)
        self.head_token_ids = head_token_ids
        head_step_ids = np.asarray(self.head_step_ids, dtype=np.int64).reshape(-1)
        self.head_step_ids = head_step_ids
        p = int(head_token_ids.shape[0])
        if head_step_ids.shape != (p,):
            raise ValueError(
                f"head_step_ids must have shape [{p}] (one step id per "
                f"head-output token), got {head_step_ids.shape}"
            )
        if p > 0:
            if L == 0:
                raise ValueError("head_token_ids require L > 0 when P > 0")
            if int(head_token_ids.min()) < 0 or int(head_token_ids.max()) >= L:
                raise ValueError(
                    f"head_token_ids must be in [0, {L}), got "
                    f"min={int(head_token_ids.min())} max={int(head_token_ids.max())}"
                )
            if bool(np.any(head_token_ids[1:] <= head_token_ids[:-1])):
                raise ValueError("head_token_ids must be strictly increasing")
            diffs = np.diff(head_step_ids)
            if int(head_step_ids[0]) < 0 or bool(np.any(diffs < 0)) or (
                self.step_sequence_ids is None and (int(head_step_ids[0]) != 0 or bool(np.any(diffs > 1)))
            ):
                raise ValueError(
                    "head_step_ids must be dense non-decreasing step ids "
                    f"starting at 0, got {head_step_ids.tolist()}"
                )
            head_sequences = self.sequence_ids[head_token_ids]
            if np.any((diffs == 0) & (head_sequences[1:] != head_sequences[:-1])):
                raise ValueError("All head tokens of a step must belong to one sequence")
        if self.step_sequence_ids is None:
            first = np.ones(p, dtype=bool)
            first[1:] = head_step_ids[1:] != head_step_ids[:-1]
            self.step_sequence_ids = self.sequence_ids[head_token_ids[first]]
        else:
            ids = _sequence_id_array(self.step_sequence_ids)
            if ids.ndim != 1 or (ids.size and (ids.min() < 0 or ids.max() >= self.B
                                               or np.any(np.diff(ids) < 0))):
                raise ValueError("step_sequence_ids must be ordered sequence IDs in [0, B)")
            if p and (head_step_ids[-1] >= ids.size
                      or np.any(ids[head_step_ids] != self.sequence_ids[head_token_ids])):
                raise ValueError("head_step_ids must identify steps in the matching sequence")
            self.step_sequence_ids = ids
        counts = self.step_counts()
        if int(counts.sum()) != self.N:
            raise ValueError(
                f"step count [{self.N}] must equal sum of per-sequence step "
                f"counts from sequence_ids [{int(counts.sum())}] (B={self.B})"
            )

    @property
    def L(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def N(self) -> int:
        """Number of steps."""
        assert self.step_sequence_ids is not None
        return int(self.step_sequence_ids.size)

    @property
    def P(self) -> int:
        """Number of head-output tokens; passive steps contribute none."""
        return int(self.head_token_ids.shape[0])

    @property
    def S(self) -> int:
        """Max steps in the batch (convenience; rows may be shorter)."""
        if self.B <= 0:
            return 0
        counts = self.step_counts()
        return int(counts.max()) if counts.size else 0

    def step_counts(self) -> np.ndarray:
        """Steps per sequence, including steps without readouts."""
        return step_counts_from_group_id(group_id=self.step_sequence_ids, B=self.B)

    def to_tensors(self, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
        """Move arrays to tensors; reuse them inside ``reuse_token_tensors``."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        elif dev.type == "cpu":
            dev = torch.device("cpu")
        cache = _tensor_cache.get()
        key = (id(self), dev, torch.is_inference_mode_enabled())
        if cache is not None and key in cache:
            return cache[key][1]

        def _long(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.int64)).to(dev)

        def _float(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)

        tensors = {
            "token_ids": _long(self.token_ids),
            "numeric_values": _float(self.numeric_values),
            "numeric_mask": torch.from_numpy(self.numeric_mask).to(dev),
            "sequence_ids": _long(self.sequence_ids),
            "head_token_ids": _long(self.head_token_ids),
            "head_step_ids": _long(self.head_step_ids),
        }
        assert self.tokenizer_ids is not None
        tensors["tokenizer_ids"] = _long(self.tokenizer_ids)
        if cache is not None:
            # Retain the batch as well, so its id cannot be recycled in scope.
            cache[key] = (self, tensors)
        return tensors


def empty_token_batch(*, B: int = 0) -> TokenBatch:
    """Empty batch (L=0, N=0); all ``B`` sequences have zero steps."""
    return TokenBatch(
        token_ids=np.zeros(0, dtype=np.int64),
        numeric_values=np.zeros(0, dtype=np.float32),
        numeric_mask=np.zeros(0, dtype=bool),
        sequence_ids=np.zeros(0, dtype=np.int64),
        head_token_ids=np.zeros(0, dtype=np.int64),
        head_step_ids=np.zeros(0, dtype=np.int64),
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
    if arr.dtype.kind == "f":
        return arr.astype(np.float32, copy=False)
    if arr.dtype.kind in "iub":
        return arr.astype(np.int64, copy=False)
    return arr


def _stack_objective_fields(steps: Sequence[StepTokens]) -> dict[str, np.ndarray]:
    """Stack per-step ``objective_fields`` into ``[N]`` / ``[N, ...]`` arrays.

    Ragged float vector columns are right-padded with ``-inf`` — the sentinel
    for actions that do not exist, which objectives exclude. Ragged integer
    columns raise: there is no integer sentinel, so the caller must pad them
    explicitly or store the field as float. ``group_id`` is not a column:
    packing returns that tensor beside this dict. A scalar ``action`` may be
    ``None`` for an automatic transition. Only here is it replaced with zero
    and paired with the derived ``action_executed`` mask; source steps retain
    their nullable actions.
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
    if "action_executed" in keys:
        raise ValueError(
            "action_executed is derived during packing; store action=None "
            "for an automatic transition instead."
        )

    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        raw = [st.objective_fields.get(key) for st in steps]
        missing = [
            i
            for i, st in enumerate(steps)
            if key not in st.objective_fields or (raw[i] is None and key != "action")
        ]
        if missing:
            raise KeyError(
                f"objective field {key!r} is missing on steps {missing}; "
                "every step must provide every stacked key"
            )
        arrays = [_as_field_array(v) for v in raw]
        if key == "action":
            out["action_executed"] = np.asarray([v is not None for v in raw])
        present = arrays
        # Column dtype is decided by *every* step, not the first one: a single
        # float anywhere makes the column float32, so int-typed steps can
        # never truncate later float values.
        dtype = (
            np.float32
            if any(a.dtype.kind == "f" for a in present)
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
            out[key] = np.asarray(arrays, dtype=dtype)
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
            if not ragged:
                out[key] = np.stack(arrays, dtype=dtype, casting="unsafe")
                continue
            # Pad ragged float vectors with -inf, the library-wide sentinel for
            # actions that do not exist (SP / SV exclude non-finite entries).
            # Finite zero padding would silently train toward fabricated values.
            buf = np.full((n, *max_shape), -np.inf, dtype=dtype)
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
        if arr.dtype.kind == "f":
            tensors[k] = torch.from_numpy(arr.astype(np.float32, copy=False))
        else:
            tensors[k] = torch.from_numpy(arr.astype(np.int64, copy=False))
    return tensors


def pack_token_batch(
    *,
    steps: Sequence[StepTokens],
    sequence_ids: Sequence[int] | None = None,
    batch_size: int | None = None,
    continuing: Sequence[bool] | None,
) -> tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor]:
    """Pack per-step :class:`StepTokens` into model and objective inputs.

    Returns ``(inputs, objective_data, group_id)``. ``group_id`` is
    int64 ``[N]``, one id per step. It is not a key in ``objective_data``.
    Move ``objective_data`` with :func:`to_device` and ``group_id`` with
    ``Tensor.to``.

    Ragged float vector columns in ``objective_data`` are right-padded with
    ``-inf``, the sentinel objectives exclude as "action does not exist".
    Ragged integer columns raise — there is no integer sentinel.

    Nullable scalar ``action`` fields become numeric tensors with a derived
    ``action_executed`` mask. This mask belongs only to the batch; it is not
    an input column or a stored field on :class:`StepTokens`.

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

    if sequence_ids is None:
        seq_per_step = [0] * len(steps)
    else:
        if len(sequence_ids) != len(steps):
            raise ValueError(
                f"sequence_ids length ({len(sequence_ids)}) must match "
                f"steps ({len(steps)})"
            )
        ids = _sequence_id_array(sequence_ids)
        if ids.ndim != 1:
            raise ValueError("sequence_ids must be a 1-D sequence")
        seq_per_step = ids.tolist()
        if any(s < 0 for s in seq_per_step):
            raise ValueError(
                f"sequence_ids must be >= 0, got min={min(seq_per_step)}"
            )

    numeric_mask: list[np.ndarray] = []
    token_ids: list[np.ndarray] = []
    numeric_values: list[np.ndarray] = []
    tokenizer_ids: list[np.ndarray] = []
    head_token_masks: list[np.ndarray] = []
    step_lengths: list[int] = []

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

    tokenizer_names = steps[0].tokenizer_names
    for st, sid in zip(steps, seq_per_step):
        if st.tokenizer_names != tokenizer_names:
            raise ValueError("All steps must use the same tokenizer_names in the same order")
        emit_group_start = st.group_start_token_ids is not None and not started[sid]
        pt = 0
        if emit_group_start:
            assert st.group_start_numeric_mask is not None
            assert st.group_start_token_ids is not None
            assert st.group_start_numeric_values is not None
            pt = int(st.group_start_token_ids.shape[0])
            numeric_mask.append(st.group_start_numeric_mask)
            token_ids.append(st.group_start_token_ids)
            numeric_values.append(st.group_start_numeric_values)
            assert st.group_start_tokenizer_ids is not None
            tokenizer_ids.append(st.group_start_tokenizer_ids)
            head_token_masks.append(np.zeros(pt, dtype=bool))
        t = st.T
        numeric_mask.append(st.numeric_mask)
        token_ids.append(st.token_ids)
        numeric_values.append(st.numeric_values)
        assert st.tokenizer_ids is not None
        tokenizer_ids.append(st.tokenizer_ids)
        head_token_masks.append(st.head_token_mask)
        step_lengths.append(pt + t)
        started[sid] = True

    # Compute packed metadata in bulk, including zero-token/passive steps.
    # Prefix tokens count toward their first step's length but never read out.
    lengths = np.asarray(step_lengths, dtype=np.int64)
    head_token_ids = np.flatnonzero(np.concatenate(head_token_masks))
    head_step_ids = np.searchsorted(np.cumsum(lengths), head_token_ids, side="right")
    fields = _stack_objective_fields(steps)
    fields["head_output_count"] = np.bincount(head_step_ids, minlength=len(steps))
    group_id = torch.tensor(seq_per_step, dtype=torch.int64)
    inputs = TokenBatch(
        numeric_mask=np.concatenate(numeric_mask),
        token_ids=np.concatenate(token_ids),
        numeric_values=np.concatenate(numeric_values),
        sequence_ids=np.repeat(seq_per_step, lengths),
        head_token_ids=head_token_ids,
        head_step_ids=head_step_ids,
        step_sequence_ids=np.asarray(seq_per_step, dtype=np.int64),
        tokenizer_ids=np.concatenate(tokenizer_ids),
        tokenizer_names=tokenizer_names,
        B=B,
    )
    return inputs, _fields_to_tensors(fields), group_id
