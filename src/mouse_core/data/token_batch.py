"""StepTokens (one step) and TensorDict (packed multi-sequence batch)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, NonTensorData


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


def _numeric_field_ids(value, mask, names):
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("numeric_field_names must contain unique nonempty field identities")
    mask = np.asarray(mask, dtype=bool)
    if value is None:
        if mask.any():
            raise ValueError("Numeric tokens require numeric_field_ids and numeric_field_names")
        return np.full(mask.shape, -1, dtype=np.int64)
    ids = _sequence_id_array(value)
    if ids.shape != mask.shape or np.any(ids[~mask] != -1):
        raise ValueError("numeric_field_ids must match tokens, with -1 on nonnumeric tokens")
    if np.any(ids[mask] < 0) or np.any(ids[mask] >= len(names)):
        raise ValueError("numeric_field_ids must identify a configured numeric input field")
    return ids


def _position_arrays(values, mask, shape):
    """Optional position scalars stay float64 until sinusoidal features are formed."""
    if (values is None) != (mask is None):
        raise ValueError("position_values and position_mask must be supplied together")
    if values is None:
        return np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=bool)
    values, mask = np.asarray(values, dtype=np.float64), np.asarray(mask)
    if values.shape != shape or mask.shape != shape:
        raise ValueError("position_values and position_mask must match token_ids shape")
    if mask.dtype != np.bool_:
        raise TypeError("position_mask must be boolean")
    if not np.all(np.isfinite(values[mask])):
        raise ValueError("position_values must contain finite numeric scalars")
    return values, mask


@dataclass
class StepTokens:
    """One step's token IDs, numeric payloads, and tokenizer-marked head tokens.

    ``tokenizer_ids`` selects a name in ``tokenizer_names`` for each token;
    IDs remain local to that tokenizer's vocabulary and embedding table.
    ``numeric_mask`` marks numeric tokens even when their scaled value is zero.
    Optional ``position_values`` / ``position_mask`` carry numeric context per token.
    Omitted arrays become zeros and a false mask; zero can be present.
    A passive step can contain no head tokens, or even no input tokens, while
    retaining its objective labels.
    """

    token_ids: np.ndarray  # [T] vocabulary rows
    numeric_values: np.ndarray  # [T] scaled values; zero for nonnumeric tokens
    numeric_mask: np.ndarray  # [T] bool
    head_token_mask: np.ndarray  # [T] bool
    objective_fields: dict[str, Any] = field(default_factory=dict)
    tokenizer_ids: np.ndarray | None = None
    numeric_field_ids: np.ndarray | None = None
    numeric_field_names: tuple[str, ...] = ()
    tokenizer_names: tuple[str, ...] = ("default",)
    position_values: np.ndarray | None = None
    position_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids, dtype=np.int64)
        if token_ids.ndim != 1:
            raise ValueError("StepTokens requires a 1-D token_ids array")
        self.position_values, self.position_mask = _position_arrays(self.position_values, self.position_mask, token_ids.shape)
        self.numeric_field_names = tuple(self.numeric_field_names)
        self.numeric_field_ids = _numeric_field_ids(self.numeric_field_ids, self.numeric_mask, self.numeric_field_names)
        self.tokenizer_names = tuple(self.tokenizer_names)
        self.tokenizer_ids = _tokenizer_ids(self.tokenizer_ids, token_ids.shape, self.tokenizer_names)
        for name, dtype in (("token_ids", np.int64), ("numeric_values", np.float32),
                            ("numeric_mask", np.bool_), ("head_token_mask", np.bool_)):
            array = np.asarray(getattr(self, name), dtype=dtype)
            if array.shape != token_ids.shape:
                raise ValueError(f"{name} must have shape [{token_ids.size}], got {array.shape}")
            setattr(self, name, array)

    @property
    def T(self) -> int:
        return int(self.token_ids.shape[0])


def make_token_batch(
    *, token_ids, numeric_values, numeric_mask, sequence_ids,
    head_token_mask, num_sequences: int,
    tokenizer_ids=None, tokenizer_names=("default",),
    numeric_field_ids=None, numeric_field_names=(),
    position_values=None, position_mask=None,
) -> TensorDict:
    """Build CPU tensor inputs without scanning or validating batch contents.

    Token, prediction and step arrays have independent lengths, so batch_size
    is empty. The producer owns their alignment; this is an ordinary TensorDict.

    Token arrays have shape [L]: token_ids, tokenizer_ids, numeric_values,
    numeric_mask, numeric_field_ids, position_values, position_mask, sequence_ids.
    sequence_ids defines attention boundaries. Token IDs are local to the table
    selected by tokenizer_ids; numeric/position masks distinguish absent values
    from zero. Numeric-field IDs select Fourier projections (-1 when absent).
    position_values stays float64 until the position features are formed.

    head_token_mask [L] selects tokens passed to prediction heads. Step mappings
    belong to objective_data, not model inputs. Names and num_sequences are
    non-tensor metadata; the count includes empty decode sequences.
    """
    ids = torch.as_tensor(token_ids, dtype=torch.int64, device="cpu")
    shape = ids.shape
    return TensorDict({
        "token_ids": ids,
        "numeric_values": torch.as_tensor(numeric_values, dtype=torch.float32, device="cpu"),
        "numeric_mask": torch.as_tensor(numeric_mask, dtype=torch.bool, device="cpu"),
        "sequence_ids": torch.as_tensor(sequence_ids, dtype=torch.int64, device="cpu"),
        "head_token_mask": torch.as_tensor(head_token_mask, dtype=torch.bool, device="cpu"),
        "tokenizer_ids": (torch.zeros(shape, dtype=torch.int64, device="cpu") if tokenizer_ids is None
                          else torch.as_tensor(tokenizer_ids, dtype=torch.int64, device="cpu")),
        "numeric_field_ids": (torch.full(shape, -1, dtype=torch.int64, device="cpu") if numeric_field_ids is None
                              else torch.as_tensor(numeric_field_ids, dtype=torch.int64, device="cpu")),
        "position_values": (torch.zeros(shape, dtype=torch.float64, device="cpu") if position_values is None
                            else torch.as_tensor(position_values, dtype=torch.float64, device="cpu")),
        "position_mask": (torch.zeros(shape, dtype=torch.bool, device="cpu") if position_mask is None
                          else torch.as_tensor(position_mask, dtype=torch.bool, device="cpu")),
        "tokenizer_names": NonTensorData(tuple(tokenizer_names)),
        "numeric_field_names": NonTensorData(tuple(numeric_field_names)),
        "num_sequences": NonTensorData(num_sequences),
    }, batch_size=[], device="cpu")


def empty_token_batch(*, num_sequences: int = 0) -> TensorDict:
    """No tokens or steps, retaining empty decode sequences."""
    return make_token_batch(
        token_ids=[], numeric_values=[], numeric_mask=[], sequence_ids=[],
        head_token_mask=[],
        num_sequences=num_sequences,
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
    step_sequence_ids belongs to objective_data. A scalar ``action`` may be
    ``None`` for an automatic transition. Packing represents it as NaN in
    the float64 action tensor; source steps retain their nullable actions.
    """
    n = len(steps)
    keys: set[str] = set()
    for st in steps:
        keys.update(st.objective_fields)
    if "group_id" in keys:
        raise ValueError(
            "group_id is not an objective column; use objective_data['step_sequence_ids']."
        )
    if "head_output_count" in keys:
        raise ValueError("head_output_count is not an objective column; use objective_data['head_step_ids'].")
    if "action_executed" in keys:
        raise ValueError(
            "action_executed is not supported; store action=None "
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
        if key == "action":
            # Float64 preserves practical discrete IDs while carrying the missing
            # value in the same tensor. Do not replace NaN with an action index.
            arrays = [np.asarray(np.nan if v is None else
                                 v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v,
                                 dtype=np.float64) for v in raw]
        else:
            arrays = [_as_field_array(v) for v in raw]
        present = arrays
        # Column dtype is decided by *every* step, not the first one: a single
        # float anywhere makes the column float32, so int-typed steps can
        # never truncate later float values.
        dtype = np.float64 if key == "action" else (
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


def _fields_to_tensors(fields: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """CPU tensors from stacked numpy objective columns."""
    tensors: dict[str, torch.Tensor] = {}
    for k, v in fields.items():
        arr = np.asarray(v)
        if arr.dtype.kind == "f":
            tensors[k] = torch.from_numpy(arr.astype(np.float64 if k == "action" else np.float32, copy=False))
        else:
            tensors[k] = torch.from_numpy(arr.astype(np.int64, copy=False))
    return tensors


def pack_token_batch(
    *,
    steps: Sequence[StepTokens],
    sequence_ids: Sequence[int] | None = None,
    batch_size: int | None = None,
) -> tuple[TensorDict, TensorDict]:
    """Pack per-step :class:`StepTokens` into model and objective inputs.

    Returns ``(inputs, objective_data)``. Step sequence boundaries live in
    ``objective_data["step_sequence_ids"]``.

    Ragged float vector columns in ``objective_data`` are right-padded with
    ``-inf``, the sentinel objectives exclude as "action does not exist".
    Ragged integer columns raise — there is no integer sentinel.

    Nullable scalar ``action`` fields become numeric tensors with a derived
    missing-action NaN. This representation belongs only to the batch; it is not
    an input column or a stored field on :class:`StepTokens`.

    """
    empty_objective = TensorDict({"step_sequence_ids": torch.empty(0, dtype=torch.int64, device="cpu"),
                                  "head_step_ids": torch.empty(0, dtype=torch.int64, device="cpu")}, batch_size=[], device="cpu")
    if not steps:
        if batch_size is None:
            return empty_token_batch(num_sequences=0), empty_objective
        return empty_token_batch(num_sequences=batch_size), empty_objective

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

    position_values: list[np.ndarray] = []
    position_mask: list[np.ndarray] = []
    numeric_field_ids: list[np.ndarray] = []
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

    tokenizer_names = steps[0].tokenizer_names
    numeric_field_names = steps[0].numeric_field_names
    for st in steps:
        if st.numeric_field_names != numeric_field_names:
            raise ValueError("All steps must use the same numeric_field_names")
        if st.tokenizer_names != tokenizer_names:
            raise ValueError("All steps must use the same tokenizer_names in the same order")
        t = st.T
        assert st.position_values is not None
        assert st.position_mask is not None
        assert st.numeric_field_ids is not None
        position_values.append(st.position_values)
        position_mask.append(st.position_mask)
        numeric_field_ids.append(st.numeric_field_ids)
        numeric_mask.append(st.numeric_mask)
        token_ids.append(st.token_ids)
        numeric_values.append(st.numeric_values)
        assert st.tokenizer_ids is not None
        tokenizer_ids.append(st.tokenizer_ids)
        head_token_masks.append(st.head_token_mask)
        step_lengths.append(t)

    # Compute packed metadata in bulk, including zero-token/passive steps.
    lengths = np.asarray(step_lengths, dtype=np.int64)
    head_token_mask = np.concatenate(head_token_masks)
    head_token_ids = np.flatnonzero(head_token_mask)
    head_step_ids = np.searchsorted(np.cumsum(lengths), head_token_ids, side="right")
    fields = _stack_objective_fields(steps)
    inputs = make_token_batch(
        position_values=np.concatenate(position_values),
        position_mask=np.concatenate(position_mask),
        numeric_field_ids=np.concatenate(numeric_field_ids),
        numeric_field_names=numeric_field_names,
        numeric_mask=np.concatenate(numeric_mask),
        token_ids=np.concatenate(token_ids),
        numeric_values=np.concatenate(numeric_values),
        sequence_ids=np.repeat(seq_per_step, lengths),
        head_token_mask=head_token_mask,
        tokenizer_ids=np.concatenate(tokenizer_ids),
        tokenizer_names=tokenizer_names,
        num_sequences=B,
    )
    objective_data = _fields_to_tensors(fields)
    objective_data["head_step_ids"] = torch.from_numpy(head_step_ids)
    objective_data["step_sequence_ids"] = torch.tensor(seq_per_step, dtype=torch.int64, device="cpu")
    return inputs, TensorDict(objective_data, batch_size=[], device="cpu")
