"""Tokenizer — independent events → token arrays → marked StepTokens.

I/O
---
* ``tokenizer(event)`` emits token arrays from one provider/payload event.
* ``tokenizer.encode(events)`` assembles complete steps by their markers.

All fields emit into one token-ID stream, with ``tokenizer_ids`` selecting
the embedding table for each token. ``numeric`` fields also emit a
scaled scalar in ``numeric_values`` and set ``numeric_mask=True``. Text,
literal-token, and image fields set the mask to false. Pack ragged event sequences
with :meth:`Tokenizer.pack_rows`; pack already-tokenized steps with
:func:`~mouse_core.data.token_batch.pack_token_batch`.
:func:`save_tokenizer` writes the packing spec as ``tokenizer.json``;
:func:`load_tokenizer` is the one recall path. Hub upload saves it alongside
the model via :func:`~mouse_core.models.io.push_model_to_hub`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mouse_core.data.events import EventLog, INDEX_FIELDS
from mouse_core.data.step_sampling import _StepBoundary
from mouse_core.data.io_fields import coerce_io_fields
from mouse_core.data.modality import (
    KIND_IMAGE,
    KIND_NUMERIC,
    KIND_TEXT,
    KIND_TOKEN,
    _field_label,
    render_text_format,
    TokenizerModalitySpec,
    copy_keep_fields,
    resolve_tokenizer_modalities,
    unwrap_scalar,
    when_as_bool,
)
from mouse_core.data.token_batch import (
    StepTokens,
    TokenBatch,
    pack_token_batch,
)


def _objective_config(objective_fields):
    pairs = coerce_io_fields(
        objective_fields or (),
        who="tokenizer objective_fields",
        allow_empty=True,
    )
    missing_values = {}
    for spec, (_, output_name) in zip(objective_fields or (), pairs):
        if "missing_value" in spec:
            value = spec["missing_value"]
            if value is None:
                if output_name != "action":
                    raise ValueError("Only action labels may use missing_value=None")
            elif isinstance(value, list) and not value:
                value = []
            elif not isinstance(value, (bool, int, float)) or not math.isfinite(value):
                raise ValueError("objective missing_value must be a finite numeric scalar or an empty list")
            missing_values[output_name] = value
    return pairs, missing_values


def _register_special_tokens(tokenizer, specs):
    tokens = list(dict.fromkeys(render_text_format(spec, field=None) for spec in specs if spec.special))
    if tokens:
        tokenizer.add_tokens(tokens, special_tokens=True)
    result = {}
    for token in tokens:
        ids = _tokenize_ids(tokenizer, token)
        if len(ids) != 1:
            raise ValueError(f"Special token {token!r} must encode to exactly one ID")
        result[token] = ids[0]
    return result


class Tokenizer:
    """Encode each event using an ordered field list, then assemble marked steps.

    ``tokenizer(event)`` returns an event's token arrays, possibly empty. It does
    not interpret provider roles. Each field's ``provider=None`` matches all
    events; a named provider filters that field. ``head_output=True`` marks
    exactly the tokens that field emits, including optional special tokens.
    Matching fields emit in ``input_fields`` order, regardless of payload key
    order. Each field is tokenized separately; text never merges across fields
    or events, and earlier events never supply missing input values.
    No readout is appended implicitly. ``encode(events)`` groups
    those arrays using ``step_field`` and returns completed ``StepTokens`` for
    a finalized sample. Pass ``transform=tokenizer.encode`` to a data loader.

    The required ``tokenizers={name: pretrained_name_or_callable}`` dictionary
    defines the backends, even when there is only one. It includes image
    backends as callables returning discrete IDs. Set ``tokenizer=name``
    on each input field. A lone backend or one named
    ``default`` is used for fields without a selector. IDs stay local to each
    backend; ``tokenizer_names[tokenizer_ids[i]]`` identifies token i's table.
    Dictionary names are sorted so insertion order cannot change routing.

    Standard fast Hugging Face backends encode each field directly in Rust,
    without computing unused offsets or Python output dictionaries. HF owns
    any internal tokenization cache; this class retains no text-to-ID cache.
    Custom backends and HF encoding overrides keep their callable behavior.

    A changed marker starts a step before that event. Missing markers continue
    the current step; the first event requires a marker. ``episode_field``
    separates repeated step indices in different episodes. Use ``stream()``
    for chunks that can end mid-step, and explicitly flush a completed tail.

    Objective fields select payload fields or envelope metadata. A qualified
    name such as ``agent.action`` selects one provider's payload. Repeated
    labels use the last value in the step. Missing defaults apply only when
    the complete step has no value. No labels or metadata are implicit inputs.
    """

    def __init__(self, *, tokenizers: Mapping[str, Any], input_fields, step_field="step_index",
                 episode_field="episode_index", objective_fields=None, hub_kwargs=None):
        if not isinstance(tokenizers, Mapping) or not tokenizers:
            raise TypeError("tokenizers must be a nonempty dictionary of names to backends")
        if any(not isinstance(name, str) or not name for name in tokenizers):
            raise ValueError("tokenizer names must be nonempty strings")
        _StepBoundary(step_field, episode_field)
        self.step_field = step_field
        self.episode_field = episode_field
        self.objective_fields, self.objective_missing_values = _objective_config(objective_fields)
        self.input_fields = tuple(field if isinstance(field, TokenizerModalitySpec) else TokenizerModalitySpec(**field)
                                  for field in input_fields)
        self._provider_meta = {}
        for provider in dict.fromkeys(spec.provider for spec in self.input_fields):
            fields = [spec for spec in self.input_fields if spec.provider == provider]
            _, meta = resolve_tokenizer_modalities(fields)
            self._provider_meta[provider] = tuple(meta)
        meta_by_spec = {id(meta.spec): meta for group in self._provider_meta.values() for meta in group}
        self._meta = tuple(enumerate(meta_by_spec[id(spec)] for spec in self.input_fields))
        self._default_meta = tuple((index, meta) for index, meta in self._meta if meta.spec.provider is None)
        self._provider_meta = {
            provider: tuple((index, meta) for index, meta in self._meta
                            if meta.spec.provider is None or meta.spec.provider == provider)
            for provider in self._provider_meta if provider is not None
        }
        sources = dict(tokenizers)
        self.tokenizer_names = tuple(sorted(sources))
        default = "default" if "default" in sources else self.tokenizer_names[0] if len(sources) == 1 else None
        field_tokenizers = []
        for spec in self.input_fields:
            name = spec.tokenizer or default
            if name not in sources:
                raise ValueError(f"Input field {_field_label(spec)!r} must select a configured tokenizer; got {name!r}")
            field_tokenizers.append(name)
        self.tokenizer_sources = {name: str(value) if isinstance(value, (str, Path)) else None
                                  for name, value in sources.items()}
        self.tokenizers: dict[str, Any] = {}
        for name, source in sources.items():
            if isinstance(source, (str, Path)):
                from transformers import AutoTokenizer
                source = AutoTokenizer.from_pretrained(str(source), **dict(hub_kwargs or {}))
            if not callable(source):
                raise TypeError(f"Tokenizer {name!r} must be a callable backend or pretrained identifier")
            self.tokenizers[name] = source
        self._backends = tuple(self.tokenizers[name] for name in field_tokenizers)
        self._routes = tuple(self.tokenizer_names.index(name) for name in field_tokenizers)
        self.special_token_ids = {}
        for name in self.tokenizer_names:
            specs = [spec for spec, selected in zip(self.input_fields, field_tokenizers) if selected == name]
            ids = _register_special_tokens(self.tokenizers[name], specs)
            if ids:
                self.special_token_ids[name] = ids
        encoders = {name: _TextEncoder(backend) for name, backend in self.tokenizers.items()}
        self._text_encoders = tuple(encoders[name] for name in field_tokenizers)

    def __call__(self, event: Mapping) -> dict[str, np.ndarray]:
        """Encode matching fields from one decoded event, without adding a step."""
        if not isinstance(event, Mapping) or not {"provider", "payload"} <= event.keys():
            raise TypeError("Tokenizer expects a provider/payload event")
        provider = event["provider"]
        meta = self._provider_meta.get(provider, self._default_meta)
        payload = event["payload"]
        if meta and not isinstance(payload, Mapping):
            raise TypeError("Configured event input fields require an object payload")
        values = _tokenize_fields(row=payload, meta=meta, tokenizers=self._backends,
                                  tokenizer_routes=self._routes, text_encoders=self._text_encoders)
        return values

    def stream(self, *, include_objectives: bool = True) -> EventTokenStream:
        return EventTokenStream(self, include_objectives=include_objectives)

    def encode(self, events, *, include_objectives: bool = True) -> list[StepTokens]:
        """Encode a finalized sample of raw events or persisted JSON payloads.

        The sample must contain complete steps. Unlike ``stream.push``, this
        method flushes the final step at the end of the supplied sequence.
        """
        stream = self.stream(include_objectives=include_objectives)
        steps = []
        for event in events:
            completed = stream.push(event, encoded_payload=not isinstance(events, EventLog))
            if completed is not None:
                steps.append(completed)
        last = stream.flush()
        if last is not None:
            steps.append(last)
        return steps

    def pack_rows(self, *, rows, continuing) -> TokenBatch:
        """Pack complete event steps per sequence for inference, without labels."""
        steps, sids = [], []
        for index, events in enumerate(rows):
            encoded = self.encode(events, include_objectives=False)
            steps.extend(encoded)
            sids.extend([index] * len(encoded))
        inputs, _, _ = pack_token_batch(steps=steps, sequence_ids=sids if steps else None,
                                        batch_size=len(rows), continuing=continuing)
        if not steps:
            inputs = replace(inputs, tokenizer_names=self.tokenizer_names, tokenizer_ids=np.zeros(0, dtype=np.int64))
        return inputs


class EventTokenStream:
    """Per-sequence token accumulator. Each push tokenizes exactly one event.

    A new marker returns the preceding step. ``flush()`` explicitly completes
    the current step; merely ending an input chunk does not. Create a separate
    stream per sequence; the Tokenizer itself has no mutable stream state.
    """

    def __init__(self, tokenizer: Tokenizer, *, include_objectives: bool = True):
        self.tokenizer = tokenizer
        self.include_objectives = include_objectives
        self._boundary = _StepBoundary(tokenizer.step_field, tokenizer.episode_field)
        self._pieces: list[dict[str, np.ndarray]] = []
        self._labels: dict[str, Any] = {}

    def push(self, event: Mapping, *, encoded_payload: bool = False) -> StepTokens | None:
        if not isinstance(event, Mapping) or not {"provider", "payload"} <= event.keys():
            raise TypeError("Tokenizer expects a provider/payload event")
        event = {key: value for key, value in event.items()
                 if key not in INDEX_FIELDS or value is not None}
        if encoded_payload and isinstance(event["payload"], str):
            event = {**event, "payload": json.loads(event["payload"])}
        piece = self.tokenizer(event)
        boundary = replace(self._boundary)
        changed = boundary.advance(event)
        if not changed and not self._pieces:
            raise ValueError("Cannot append events to a flushed step; start a new step marker")
        completed = self._finish() if changed and self._pieces else None
        labels = {} if changed else dict(self._labels)
        if self.include_objectives:
            payload = event["payload"]
            for name, _ in self.tokenizer.objective_fields:
                provider, separator, field = name.partition(".")
                if separator and provider == event["provider"] and isinstance(payload, Mapping):
                    if field in payload:
                        labels[name] = deepcopy(payload[field])
                elif not separator:
                    if name in event and isinstance(payload, Mapping) and name in payload:
                        raise ValueError(f"Event envelope and payload both supply objective field {name!r}")
                    if isinstance(payload, Mapping) and name in payload:
                        labels[name] = deepcopy(payload[name])
                    elif name in event:
                        labels[name] = deepcopy(event[name])
        if changed:
            self._pieces = []
        self._boundary = boundary
        self._labels = labels
        self._pieces.append(piece)
        return completed

    def _finish(self) -> StepTokens:
        pieces = self._pieces
        values = {name: np.concatenate([piece[name] for piece in pieces]) for name in
                  ("token_ids", "numeric_values", "numeric_mask", "head_token_mask", "tokenizer_ids")}
        prefixes = []
        seen_fields = set()
        for piece in pieces:
            if "group_start_token_ids" not in piece:
                continue
            ids = piece["group_start_field_ids"]
            keep = np.asarray([int(index) not in seen_fields for index in ids], dtype=bool)
            seen_fields.update(ids.tolist())
            prefixes.append({name: piece[name][keep] for name in
                             ("group_start_token_ids", "group_start_numeric_values", "group_start_numeric_mask", "group_start_tokenizer_ids")})
        if prefixes:
            values.update({name: np.concatenate([piece[name] for piece in prefixes]) for name in
                           ("group_start_token_ids", "group_start_numeric_values", "group_start_numeric_mask", "group_start_tokenizer_ids")})
        objective = copy_keep_fields(self._labels,
                                    self.tokenizer.objective_fields if self.include_objectives else (),
                                    missing_values=self.tokenizer.objective_missing_values)
        return StepTokens(**values, objective_fields=objective, tokenizer_names=self.tokenizer.tokenizer_names)

    def flush(self) -> StepTokens | None:
        if not self._pieces:
            return None
        completed = self._finish()
        self._pieces = []
        self._labels = {}
        return completed


def _field_text_value(spec: TokenizerModalitySpec, row: dict[str, Any]) -> str | None:
    assert spec.format is not None
    if spec.input_field is None:
        return render_text_format(spec, field=None)
    raw = row.get(spec.input_field)
    if raw is None:
        return None
    value = _text_scalar(spec, raw)
    if spec.lookup is not None:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"lookup field {_field_label(spec)!r} requires an integer ID")
        if not 0 <= value < len(spec.lookup):
            raise ValueError(f"lookup field {_field_label(spec)!r} ID {value} is outside [0, {len(spec.lookup)})")
        value = spec.lookup[int(value)]
    return render_text_format(spec, field=value)


def _numeric_scaled(spec: TokenizerModalitySpec, row: dict[str, Any]) -> float | None:
    """Map the field's number from ``[fourier_min, fourier_max]`` onto ``[-1, 1]``."""
    in_name = spec.input_field
    if not in_name:
        raise RuntimeError("numeric field is missing input_field")
    raw = row.get(in_name)
    if raw is None:
        return None
    scalar = _text_scalar(spec, raw)
    who = _field_label(spec)
    if isinstance(scalar, (bool, np.bool_)):
        raise TypeError(
            f"numeric field {who!r} value must be a real number, "
            f"got {type(scalar).__name__}"
        )
    if not isinstance(scalar, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"numeric field {who!r} value must be a real number, "
            f"got {type(scalar).__name__}"
        )
    value = float(scalar)
    if not math.isfinite(value):
        raise ValueError(f"numeric field {who!r} value must be finite, got {value!r}")
    lo = spec.fourier_min
    hi = spec.fourier_max
    if lo is None or hi is None:
        raise RuntimeError(f"numeric field {who!r} is missing fourier bounds")
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _text_scalar(spec: TokenizerModalitySpec, raw: Any) -> Any:
    """Scalar to interpolate, or one element selected by ``input_index``."""
    who = _field_label(spec)
    rank = _value_rank(raw)
    index = spec.input_index
    if index is None:
        if rank == 1:
            raise ValueError(
                f"field {who!r} value is a 1-D vector; set input_index= "
                "to select an element"
            )
        if rank is not None and rank != 0:
            raise ValueError(
                f"field {who!r} value has rank {rank}; expected a scalar"
            )
        return unwrap_scalar(raw)
    if rank is None or rank == 0:
        raise ValueError(
            f"field {who!r} input_index={index} requires a 1-D vector, "
            "got a scalar"
        )
    if rank != 1:
        raise ValueError(
            f"field {who!r} input_index={index} requires a 1-D vector, "
            f"got rank {rank}"
        )
    n = _vector_length(raw)
    if index >= n:
        raise ValueError(
            f"field {who!r} input_index={index} is out of range for a "
            f"vector of length {n}"
        )
    return unwrap_scalar(_vector_at(raw, index))


def _value_rank(value: Any) -> int | None:
    """Array rank, or ``None`` when ``value`` is a scalar (including ``str``)."""
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, torch.Tensor):
        return int(value.ndim)
    if isinstance(value, np.ndarray):
        return int(value.ndim)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 1
        if any(_is_nested_element(item) for item in value):
            return 2
        return 1
    return None


def _is_nested_element(item: Any) -> bool:
    if isinstance(item, (str, bytes)):
        return False
    if isinstance(item, (list, tuple, np.ndarray)):
        return True
    if isinstance(item, torch.Tensor):
        return item.ndim >= 1
    return False


def _vector_length(value: Any) -> int:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return int(value.shape[0])
    return len(value)


def _vector_at(value: Any, index: int) -> Any:
    return value[index]


def _require_one_token(spec: TokenizerModalitySpec, token_ids: list[int]) -> None:
    """``numeric`` and ``token`` formats tokenize to one id."""
    n = len(token_ids)
    if n == 1:
        return
    who = _field_label(spec) if spec.input_field is not None else spec.type
    raise ValueError(
        f"{spec.type} field {who!r} format={spec.format!r} tokenized to "
        f"{n} tokens; expected 1"
    )


def _require_max_tokens(spec: TokenizerModalitySpec, token_ids: list[int]) -> None:
    limit = spec.max_tokens
    if limit is None:
        return
    n = len(token_ids)
    if n > limit:
        raise ValueError(
            f"field {spec.input_field or 'const'!r} tokenized to {n} tokens "
            f"(max_tokens={limit})"
        )


def _tokenize_ids(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.view(-1).tolist()
    elif ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    out: list[int] = []
    for i in ids:
        if isinstance(i, torch.Tensor):
            out.append(int(i.item()))
        else:
            out.append(int(i))
    return out


class _TextEncoder:
    """Encode one field at a time; leave caching to the native HF backend."""

    def __init__(self, backend: Any):
        from transformers import PreTrainedTokenizerFast

        self.backend = backend
        self._native = (
            isinstance(backend, PreTrainedTokenizerFast)
            and not hasattr(backend, "_switch_to_input_mode")
            and all(getattr(getattr(backend, name), "__func__", None)
                    is getattr(PreTrainedTokenizerFast, name) for name in (
                        "__call__", "_encode_plus", "_convert_encoding",
                        "_get_padding_truncation_strategies", "set_truncation_and_padding",
                    ))
        )

    def __call__(self, text: str) -> list[int]:
        if not text:
            return []
        if self._native:
            native = self.backend.backend_tokenizer
            # The HF wrapper resets these settings before encoding. Use it
            # when needed rather than changing a shared backend ourselves.
            if (native.padding is None and native.truncation is None
                    and native.encode_special_tokens == self.backend.split_special_tokens):
                # Singleton input: fields never merge or execute out of order.
                # Offsets are unused; only take the independent field's IDs.
                return native.encode_batch_fast([text], add_special_tokens=False)[0].ids
        return _tokenize_ids(self.backend, text)


def _image_token_ids(image_tokenizer: Any, value: Any, *, in_name: str) -> list[int]:
    if image_tokenizer is None:
        raise RuntimeError("image_tokenizer is not configured")
    out = image_tokenizer(value)
    if isinstance(out, torch.Tensor):
        if out.ndim == 2 and out.shape[-1] > 1:
            raise TypeError("image_tokenizer must return token ids, not embeddings")
        img_ids = [int(x) for x in out.view(-1).tolist()]
    elif isinstance(out, (list, tuple, np.ndarray)):
        img_ids = [int(x) for x in np.asarray(out).ravel().tolist()]
    else:
        raise TypeError("image_tokenizer must return a sequence of token ids")
    if not img_ids:
        raise ValueError(f"image tokenizer returned no tokens for {in_name!r}")
    return img_ids


def _tokenize_fields(*, row, meta, tokenizers, tokenizer_routes, text_encoders):
    token_ids: list[int] = []
    tokenizer_ids: list[int] = []
    numeric_values: list[float] = []
    numeric_mask: list[bool] = []
    head_token_mask: list[bool] = []
    prefix_token_ids: list[int] = []
    prefix_tokenizer_ids: list[int] = []
    prefix_numeric_values: list[float] = []
    prefix_numeric_mask: list[bool] = []
    prefix_field_ids: list[int] = []

    def _emit(
        emitted_ids: list[int], *, spec: TokenizerModalitySpec, field_id: int,
        to_group_start: bool, numeric_value: float | None = None,
        head_output: bool = False,
    ) -> None:
        _require_max_tokens(spec, emitted_ids)
        is_numeric = numeric_value is not None
        value = 0.0 if numeric_value is None else numeric_value
        if to_group_start:
            prefix_field_ids.extend([field_id] * len(emitted_ids))
            prefix_token_ids.extend(emitted_ids)
            prefix_tokenizer_ids.extend([tokenizer_routes[field_id]] * len(emitted_ids))
            prefix_numeric_values.extend([value] * len(emitted_ids))
            prefix_numeric_mask.extend([is_numeric] * len(emitted_ids))
        else:
            token_ids.extend(emitted_ids)
            tokenizer_ids.extend([tokenizer_routes[field_id]] * len(emitted_ids))
            numeric_values.extend([value] * len(emitted_ids))
            numeric_mask.extend([is_numeric] * len(emitted_ids))
            head_token_mask.extend([head_output] * len(emitted_ids))

    for field_id, m in meta:
        spec = m.spec
        tokenizer = tokenizers[field_id]
        text_ids = text_encoders[field_id]
        ordinary, to_group_start = _when_routes(spec, row)
        if not ordinary and not to_group_start:
            continue
        if m.kind == KIND_TEXT:
            rendered = _field_text_value(spec, row)
            if rendered is None:
                continue
            if tokenizer is None:
                raise RuntimeError("tokenizer required to tokenize text runs")
            _emit(
                text_ids(rendered),
                spec=spec, field_id=field_id,
                to_group_start=to_group_start,
                head_output=spec.head_output,
            )
            continue

        if m.kind == KIND_NUMERIC:
            scaled = _numeric_scaled(spec, row)
            if scaled is None:
                continue
            if tokenizer is None:
                raise RuntimeError("tokenizer required to tokenize numeric runs")
            rendered = render_text_format(spec, field=None)
            emitted_ids = text_ids(rendered)
            _require_one_token(spec, emitted_ids)
            _emit(
                emitted_ids,
                spec=spec, field_id=field_id,
                to_group_start=to_group_start,
                numeric_value=scaled,
                head_output=spec.head_output,
            )
            continue

        if m.kind == KIND_TOKEN:
            if spec.input_field is not None and row.get(spec.input_field) is None:
                continue
            if tokenizer is None:
                raise RuntimeError("tokenizer required to tokenize token runs")
            rendered = render_text_format(spec, field=None)
            emitted_ids = text_ids(rendered)
            _require_one_token(spec, emitted_ids)
            _emit(
                emitted_ids,
                spec=spec, field_id=field_id,
                to_group_start=to_group_start,
                head_output=spec.head_output,
            )
            continue

        in_name = str(spec.input_field)
        value = row.get(in_name)
        if value is None:
            continue

        if m.kind == KIND_IMAGE:
            img_ids = _image_token_ids(tokenizer, value, in_name=in_name)
            _emit(
                img_ids,
                spec=spec, field_id=field_id,
                to_group_start=to_group_start,
                head_output=spec.head_output,
            )
        else:
            raise ValueError(f"unsupported modality kind {m.kind!r}")

    group_start_kwargs: dict[str, np.ndarray] = {}
    if prefix_token_ids:
        group_start_kwargs = {
            "group_start_token_ids": np.asarray(prefix_token_ids, dtype=np.int64),
            "group_start_tokenizer_ids": np.asarray(prefix_tokenizer_ids, dtype=np.int64),
            "group_start_numeric_values": np.asarray(prefix_numeric_values, dtype=np.float32),
            "group_start_numeric_mask": np.asarray(prefix_numeric_mask, dtype=bool),
        }

    if prefix_field_ids:
        group_start_kwargs["group_start_field_ids"] = np.asarray(prefix_field_ids, dtype=np.int64)

    return dict(
        token_ids=np.asarray(token_ids, dtype=np.int64),
        tokenizer_ids=np.asarray(tokenizer_ids, dtype=np.int64),
        numeric_values=np.asarray(numeric_values, dtype=np.float32),
        numeric_mask=np.asarray(numeric_mask, dtype=bool),
        head_token_mask=np.asarray(head_token_mask, dtype=bool),
        **group_start_kwargs,
    )


def _when_routes(
    spec: TokenizerModalitySpec, row: dict
) -> tuple[bool, bool]:
    """Return ``(ordinary, to_group_start)`` for one field on ``row``.

    Omit ``when`` → always ordinary. Otherwise call ``when`` with
    ``group_start=False`` (ordinary) and ``group_start=True`` (pack
    path). Ordinary wins when both are true so mid-group equals still
    appear on the step run; group-start alone rides ``group_start_*``.
    """
    when = spec.when
    if when is None:
        return True, False
    from mouse_core.data.conditions import GROUP_START_KEY

    ordinary_ctx = {**row, GROUP_START_KEY: False}
    group_ctx = {**row, GROUP_START_KEY: True}
    ordinary = when_as_bool(when(ordinary_ctx))
    wants_group = when_as_bool(when(group_ctx))
    to_group_start = wants_group and not ordinary
    return ordinary, to_group_start


TOKENIZER_FORMAT = "mouse-core-tokenizer-v2"
TOKENIZER_FILENAME = "tokenizer.json"


def _jsonable(value: Any) -> Any:
    """JSON-safe form of a tokenizer field value."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _jsonable_when(when: Any) -> str:
    """Serialize a ``when`` callable as ``module:qualname``."""
    from mouse_core.data.conditions import when_ref

    return when_ref(when)


def _input_fields_config(specs):
    input_fields: list[dict[str, Any]] = []
    for spec in specs:
        data = asdict(spec)
        field: dict[str, Any] = {}
        for key, value in data.items():
            if key.startswith("_") or value is None:
                continue
            if key in ("head_output", "special") and value is False:
                continue
            if key == "when":
                field[key] = _jsonable_when(value)
                continue
            field[key] = _jsonable(value)
        input_fields.append(field)
    return input_fields


def tokenizer_config(*, tokenizer: Tokenizer) -> dict[str, Any]:
    """Public tokenization specification, including event payload schemas."""
    objective_fields: list[dict[str, Any]] = []
    for in_name, out_name in tokenizer.objective_fields:
        entry: dict[str, Any] = {"input_field": in_name}
        if out_name != in_name:
            entry["output_field"] = out_name
        if out_name in tokenizer.objective_missing_values:
            entry["missing_value"] = tokenizer.objective_missing_values[out_name]
        objective_fields.append(entry)
    result = {
        "format": TOKENIZER_FORMAT,
        "tokenizers": {name: tokenizer.tokenizer_sources[name] for name in tokenizer.tokenizer_names},
        "objective_fields": objective_fields,
    }

    result["step_field"] = tokenizer.step_field
    result["episode_field"] = tokenizer.episode_field
    result["input_fields"] = _input_fields_config(tokenizer.input_fields)
    if tokenizer.special_token_ids:
        result["special_token_ids"] = tokenizer.special_token_ids
    return result


def save_tokenizer(*, tokenizer: Tokenizer, path: str | Path) -> None:
    """Write ``tokenizer.json`` into *path*, which may also contain model weights."""
    config = tokenizer_config(tokenizer=tokenizer)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / TOKENIZER_FILENAME).open("w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def load_tokenizer(
    *,
    repo_id_or_path: str,
    tokenizers=None,
    force_download: bool = True,
    local_dir: str | Path | None = None,
    **kwargs: Any,
) -> Tokenizer:
    """Load a tokenizer packing spec from a checkpoint directory or Hub repo.

    The JSON is the packing contract (fields, ``when`` import refs,
    ``objective_fields``, and the named ``tokenizers`` dictionary).
    Pretrained backends reload from saved identifiers. Supply custom text
    or image backends by name in ``tokenizers=``.

    Args:
        repo_id_or_path: Local tokenizer directory or Hub repo id from
            :func:`save_tokenizer` / ``push_model_to_hub(..., repo_id=)``.
        tokenizers: Optional dictionary overriding named backends. Custom
            callables must be supplied again by name; other entries reload
            from saved pretrained identifiers. Routing names cannot change.
        force_download: Bypass the Hub cache (ignored for local paths).
        local_dir: Where Hub files are saved (ignored for local paths).
        **kwargs: Hub download kwargs (``revision``, ``token``, …).
    """
    obsolete = set(kwargs) & {"tokenizer", "pretrained", "image_tokenizer"}
    if obsolete:
        raise TypeError(f"Unsupported tokenizer arguments {sorted(obsolete)}; use tokenizers={{name: backend}}")
    if tokenizers is not None and not isinstance(tokenizers, Mapping):
        raise TypeError("tokenizers overrides must be a dictionary of names to backends")
    local = Path(repo_id_or_path)
    if local.exists():
        config_path = local / TOKENIZER_FILENAME
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{config_path} is missing; save the tokenizer with "
                "save_tokenizer or push_model_to_hub(..., repo_id=)."
            )
        with config_path.open() as fh:
            config = json.load(fh)
    else:
        from huggingface_hub import hf_hub_download

        from mouse_core.models.io import _hub_repo_id_for_user

        hf_kwargs: dict[str, Any] = {"force_download": force_download, **kwargs}
        if local_dir is not None:
            hf_kwargs["local_dir"] = str(local_dir)
        hub_repo_id = _hub_repo_id_for_user(repo_id_or_path, token=kwargs.get("token"))
        config_path = Path(
            hf_hub_download(
                repo_id=hub_repo_id, filename=TOKENIZER_FILENAME, **hf_kwargs
            )
        )
        with config_path.open() as fh:
            config = json.load(fh)

    if config.get("format") != TOKENIZER_FORMAT:
        raise ValueError(
            "Unsupported tokenizer config format. Expected a MOUSE tokenizer "
            "saved with save_tokenizer or push_model_to_hub(..., repo_id=)."
        )
    saved_backends = config.get("tokenizers")
    if not isinstance(saved_backends, dict) or not saved_backends:
        raise ValueError("Saved tokenizer specification requires a nonempty tokenizers dictionary")
    if tokenizers is not None and set(tokenizers) - set(saved_backends):
        raise ValueError("tokenizers overrides must use names from the saved specification")
    tokenizers = {**saved_backends, **dict(tokenizers or {})}
    result = Tokenizer(
        input_fields=config["input_fields"],
        step_field=config["step_field"],
        episode_field=config["episode_field"],
        objective_fields=config.get("objective_fields") or (),
        tokenizers=tokenizers,
        hub_kwargs=kwargs,
    )
    if result.special_token_ids != config.get("special_token_ids", {}):
        raise ValueError("Special token IDs differ from the saved tokenizer vocabulary")
    # Preserve saved source identifiers when live callables override execution.
    result.tokenizer_sources = {name: result.tokenizer_sources[name] or source
                                for name, source in saved_backends.items()}
    return result
