"""TextTokenizer — format/tokenize one step → StepTokens (discrete ids).

I/O
---
* **in:** ``dict`` (one step; must include ``grouping_field``)
* **out:** :class:`~mouse_core.data.token_batch.StepTokens`

Tokens are tagged by modality name (``__text__`` / ``__vision__``). Pack many
steps with :func:`~mouse_core.data.token_batch.pack_token_batch`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from string import Formatter
from typing import Any

import numpy as np
import torch

from mouse_core.data.io_fields import coerce_io_fields
from mouse_core.data.modality import (
    TextTokenizerModalitySpec,
    copy_keep_fields,
    expand_tokenizer_text_spec,
    unwrap_scalar,
    values_equal,
)
from mouse_core.data.token_batch import ModalityInfo, StepTokens

# Stable modality names for text vs vision discrete streams. Resolve local
# indices via ``StepTokens.modality_names`` / ``modality_map``; they are not fixed.
NAME_TEXT = "__text__"
NAME_VISION = "__vision__"


class TextTokenizer:
    """CPU packer: format + HF/image tokenization → :class:`StepTokens`.

    Construct independently of the embedder. Alignment is by modality **name**
    (``__text__`` / ``__vision__`` / learnable ``output_field``).
    ``input_fields=`` are the tokens fed to the transformer (each
    ``{type, input_field}``; optional ``output_field``). A ``text`` field
    requires ``format=``. ``learnable`` fields have no step I/O and are
    appended after the rendered format. ``group_prefix=`` is a format string
    over the raw step dict (placeholders need not be ``input_fields``); it is
    tokenized as ``__text__`` and :func:`~mouse_core.data.token_batch.pack_token_batch`
    inserts those tokens at the start of each grouping-field segment.
    ``objective_fields=`` is a list of ``{input_field}`` dicts (optional)
    ``output_field``; defaults to the input name)
    copied into ``StepTokens.objective_fields`` (input fields are not
    auto-copied). TD / PPO / GRPO objectives read ``action``, ``reward``,
    ``episode_done``, and ``task_done`` from that keep-list. ``task_done`` is an
    objective column only — it is not interpolated into the step format and is
    not fed to the transformer.
    """

    def __init__(
        self,
        *,
        input_fields: Sequence[dict | TextTokenizerModalitySpec] | None = None,
        grouping_field: str,
        format: str | None = None,
        group_prefix: str | None = None,
        tokenizer=None,
        image_processor=None,
        objective_fields: Sequence[dict[str, Any]] | None = None,
        pretrained: str | Path | None = None,
        hub_kwargs: dict | None = None,
    ) -> None:
        if not grouping_field:
            raise ValueError("TextTokenizer requires a non-empty grouping_field")
        raw = input_fields or []
        specs: list[TextTokenizerModalitySpec] = []
        n_learnable = 0
        for m in raw:
            if isinstance(m, TextTokenizerModalitySpec):
                spec = m
            else:
                data = dict(m)
                spec = TextTokenizerModalitySpec(**data)
            specs.extend(expand_tokenizer_text_spec(spec, learnable_index=n_learnable))
            if spec.type == "learnable":
                n_learnable += 1

        has_text = any(s.type == "text" for s in specs)
        has_token = any(s.type == "token" for s in specs)
        has_image = any(s.type == "image" for s in specs)
        needs_format = has_text or has_image or has_token
        if needs_format and format is None:
            raise TypeError(
                "TextTokenizer requires format= when text, token, or image input_fields "
                "are declared"
            )
        if format is not None and not (has_text or has_token or has_image):
            raise TypeError("format= requires at least one text, token, or image input field")

        flagged = [s.output_field for s in specs if s.head_output]
        if len(flagged) != 1:
            raise ValueError(
                "TextTokenizer requires exactly one input field with "
                "head_output=True (its tokens are the step's head-output tokens "
                f"— the Q / action readout positions); got {flagged or 'none'}"
            )

        text_by_field = {
            s.output_field: s
            for s in specs
            if s.type == "text" and isinstance(s.output_field, str)
        }
        token_by_field = {
            s.output_field: s
            for s in specs
            if s.type == "token" and isinstance(s.output_field, str)
        }
        image_by_field = {
            s.output_field: s
            for s in specs
            if s.type == "image" and isinstance(s.output_field, str)
        }

        if format is not None:
            for _, name, _, _ in Formatter().parse(format):
                if name is None or name == "":
                    continue
                if (
                    name not in text_by_field
                    and name not in token_by_field
                    and name not in image_by_field
                ):
                    raise ValueError(
                        f"format placeholder {{{name}}} has no matching text/token/image input field"
                    )

        if group_prefix is not None and group_prefix == "":
            raise ValueError(
                "TextTokenizer group_prefix= must be a non-empty string"
            )
        needs_tokenizer = (format is not None and has_text) or group_prefix is not None
        if tokenizer is not None:
            tok = tokenizer
        elif pretrained is not None and needs_tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(pretrained, **dict(hub_kwargs or {}))
        elif needs_tokenizer:
            raise TypeError(
                "TextTokenizer with text input_fields or group_prefix= requires "
                "tokenizer= or pretrained="
            )
        else:
            tok = None

        if has_image:
            if image_processor is None or not callable(image_processor):
                raise TypeError(
                    "TextTokenizer with image input_fields requires image_processor= "
                    "callable that returns discrete token ids"
                )

        names: list[str] = []
        mmap: dict[str, ModalityInfo] = {}
        if has_text or has_token or group_prefix is not None:
            names.append(NAME_TEXT)
            mmap[NAME_TEXT] = ModalityInfo(type="token")
        if has_image:
            names.append(NAME_VISION)
            mmap[NAME_VISION] = ModalityInfo(type="image")
        learnable_specs = [s for s in specs if s.type == "learnable"]
        for spec in learnable_specs:
            name = str(spec.output_field)
            if name in mmap:
                raise ValueError(f"duplicate tokenizer modality name {name!r}")
            names.append(name)
            mmap[name] = ModalityInfo(type="learnable")

        self.format = format
        self.group_prefix = group_prefix
        self.input_fields: tuple[TextTokenizerModalitySpec, ...] = tuple(specs)
        self.grouping_field = grouping_field
        self._text_by_field = text_by_field
        self._token_by_field = token_by_field
        self._image_by_field = image_by_field
        self._learnable_specs = tuple(learnable_specs)
        self.tokenizer = tok
        self.image_processor = image_processor
        self.objective_fields: tuple[tuple[str, str], ...] = coerce_io_fields(
            objective_fields or (),
            who="tokenizer objective_fields",
            allow_empty=True,
        )
        self.modality_names: tuple[str, ...] = tuple(names)
        self.modality_map: dict[str, ModalityInfo] = mmap
        self._name_to_index = {n: i for i, n in enumerate(self.modality_names)}

    def __call__(self, step: dict) -> StepTokens:
        if not isinstance(step, dict):
            raise TypeError(
                f"TextTokenizer expects a step dict, got {type(step).__name__}"
            )
        return _tokenize_text_step(
            row=step,
            format_str=self.format,
            group_prefix_str=self.group_prefix,
            text_by_field=self._text_by_field,
            token_by_field=self._token_by_field,
            image_by_field=self._image_by_field,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            learnable_specs=self._learnable_specs,
            objective_fields_keep=self.objective_fields,
            grouping_field=self.grouping_field,
            name_to_index=self._name_to_index,
            modality_names=self.modality_names,
            modality_map=self.modality_map,
        )


def _field_text_value(spec: TextTokenizerModalitySpec, row: dict[str, Any]) -> str | None:
    assert isinstance(spec.input_field, str)
    assert isinstance(spec.output_field, str)
    assert spec.format is not None
    value = row.get(spec.input_field)
    if value is None:
        if spec.required:
            raise KeyError(f"Required modality {spec.input_field!r} is missing")
        return None
    if spec.skip is not None and values_equal(value, spec.skip):
        return None
    return spec.format.format_map({spec.output_field: unwrap_scalar(value)})


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


def _tokenize_text_step(
    *,
    row: dict,
    format_str: str | None,
    group_prefix_str: str | None,
    text_by_field: dict[str, TextTokenizerModalitySpec],
    token_by_field: dict[str, TextTokenizerModalitySpec],
    image_by_field: dict[str, TextTokenizerModalitySpec],
    tokenizer: Any,
    image_processor: Any,
    learnable_specs: Sequence[TextTokenizerModalitySpec],
    objective_fields_keep: Sequence[tuple[str, str]],
    grouping_field: str,
    name_to_index: dict[str, int],
    modality_names: tuple[str, ...],
    modality_map: dict[str, ModalityInfo],
) -> StepTokens:
    if grouping_field not in row:
        raise KeyError(
            f"grouping_field {grouping_field!r} missing from step "
            f"(have {sorted(row)})"
        )
    gid = int(unwrap_scalar(row[grouping_field]))

    modality_ids: list[int] = []
    ids: list[int] = []
    values: list[float] = []
    positions: list[int] = []
    head_output_mask: list[bool] = []
    # Running per-modality token count within this step, so ``positions``
    # keeps counting across separately emitted runs of the same modality.
    next_position: dict[int, int] = {}

    def _emit(token_ids: list[int], *, name: str, head_output: bool = False) -> None:
        mid = name_to_index[name]
        pos = next_position.get(mid, 0)
        for tid in token_ids:
            modality_ids.append(mid)
            ids.append(tid)
            values.append(0.0)
            positions.append(pos)
            pos += 1
            head_output_mask.append(head_output)
        next_position[mid] = pos

    if format_str is not None:
        text_buf: list[str] = []

        def flush_text() -> None:
            if not text_buf:
                return
            text = "".join(text_buf)
            text_buf.clear()
            if tokenizer is None:
                raise RuntimeError("tokenizer required to tokenize text runs")
            _emit(_tokenize_ids(tokenizer, text), name=NAME_TEXT)

        for literal, name, _fmt, _conv in Formatter().parse(format_str):
            if name is None:
                if literal:
                    text_buf.append(literal)
                continue

            if name in token_by_field or name in image_by_field:
                if literal:
                    text_buf.append(literal)
                flush_text()
                if name in token_by_field:
                    spec = token_by_field[name]
                    value = row.get(spec.input_field)
                    if value is None:
                        if spec.required:
                            raise KeyError(
                                f"Required modality {spec.input_field!r} is missing"
                            )
                        continue
                    if spec.skip is not None and values_equal(value, spec.skip):
                        continue
                    _emit(
                        [int(unwrap_scalar(value))],
                        name=NAME_TEXT,
                        head_output=spec.head_output,
                    )
                else:
                    spec = image_by_field[name]
                    value = row.get(spec.input_field)
                    if value is None:
                        if spec.required:
                            raise KeyError(f"Required modality {name!r} is missing")
                        continue
                    if spec.skip is not None and values_equal(value, spec.skip):
                        continue
                    if image_processor is None:
                        raise RuntimeError("image_processor is not configured")
                    out = image_processor(value)
                    if isinstance(out, torch.Tensor):
                        if out.ndim == 2 and out.shape[-1] > 1:
                            raise TypeError(
                                "image_processor must return token ids, not embeddings"
                            )
                        img_ids = [int(x) for x in out.view(-1).tolist()]
                    elif isinstance(out, (list, tuple, np.ndarray)):
                        img_ids = [int(x) for x in np.asarray(out).ravel().tolist()]
                    else:
                        raise TypeError(
                            "image_processor must return a sequence of token ids"
                        )
                    _emit(img_ids, name=NAME_VISION, head_output=spec.head_output)
                continue

            if literal:
                text_buf.append(literal)
            spec = text_by_field[name]
            rendered = _field_text_value(spec, row)
            if rendered is None:
                continue
            if spec.head_output:
                # Tokenize the head-output field as its own run so its token
                # boundaries (and therefore the head-output mask) are exact.
                flush_text()
                if tokenizer is None:
                    raise RuntimeError("tokenizer required to tokenize text runs")
                _emit(_tokenize_ids(tokenizer, rendered), name=NAME_TEXT, head_output=True)
                continue
            text_buf.append(rendered)

        flush_text()

    for spec in learnable_specs:
        n = int(spec.tokens or 1)
        _emit(
            list(range(n)),
            name=str(spec.output_field),
            head_output=spec.head_output,
        )

    if not modality_ids:
        raise ValueError(
            "step has no tokens after skips; ensure the step format still "
            "produces at least one token"
        )

    group_prefix_kwargs: dict[str, np.ndarray] = {}
    if group_prefix_str is not None:
        if NAME_TEXT not in name_to_index:
            raise RuntimeError(
                "group_prefix= requires a text or token input field so __text__ exists"
            )
        if tokenizer is None:
            raise RuntimeError("tokenizer required to tokenize group_prefix=")
        rendered = _render_group_prefix(group_prefix_str, row)
        group_prefix_ids = _tokenize_ids(tokenizer, rendered)
        if not group_prefix_ids:
            raise ValueError(
                f"group_prefix= {group_prefix_str!r} tokenized to no tokens "
                "for this step"
            )
        mid = name_to_index[NAME_TEXT]
        group_prefix_kwargs = {
            "group_prefix_modality_ids": np.full(
                len(group_prefix_ids), mid, dtype=np.int64
            ),
            "group_prefix_ids": np.asarray(group_prefix_ids, dtype=np.int64),
            "group_prefix_values": np.zeros(len(group_prefix_ids), dtype=np.float32),
            "group_prefix_positions": np.arange(len(group_prefix_ids), dtype=np.int64),
        }

    return StepTokens(
        modality_ids=np.asarray(modality_ids, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        positions=np.asarray(positions, dtype=np.int64),
        modality_names=modality_names,
        modality_map=dict(modality_map),
        grouping_id=gid,
        grouping_field=grouping_field,
        head_output_mask=np.asarray(head_output_mask, dtype=bool),
        objective_fields=copy_keep_fields(row, objective_fields_keep),
        **group_prefix_kwargs,
    )


def _render_group_prefix(group_prefix: str, row: dict[str, Any]) -> str:
    mapping: dict[str, Any] = {}
    for _, name, _, _ in Formatter().parse(group_prefix):
        if name is None or name == "":
            continue
        if name not in row:
            raise KeyError(
                f"group_prefix placeholder {{{name}}} missing from step "
                f"(have {sorted(row)})"
            )
        mapping[name] = unwrap_scalar(row[name])
    return group_prefix.format_map(mapping)
