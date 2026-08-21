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

from mouse_core.data.modality import (
    TextTokenizerModalitySpec,
    expand_tokenizer_text_spec,
    unwrap_scalar,
    values_equal,
)
from mouse_core.data.token_batch import ModalityInfo, StepTokens

# Stable modality names for text vs vision discrete streams.
NAME_TEXT = "__text__"
NAME_VISION = "__vision__"

# Back-compat integer indices into modality_names when both are present.
# Prefer resolving via modality_names / modality_map.
MODALITY_TEXT = 0
MODALITY_VISION = 1


class TextTokenizer:
    """CPU packer: format + HF/image tokenization → :class:`StepTokens`.

    Construct independently of the embedder. Alignment is by modality **name**
    (``__text__`` / ``__vision__``). ``input_fields=`` are the tokens fed to the
    transformer. ``objective_fields=`` is an explicit keep-list of step dict keys
    copied into ``StepTokens.objective_fields`` (input fields are not
    auto-copied). TD / PPO / GRPO objectives read ``action``, ``reward``,
    ``episode_done``, and ``task_done`` from that keep-list. ``task_done`` is an
    objective column only — it is not interpolated into the step format and is
    not fed to the transformer.
    """

    def __init__(
        self,
        *,
        input_fields: list[dict | TextTokenizerModalitySpec] | None = None,
        grouping_field: str,
        format: str | None = None,
        tokenizer=None,
        image_processor=None,
        objective_fields: Sequence[str] | None = None,
        pretrained: str | Path | None = None,
        hub_kwargs: dict | None = None,
    ) -> None:
        if not grouping_field:
            raise ValueError("TextTokenizer requires a non-empty grouping_field")
        raw = input_fields or []
        specs: list[TextTokenizerModalitySpec] = []
        for m in raw:
            if isinstance(m, TextTokenizerModalitySpec):
                spec = m
            else:
                data = dict(m)
                if "input_field" in data or "output_field" in data:
                    raise TypeError(
                        "tokenizer input_fields use field= "
                        "(not input_field=/output_field=); "
                        "rename with Selector before tokenize"
                    )
                spec = TextTokenizerModalitySpec(**data)
            specs.extend(expand_tokenizer_text_spec(spec))

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

        text_by_field = {
            s.field: s
            for s in specs
            if s.type == "text" and isinstance(s.field, str)
        }
        token_by_field = {
            s.field: s
            for s in specs
            if s.type == "token" and isinstance(s.field, str)
        }
        image_by_field = {
            s.field: s
            for s in specs
            if s.type == "image" and isinstance(s.field, str)
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

        needs_tokenizer = format is not None and has_text
        if tokenizer is not None:
            tok = tokenizer
        elif pretrained is not None and needs_tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(pretrained, **dict(hub_kwargs or {}))
        elif needs_tokenizer:
            raise TypeError("TextTokenizer with text input_fields requires tokenizer= or pretrained=")
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
        if has_text or has_token:
            names.append(NAME_TEXT)
            mmap[NAME_TEXT] = ModalityInfo(type="token")
        if has_image:
            names.append(NAME_VISION)
            mmap[NAME_VISION] = ModalityInfo(type="image")

        self.format = format
        self.input_fields: tuple[TextTokenizerModalitySpec, ...] = tuple(specs)
        self.grouping_field = grouping_field
        self._text_by_field = text_by_field
        self._token_by_field = token_by_field
        self._image_by_field = image_by_field
        self.tokenizer = tok
        self.image_processor = image_processor
        self.objective_fields: tuple[str, ...] = tuple(objective_fields or ())
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
            text_by_field=self._text_by_field,
            token_by_field=self._token_by_field,
            image_by_field=self._image_by_field,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            objective_fields_keep=self.objective_fields,
            grouping_field=self.grouping_field,
            name_to_index=self._name_to_index,
            modality_names=self.modality_names,
            modality_map=self.modality_map,
        )


def _copy_keep_fields(row: dict, keep: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in keep:
        value = row.get(field)
        if value is None:
            out[field] = 0
            continue
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.ndim >= 1 and arr.size != 1:
            if np.issubdtype(arr.dtype, np.floating):
                out[field] = arr.astype(np.float32).ravel()
            elif arr.dtype == np.uint8:
                out[field] = arr.astype(np.uint8).ravel()
            else:
                out[field] = arr.astype(np.int64).ravel()
        else:
            sample = unwrap_scalar(value)
            if isinstance(sample, (float, np.floating)):
                out[field] = float(sample)
            else:
                out[field] = int(sample)
    return out


def _field_text_value(spec: TextTokenizerModalitySpec, row: dict[str, Any]) -> str | None:
    assert isinstance(spec.field, str)
    assert spec.format is not None
    value = row.get(spec.field)
    if value is None:
        if spec.required:
            raise KeyError(f"Required modality {spec.field!r} is missing")
        return None
    if spec.skip is not None and values_equal(value, spec.skip):
        return None
    return spec.format.format_map({spec.field: unwrap_scalar(value)})


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
    text_by_field: dict[str, TextTokenizerModalitySpec],
    token_by_field: dict[str, TextTokenizerModalitySpec],
    image_by_field: dict[str, TextTokenizerModalitySpec],
    tokenizer: Any,
    image_processor: Any,
    objective_fields_keep: Sequence[str],
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

    def _emit(token_ids: list[int], *, name: str) -> None:
        mid = name_to_index[name]
        for tid in token_ids:
            modality_ids.append(mid)
            ids.append(tid)
            values.append(0.0)

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
                    value = row.get(name)
                    if value is None:
                        if spec.required:
                            raise KeyError(f"Required modality {name!r} is missing")
                        continue
                    if spec.skip is not None and values_equal(value, spec.skip):
                        continue
                    _emit([int(unwrap_scalar(value))], name=NAME_TEXT)
                else:
                    spec = image_by_field[name]
                    value = row.get(name)
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
                    _emit(img_ids, name=NAME_VISION)
                continue

            if literal:
                text_buf.append(literal)
            spec = text_by_field[name]
            rendered = _field_text_value(spec, row)
            if rendered is None:
                continue
            text_buf.append(rendered)

        flush_text()

    if not modality_ids:
        raise ValueError(
            "step has no tokens after skips; ensure the step format still "
            "produces at least one token"
        )

    return StepTokens(
        modality_ids=np.asarray(modality_ids, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        modality_names=modality_names,
        modality_map=dict(modality_map),
        grouping_id=gid,
        grouping_field=grouping_field,
        objective_fields=_copy_keep_fields(row, objective_fields_keep),
    )
