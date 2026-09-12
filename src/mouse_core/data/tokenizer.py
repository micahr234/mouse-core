"""Tokenizer — one step dict → StepTokens (text / discrete / continuous).

I/O
---
* **in:** ``dict`` (one step; must include ``grouping_field``)
* **out:** :class:`~mouse_core.data.token_batch.StepTokens`

``text`` / ``token`` fields share the ``__text__`` stream. Every other
field is tagged by ``output_field`` (modality name). Pack many steps with
:func:`~mouse_core.data.token_batch.pack_token_batch`.
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
    KIND_DISCRETE,
    KIND_FOURIER,
    KIND_IMAGE,
    KIND_LEARNABLE,
    KIND_TEXT,
    KIND_TOKEN,
    NAME_TEXT,
    TEXT_FORMAT_KEY,
    TokenizerModalityMeta,
    TokenizerModalitySpec,
    copy_keep_fields,
    resolve_tokenizer_modalities,
    unwrap_scalar,
    values_equal,
)
from mouse_core.data.token_batch import ModalityInfo, StepTokens


class Tokenizer:
    """CPU packer: one step dict → :class:`StepTokens`.

    Construct independently of the embedder. Alignment is by modality
    **name**: ``__text__`` for ``text`` / ``token`` fields (and
    ``group_prefix=``), ``output_field`` for every other type.
    ``input_fields=`` are the tokens fed to the transformer, emitted in
    list order. Each field is its own tokenize/emit run (no BPE merge
    across ``text`` fields). Exactly one input field must set
    ``head_output=True``: its tokens are the step's head-output tokens —
    the positions the model reads Q / action outputs from. Every step
    must emit at least one (never skip that field); a step may emit
    several (e.g. ``learnable`` with ``tokens > 1``), and the DQN
    objectives then train each of them toward the same per-step target.

    A ``text`` field requires ``format=``. ``input_field=`` reads the
    step into exactly one placeholder ``{field}``.
    Omit ``input_field=`` and the field is a const: ``output_field=``
    names it and ``format=`` is the literal string (no placeholders;
    ``{{`` / ``}}`` for a literal brace, as in every ``format=``).
    ``skip=`` / ``format_skipped=`` replace the run with that literal
    when the step value matches ``skip``. A ``required=False`` field
    whose value is missing / ``None`` emits nothing (``format_skipped=``
    does not apply). ``max_tokens=`` (``text`` / ``image``) raises if
    that run is longer. ``learnable`` fields have no step I/O and emit
    in list order. ``group_prefix=`` is a format string over the raw
    step dict (placeholders need not be ``input_fields``); it is
    tokenized as ``__text__`` and :func:`~mouse_core.data.token_batch.pack_token_batch`
    inserts those tokens at the start of each grouping-field segment.
    ``objective_fields=`` is a list of ``{input_field}`` dicts (optional
    ``output_field``; defaults to the input name)
    copied into ``StepTokens.objective_fields`` (input fields are not
    auto-copied). ``grouping_field`` names the step key used for
    attention isolation (typically ``task_index``).

    TD / PPO / GRPO objectives read ``action``, ``reward``,
    ``episode_done``, and ``task_done`` from that keep-list (plus extras
    such as ``old_log_prob``). ``task_done`` is an objective column only
    — it is not interpolated into a field format and is not fed to the
    transformer.
    """

    def __init__(
        self,
        *,
        input_fields: Sequence[dict[str, Any] | TokenizerModalitySpec] | None = None,
        grouping_field: str,
        group_prefix: str | None = None,
        tokenizer=None,
        image_tokenizer=None,
        objective_fields: Sequence[dict[str, Any]] | None = None,
        pretrained: str | Path | None = None,
        hub_kwargs: dict | None = None,
    ) -> None:
        if not grouping_field:
            raise ValueError("Tokenizer requires a non-empty grouping_field")
        specs, meta = resolve_tokenizer_modalities(input_fields)
        has_text = any(m.kind == KIND_TEXT for m in meta)
        has_token = any(m.kind == KIND_TOKEN for m in meta)
        has_image = any(m.kind == KIND_IMAGE for m in meta)

        flagged = [m.spec.output_field for m in meta if m.spec.head_output]
        if len(flagged) != 1:
            raise ValueError(
                "Tokenizer requires exactly one input field with "
                "head_output=True (its tokens are the step's head-output tokens "
                f"— the Q / action readout positions); got {flagged or 'none'}"
            )

        if group_prefix is not None and group_prefix == "":
            raise ValueError("Tokenizer group_prefix= must be a non-empty string")
        needs_tokenizer = has_text or group_prefix is not None
        if tokenizer is not None:
            tok = tokenizer
        elif pretrained is not None and needs_tokenizer:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(pretrained, **dict(hub_kwargs or {}))
        elif needs_tokenizer:
            raise TypeError(
                "Tokenizer with text input_fields or group_prefix= requires "
                "tokenizer= or pretrained="
            )
        else:
            tok = None

        if has_image:
            if image_tokenizer is None or not callable(image_tokenizer):
                raise TypeError(
                    "Tokenizer with image input_fields requires image_tokenizer= "
                    "callable that returns discrete token ids"
                )

        names: list[str] = []
        mmap: dict[str, ModalityInfo] = {}
        if has_text or has_token or group_prefix is not None:
            names.append(NAME_TEXT)
            mmap[NAME_TEXT] = ModalityInfo(type="token")
        for m in meta:
            if m.kind in (KIND_TEXT, KIND_TOKEN):
                continue
            if m.name in mmap:
                raise ValueError(f"duplicate tokenizer modality name {m.name!r}")
            names.append(m.name)
            if m.kind == KIND_FOURIER:
                mmap[m.name] = ModalityInfo(type="fourier", dim=m.dim)
            elif m.kind == KIND_DISCRETE:
                mmap[m.name] = ModalityInfo(type="discrete")
            elif m.kind == KIND_IMAGE:
                mmap[m.name] = ModalityInfo(type="image")
            elif m.kind == KIND_LEARNABLE:
                mmap[m.name] = ModalityInfo(type="learnable")
            else:
                raise ValueError(f"unsupported modality kind {m.kind!r}")

        self.group_prefix = group_prefix
        self.input_fields: tuple[TokenizerModalitySpec, ...] = tuple(specs)
        self._meta: tuple[TokenizerModalityMeta, ...] = tuple(meta)
        self.grouping_field = grouping_field
        self.tokenizer = tok
        self.image_tokenizer = image_tokenizer
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
                f"Tokenizer expects a step dict, got {type(step).__name__}"
            )
        return _tokenize_step(
            row=step,
            meta=self._meta,
            group_prefix_str=self.group_prefix,
            tokenizer=self.tokenizer,
            image_tokenizer=self.image_tokenizer,
            objective_fields_keep=self.objective_fields,
            grouping_field=self.grouping_field,
            name_to_index=self._name_to_index,
            modality_names=self.modality_names,
            modality_map=self.modality_map,
        )


def _field_text_value(spec: TokenizerModalitySpec, row: dict[str, Any]) -> str | None:
    assert isinstance(spec.output_field, str)
    assert spec.format is not None
    # Const ``format=`` and ``format_skipped=`` are validated as
    # placeholder-free format strings, so render them through ``format_map``
    # too: ``{{`` / ``}}`` un-escape the same way as in a step-backed format.
    if spec.input_field is None:
        return spec.format.format_map({})
    raw = row.get(spec.input_field)
    if raw is None:
        if spec.required:
            raise KeyError(f"Required modality {spec.input_field!r} is missing")
        return None
    if spec.skip is not None and values_equal(raw, spec.skip):
        assert spec.format_skipped is not None
        return spec.format_skipped.format_map({})
    return spec.format.format_map({TEXT_FORMAT_KEY: unwrap_scalar(raw)})


def _require_max_tokens(spec: TokenizerModalitySpec, token_ids: list[int]) -> None:
    limit = spec.max_tokens
    if limit is None:
        return
    n = len(token_ids)
    if n > limit:
        raise ValueError(
            f"field {spec.output_field!r} tokenized to {n} tokens "
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


def _tokenize_step(
    *,
    row: dict,
    meta: Sequence[TokenizerModalityMeta],
    group_prefix_str: str | None,
    tokenizer: Any,
    image_tokenizer: Any,
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

    def _emit(
        token_ids: list[int],
        *,
        spec: TokenizerModalitySpec,
        name: str,
        token_values: list[float] | None = None,
        head_output: bool = False,
    ) -> None:
        _require_max_tokens(spec, token_ids)
        mid = name_to_index[name]
        pos = next_position.get(mid, 0)
        vals = token_values if token_values is not None else [0.0] * len(token_ids)
        if len(vals) != len(token_ids):
            raise ValueError(
                f"token values length {len(vals)} != token id length {len(token_ids)}"
            )
        for tid, val in zip(token_ids, vals, strict=True):
            modality_ids.append(mid)
            ids.append(tid)
            values.append(val)
            positions.append(pos)
            pos += 1
            head_output_mask.append(head_output)
        next_position[mid] = pos

    for m in meta:
        spec = m.spec
        if m.kind == KIND_LEARNABLE:
            n = int(m.n_learnable)
            _emit(
                list(range(n)),
                spec=spec,
                name=m.name,
                head_output=spec.head_output,
            )
            continue

        if m.kind == KIND_TEXT:
            rendered = _field_text_value(spec, row)
            if rendered is None:
                continue
            if tokenizer is None:
                raise RuntimeError("tokenizer required to tokenize text runs")
            _emit(
                _tokenize_ids(tokenizer, rendered),
                spec=spec,
                name=NAME_TEXT,
                head_output=spec.head_output,
            )
            continue

        in_name = str(spec.input_field)
        value = row.get(in_name)
        if value is None:
            if spec.required:
                raise KeyError(
                    f"Required input field {in_name!r} is missing from step"
                )
            continue
        if spec.skip is not None and values_equal(value, spec.skip):
            continue

        if m.kind == KIND_TOKEN:
            _emit(
                [int(unwrap_scalar(value))],
                spec=spec,
                name=NAME_TEXT,
                head_output=spec.head_output,
            )
        elif m.kind == KIND_DISCRETE:
            _emit(
                [int(unwrap_scalar(value))],
                spec=spec,
                name=m.name,
                head_output=spec.head_output,
            )
        elif m.kind == KIND_FOURIER:
            if m.dim == 1:
                vals = [float(unwrap_scalar(value))]
            else:
                arr = np.asarray(value, dtype=np.float32).ravel()
                if arr.size != m.dim:
                    raise ValueError(
                        f"continuous field {in_name!r} declared dim={m.dim} but the "
                        f"step value has {arr.size} elements"
                    )
                vals = [float(v) for v in arr]
            _emit(
                list(range(len(vals))),
                spec=spec,
                name=m.name,
                token_values=vals,
                head_output=spec.head_output,
            )
        elif m.kind == KIND_IMAGE:
            img_ids = _image_token_ids(image_tokenizer, value, in_name=in_name)
            _emit(img_ids, spec=spec, name=m.name, head_output=spec.head_output)
        else:
            raise ValueError(f"unsupported modality kind {m.kind!r}")

    if not modality_ids:
        raise ValueError(
            "step has no tokens after skips; ensure at least one input field "
            "still produces a token"
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
