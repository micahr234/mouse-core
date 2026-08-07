"""NumericTokenizer — one step dict → StepTokens (discrete / continuous payloads).

I/O
---
* **in:** ``dict`` (one step; must include ``grouping_field``)
* **out:** :class:`~mouse_core.data.token_batch.StepTokens`

Tokens are tagged by modality **name** (``field``). Pack many steps with
:func:`~mouse_core.data.token_batch.pack_token_batch`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mouse_core.data.modality import (
    KIND_DISCRETE,
    KIND_FOURIER,
    KIND_IMAGE,
    KIND_LEARNABLE,
    NumericTokenizerModalitySpec,
    TokenizerModalityMeta,
    resolve_tokenizer_numeric_modalities,
    unwrap_scalar,
    values_equal,
)
from mouse_core.data.token_batch import ModalityInfo, StepTokens


class NumericTokenizer:
    """CPU packer: one step dict → :class:`StepTokens`.

    Construct independently of the embedder. Alignment is by modality **name**
    (``field``), not list order. ``step_fields=`` is an explicit keep-list of
    step dict keys copied into ``StepTokens.step_fields`` (modalities are not
    auto-copied). Rename step keys with :class:`~mouse_core.data.selector.Selector`
    before tokenize.
    """

    def __init__(
        self,
        *,
        modalities: list[dict[str, Any] | NumericTokenizerModalitySpec] | None = None,
        grouping_field: str,
        image_tokenizer: Callable[[Any], Sequence[int]] | None = None,
        step_fields: Sequence[str] | None = None,
    ) -> None:
        if not grouping_field:
            raise ValueError("NumericTokenizer requires a non-empty grouping_field")
        specs, meta = resolve_tokenizer_numeric_modalities(modalities)
        if any(m.kind == KIND_IMAGE for m in meta) and image_tokenizer is None:
            raise TypeError(
                "NumericTokenizer with type='image' modalities requires image_tokenizer="
            )
        self.modalities: list[NumericTokenizerModalitySpec] = list(specs)
        self._meta: tuple[TokenizerModalityMeta, ...] = tuple(meta)
        self.modality_names: tuple[str, ...] = tuple(m.name for m in meta)
        self.modality_map: dict[str, ModalityInfo] = {
            m.name: ModalityInfo(
                type=m.kind if m.kind != KIND_FOURIER else "fourier",
                dim=m.dim,
            )
            for m in meta
        }
        # Normalize continuous → fourier in map type for embedder agreement.
        for m in meta:
            if m.kind == KIND_FOURIER:
                self.modality_map[m.name] = ModalityInfo(type="fourier", dim=m.dim)
        self._name_to_index = {n: i for i, n in enumerate(self.modality_names)}
        self.grouping_field = grouping_field
        self.image_tokenizer = image_tokenizer
        self.step_fields: tuple[str, ...] = tuple(step_fields or ())

    def __call__(self, step: dict) -> StepTokens:
        if not isinstance(step, dict):
            raise TypeError(
                f"NumericTokenizer expects a step dict, got {type(step).__name__}"
            )
        return _tokenize_numeric_step(
            step,
            self._meta,
            self._name_to_index,
            self.modality_names,
            self.modality_map,
            self.image_tokenizer,
            self.step_fields,
            grouping_field=self.grouping_field,
        )


def _copy_keep_fields(
    row: dict,
    keep: Sequence[str],
) -> dict[str, Any]:
    """Copy keep-list columns from the step.

    Every ``step_fields`` key must be present (and not ``None``) on the step.
    There is no silent default: a missing objective column (e.g.
    ``old_log_prob`` or ``advantage``) would otherwise train on zeros.
    """
    out: dict[str, Any] = {}
    for field in keep:
        value = row.get(field)
        if value is None:
            raise KeyError(
                f"step_fields key {field!r} is missing from step "
                f"(have {sorted(row)}); stamp it on the row before tokenizing"
            )
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            if np.issubdtype(arr.dtype, np.floating):
                out[field] = arr.astype(np.float32).ravel()
            else:
                out[field] = arr.astype(np.int64).ravel()
            continue
        if isinstance(value, np.ndarray) and value.ndim > 0:
            if np.issubdtype(value.dtype, np.floating):
                out[field] = value.astype(np.float32).ravel()
            else:
                out[field] = value.astype(np.int64).ravel()
            continue
        sample = unwrap_scalar(value)
        if isinstance(sample, (float, np.floating)):
            out[field] = float(sample)
        else:
            out[field] = int(sample)
    return out


def _tokenize_numeric_step(
    row: dict,
    meta: Sequence[TokenizerModalityMeta],
    name_to_index: dict[str, int],
    modality_names: tuple[str, ...],
    modality_map: dict[str, ModalityInfo],
    image_tokenizer: Callable[[Any], Sequence[int]] | None,
    step_fields_keep: Sequence[str],
    *,
    grouping_field: str,
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

    def _emit(*, name: str, token_id: int, value: float = 0.0) -> None:
        modality_ids.append(name_to_index[name])
        ids.append(token_id)
        values.append(value)

    for m in meta:
        field = str(m.spec.field)
        name = m.name
        spec = m.spec
        if m.kind == KIND_LEARNABLE:
            for i in range(m.n_learnable):
                _emit(name=name, token_id=i)
            continue

        value = row.get(field)
        if value is None:
            if spec.required:
                raise KeyError(
                    f"Required modality field {field!r} is missing from step"
                )
            continue
        if spec.skip is not None and values_equal(value, spec.skip):
            continue

        if m.kind == KIND_DISCRETE:
            _emit(name=name, token_id=int(unwrap_scalar(value)))
        elif m.kind == KIND_FOURIER:
            if m.dim == 1:
                vals = [float(unwrap_scalar(value))]
            else:
                arr = np.asarray(value, dtype=np.float32).ravel()
                if arr.size < m.dim:
                    pad = np.zeros(m.dim - arr.size, dtype=np.float32)
                    arr = np.concatenate([arr, pad])
                vals = [float(arr[i]) for i in range(m.dim)]
            for i, v in enumerate(vals):
                _emit(name=name, token_id=i, value=v)
        elif m.kind == KIND_IMAGE:
            if image_tokenizer is None:
                raise RuntimeError("image_tokenizer is not configured")
            img_ids = list(image_tokenizer(value))
            if not img_ids:
                raise ValueError(
                    f"image tokenizer returned no tokens for {field!r}"
                )
            for tid in img_ids:
                _emit(name=name, token_id=int(tid))

    if not modality_ids:
        raise ValueError(
            "step has no tokens after skips; ensure at least one modality "
            "is present (e.g. add a learnable modality)"
        )

    return StepTokens(
        modality_ids=np.asarray(modality_ids, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        modality_names=modality_names,
        modality_map=dict(modality_map),
        grouping_id=gid,
        grouping_field=grouping_field,
        step_fields=_copy_keep_fields(row, step_fields_keep),
    )
