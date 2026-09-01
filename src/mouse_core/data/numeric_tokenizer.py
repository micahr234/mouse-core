"""NumericTokenizer — one step dict → StepTokens (discrete / continuous payloads).

I/O
---
* **in:** ``dict`` (one step; must include ``grouping_field``)
* **out:** :class:`~mouse_core.data.token_batch.StepTokens`

Tokens are tagged by ``output_field`` (modality name). Pack many steps with
:func:`~mouse_core.data.token_batch.pack_token_batch`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from mouse_core.data.io_fields import coerce_io_fields
from mouse_core.data.modality import (
    KIND_DISCRETE,
    KIND_FOURIER,
    KIND_IMAGE,
    KIND_LEARNABLE,
    NumericTokenizerModalitySpec,
    TokenizerModalityMeta,
    copy_keep_fields,
    resolve_tokenizer_numeric_modalities,
    unwrap_scalar,
    values_equal,
)
from mouse_core.data.token_batch import ModalityInfo, StepTokens


class NumericTokenizer:
    """CPU packer: one step dict → :class:`StepTokens`.

    Construct independently of the embedder. Alignment is by ``output_field``
    name, not list order. ``input_fields=`` are the tokens fed to the
    transformer (each ``{type, input_field}``; optional ``output_field``).
    ``objective_fields=`` is a list of ``{input_field}`` dicts (optional
    ``output_field``; defaults to the input name)
    copied into ``StepTokens.objective_fields`` (input fields are not
    auto-copied). ``grouping_field`` names the step key used for attention
    isolation (typically ``task_index``).

    TD / PPO / GRPO objectives read ``action``, ``reward``, ``episode_done``,
    and ``task_done`` from that keep-list (plus extras such as ``old_log_prob``).
    ``task_done`` is an objective column only — it is not an input field and is
    not fed to the transformer.
    """

    def __init__(
        self,
        *,
        input_fields: list[dict[str, Any] | NumericTokenizerModalitySpec] | None = None,
        grouping_field: str,
        image_tokenizer: Callable[[Any], Sequence[int]] | None = None,
        objective_fields: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if not grouping_field:
            raise ValueError("NumericTokenizer requires a non-empty grouping_field")
        specs, meta = resolve_tokenizer_numeric_modalities(input_fields)
        if any(m.kind == KIND_IMAGE for m in meta) and image_tokenizer is None:
            raise TypeError(
                "NumericTokenizer with type='image' input_fields requires image_tokenizer="
            )
        self.input_fields: list[NumericTokenizerModalitySpec] = list(specs)
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
        self.objective_fields: tuple[tuple[str, str], ...] = coerce_io_fields(
            objective_fields or (),
            who="tokenizer objective_fields",
            allow_empty=True,
        )

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
            self.objective_fields,
            grouping_field=self.grouping_field,
        )


def _tokenize_numeric_step(
    row: dict,
    meta: Sequence[TokenizerModalityMeta],
    name_to_index: dict[str, int],
    modality_names: tuple[str, ...],
    modality_map: dict[str, ModalityInfo],
    image_tokenizer: Callable[[Any], Sequence[int]] | None,
    objective_fields_keep: Sequence[tuple[str, str]],
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
        name = m.name
        spec = m.spec
        if m.kind == KIND_LEARNABLE:
            for i in range(m.n_learnable):
                _emit(name=name, token_id=i)
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

        if m.kind == KIND_DISCRETE:
            _emit(name=name, token_id=int(unwrap_scalar(value)))
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
            for i, v in enumerate(vals):
                _emit(name=name, token_id=i, value=v)
        elif m.kind == KIND_IMAGE:
            if image_tokenizer is None:
                raise RuntimeError("image_tokenizer is not configured")
            img_ids = list(image_tokenizer(value))
            if not img_ids:
                raise ValueError(
                    f"image tokenizer returned no tokens for {in_name!r}"
                )
            for tid in img_ids:
                _emit(name=name, token_id=int(tid))

    if not modality_ids:
        raise ValueError(
            "step has no tokens after skips; ensure at least one input field "
            "is present (e.g. add a learnable input field)"
        )

    return StepTokens(
        modality_ids=np.asarray(modality_ids, dtype=np.int64),
        ids=np.asarray(ids, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        modality_names=modality_names,
        modality_map=dict(modality_map),
        grouping_id=gid,
        grouping_field=grouping_field,
        objective_fields=copy_keep_fields(row, objective_fields_keep),
    )
