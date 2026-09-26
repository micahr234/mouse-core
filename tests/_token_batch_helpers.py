"""Shared helpers: list[list[dict]] → TokenBatch via per-step compose + pack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from mouse_core.data import Tokenizer, compose, pack_token_batch
from mouse_core.data.token_batch import StepTokens, TokenBatch

DEFAULT_GROUPING_FIELD = "grouping_id"
DEFAULT_TOKEN_VOCAB = 32


def token_tokenizer(
    *fields: str,
    grouping_field: str = DEFAULT_GROUPING_FIELD,
    objective_fields: list[dict[str, Any]] | list[str] | None = None,
    **kwargs: Any,
) -> Tokenizer:
    """Pack integer step fields as ``type="token"`` (shared ``__text__`` stream).

    The last listed field is ``head_output=True``. Reward floats stay on
    ``objective_fields`` only.
    """
    input_fields: list[dict[str, Any]] = [
        {"type": "token", "input_field": name} for name in fields
    ]
    if input_fields:
        input_fields[-1]["head_output"] = True
    if input_fields and not any(
        field.get("input_field") == "episode_index"
        or field.get("output_field") == "episode_index"
        for field in input_fields
    ):
        head_at = next(
            i for i, field in enumerate(input_fields) if field.get("head_output")
        )
        input_fields.insert(
            head_at,
            {
                "type": "token",
                "input_field": "episode_index",
                "when_field": "step_index",
                "when_equals": 0,
            },
        )
    if objective_fields is None:
        resolved = [{"input_field": name} for name in fields]
    elif objective_fields and isinstance(objective_fields[0], str):
        resolved = [{"input_field": name} for name in cast(list[str], objective_fields)]
    else:
        resolved = cast(list[dict[str, Any]], objective_fields)
    return Tokenizer(
        input_fields=input_fields,
        objective_fields=resolved,
        grouping_field=grouping_field,
        **kwargs,
    )


def _ensure_grouping_field(step: dict, grouping_field: str) -> dict:
    """Stamp a constant grouping value when the step has no isolation column."""
    if grouping_field in step:
        return step
    out = dict(step)
    out[grouping_field] = 0
    return out


def batch_to_packed(
    tokenizer: Callable[[dict], StepTokens],
    batch: list[list[dict]],
    *,
    grouping_field: str = DEFAULT_GROUPING_FIELD,
) -> tuple[TokenBatch, dict[str, torch.Tensor]]:
    """Tokenize a ragged ``list[list[dict]]`` into ``(inputs, objective_data)``."""
    transform = compose(
        stages=(lambda step: _ensure_grouping_field(step, grouping_field), tokenizer),
    )
    steps: list[StepTokens] = []
    sids: list[int] = []
    for b, seq in enumerate(batch):
        for step in seq:
            steps.append(transform(step))
            sids.append(b)
    return pack_token_batch(
        steps=steps,
        sequence_ids=sids if steps else None,
        batch_size=len(batch),
        grouping_field=grouping_field,
    )


def batch_to_token_batch(
    tokenizer: Callable[[dict], StepTokens],
    batch: list[list[dict]],
    *,
    grouping_field: str = DEFAULT_GROUPING_FIELD,
) -> TokenBatch:
    """Tokenize a ragged ``list[list[dict]]`` to model inputs only."""
    inputs, _ = batch_to_packed(
        tokenizer, batch, grouping_field=grouping_field
    )
    return inputs
