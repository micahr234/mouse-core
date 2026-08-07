"""Shared helpers: list[list[dict]] → TokenBatch via per-step compose + pack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, is_dataclass

from mouse_core.data import Grouper, NumericTokenizer, compose, pack_token_batch
from mouse_core.data.token_batch import StepTokens, TokenBatch

DEFAULT_GROUPING_FIELD = "grouping_id"


def _tokenizer_modalities_from_encoder(encoder) -> list[dict]:
    """Map embedder modality specs to tokenizer packing specs (by name)."""
    out: list[dict] = []
    for m in encoder.modalities:
        data = asdict(m) if is_dataclass(m) else dict(m)
        kind = str(data["type"]).lower()
        if kind == "learnable":
            out.append({"type": "learnable", "tokens": data.get("tokens")})
            continue
        name = data.get("field")
        entry: dict = {
            "type": kind,
            "field": name,
        }
        if data.get("dim") is not None:
            entry["dim"] = data["dim"]
        out.append(entry)
    return out


def tok_from_encoder(
    encoder,
    *,
    grouping_field: str = DEFAULT_GROUPING_FIELD,
    step_fields: list[str] | None = None,
    **kwargs,
) -> NumericTokenizer:
    # Default keep-list: non-learnable modality names (common for tests/objectives).
    if step_fields is None:
        step_fields = []
        for m in encoder.modalities:
            data = asdict(m) if is_dataclass(m) else dict(m)
            if str(data["type"]).lower() == "learnable":
                continue
            name = data.get("field")
            if isinstance(name, str):
                step_fields.append(name)
    return NumericTokenizer(
        modalities=_tokenizer_modalities_from_encoder(encoder),
        step_fields=step_fields,
        grouping_field=grouping_field,
        **kwargs,
    )


def _ensure_grouping_field(step: dict, grouping_field: str) -> dict:
    """Stamp a constant grouping value when no Grouper ran (no isolation)."""
    if grouping_field in step:
        return step
    out = dict(step)
    out[grouping_field] = 0
    return out


def batch_to_token_batch(
    tokenizer: Callable[[dict], StepTokens],
    batch: list[list[dict]],
    *,
    grouper: Grouper | None = None,
    grouping_field: str = DEFAULT_GROUPING_FIELD,
) -> TokenBatch:
    """Tokenize a ragged ``list[list[dict]]`` with per-step transform + pack."""
    if grouper is None:
        transform = compose(
            lambda step: _ensure_grouping_field(step, grouping_field),
            tokenizer,
        )
    else:
        transform = compose(grouper, tokenizer)
    steps: list[StepTokens] = []
    sids: list[int] = []
    for b, seq in enumerate(batch):
        for step in seq:
            steps.append(transform(step))
            sids.append(b)
    return pack_token_batch(
        steps,
        sequence_ids=sids if steps else None,
        batch_size=len(batch),
        grouping_field=grouping_field,
    )
