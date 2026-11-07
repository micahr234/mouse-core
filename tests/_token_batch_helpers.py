"""Encode synthetic measurement payloads as events, then pack model test batches."""

from __future__ import annotations
from tensordict import TensorDict

from collections.abc import Callable
from typing import Any, cast

import torch

from mouse_core.data import Tokenizer, pack_token_batch
from tests._event_fixtures import encode_measurement
from mouse_core.data.token_batch import StepTokens


DEFAULT_TOKEN_VOCAB = 32


class IntIdTokenizer:
    """Map a rendered integer string back to that integer as one token id."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ):
        return {"input_ids": torch.tensor([[int(text)]], dtype=torch.long)}


def token_tokenizer(
    *fields: str,
    objective_fields: list[dict[str, Any]] | list[str] | None = None,
    **kwargs: Any,
) -> Tokenizer:
    """Pack integer step fields as ``type="text"`` with formats using each field name.

    ``IntIdTokenizer`` turns that rendered integer into one vocab id, so an
    action value of ``1`` is still token id ``1``. The last listed field is
    ``head_output=True``. Reward floats stay on ``objective_fields`` only.
    """
    input_fields: list[dict[str, Any]] = [
        {"provider": "sensor", "type": "text", "input_field": name, "format": "{" + name + "}"} for name in fields
    ]
    if input_fields:
        input_fields[-1]["head_output"] = True
    if input_fields and not any(
        field.get("input_field") == "episode_index"
        for field in input_fields
    ):
        head_at = next(
            i for i, field in enumerate(input_fields) if field.get("head_output")
        )
        input_fields.insert(
            head_at,
            {"provider": "sensor",
                "type": "text",
                "input_field": "episode_index",
                "format": lambda ctx: f"{ctx['episode_index']}" if ('step_index' in ctx and ctx['step_index'] == 0) else "",
            },
        )
    if objective_fields is None:
        resolved = [{"input_field": name} for name in fields]
    elif objective_fields and isinstance(objective_fields[0], str):
        resolved = [{"input_field": name} for name in cast(list[str], objective_fields)]
    else:
        resolved = cast(list[dict[str, Any]], objective_fields)
    if "tokenizers" not in kwargs:
        kwargs["tokenizers"] = {"default": IntIdTokenizer()}
    return Tokenizer(
        input_fields=input_fields,
        objective_fields=resolved,
        **kwargs,
    )


def batch_to_packed(
    tokenizer: Tokenizer | Callable[[dict], StepTokens],
    batch: list[list[dict]],
) -> tuple[TensorDict, dict[str, torch.Tensor]]:
    """Tokenize a ragged ``list[list[dict]]`` into ``(inputs, objective_data)``."""
    steps: list[StepTokens] = []
    sids: list[int] = []
    for b, seq in enumerate(batch):
        for step in seq:
            steps.append(encode_measurement(tokenizer, step) if isinstance(tokenizer, Tokenizer) else tokenizer(step))
            sids.append(b)
    return pack_token_batch(
        steps=steps,
        sequence_ids=sids if steps else None,
        batch_size=len(batch),
    )


def batch_to_token_batch(
    tokenizer: Tokenizer | Callable[[dict], StepTokens],
    batch: list[list[dict]],
) -> TensorDict:
    """Tokenize a ragged ``list[list[dict]]`` to model inputs only."""
    inputs, _ = batch_to_packed(tokenizer, batch)
    return inputs


def embed_batch(embedding, batch):
    tensors = batch.to(next(embedding.parameters()).device)
    return embedding(
        tensors["token_ids"], numeric_values=tensors["numeric_values"],
        numeric_mask=tensors["numeric_mask"],
        numeric_field_ids=tensors["numeric_field_ids"], numeric_field_names=batch['numeric_field_names'],
        tokenizer_ids=tensors["tokenizer_ids"], tokenizer_names=batch['tokenizer_names'],
    )
