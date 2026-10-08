"""Shared helpers: list[list[dict]] → TokenBatch via per-step compose + pack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from mouse_core.data import Tokenizer, pack_token_batch
from mouse_core.data.token_batch import StepTokens, TokenBatch

def when_episode_done_nonzero(ctx):
    return "episode_done" in ctx and ctx["episode_done"] != 0


def when_group_start(ctx):
    return bool(ctx["group_start"])


def when_reward_nonzero(ctx):
    return "reward" in ctx and ctx["reward"] != 0.0


def when_step_index_zero(ctx):
    return "step_index" in ctx and ctx["step_index"] == 0


def when_step_index_zero_or_group_start(ctx):
    return (ctx.get("step_index") == 0) | bool(ctx["group_start"])


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
    """Pack integer step fields as ``type="text"`` with ``format="{field}"``.

    ``IntIdTokenizer`` turns that rendered integer into one vocab id, so an
    action value of ``1`` is still token id ``1``. The last listed field is
    ``head_output=True``. Reward floats stay on ``objective_fields`` only.
    """
    input_fields: list[dict[str, Any]] = [
        {"type": "text", "input_field": name, "format": "{field}"} for name in fields
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
            {
                "type": "text",
                "input_field": "episode_index",
                "format": "{field}",
                "when": when_step_index_zero,
            },
        )
    if objective_fields is None:
        resolved = [{"input_field": name} for name in fields]
    elif objective_fields and isinstance(objective_fields[0], str):
        resolved = [{"input_field": name} for name in cast(list[str], objective_fields)]
    else:
        resolved = cast(list[dict[str, Any]], objective_fields)
    if "tokenizer" not in kwargs:
        kwargs["tokenizer"] = IntIdTokenizer()
    return Tokenizer(
        input_fields=input_fields,
        objective_fields=resolved,
        **kwargs,
    )


def batch_to_packed(
    tokenizer: Callable[[dict], StepTokens],
    batch: list[list[dict]],
) -> tuple[TokenBatch, dict[str, torch.Tensor], torch.Tensor]:
    """Tokenize a ragged ``list[list[dict]]`` into ``(inputs, objective_data, group_id)``."""
    steps: list[StepTokens] = []
    sids: list[int] = []
    for b, seq in enumerate(batch):
        for step in seq:
            steps.append(tokenizer(step))
            sids.append(b)
    return pack_token_batch(
        steps=steps,
        group_ids=sids if steps else None,
        batch_size=len(batch),
        continuing=None,
    )


def batch_to_token_batch(
    tokenizer: Callable[[dict], StepTokens],
    batch: list[list[dict]],
) -> TokenBatch:
    """Tokenize a ragged ``list[list[dict]]`` to model inputs only."""
    inputs, _, _ = batch_to_packed(tokenizer, batch)
    return inputs
