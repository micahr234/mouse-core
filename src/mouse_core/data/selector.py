"""Selector — keep and optionally rename fields on one step.

I/O
---
* **in:** ``dict`` (one step)
* **out:** ``dict`` (only mapped keys, under their output names)

``fields`` is a list of ``{input_field, output_field}`` dicts. This is the last
pipeline stage that remaps field names; tokenizers and embedders use a single
``field`` name thereafter. Include the tokenizer ``grouping_field`` in ``fields``
so attention isolation survives this keep-list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mouse_core.data.io_fields import coerce_io_fields


class Selector:
    """Callable that keeps listed inputs and writes them under output names."""

    def __init__(self, *, fields: Sequence[Mapping[str, Any]]) -> None:
        self.fields = coerce_io_fields(fields, who="Selector")

    def __call__(self, step: dict) -> dict:
        out: dict = {}
        for in_name, out_name in self.fields:
            if in_name not in step:
                raise KeyError(
                    f"Selector input field {in_name!r} missing from step "
                    f"(have {sorted(step)})"
                )
            out[out_name] = step[in_name]
        return out
