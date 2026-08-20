"""Grouper — copy listed fields onto output names on one step.

I/O
---
* **in:** ``dict`` (one step)
* **out:** ``dict`` (copy with mapped keys set; other keys left in place)

``fields`` is a list of ``{input_field, output_field}`` dicts. Values are
copied as-is (no int cast). Same names replace the key in place; different
names write the output and leave the input.

The tokenizer's ``grouping_field`` should match a Grouper output name when
this column is used for attention isolation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mouse_core.data.io_fields import coerce_io_fields
from mouse_core.data.modality import unwrap_scalar


class Grouper:
    """Callable that copies listed inputs onto output names."""

    def __init__(self, *, fields: Sequence[Mapping[str, Any]]) -> None:
        self.fields = coerce_io_fields(fields, who="Grouper")

    def __call__(self, step: dict) -> dict:
        out = dict(step)
        for in_name, out_name in self.fields:
            if in_name not in step:
                raise KeyError(
                    f"Grouper input field {in_name!r} missing from step "
                    f"(have {sorted(step)})"
                )
            out[out_name] = unwrap_scalar(step[in_name])
        return out
