"""Grouper — copy ``input_field`` onto ``output_field`` on one step.

I/O
---
* **in:** ``dict`` (one step)
* **out:** ``dict`` (copy with ``output_field`` set from ``input_field``)

Both ``input_field`` and ``output_field`` are required. Values are copied as-is
(no int cast). Same names replace the key in place.

The tokenizer's ``grouping_field`` should match ``output_field`` when this
column is used for attention isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

from mouse_core.data.modality import unwrap_scalar


@dataclass(frozen=True)
class Grouper:
    """Callable that copies one step field onto another name."""

    input_field: str
    output_field: str

    def __call__(self, step: dict) -> dict:
        if self.input_field not in step:
            raise KeyError(
                f"Grouper input_field {self.input_field!r} missing from step "
                f"(have {sorted(step)})"
            )
        out = dict(step)
        out[self.output_field] = unwrap_scalar(step[self.input_field])
        return out
