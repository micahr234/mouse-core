"""Selector — keep and optionally rename fields on one step.

I/O
---
* **in:** ``dict`` (one step)
* **out:** ``dict`` (only mapped keys, under their output names)

``fields`` maps input step keys → output names. This is the last pipeline stage
that remaps field names; tokenizers and embedders use a single ``field`` name
thereafter. Include the Grouper ``output_field`` in ``fields`` when a
:class:`Grouper` runs earlier in the composed pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Selector:
    """Callable that keeps listed inputs and writes them under output names."""

    fields: Mapping[str, str]

    def __post_init__(self) -> None:
        mapping = dict(self.fields)
        if not mapping:
            raise ValueError("Selector requires a non-empty fields mapping")
        outputs = list(mapping.values())
        if len(outputs) != len(set(outputs)):
            raise ValueError(
                f"Selector fields has duplicate output names: {outputs!r}"
            )
        object.__setattr__(self, "fields", mapping)

    def __call__(self, step: dict) -> dict:
        out: dict = {}
        for in_name, out_name in self.fields.items():
            if in_name not in step:
                raise KeyError(
                    f"Selector input field {in_name!r} missing from step "
                    f"(have {sorted(step)})"
                )
            out[out_name] = step[in_name]
        return out
