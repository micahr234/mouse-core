"""Shared coerce for Selector ``fields=`` lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def coerce_io_fields(
    fields: Sequence[Mapping[str, Any]],
    *,
    who: str,
) -> tuple[tuple[str, str], ...]:
    """Return ``(input_field, output_field)`` pairs from a list of dicts."""
    if isinstance(fields, Mapping):
        raise TypeError(
            f"{who} fields is a list of dicts with input_field=/output_field= "
            "(not a name→name mapping)"
        )
    pairs: list[tuple[str, str]] = []
    for spec in fields:
        if not isinstance(spec, Mapping):
            raise TypeError(
                f"{who} fields entries must be dicts with input_field= and "
                f"output_field=, got {type(spec).__name__}"
            )
        data = dict(spec)
        if "field" in data and "input_field" not in data:
            raise TypeError(
                f"{who} fields use input_field=/output_field= (not field=)"
            )
        unknown = set(data) - {"input_field", "output_field"}
        if unknown:
            raise TypeError(
                f"{who} field spec unknown keys {sorted(unknown)}; "
                "only input_field= and output_field= are allowed"
            )
        try:
            in_name = data["input_field"]
            out_name = data["output_field"]
        except KeyError as exc:
            raise TypeError(
                f"{who} field spec requires input_field= and output_field="
            ) from exc
        if not isinstance(in_name, str) or not isinstance(out_name, str):
            raise TypeError(
                f"{who} input_field=/output_field= must be strings, "
                f"got {in_name!r} → {out_name!r}"
            )
        pairs.append((in_name, out_name))
    if not pairs:
        raise ValueError(f"{who} requires a non-empty fields list")
    outputs = [out_name for _, out_name in pairs]
    if len(outputs) != len(set(outputs)):
        raise ValueError(
            f"{who} fields has duplicate output names: {outputs!r}"
        )
    return tuple(pairs)
