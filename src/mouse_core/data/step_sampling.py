"""Step boundaries shared by event tokenization and source-event sampling."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mouse_core.data.events import EventLog


def _event_field(event: Mapping, field: str | None, *, encoded_payload: bool = False) -> Any:
    if field is None:
        return None
    if field.startswith("payload."):
        payload = event["payload"]
        if encoded_payload and isinstance(payload, str):
            payload = json.loads(payload)
        return payload.get(field.removeprefix("payload.")) if isinstance(payload, Mapping) else None
    return event.get(field)


def _validate_marker(value: Any, field: str) -> None:
    if value is not None and (type(value) not in (str, int, float)
                              or isinstance(value, float) and not math.isfinite(value)):
        raise ValueError(f"Step boundary field {field!r} must contain finite scalar markers")


@dataclass
class _StepBoundary:
    step_field: str
    episode_field: str | None
    step: Any = None
    episode: Any = None
    started: bool = False

    def __post_init__(self):
        for field in (self.step_field, self.episode_field):
            if field is not None and (not isinstance(field, str) or not field or field == "payload."):
                raise ValueError("Boundary fields must be nonempty field names")
        if self.step_field is None:
            raise ValueError("step_field is required")

    def advance(self, event: Mapping, *, encoded_payload: bool = False) -> bool:
        """A changed marker starts a step before this event; omissions continue it."""
        step = _event_field(event, self.step_field, encoded_payload=encoded_payload)
        episode = _event_field(event, self.episode_field, encoded_payload=encoded_payload)
        _validate_marker(step, self.step_field)
        if self.episode_field is not None:
            _validate_marker(episode, self.episode_field)
        new_episode = episode is not None and episode != self.episode
        if (not self.started or new_episode) and step is None:
            raise ValueError(f"The first event of a sequence or episode requires {self.step_field!r}")
        changed = not self.started or new_episode or step is not None and step != self.step
        if step is not None:
            self.step = step
        if episode is not None:
            self.episode = episode
        self.started = True
        return changed


def _boundary_events(events, fields):
    """Project header columns in batches, without reading or decoding payloads."""
    if hasattr(events, "column_names") and all(not f.startswith("payload.") for f in fields if f):
        columns = [f for f in dict.fromkeys(fields) if f in events.column_names]
        if not columns:
            # Preserve missing-marker errors, including on nonempty datasets.
            yield from ({} for _ in range(len(events)))
            return
        for batch in events.select_columns(columns).iter(batch_size=8192):
            yield from (dict(zip(columns, row)) for row in zip(*(batch[f] for f in columns)))
    else:
        yield from events



def _step_spans(events, *, step_field: str, episode_field: str | None):
    boundary = _StepBoundary(step_field, episode_field)
    start, episode, end = 0, None, 0
    for index, event in enumerate(_boundary_events(events, (step_field, episode_field))):
        changed = boundary.advance(event, encoded_payload=not isinstance(events, EventLog))
        if changed and index:
            yield start, index, episode
        if changed:
            start, episode = index, boundary.episode
        end = index + 1
    if end:
        yield start, end, episode



def _validate_step_ranges(events, ranges, *, step_field: str, episode_field: str | None):
    spans = list(_step_spans(events, step_field=step_field, episode_field=episode_field))
    starts = {start: index for index, (start, _, _) in enumerate(spans)}
    ends = {end: index for index, (_, end, _) in enumerate(spans)}
    # Count episode transitions, so even a repeated ID cannot hide a crossing.
    episodes = []
    generation = 0
    for index, (_, _, episode) in enumerate(spans):
        if index and episode != spans[index - 1][2]:
            generation += 1
        episodes.append(generation)
    for start, end in ranges:
        if start not in starts or end not in ends:
            raise ValueError("sample_ranges must select whole steps; a boundary cuts through a step")
        if episodes[starts[start]] != episodes[ends[end]]:
            raise ValueError("sample_ranges must stay within one episode")
