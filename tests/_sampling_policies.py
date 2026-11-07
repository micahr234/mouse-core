"""Sampling policies owned by test scenarios, outside the Core API."""
import json
from mouse_core.data import EventLog, EnvironmentProtocol
from mouse_core.data.step_sampling import _boundary_events, _step_spans, _validate_marker
from mouse_core.data.event_transform import (_record, _headers, _validate_coordinator_payload,
                                             _validate_environment_payload)
_DEFAULT_PROTOCOL = EnvironmentProtocol()

def episode_sample_ranges(events, *, episode_field="episode_index"):
    """Header-only candidate episodes, including an unfinished final episode.

    Missing markers continue the current episode. Each ID must occupy one
    contiguous run. Payload eligibility and validation belong to the loader's
    sample callbacks, not this boundary scan.
    """
    if not isinstance(episode_field, str) or not episode_field or episode_field.startswith("payload."):
        raise ValueError("episode_field must name a header column")
    start, previous, end = 0, None, 0
    seen = set()
    for index, event in enumerate(_boundary_events(events, (episode_field,))):
        marker = event.get(episode_field)
        _validate_marker(marker, episode_field)
        if marker is None:
            if previous is None:
                raise ValueError(f"The first event requires {episode_field!r}")
        elif marker != previous:
            if marker in seen:
                raise ValueError("Episode indices must form one contiguous run each")
            if index:
                yield start, index
            seen.add(marker)
            start, previous = index, marker
        end = index + 1
    if end:
        yield start, end



def step_sample_ranges(events, *, steps_per_sample: int, step_field: str = "step_index",
                       episode_field: str = "episode_index", include_final_step: bool = True):
    """Yield sliding windows of exactly N whole steps, within one episode.

    Use ``partial(step_sample_ranges, steps_per_sample=N)`` as a loader's
    ``sample_ranges``. Short episodes produce no windows; windows never shrink
    to fit a budget. Each step is a contiguous marker run, including events
    without a marker. Episode IDs are required at the beginning of each run.

    A finalized dataset includes its last step. For a live snapshot whose tail
    may still be receiving events, pass ``include_final_step=False``; the next
    marker is then required to establish that the preceding step is complete.
    """
    if type(steps_per_sample) is not int or steps_per_sample < 1:
        raise ValueError("steps_per_sample must be a positive integer")
    spans = list(_step_spans(events, step_field=step_field, episode_field=episode_field))
    if not include_final_step:
        spans = spans[:-1]
    if any(episode is None for _, _, episode in spans):
        raise ValueError(f"Step sampling requires {episode_field!r}")
    episode_start = 0
    for index, (_, end, episode) in enumerate(spans):
        if index and episode != spans[index - 1][2]:
            episode_start = index
        first = index + 1 - steps_per_sample
        if first >= episode_start:
            yield spans[first][0], end



def environment_episode_ranges(events, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    """Select indexed episodes after checking their start and final result.

    An index change defines the episode boundary; it does not prove that the
    episode is complete. Require a reset coordinator first and a terminated or
    truncated environment result last. Only an unfinished final episode may
    be omitted. Events after an ending and resets within an index are errors.
    """
    start, episode, ended = 0, None, False
    end = 0
    seen = set()
    normalized = isinstance(events, EventLog)
    sources = (record[0] for record in events._records) if normalized else events
    for index, event in enumerate(sources):
        event = _record(event, encoded_payload=not normalized, normalized=normalized)
        end = index + 1
        marker = _headers(event)["episode_index"]
        payload = event["payload"]
        is_start = event["provider"] == "coordinator" and payload.get(protocol.episode_start, False)
        if marker != episode:
            if episode is not None:
                if not ended:
                    raise ValueError("A new episode starts before the previous episode ends")
                yield start, index
            if marker in seen:
                raise ValueError("Episode indices must form one contiguous run each")
            if not is_start:
                raise ValueError("Episode samples must begin with a reset coordinator")
            seen.add(marker)
            start, episode, ended = index, marker, False
        elif ended:
            raise ValueError("Events cannot follow the final result within an episode")
        elif is_start:
            raise ValueError("An episode index cannot contain multiple episode starts")
        if event["provider"] == "coordinator":
            _validate_coordinator_payload(payload, protocol=protocol)
        if event["provider"] == "environment":
            _validate_environment_payload(payload, protocol=protocol)
            ended = payload.get(protocol.terminated, False) or payload.get(protocol.truncated, False)
    if ended:
        yield start, end



def environment_episode_is_complete(events, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    """Eligibility for a validated whole episode; unfinished candidates are skipped.

    The test scenario selects its own protocol flags.
    """
    if not len(events):
        return False
    event = events[-1]
    if event["provider"] != "environment":
        return False
    payload = event["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return bool(payload.get(protocol.terminated, False) or payload.get(protocol.truncated, False))



