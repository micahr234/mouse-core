"""Read coordinator, agent, and environment events without changing the log."""
from collections.abc import Mapping
from copy import deepcopy
import json

from mouse_core.data.events import EventLog, EVENT_FIELDS, INDEX_FIELDS, _indices, _json


_INDICES = {"episode_index", "step_index"}
_COORDINATION = _INDICES | {"episode_start",
                            "component", "environment_step"}
_ENDINGS = {"terminated", "truncated", "inner_terminated", "inner_truncated"}
_RESULTS = _ENDINGS | {"reward", "action_requested"}
_DERIVED = {"events", "repeat_episode_start"}


def _record(event, *, encoded_payload, normalized=False):
    if not {"provider", "payload"} <= set(event):
        raise ValueError("Expected provider/payload event records")
    payload = event["payload"]
    if encoded_payload and isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("Environment protocol payloads must be JSON objects")
    # Arrow fills optional envelope columns with nulls. Restore their absence
    # so live and persisted views of sparse metadata remain identical.
    record = {key: value for key, value in event.items() if key not in INDEX_FIELDS or value is not None}
    if normalized and type(event["provider"]) is str:
        # EventLog owns already-normalized snapshots. Decoders only read them;
        # _observation copies each returned view before it leaves this module.
        return record
    return json.loads(_json({**record, "payload": payload}))


def _validate_environment_payload(payload):
    if set(payload) & (_COORDINATION | _DERIVED | {"action"}):
        raise ValueError("Environment cannot supply coordinator or agent fields")
    for name in _ENDINGS | {"action_requested"}:
        if name in payload and type(payload[name]) is not bool:
            raise ValueError(f"{name} must be boolean when present")
    if "reward" in payload and type(payload["reward"]) not in (int, float):
        raise ValueError("Reward must be finite numeric data or absent")


def _validate_coordinator_payload(payload):
    if set(payload) & (_INDICES | _RESULTS | _DERIVED | {"action", "observation"}):
        raise ValueError("Coordinator cannot supply environment results or actions")
    if "episode_start" in payload and type(payload["episode_start"]) is not bool:
        raise ValueError("episode_start must be boolean when present")


def _validate_agent_payload(payload):
    if payload.get("action") is None:
        raise ValueError("Agent must emit a real action")
    if set(payload) & (_COORDINATION | _RESULTS | _DERIVED):
        raise ValueError("Agent cannot supply coordinator or environment fields")


def _headers(event):
    indices = _indices({key: event[key] for key in INDEX_FIELDS if key in event})
    if "episode_index" not in indices:
        raise ValueError("Environment events require top-level episode_index")
    return indices


def _split_step(events):
    """Read one output packet with optional coordinator and agent events."""
    providers = [event["provider"] for event in events]
    if providers not in (["environment"], ["agent", "environment"],
                         ["coordinator", "environment"],
                         ["coordinator", "agent", "environment"]):
        raise ValueError("A step must contain optional coordinator, optional agent, then one environment output")
    coordinator = events[0] if providers[0] == "coordinator" else None
    agent = events[-2] if len(events) > 1 and providers[-2] == "agent" else None
    return coordinator, agent, events[-1]


def _frame(environment, coordinator):
    if environment["provider"] != "environment":
        raise ValueError("Each step requires an environment output")
    indices = _headers(environment)
    control = {}
    if coordinator is not None:
        if coordinator["provider"] != "coordinator":
            raise ValueError("Coordinator must precede agent and environment events")
        if _headers(coordinator) != indices:
            raise ValueError("Coordinator and environment sampling indices must match")
        control = coordinator["payload"]
        _validate_coordinator_payload(control)
    result = environment["payload"]
    _validate_environment_payload(result)
    if set(result) & set(control):
        raise ValueError("Coordinator fields collide with environment output fields")
    if result.get("action_requested", False) and any(result.get(name, False) for name in _ENDINGS):
        raise ValueError("An ended episode cannot request an action")
    return {**indices, **result, **control}


def _validate_agent(agent, coordinator, previous):
    _validate_agent_payload(agent["payload"])
    if previous is None or not previous.get("action_requested", False):
        raise ValueError("An agent must respond to the previous environment action request")
    if _headers(agent)["episode_index"] != previous["episode_index"]:
        raise ValueError("Agent must belong to the requesting environment episode")
    if coordinator is not None and _headers(agent) != _headers(coordinator):
        raise ValueError("Agent and coordinator sampling indices must match")


def _observation(previous, agent, environment, coordinator):
    row = _frame(environment, coordinator)
    new_episode = previous is None or row["episode_index"] != previous["episode_index"]
    if new_episode:
        if agent is not None or not row.get("episode_start", False):
            raise ValueError("An episode must begin with episode_start=True and no agent")
        if previous is not None and (row["episode_index"] <= previous["episode_index"]
                                     or not (previous.get("terminated", False) or previous.get("truncated", False))):
            raise ValueError("Cannot start a new episode before the previous episode ends")
    else:
        assert previous is not None
        if row.get("episode_start", False) or previous.get("terminated", False) or previous.get("truncated", False):
            raise ValueError("Episode reset and ending flags do not match event order")
        if previous.get("action_requested", False) != (agent is not None):
            raise ValueError("An action request must receive exactly one agent event")
    if agent is not None:
        _validate_agent(agent, coordinator, previous)
        if _headers(agent) != _headers(environment):
            raise ValueError("Agent and environment sampling indices must match")
        incoming = agent["payload"]
        if set(incoming) & set(row):
            raise ValueError("Agent fields collide with observation fields")
        row.update(incoming)
    source_fields = set(EVENT_FIELDS + INDEX_FIELDS)
    overlays = {key: value for key, value in environment.items() if key not in source_fields}
    if set(overlays) & (set(row) | _COORDINATION | _RESULTS | _DERIVED | {"action"}):
        raise ValueError("Event overlays cannot replace observation or derived fields")
    row.update(overlays)
    packet = [event for event in (coordinator, agent, environment) if event is not None]
    # _record has already normalized every source record. Serialize the packet
    # directly instead of walking all nested payloads through _plain again.
    row["events"] = json.dumps(
        [{key: event[key] for key in EVENT_FIELDS + INDEX_FIELDS if key in event} for event in packet],
        separators=(",", ":"), allow_nan=False,
    )
    return deepcopy(row)


def environment_decisions(events):
    """Yield sparse views of optional coordinator → optional agent → environment steps.

    The environment requests the next action. Each source event has top-level
    episode sampling indices. Payload fields are copied only when present;
    no absent rewards, flags, actions, or repeat indices are reconstructed.
    An unfinished tail stays recorded without inventing an observation.
    """
    previous = agent = coordinator = None
    normalized = isinstance(events, EventLog)
    for source in events._events if normalized else events:
        event = _record(source, encoded_payload=not normalized, normalized=normalized)
        provider = event["provider"]
        if provider == "coordinator":
            if coordinator is not None or agent is not None:
                raise ValueError("Coordinator must start a step and finish with an environment output")
            _headers(event)
            _validate_coordinator_payload(event["payload"])
            coordinator = event
        elif provider == "agent":
            if agent is not None:
                raise ValueError("An action request can receive only one agent event")
            _validate_agent(event, coordinator, previous)
            agent = event
        elif provider == "environment":
            _validate_environment_payload(event["payload"])
            row = _observation(previous, agent, event, coordinator)
            previous = _frame(event, coordinator)
            yield row
            agent = coordinator = None
        else:
            raise ValueError("Environment protocol requires environment, coordinator or agent providers")


def environment_episode_ranges(events):
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
    for index, event in enumerate(events._events if normalized else events):
        event = _record(event, encoded_payload=not normalized, normalized=normalized)
        end = index + 1
        marker = _headers(event)["episode_index"]
        payload = event["payload"]
        is_start = event["provider"] == "coordinator" and payload.get("episode_start", False)
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
            _validate_coordinator_payload(payload)
        if event["provider"] == "environment":
            _validate_environment_payload(payload)
            ended = payload.get("terminated", False) or payload.get("truncated", False)
    if ended:
        yield start, end


def latest_environment_decision(log):
    """Read a completed observation from at most six source events."""
    if not len(log):
        raise ValueError("The log has no completed environment observation")
    normalized = isinstance(log, EventLog)
    records = log._events if normalized else log
    tail = [_record(records[-offset], encoded_payload=False, normalized=normalized)
            for offset in range(min(len(log), 6), 0, -1)]
    if tail[-1]["provider"] != "environment":
        raise ValueError("The latest event is not an environment observation")
    boundaries = [index for index, event in enumerate(tail[:-1])
                  if event["provider"] == "environment"]
    previous = None
    start = 0
    if boundaries:
        previous_end = boundaries[-1]
        previous_start = boundaries[-2] + 1 if len(boundaries) > 1 else 0
        previous_coordinator, _, previous_environment = _split_step(tail[previous_start:previous_end + 1])
        previous = _frame(previous_environment, previous_coordinator)
        start = previous_end + 1
    coordinator, agent, environment = _split_step(tail[start:])
    return _observation(previous, agent, environment, coordinator)


class EventTransform:
    """Encode read-only observation views with an explicitly selected decoder."""
    def __init__(self, *, encode, decode=environment_decisions):
        self.encode = encode
        self.decode = decode

    def __call__(self, events):
        return [self.encode(fields) for fields in self.decode(events)]

    def reseed(self, *, generation):
        for stage in (self.decode, self.encode):
            reseed = getattr(stage, 'reseed', None)
            if callable(reseed):
                reseed(generation=generation)
