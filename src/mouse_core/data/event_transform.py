"""Read coordinator, agent, and environment events without changing the log."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json

from mouse_core.data.events import EventLog, EVENT_FIELDS, INDEX_FIELDS, _indices, _json


@dataclass(frozen=True)
class EnvironmentProtocol:
    """Explicit payload field names; underscores have no special meaning.

    Different environments may render their ending flags as model inputs or
    keep them as metadata. Readers use exactly the names selected here.
    """

    episode_start: str = "episode_start"
    terminated: str = "terminated"
    truncated: str = "truncated"

    def __post_init__(self):
        names = (self.episode_start, self.terminated, self.truncated)
        reserved = {"episode_index", "step_index", "component", "environment_step",
                    "inner_terminated", "inner_truncated", "reward", "action_requested",
                    "events", "repeat_episode_start", "action", "observation"}
        if any(not isinstance(name, str) or not name or name in reserved for name in names):
            raise ValueError("Protocol field names must be nonempty, non-reserved strings")
        if len(set(names)) != len(names):
            raise ValueError("Protocol field names must be distinct")


_DEFAULT_PROTOCOL = EnvironmentProtocol()


_INDICES = {"episode_index", "step_index"}
_COORDINATION = _INDICES | {"component", "environment_step"}
_INNER_ENDINGS = {"inner_terminated", "inner_truncated"}
_DERIVED = {"events", "repeat_episode_start"}


def _coordination(protocol):
    return _COORDINATION | {protocol.episode_start}


def _endings(protocol):
    return _INNER_ENDINGS | {protocol.terminated, protocol.truncated}


def _results(protocol):
    return _endings(protocol) | {"reward", "action_requested"}



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
        # Public decoders copy each returned view before it leaves this module.
        return record
    return json.loads(_json({**record, "payload": payload}))


def _validate_environment_payload(payload, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    if set(payload) & (_coordination(protocol) | _DERIVED | {"action"}):
        raise ValueError("Environment cannot supply coordinator or agent fields")
    for name in _endings(protocol) | {"action_requested"}:
        if name in payload and type(payload[name]) is not bool:
            raise ValueError(f"{name} must be boolean when present")
    if "reward" in payload and type(payload["reward"]) not in (int, float):
        raise ValueError("Reward must be finite numeric data or absent")


def _validate_coordinator_payload(payload, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    if set(payload) & (_INDICES | _results(protocol) | _DERIVED | {"action", "observation"}):
        raise ValueError("Coordinator cannot supply environment results or actions")
    if protocol.episode_start in payload and type(payload[protocol.episode_start]) is not bool:
        raise ValueError(f"{protocol.episode_start} must be boolean when present")


def _validate_agent_payload(payload, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    if payload.get("action") is None:
        raise ValueError("Agent must emit a real action")
    if set(payload) & (_coordination(protocol) | _results(protocol) | _DERIVED):
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


def _frame(environment, coordinator, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
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
        _validate_coordinator_payload(control, protocol=protocol)
    result = environment["payload"]
    _validate_environment_payload(result, protocol=protocol)
    if set(result) & set(control):
        raise ValueError("Coordinator fields collide with environment output fields")
    if result.get("action_requested", False) and any(result.get(name, False) for name in _endings(protocol)):
        raise ValueError("An ended episode cannot request an action")
    return {**indices, **result, **control}


def _validate_agent(agent, coordinator, previous, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    _validate_agent_payload(agent["payload"], protocol=protocol)
    if previous is None or not previous.get("action_requested", False):
        raise ValueError("An agent must respond to the previous environment action request")
    if _headers(agent)["episode_index"] != previous["episode_index"]:
        raise ValueError("Agent must belong to the requesting environment episode")
    if coordinator is not None and _headers(agent) != _headers(coordinator):
        raise ValueError("Agent and coordinator sampling indices must match")


def _observation_fields(previous, agent, environment, coordinator, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    row = _frame(environment, coordinator, protocol=protocol)
    new_episode = previous is None or row["episode_index"] != previous["episode_index"]
    if new_episode:
        if agent is not None or not row.get(protocol.episode_start, False):
            raise ValueError(f"An episode must begin with {protocol.episode_start}=True and no agent")
        if previous is not None and (row["episode_index"] <= previous["episode_index"]
                                     or not (previous.get(protocol.terminated, False) or previous.get(protocol.truncated, False))):
            raise ValueError("Cannot start a new episode before the previous episode ends")
    else:
        assert previous is not None
        if row.get(protocol.episode_start, False) or previous.get(protocol.terminated, False) or previous.get(protocol.truncated, False):
            raise ValueError("Episode reset and ending flags do not match event order")
        if previous.get("action_requested", False) != (agent is not None):
            raise ValueError("An action request must receive exactly one agent event")
    if agent is not None:
        _validate_agent(agent, coordinator, previous, protocol=protocol)
        if _headers(agent) != _headers(environment):
            raise ValueError("Agent and environment sampling indices must match")
        incoming = agent["payload"]
        if set(incoming) & set(row):
            raise ValueError("Agent fields collide with observation fields")
        row.update(incoming)
    source_fields = set(EVENT_FIELDS + INDEX_FIELDS)
    overlays = {key: value for key, value in environment.items() if key not in source_fields}
    if set(overlays) & (set(row) | _coordination(protocol) | _results(protocol) | _DERIVED | {"action"}):
        raise ValueError("Event overlays cannot replace observation or derived fields")
    row.update(overlays)
    return row


def _packet_json(packet, *, payload_json=None):
    if payload_json is not None:
        # Only owned EventLog payloads take this path. They were validated and
        # encoded at ingestion; encode the small envelope without revisiting
        # nested observations, maps, or other payload values.
        return "[" + ",".join(
            "{" + ",".join(
                json.dumps(key) + ":" + (
                    encoded if key == "payload" else
                    json.dumps(event[key], separators=(",", ":"), allow_nan=False)
                )
                for key in EVENT_FIELDS + INDEX_FIELDS if key in event
            ) + "}"
            for event, encoded in zip(packet, payload_json, strict=True)
        ) + "]"
    return json.dumps(
        [{key: event[key] for key in EVENT_FIELDS + INDEX_FIELDS if key in event} for event in packet],
        separators=(",", ":"), allow_nan=False,
    )


def _environment_steps(events, *, require_complete=False, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    previous = agent = coordinator = None
    normalized = isinstance(events, EventLog)
    sources = (record[0] for record in events._records) if normalized else events
    for source in sources:
        event = _record(source, encoded_payload=not normalized, normalized=normalized)
        provider = event["provider"]
        if provider == "coordinator":
            if coordinator is not None or agent is not None:
                raise ValueError("Coordinator must start a step and finish with an environment output")
            _headers(event)
            _validate_coordinator_payload(event["payload"], protocol=protocol)
            coordinator = event
        elif provider == "agent":
            if agent is not None:
                raise ValueError("An action request can receive only one agent event")
            _validate_agent(event, coordinator, previous, protocol=protocol)
            agent = event
        elif provider == "environment":
            _validate_environment_payload(event["payload"], protocol=protocol)
            row = _observation_fields(previous, agent, event, coordinator, protocol=protocol)
            previous = _frame(event, coordinator, protocol=protocol)
            yield row, [item for item in (coordinator, agent, event) if item is not None]
            agent = coordinator = None
        else:
            raise ValueError("Environment protocol requires environment, coordinator or agent providers")
    if require_complete and (
        previous is None or agent is not None or coordinator is not None
        or not (previous.get(protocol.terminated, False) or previous.get(protocol.truncated, False))
    ):
        raise ValueError("Replay ends with an incomplete episode")


def validate_environment_events(events, *, require_complete=False, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL) -> None:
    """Validate the environment protocol without constructing decoded snapshots.

    require_complete rejects empty logs and unfinished final episodes or steps.
    Generic EventLog/Datastore recording remains independent of this protocol.
    """
    for _ in _environment_steps(events, require_complete=require_complete, protocol=protocol):
        pass


def environment_decisions(events, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    """Yield sparse views of optional coordinator → optional agent → environment steps.

    The environment requests the next action. Each source event has top-level
    episode sampling indices. Payload fields are copied only when present;
    no absent rewards, flags, actions, or repeat indices are reconstructed.
    An unfinished tail stays recorded without inventing an observation.
    """
    for row, packet in _environment_steps(events, protocol=protocol):
        row["events"] = _packet_json(packet)
        yield deepcopy(row)


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


def _latest_environment_step(log, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    if not len(log):
        raise ValueError("The log has no completed environment observation")
    normalized = isinstance(log, EventLog)
    # Capture owned records and their encodings together. Later appends cannot
    # change which payload belongs to this completed packet.
    records = log._records[-6:] if normalized else None
    if records is not None:
        tail = [_record(event, encoded_payload=False, normalized=True) for event, _ in records]
    else:
        tail = [_record(log[-offset], encoded_payload=True)
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
        previous = _frame(previous_environment, previous_coordinator, protocol=protocol)
        start = previous_end + 1
    coordinator, agent, environment = _split_step(tail[start:])
    row = _observation_fields(previous, agent, environment, coordinator, protocol=protocol)
    packet = [event for event in (coordinator, agent, environment) if event is not None]
    encoded = [payload for _, payload in records[start:]] if records is not None else None
    return row, packet, encoded


def validate_latest_environment_step(log, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL) -> None:
    """Validate the last completed step without copying or serializing a view."""
    _latest_environment_step(log, protocol=protocol)


def latest_environment_packet(log, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL) -> str:
    """Return only the validated latest step's source event packet as JSON."""
    _, packet, encoded = _latest_environment_step(log, protocol=protocol)
    return _packet_json(packet, payload_json=encoded)


def latest_environment_decision(log, *, protocol: EnvironmentProtocol = _DEFAULT_PROTOCOL):
    """Read a completed observation from at most six source events."""
    row, packet, encoded = _latest_environment_step(log, protocol=protocol)
    row["events"] = _packet_json(packet, payload_json=encoded)
    return deepcopy(row)


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
