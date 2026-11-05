"""Explicit protocol names preserve payloads and do not control tokenization."""
from functools import partial
import json

import pytest

from mouse_core.data import (
    DataLoader, EnvironmentProtocol, EventLog, Tokenizer, environment_decisions,
    environment_episode_ranges, latest_environment_decision, latest_environment_packet,
    validate_environment_events, validate_latest_environment_step,
)
from tests._token_batch_helpers import IntIdTokenizer


def recorded(protocol, ending):
    log = EventLog()
    for episode in range(2):
        log.event(provider='coordinator', episode_index=episode, step_index=0,
                  payload={protocol.episode_start: True})
        log.event(provider='environment', episode_index=episode, step_index=0,
                  payload={'observation': 1, '_label': 7, 'action_requested': True})
        log.event(provider='agent', episode_index=episode, step_index=1,
                  payload={'action': 0})
        log.event(provider='environment', episode_index=episode, step_index=1,
                  payload={'observation': 2, '_label': 8, ending: True})
    return log


@pytest.mark.parametrize('prefix', ['_', 'custom_'])
@pytest.mark.parametrize('ending', ['terminated', 'truncated'])
def test_readers_use_explicit_names_and_preserve_source_events(prefix, ending):
    protocol = EnvironmentProtocol(**{name: prefix + name for name in
                                     ('episode_start', 'terminated', 'truncated')})
    log = recorded(protocol, getattr(protocol, ending))
    original = log.events
    expected = list(environment_decisions(log, protocol=protocol))
    for source in (log, log.events, log.to_dataset(), log.to_store().to_dataset()):
        validate_environment_events(source, require_complete=True, protocol=protocol)
        validate_latest_environment_step(source, protocol=protocol)
        assert list(environment_episode_ranges(source, protocol=protocol)) == [(0, 4), (4, 8)]
        assert list(environment_decisions(source, protocol=protocol)) == expected
        assert latest_environment_decision(source, protocol=protocol) == expected[-1]
        assert json.loads(latest_environment_packet(source, protocol=protocol)) == original[-2:]
    assert log.events == original
    assert expected[-1][getattr(protocol, ending)] is True
    assert ending not in expected[-1]
    # No automatic underscore detection or fallback to another payload schema.
    with pytest.raises(ValueError, match='reset coordinator'):
        list(environment_episode_ranges(log))
    with pytest.raises(ValueError, match='episode_start=True'):
        list(environment_decisions(log))
    with pytest.raises(ValueError, match='incomplete episode'):
        validate_environment_events(log[:-1], require_complete=True, protocol=protocol)


def test_underscore_payloads_can_be_explicit_inputs_and_objective_aliases():
    tokenizer = Tokenizer(tokenizers={'default': IntIdTokenizer()}, input_fields=[
        dict(type='text', provider='environment', input_field='_label', format='{field}', head_output=True),
    ], objective_fields=[dict(input_field='_label', output_field='target')])
    event = dict(provider='environment', episode_index=0, step_index=0,
                 payload={'_label': 7, 'observation': 99})
    step = tokenizer.encode([event])[0]
    assert step.token_ids.tolist() == [7]
    assert step.objective_fields == {'target': 7}


@pytest.mark.parametrize('workers', [0, 4])
def test_custom_episode_sampling_survives_shared_memory_workers(workers):
    protocol = EnvironmentProtocol(episode_start='_episode_start', terminated='_terminated', truncated='_truncated')
    log = recorded(protocol, protocol.terminated)
    tokenizer = Tokenizer(tokenizers={'default': IntIdTokenizer()}, input_fields=[
        dict(type='text', provider='environment', input_field='observation', format='{field}', head_output=True),
    ], objective_fields=[dict(input_field='_label', output_field='target')])
    loader = DataLoader(stores=[log.to_store()], token_budget=4, prefetch=4, num_workers=workers,
                        sample_ranges=partial(environment_episode_ranges, protocol=protocol),
                        transform=tokenizer.encode)
    try:
        inputs, labels = loader.next_batch()
        assert inputs['token_ids'].tolist() == [1, 2, 1, 2]
        assert labels['target'].tolist() == [7, 8, 7, 8]
    finally:
        loader.close()


@pytest.mark.parametrize('changes', [
    {'episode_start': ''}, {'terminated': 'truncated'}, {'truncated': 'reward'},
    {'episode_start': 'step_index'}, {'terminated': None},
])
def test_protocol_rejects_ambiguous_field_names(changes):
    with pytest.raises(ValueError, match='Protocol field names'):
        EnvironmentProtocol(**changes)
