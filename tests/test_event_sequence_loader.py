from functools import partial as bind
from tests._event_fixtures import encode_measurement
"""Storage and sampling know source events, not a decoder's model steps."""
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from mouse_core.data import Datastore, DataLoader, EpochDataLoader, EventLog, EventTransform, StepTokens
from mouse_core.objectives import DqnObjective, boundary_discount, nstep_gate
from tests._token_batch_helpers import token_tokenizer


OFFLINE_EXAMPLES = [
    '02_train_offline_dqn.ipynb',
    '10_train_offline_reasoning_dqn.ipynb',
    '13_train_offline_n_step_dqn.ipynb',
    '14_train_offline_value_gap_dqn.ipynb',
    '18_train_offline_cartpole_dqn.ipynb',
    '21_train_dqn_cosine.ipynb',
]


def source(count=6):
    store = Datastore()
    for episode in range(2):
        for i in range(count):
            store.append(data=dict(provider='sensor', payload={'episode_index': episode,
                                   'event_index': i, 'task_id': 'same-task',
                                   'sensor': i, 'timestamp': str(100-i)}))
    return store


class EncodeSample:
    def __init__(self):
        self.seen = []

    def __call__(self, events):
        # This protocol deliberately has neither environment outputs nor
        # action-request fields. The loader must not try to replay it.
        self.seen.append(events)
        assert {e['provider'] for e in events} == {'sensor'}
        assert [json.loads(e['payload'])['event_index'] for e in events] == list(range(len(events)))
        assert len({json.loads(e['payload'])['episode_index'] for e in events}) == 1
        return [StepTokens(token_ids=np.array([1,2]), numeric_values=np.zeros(2),
                           numeric_mask=np.zeros(2, dtype=bool),
                           head_token_mask=np.array([False,True]),
                           objective_fields={'final_source_event': json.loads(events[-1]['payload'])['event_index']})]


def test_source_event_budget_is_independent_of_tokenizer_readout_count():
    store, encode = source(), EncodeSample()
    original = store[:]
    with DataLoader(stores=store, events_budget=6, batch_size=1, sample_field='payload.episode_index',
                    transform=encode, index_field='source_offset', num_workers=0, seed=1) as loader:
        inputs, data, group = loader.next_batch()
    assert inputs.N == 1 and group.tolist() == [0]
    assert data['final_source_event'].tolist() == [5]
    assert all(len(events) == 6 for events in encode.seen)
    for events in encode.seen:
        offset = json.loads(events[0]['payload'])['episode_index'] * 6
        assert [e['source_offset'] for e in events] == list(range(offset, offset+6))
        assert [json.loads(e['payload'])['timestamp'] for e in events] == ['100','99','98','97','96','95']
        assert [json.loads(e['payload'])['sensor'] for e in events] == list(range(6))
    assert store[:] == original
    with DataLoader(stores=store, events_budget=5, batch_size=1, sample_field='payload.episode_index',
                    transform=encode, num_workers=0) as loader:
        with pytest.raises(ValueError, match='6 source events'):
            loader.next_batch()


def test_epoch_loader_encodes_whole_event_samples_once():
    encode = EncodeSample()
    loader = EpochDataLoader(stores=[source()], transform=encode, token_budget=4,
                             sample_field='payload.episode_index', epochs=3)
    assert len(encode.seen) == 2
    assert len(loader) == 3
    _, data, groups = loader.next_batch()
    assert data['final_source_event'].tolist() == [5,5]
    assert groups.tolist() == [0,1]
    loader.close()


def test_per_event_weights_use_source_counts():
    with DataLoader(stores=[source(3), source(6)], events_budget=6, batch_size=1,
                    sample_field='payload.episode_index', weight_mode='per_event',
                    transform=EncodeSample(), num_workers=0) as loader:
        np.testing.assert_allclose(loader._probs, [1/3,2/3])


@pytest.mark.parametrize('field', ['provider', 'payload'])
def test_loader_index_overlay_cannot_overwrite_source_fields(field):
    with pytest.raises(ValueError, match='index_field cannot overwrite'):
        DataLoader(stores=source(), events_budget=6, batch_size=1,
                   sample_field='payload.episode_index', transform=EncodeSample(),
                   index_field=field, num_workers=0)


def test_old_step_loader_api_is_removed():
    with pytest.raises(TypeError, match='samples_budget'):
        DataLoader(stores=source(), samples_budget=6, batch_size=1,  # type: ignore[call-arg]
                   sample_field='payload.episode_index', transform=EncodeSample(), num_workers=0)
    with pytest.raises(ValueError, match='per_event'):
        DataLoader(stores=source(), events_budget=6, batch_size=1,
                   sample_field='payload.episode_index', weight_mode='per_step',
                   transform=EncodeSample(), num_workers=0)
    with pytest.raises(TypeError, match='replay'):
        Datastore(replay=lambda events: [])  # type: ignore[call-arg]


def test_explicit_windows_are_source_event_slices():
    seen = []
    def encode(events):
        seen.append([json.loads(e['payload'])['event_index'] for e in events])
        return [StepTokens(token_ids=np.array([1]), numeric_values=np.zeros(1),
                           numeric_mask=np.zeros(1, dtype=bool), head_token_mask=np.ones(1, dtype=bool),
                           objective_fields={'source_start': json.loads(events[0]['payload'])['event_index']})]
    loader = EpochDataLoader(stores=[source()], transform=encode, token_budget=2,
                             sample_ranges=lambda dataset: [(0,3),(2,6),(6,9),(8,12)])
    assert seen == [[0,1,2],[2,3,4,5]] * 2
    loader.close()
    # Explicit policies may cross any domain boundary; storage does not interpret it.
    loader = EpochDataLoader(stores=[source()], transform=encode, token_budget=2,
                             sample_ranges=lambda dataset: [(2,8)])
    assert seen[-1] == [2,3,4,5,0,1]
    loader.close()


def _offline_sample_ranges(filename):
    path = Path(__file__).resolve().parents[1] / 'examples' / filename
    notebook = json.loads(path.read_text())
    imports = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
               for node in ast.parse(''.join(cell['source'])).body
               if isinstance(node, ast.ImportFrom) and node.module == 'mouse_core.data']
    assert any(alias.name == 'environment_episode_ranges' for node in imports for alias in node.names)
    assert 'sample_ranges=environment_episode_ranges' in path.read_text()
    from mouse_core.data import environment_episode_ranges
    return environment_episode_ranges



def _repeated_tasks_with_unfinished_tail():
    log = EventLog()
    for episode in range(2):
        log.event(provider='coordinator', episode_index=episode, payload={
            'episode_start': True,
        })
        log.event(provider='environment', episode_index=episode, payload={
            'observation': 0,
            'reward': 0.0,
            'terminated': False,
            'truncated': False,
            "action_requested": True,
        })
        log.event(provider='coordinator', episode_index=episode, payload={
        })
        log.event(provider='agent', episode_index=episode, payload={
            'action': 0,
        })
        log.event(provider='environment', episode_index=episode, payload={
            'observation': 1,
            'reward': float(episode + 1),
            'inner_terminated': episode == 0,
            'inner_truncated': episode == 1,
            'terminated': False,
            'truncated': False,
        })
        log.event(provider='coordinator', episode_index=episode, payload={
        })
        log.event(provider='environment', episode_index=episode, payload={
            'observation': 2,
            'reward': 0.0,
            'terminated': False,
            'truncated': False,
            "action_requested": True,
        })
        log.event(provider='coordinator', episode_index=episode, payload={
        })
        log.event(provider='agent', episode_index=episode, payload={
            'action': 1,
        })
        log.event(provider='environment', episode_index=episode, payload={
            'observation': 3,
            'reward': float(episode + 2),
            'inner_terminated': episode == 0,
            'inner_truncated': episode == 1,
            'terminated': episode == 0,
            'truncated': episode == 1,
        })
    log.event(provider='coordinator', episode_index=2, payload={
        'episode_start': True,
    })
    log.event(provider='environment', episode_index=2, payload={
        'observation': 0,
        'reward': 0.0,
        'terminated': False,
        'truncated': False,
        "action_requested": True,
    })
    log.event(provider='coordinator', episode_index=2, payload={
    })
    log.event(provider='agent', episode_index=2, payload={
        'action': 0,
    })
    log.event(provider='environment', episode_index=2, payload={
        'observation': 1,
        'reward': 99.0,
        'terminated': False,
        'truncated': False,
        "action_requested": True,
    })
    log.event(provider='coordinator', episode_index=2, payload={
    })
    log.event(provider='agent', episode_index=2, payload={
        'action': 1,
    })
    return log.to_store()

@pytest.mark.parametrize('filename', OFFLINE_EXAMPLES)
def test_offline_examples_sample_complete_tasks_and_preserve_unfinished_events(filename):
    sample_ranges = _offline_sample_ranges(filename)
    store = _repeated_tasks_with_unfinished_tail()
    original = store[:]
    assert list(sample_ranges(store.to_dataset())) == [(0, 10), (10, 20)]
    tokenizer = token_tokenizer('observation', objective_fields=[
        {'input_field': 'action', 'missing_value': None},
        *[{'input_field': key} for key in ('reward', 'terminated', 'truncated', 'episode_index')],
    ])
    transform = EventTransform(encode=bind(encode_measurement, tokenizer))
    seen = []

    def encode(events):
        seen.append(events)
        return transform(events)

    objective = DqnObjective(
        discount=boundary_discount(gamma_step=1., gamma_terminated=0., gamma_truncated=0.),
        reward=None, value=None, temperature=0., cross_group_backups='fault',
        gate=nstep_gate(n=10),
    )
    with DataLoader(stores=store, events_budget=20, batch_size=1,
                    sample_ranges=sample_ranges, transform=encode,
                    num_workers=0, seed=5) as loader:
        inputs, data, groups = loader.next_batch()
    assert inputs.N == 8
    assert groups.tolist() == [0] * 4 + [1] * 4
    values = torch.full((inputs.N, 2), 1000., requires_grad=True)
    losses, metrics = objective(objective_data=data, group_id=groups,
        predictions=values, backup_value_predictions=values.detach(),
        backup_action_predictions=values.detach())
    assert torch.isfinite(losses['action_value'])
    losses['action_value'].backward()
    assert values.grad is not None and values.grad.abs().sum() > 0
    backup = metrics['backup']
    assert isinstance(backup, torch.Tensor)
    for first in (0, 4):
        episode = int(data['episode_index'][first])
        assert episode in (0, 1)
        # Both repeats contribute; their inner boundaries do not end the task.
        assert backup[first].item() == float(2 * episode + 3)
        assert data['terminated'][first + 3].item() == (episode == 0)
        assert data['truncated'][first + 3].item() == (episode == 1)
    assert seen
    for events in seen:
        first, last = json.loads(events[0]['payload']), json.loads(events[-1]['payload'])
        assert first['episode_start']
        assert last['terminated'] or last['truncated']
        assert len(events) == 10
    assert store[:] == original
    assert [event['provider'] for event in store[-6:]] == ['environment', 'coordinator', 'agent', 'environment', 'coordinator', 'agent']
    assert json.loads(store[-3:][0]['payload'])['reward'] == 99.

    partial = Datastore()
    partial.from_dataset(ds=store.to_dataset().select(range(20, 27)))
    with DataLoader(stores=partial, events_budget=20, batch_size=1,
                    sample_ranges=sample_ranges, transform=transform,
                    num_workers=0, seed=5) as loader:
        with pytest.raises(ValueError, match='no eligible samples'):
            loader.next_batch()
    assert len(partial) == 7
