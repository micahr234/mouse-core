"""Reasoning adds decision metadata to the ordinary offline DQN input pipeline."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from mouse_core.data import Augmenter, Tokenizer, compose, pack_token_batch

EXAMPLES = Path(__file__).parents[1] / 'examples'


class CharacterBackend:
    def __len__(self):
        return 256

    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode())}


def pipeline(filename, compute_tokens):
    cells = json.loads((EXAMPLES / filename).read_text())['cells']
    namespace = dict(
        Tokenizer=lambda **kwargs: Tokenizer(**{**kwargs, 'tokenizers': {**kwargs['tokenizers'], 'text': CharacterBackend()}}),
        PRETRAINED='test-owned', MAX_STEPS_PER_EPISODE=5, REPEAT_COUNT=2,
        MAX_ACTIONS=4, MAX_OBS_DISCRETE=64, COMPUTE_TOKENS=compute_tokens,
        Augmenter=Augmenter, compose=compose,
    )
    exec(''.join(cells[7]['source']), namespace)
    tree = ast.parse(''.join(cells[9]['source']))
    tree.body = [n for n in tree.body if not isinstance(n, ast.Assign) or
                 not any(isinstance(t, ast.Name) and t.id == 'loader' for t in n.targets)]
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace


def repeated_events():
    rows = [
        (0, 'coordinator', {'_episode_start': True}),
        (0, 'environment', {'repeat_episode_index': 0, 'observation': 6, 'action_requested': True}),
        (1, 'agent', {'action': 2}),
        (1, 'environment', {'observation': 8, 'reward': 1., 'inner_terminated': True}),
        (2, 'environment', {'repeat_episode_index': 1, 'observation': 6, 'action_requested': True}),
        (3, 'agent', {'action': 1}),
        (3, 'environment', {'observation': 9, 'inner_truncated': True, '_truncated': True}),
    ]
    return [dict(episode_index=3, step_index=step, provider=provider, payload=payload)
            for step, provider, payload in rows]


def packed(steps):
    return pack_token_batch(steps=steps, sequence_ids=[0] * len(steps))


@pytest.mark.parametrize('compute_tokens', [0, 3])
def test_reasoning_matches_dqn_fields_augmentation_and_labels(compute_tokens):
    baseline = pipeline('02_train_offline_dqn.ipynb', compute_tokens)
    reasoning = pipeline('25_train_reasoning_dqn.ipynb', compute_tokens)
    events = repeated_events()
    assert baseline['sample_eligible'](events) and reasoning['sample_eligible'](events)
    inputs, labels = packed(baseline['train_transform'](events))
    actual, actual_labels = packed(reasoning['train_transform'](events))
    for key in ('head_mask', 'step_readout_mask', 'sequence_ids'):
        np.testing.assert_array_equal(actual[key], inputs[key])
    np.testing.assert_array_equal(actual['token_ids'], inputs['token_ids'])
    np.testing.assert_array_equal(actual['tokenizer_ids'], inputs['tokenizer_ids'])
    assert actual['tokenizer_names'] == inputs['tokenizer_names'] == ('text',)
    assert (actual['token_ids'][actual['head_mask']] == ord('\n')).all()
    assert 'autoregressive_mask' not in actual
    for key in ('action', 'reward', 'terminated', 'truncated', 'step_ids', 'sequence_ids'):
        np.testing.assert_array_equal(actual_labels[key], labels[key])
    assert actual['head_mask'].sum() == 2  # Passive endings have no decision.
    assert labels['truncated'].sum() == 1  # Only the outer task ends the return.
    # The real notebook transform must actually apply its shared event augmenter.
    augmented = reasoning['augmenter'].events(events)
    direct, direct_labels = packed(reasoning['tokenizer'].encode(augmented))
    np.testing.assert_array_equal(actual['token_ids'], direct['token_ids'])
    np.testing.assert_array_equal(actual_labels['action'], direct_labels['action'])


def test_inference_keeps_passive_boundary_context_and_matches_training_fields():
    namespace = pipeline('25_train_reasoning_dqn.ipynb', 2)
    cells = json.loads((EXAMPLES / '15_inference.ipynb').read_text())['cells']
    exec(''.join(cells[20]['source']), namespace)
    events = repeated_events()
    tokenizer = namespace['tokenizer']
    expected, _ = packed(tokenizer.encode(events))
    chunks = [events[:2], events[2:5], events[5:]]
    actual = [namespace['encode_reasoning_events'](events=chunk, token_spec=tokenizer) for chunk in chunks]
    np.testing.assert_array_equal(np.concatenate([batch['token_ids'] for batch in actual]), expected['token_ids'])
    np.testing.assert_array_equal(np.concatenate([batch['head_mask'] for batch in actual]), expected['head_mask'])
    text = bytes(expected['token_ids'].tolist()).decode()
    assert 'episode 1\nx,6' in text and 'episode 2\nx,6' in text
    assert 'c,8,r=1,terminated\n\nepisode 2' in text
    assert 'b,9,truncated\n\n' in text
