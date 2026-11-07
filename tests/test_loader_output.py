"""The example loader returns aligned tokens, sparse readouts, and step labels."""
from functools import partial

import numpy as np
import pytest
import torch

from mouse_core.data import DataLoader
from tests._sampling_policies import environment_episode_ranges, step_sample_ranges
from tests.test_collection_events import _collect_outputs, _notebook_transform, FROZENLAKE_PROTOCOL


environment_episode_ranges = partial(environment_episode_ranges, protocol=FROZENLAKE_PROTOCOL)

@pytest.mark.parametrize('window', [None, 2])
@pytest.mark.parametrize('augment', [False, True])
def test_example_next_batch_keeps_complete_samples_and_aligned_labels(window, augment):
    log, _ = _collect_outputs('01_collect_dataset.ipynb', [
        (0, 0., False, False, False, False),
        (1, 0., False, False, False, False),
        (2, 0., True, False, False, False),
        (3, 0., False, False, False, False),
        (4, 1., True, False, True, False),
    ])
    tokenizer = _notebook_transform('02_train_offline_dqn.ipynb', tokenizer_only=True)
    # Retain source indices as labels to check sampled boundaries, not tokens.
    tokenizer.objective_fields += (('step_index', 'step_index'),)
    transform = (_notebook_transform('02_train_offline_dqn.ipynb') if augment else tokenizer.encode)
    if augment:
        # The real augmented pipeline is also checked below with complete episodes.
        sample_ranges = environment_episode_ranges
        expected_size = 5
    else:
        sample_ranges = (environment_episode_ranges if window is None else
                         partial(step_sample_ranges, steps_per_sample=window))
        expected_size = window or 5
    with DataLoader(stores=log.to_store(), token_budget=2100,
                    sample_ranges=sample_ranges, transform=transform, num_workers=0, seed=7) as loader:
        inputs, objective_data = loader.next_batch()
        group_id = objective_data["step_sequence_ids"]
    assert inputs["num_sequences"] >= 2
    assert objective_data["step_sequence_ids"].shape[0] == inputs["num_sequences"] * expected_size
    assert inputs["token_ids"].shape[0] <= 2100
    assert group_id.dtype == torch.int64
    assert group_id.shape == (objective_data["step_sequence_ids"].shape[0],)
    np.testing.assert_array_equal(group_id.numpy(), objective_data["step_sequence_ids"])
    for key, values in objective_data.items():
        if key == "head_step_ids":
            assert len(values) == int(inputs["head_token_mask"].sum())
            continue
        assert values.shape[0] == objective_data["step_sequence_ids"].shape[0] and values.device.type == 'cpu'
    assert 'group_id' not in objective_data
    assert bool((inputs['token_ids'][inputs["head_token_mask"].nonzero(as_tuple=True)[0]] == ord('\n')).all())
    np.testing.assert_array_equal(inputs['sequence_ids'][inputs["head_token_mask"].nonzero(as_tuple=True)[0]],
                                  group_id[objective_data["head_step_ids"]].numpy())
    for group in range(inputs["num_sequences"]):
        start = group * expected_size
        indices = (list(range(5)) if augment else
                   objective_data['step_index'][start:start + expected_size].tolist())
        assert len(indices) == expected_size
        assert indices == list(range(indices[0], indices[0] + expected_size))
        expected_heads = [start + local for local, index in enumerate(indices) if index in (0, 1, 3)]
        actual_heads = objective_data["head_step_ids"][(objective_data["head_step_ids"] >= start) &
                                           (objective_data["head_step_ids"] < start + expected_size)]
        np.testing.assert_array_equal(actual_heads, expected_heads)
        assert (~torch.isnan(objective_data["action"]))[start:start + expected_size].tolist() == [
            index in (1, 2, 4) for index in indices]
        assert objective_data['reward'][start:start + expected_size].tolist() == [
            float(index == 4) for index in indices]
        assert objective_data['terminated'][start:start + expected_size].tolist() == [
            index == 4 for index in indices]
        assert not objective_data['truncated'][start:start + expected_size].any()
