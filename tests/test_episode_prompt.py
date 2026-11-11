"""Outer-episode prompts are event text, including through sampling and caching."""

from functools import partial

import numpy as np
import pytest
import torch

from mouse_core.data import DataLoader
from tests._sampling_policies import environment_episode_ranges
from tests.test_collection_events import (
    FROZENLAKE_PROTOCOL,
    _collect_outputs,
    _notebook_transform,
)


@pytest.mark.parametrize("num_workers", [0, 4])
def test_complete_outer_episodes_each_get_one_visible_prompt(num_workers):
    episode = [
        (0, 0.0, False, False, False, False),
        (1, 0.0, False, False, False, False),
        (2, 0.0, True, False, False, False),
        (3, 0.0, False, False, False, False),
        (4, 1.0, True, False, True, False),
    ]
    log, _ = _collect_outputs("01_collect_dataset.ipynb", episode * 2)
    tokenizer = _notebook_transform("02_train_offline_dqn.ipynb", tokenizer_only=True)
    prompt_field = next(
        field
        for field in tokenizer.input_fields
        if field.input_field == "_episode_start"
    )
    prompt = prompt_field.format
    assert isinstance(prompt, str) and prompt_field.provider == "coordinator"
    ranges = partial(environment_episode_ranges, protocol=FROZENLAKE_PROTOCOL)
    samples = [log[start:end] for start, end in ranges(log)]
    assert len(samples) == 2
    with DataLoader(
        stores=log.to_store(),
        events_budget=len(log),
        sample_ranges=ranges,
        transform=tokenizer.encode,
        num_workers=num_workers,
        prefetch=4,
        seed=7,
    ) as loader:
        inputs, objective = loader.next_batch()
    assert inputs["num_sequences"] == 2
    for sequence in range(2):
        selected = inputs["sequence_ids"] == sequence
        text = "".join(chr(token) for token in inputs["token_ids"][selected].tolist())
        assert text.startswith(prompt) and text.count(prompt) == 1
        assert text.count("episode 1\n") == text.count("episode 2\n") == 1
        # Inner resets retain the same attention sequence as their prompt.
        assert inputs["head_mask"][selected].sum().item() == 3
        assert (objective["sequence_ids"] == sequence).sum().item() == 5

    for events in samples:
        full, _ = tokenizer.pack_rows(rows=[events])
        first, _ = tokenizer.pack_rows(rows=[events[:2]])
        rest, _ = tokenizer.pack_rows(rows=[events[2:]])
        torch.testing.assert_close(
            full["token_ids"], torch.cat((first["token_ids"], rest["token_ids"]))
        )
        assert "".join(map(chr, first["token_ids"].tolist())).startswith(prompt)
        assert prompt not in "".join(map(chr, rest["token_ids"].tolist()))
        # A caller omitting the start event gets exactly those events, no invented prefix.
        cropped, _ = tokenizer.pack_rows(rows=[events[1:]])
        assert prompt not in "".join(map(chr, cropped["token_ids"].tolist()))
        np.testing.assert_array_equal(
            tokenizer(events[0])["token_ids"], list(map(ord, prompt))
        )
