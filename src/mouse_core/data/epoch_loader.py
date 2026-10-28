"""Finite shuffled passes over whole samples, packed within a token budget."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np
import torch
from mouse_core.data.dataloader import _event_sample_ranges, _segment_token_count, _tokenize_event_window
from mouse_core.data.token_batch import pack_token_batch


class EpochDataLoader:
    def __init__(self, *, stores, transform, token_budget, sample_field=None, sample_ranges=None, epochs=1,
                 num_workers=0, seed=0, step_field=None, episode_field="episode_index"):
        if epochs < 1 or token_budget < 1 or num_workers < 0:
            raise ValueError("epochs and token_budget must be positive; num_workers must be nonnegative")
        if (sample_field is None) == (sample_ranges is None):
            raise ValueError("Pass exactly one of sample_field or sample_ranges")
        windows = []
        for store in stores:
            ds = store.to_dataset()
            windows.extend((ds, int(start), int(end))
                           for start, end in _event_sample_ranges(ds=ds, sample_field=sample_field,
                               sample_ranges=sample_ranges, step_field=step_field, episode_field=episode_field))
        if not windows:
            raise ValueError("EpochDataLoader requires at least one nonempty sample")

        def prepare(item):
            index, (ds, start, end) = item
            pipeline = deepcopy(transform) if num_workers else transform
            reseed = getattr(pipeline, "reseed", None)
            if callable(reseed):
                reseed(generation=index)
            return _tokenize_event_window(ds=ds, start=start, end=end, index_field=None, transform=pipeline)

        if num_workers:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                self.samples = list(pool.map(prepare, enumerate(windows)))
        else:
            self.samples = [prepare(item) for item in enumerate(windows)]
        costs = [_segment_token_count(sample) for sample in self.samples]
        if any(cost > token_budget for cost in costs):
            raise ValueError("A complete sample exceeds token_budget; increase the budget")
        self.batches = []
        self.epoch_ends = []
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            batch, used = [], 0
            for index in rng.permutation(len(self.samples)):
                if batch and used + costs[index] > token_budget:
                    self.batches.append(batch)
                    batch, used = [], 0
                batch.append(int(index))
                used += costs[index]
            if batch:
                self.batches.append(batch)
            self.epoch_ends.append(len(self.batches))
        self.position = 0
        self.sample_objective_fields = {}

    @property
    def remaining_in_epoch(self):
        return next((end - self.position for end in self.epoch_ends
                     if end > self.position), 0)

    @property
    def exhausted(self):
        return self.position >= len(self.batches)

    def __len__(self):
        return len(self.batches)

    def next_batch(self):
        if self.exhausted:
            raise StopIteration
        steps, sequence_ids = [], []
        indices = self.batches[self.position]
        for group, index in enumerate(indices):
            sample = self.samples[index]
            steps.extend(sample)
            sequence_ids.extend([group] * len(sample))
        batch = pack_token_batch(steps=steps, sequence_ids=sequence_ids,
                                 batch_size=len(indices), continuing=None)
        inputs, data, groups = batch
        for name, values in self.sample_objective_fields.items():
            data[name] = torch.cat([values[index] for index in indices])
        self.position += 1
        return inputs, data, groups

    def set_sample_objective_fields(self, name, values):
        if len(values) != len(self.samples) or any(
            value.ndim != 1 or len(value) != len(sample)
            for value, sample in zip(values, self.samples)
        ):
            raise ValueError("objective fields must align with every sample step")
        self.sample_objective_fields[name] = [value.detach().cpu().clone() for value in values]

    def close(self):
        self.sample_objective_fields.clear()
        self.samples.clear()
        self.batches.clear()
