"""Whole-episode sampling from a live reasoning dataset; no generation here."""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING
import numpy as np
from tensordict import TensorDict

from mouse_core.data.reasoning_dataset import ReasoningDataset, pack_reasoning_episodes


if TYPE_CHECKING:
    from mouse_core.models.sequential_reasoner import ReasoningBatch


class ReasoningDataLoader:
    """Sequential cycling or uniform random sampling with replacement.

    Exactly one batch budget must be enabled. Token-budget batches can exceed
    the budget by their final whole episode. A small dataset can serve larger
    batches by repeating episodes. Each loader has its own RNG and cursor.
    Writes are observed on the next call; no refresh, workers, or prefetch is used.
    """

    def __init__(self, *, dataset: ReasoningDataset, sample_budget: int | None,
                 token_budget: int | None, sampling: Literal['sequential', 'random'],
                 seed: int | np.random.SeedSequence) -> None:
        if not isinstance(dataset, ReasoningDataset):
            raise TypeError('dataset must be a ReasoningDataset')
        if (sample_budget is None) == (token_budget is None):
            raise ValueError('Set exactly one of sample_budget or token_budget')
        for name, value in [('sample_budget', sample_budget), ('token_budget', token_budget)]:
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{name} must be a positive integer or None')
        if sampling not in ('sequential', 'random'):
            raise ValueError('sampling must be sequential or random')
        self.dataset = dataset
        self.sample_budget, self.token_budget = sample_budget, token_budget
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self._cursor = dataset.first_id
        self.sampled_steps: list[int] = []

    def next_batch(self) -> tuple[ReasoningBatch, TensorDict]:
        if not len(self.dataset):
            raise ValueError('Reasoning dataset is empty; add generated episodes before sampling')
        chosen, tokens = [], 0
        while ((self.sample_budget is None or len(chosen) < self.sample_budget)
               and (self.token_budget is None or tokens < self.token_budget)):
            if self.sampling == 'sequential':
                self._cursor = max(self._cursor, self.dataset.first_id)
                if self._cursor >= self.dataset.next_id:
                    self._cursor = self.dataset.first_id
                episode = self.dataset[self._cursor - self.dataset.first_id]
                self._cursor += 1
            else:
                episode = self.dataset[int(self.rng.integers(len(self.dataset)))]
            chosen.append(episode)
            tokens += episode.token_count
        batch = pack_reasoning_episodes(chosen)
        self.sampled_steps = [episode.generated_step for episode in chosen]
        return batch

    def close(self) -> None:
        """Reset this loader without deleting the caller-owned dataset."""
        self.sampled_steps.clear()
        self._cursor = self.dataset.first_id
