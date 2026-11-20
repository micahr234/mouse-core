"""CPU-resident, FIFO storage of complete generated reasoning episodes.

Only sparse detached reasoning vectors, context tokens, positions, and objective
labels are retained. The dataset owns no model, environment, generator, or policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from mouse_core.models.sequential_reasoner import ReasoningBatch


@dataclass(frozen=True)
class ReasoningEpisode:
    tokens: TensorDict
    latent_values: torch.Tensor
    data: TensorDict
    initial_positions: torch.Tensor
    update_positions: torch.Tensor
    readout_positions: torch.Tensor
    head_token_ids: tuple[int, ...]
    source_token_count: int
    lengths: torch.Tensor
    generated_step: int

    def __post_init__(self) -> None:
        if type(self.generated_step) is not int or self.generated_step < 0:
            raise ValueError('generated_step must be a nonnegative integer')
        tensors = [*self.tokens.values(), *self.data.values(), self.latent_values,
                   self.initial_positions, self.update_positions, self.readout_positions, self.lengths]
        if any(t.device.type != 'cpu' or t.requires_grad for t in tensors if isinstance(t, torch.Tensor)):
            raise ValueError('Reasoning episodes must contain detached CPU tensors')
        if 'latent_values' in self.tokens.keys():
            raise ValueError('Reasoning episodes store sparse latent_values separately from tokens')
        if self.token_count < 1 or not len(self.lengths):
            raise ValueError('Reasoning episodes must contain tokens and decisions')
        if self.latent_values.ndim != 2 or len(self.latent_values) != int(self.tokens['latent_mask'].sum()):
            raise ValueError('Sparse latent_values must align with latent_mask')
        if (len(self.data['step_ids']) != len(self.lengths)
                or self.initial_positions.shape != self.lengths.shape
                or self.readout_positions.shape != self.lengths.shape
                or self.update_positions.ndim != 2 or len(self.update_positions) != len(self.lengths)):
            raise ValueError('Reasoning positions, lengths, and step_ids must align')

    @property
    def token_count(self) -> int:
        return len(self.tokens['token_ids'])


def split_reasoning_episodes(inputs: TensorDict, generated: ReasoningBatch, data: TensorDict,
                             generated_step: int) -> Iterator[ReasoningEpisode]:
    """Detach episodes onto CPU; store latent vectors only at reasoning positions."""
    if generated.cached_prefix:
        raise ValueError('Reasoning replay requires complete episode histories')
    generated = generated.to('cpu')
    inputs, data = inputs.to('cpu'), data.detach().to('cpu')
    data = generated.align_objective_data(inputs, data)
    if len(generated.lengths) != len(data['step_ids']):
        raise ValueError('Reasoning decisions must align with objective step_ids')
    tokens = generated.tokens
    for group in range(tokens['num_sequences']):
        token_rows = (tokens['sequence_ids'] == group).nonzero().flatten()
        source_rows = (inputs['sequence_ids'] == group).nonzero().flatten()
        step_rows = (data['sequence_ids'] == group).nonzero().flatten()
        decisions = (tokens['sequence_ids'][generated.initial_positions] == group).nonzero().flatten()
        if not len(token_rows) or not len(step_rows) or not len(decisions):
            raise ValueError('Reasoning replay requires nonempty complete episodes with decisions')
        token_start, source_start, step_start = int(token_rows[0]), int(source_rows[0]), int(step_rows[0])
        episode_tokens = tokens.clone(recurse=False)
        for key, value in tokens.items():
            if isinstance(value, torch.Tensor):
                episode_tokens[key] = value[token_rows].detach().clone()
        latent_values = episode_tokens.pop('latent_values')[episode_tokens['latent_mask']].clone()
        episode_tokens['sequence_ids'].zero_()
        episode_tokens.set_non_tensor('num_sequences', 1)
        episode_data = TensorDict({key: value[decisions if key == 'step_ids' else step_rows].clone()
                                  for key, value in data.items()}, batch_size=[], device='cpu')
        episode_data['step_ids'] -= step_start
        episode_data['sequence_ids'].zero_()
        positions = generated.update_positions[decisions].clone() - token_start
        valid = torch.arange(positions.shape[1], device=generated.lengths.device) < generated.lengths[decisions, None]
        positions[~valid] = 0
        yield ReasoningEpisode(episode_tokens, latent_values, episode_data,
            generated.initial_positions[decisions].clone() - token_start, positions,
            generated.readout_positions[decisions].clone() - token_start,
            tuple(generated.head_token_ids[i] - source_start for i in decisions.tolist()),
            len(source_rows), generated.lengths[decisions].clone(), generated_step)


def pack_reasoning_episodes(episodes: Sequence[ReasoningEpisode]) -> tuple[ReasoningBatch, TensorDict]:
    """Rebase token, decision, and objective-step indices independently."""
    from mouse_core.models.sequential_reasoner import ReasoningBatch

    if not episodes:
        raise ValueError('Cannot pack an empty reasoning batch')
    token_parts, data_parts = [], []
    initial, updates, readouts, heads, lengths = [], [], [], [], []
    token_offset = step_offset = source_offset = 0
    names = ('tokenizer_names', 'numeric_field_names')
    for group, episode in enumerate(episodes):
        if any(episode.tokens[name] != episodes[0].tokens[name] for name in names):
            raise ValueError('Reasoning episodes must use the same tokenizer field identities')
        tokens = episode.tokens.clone()
        tokens['sequence_ids'].fill_(group)
        latent = episode.latent_values.new_zeros((episode.token_count, episode.latent_values.shape[-1]))
        latent[tokens['latent_mask']] = episode.latent_values
        tokens['latent_values'] = latent
        token_parts.append(tokens)
        data = episode.data.clone()
        data['sequence_ids'].fill_(group)
        data['step_ids'] += step_offset
        data_parts.append(data)
        initial.append(episode.initial_positions + token_offset)
        positions = episode.update_positions + token_offset
        valid = torch.arange(positions.shape[1], device=episode.lengths.device) < episode.lengths[:, None]
        positions[~valid] = 0
        updates.append(positions)
        readouts.append(episode.readout_positions + token_offset)
        heads.extend(head + source_offset for head in episode.head_token_ids)
        lengths.append(episode.lengths)
        token_offset += episode.token_count
        step_offset += len(data['sequence_ids'])
        source_offset += episode.source_token_count
    tokens = token_parts[0].clone()
    for key, value in tokens.items():
        if isinstance(value, torch.Tensor):
            tokens[key] = torch.cat([part[key] for part in token_parts])
    tokens.set_non_tensor('num_sequences', len(episodes))
    data = TensorDict({key: torch.cat([part[key] for part in data_parts])
                      for key in data_parts[0].keys()}, batch_size=[], device='cpu')
    lengths = torch.cat(lengths)
    # Forward rebuilds output states with live weights; cached outputs are unnecessary.
    states = episodes[0].latent_values.new_empty((len(lengths), 0, episodes[0].latent_values.shape[-1]))
    trace = ReasoningBatch(tokens, torch.cat(initial), torch.cat(updates), torch.cat(readouts),
                           tuple(heads), states, lengths)
    return trace, data


class ReasoningDataset:
    """A token-capacity FIFO bank of complete episodes in CPU RAM.

    Append never waits for a full bank. Sampling does not remove entries; only
    new arrivals evict oldest episodes. Capacity counts context and reasoning
    positions, while only reasoning positions store embedding vectors.
    Entries have monotonically increasing IDs so independent loaders can retain
    cursors across eviction or clear(). Tensor payloads are read-only by contract.
    This is synchronous storage; concurrent writes and reads are unsupported.
    """

    def __init__(self, *, capacity_tokens: int) -> None:
        if type(capacity_tokens) is not int or capacity_tokens < 1:
            raise ValueError('capacity_tokens must be a positive integer')
        self.capacity_tokens = capacity_tokens
        self._episodes: deque[ReasoningEpisode] = deque()
        self.token_count = 0
        self.first_id = self.next_id = 0
        self._schema = None

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[ReasoningEpisode]:
        return iter(self._episodes)

    def __getitem__(self, index: int) -> ReasoningEpisode:
        return self._episodes[index]

    def _check(self, episode: ReasoningEpisode):
        if not isinstance(episode, ReasoningEpisode):
            raise TypeError('ReasoningDataset accepts ReasoningEpisode entries')
        if episode.token_count > self.capacity_tokens:
            raise ValueError(f'Reasoning episode has {episode.token_count} tokens, exceeding capacity_tokens={self.capacity_tokens}')
        schema = (episode.tokens['tokenizer_names'], episode.tokens['numeric_field_names'],
                  episode.latent_values.shape[1:], episode.latent_values.dtype,
                  episode.update_positions.shape[1:],
                  tuple(sorted((key, value.dtype, value.shape[1:]) for key, value in episode.tokens.items()
                               if isinstance(value, torch.Tensor))),
                  tuple(sorted((key, value.dtype, value.shape[1:]) for key, value in episode.data.items())))
        if self._schema is not None and schema != self._schema:
            raise ValueError('Reasoning episodes must share token/objective schemas and reasoning dimensions')
        return schema

    def append(self, episode: ReasoningEpisode) -> None:
        """Take ownership of an episode's immutable tensor payloads."""
        self._schema = self._check(episode)
        self._episodes.append(episode)
        self.next_id += 1
        self.token_count += episode.token_count
        while self.token_count > self.capacity_tokens:
            self.token_count -= self._episodes.popleft().token_count
            self.first_id += 1

    def add(self, inputs: TensorDict, generated: ReasoningBatch, objective_data: TensorDict,
            *, generated_step: int) -> tuple[ReasoningEpisode, ...]:
        """Detach and split a complete generated batch, align labels, then append.

        Inputs and objective_data are the original source batch. The returned
        records describe all arrivals, including any immediately evicted by FIFO.
        """
        episodes = tuple(split_reasoning_episodes(inputs, generated, objective_data, generated_step))
        schemas = [self._check(episode) for episode in episodes]
        if schemas and any(schema != schemas[0] for schema in schemas):
            raise ValueError('Reasoning episodes must share token/objective schemas and reasoning dimensions')
        for episode in episodes:
            self.append(episode)
        return episodes

    def clear(self) -> None:
        self._episodes.clear()
        self.token_count = 0
        self.first_id = self.next_id
        self._schema = None

    def memory_bytes(self) -> int:
        """Stored tensor payload bytes, excluding Python overhead and live batches."""
        total = 0
        for episode in self._episodes:
            tensors = [*episode.tokens.values(), *episode.data.values(), episode.latent_values,
                       episode.initial_positions, episode.update_positions,
                       episode.readout_positions, episode.lengths]
            total += sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))
        return total
