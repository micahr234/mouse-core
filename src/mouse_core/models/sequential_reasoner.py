"""Sequential reasoning generates complete token batches for ordinary forward calls."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import math
import torch
from tensordict import TensorDict
from torch import nn

from mouse_core.models.backbone.base import BaseBackbone
from mouse_core.models.backbone.continuous_decode import ContinuousDecodeSession
from mouse_core.models.embedding import MultiEmbedding
from mouse_core.models.generation import generate


@dataclass(frozen=True)
class _ReasoningTrace:
    """Internal rollout states before materializing full token sequences.

    Each decision uses fixed depth or generates until an environment action is selected.
    """

    head_token_ids: tuple[int, ...]
    states: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True)
class ReasoningBatch:
    """Complete generated token sequences ready for either network's forward.

    ``tokens`` contains ordinary token data plus detached ``latent_values`` and
    ``latent_mask``. Episode rows retain only the reasoning states actually generated.
    Positions identify transformer updates and their tokenizer-defined starts;
    readout_positions mark the final reasoning tokens in the episode rows.
    ``states`` includes an initial-embedding slot, followed by generated outputs,
    for incremental inference/diagnostics only. Forward recomputes the outputs.
    Residual mode emits a final feedback token after the last update; repeated-token
    mode has exactly one input marker per update and no unused final token.
    """

    tokens: TensorDict
    initial_positions: torch.Tensor
    update_positions: torch.Tensor
    readout_positions: torch.Tensor
    head_token_ids: tuple[int, ...]
    states: torch.Tensor
    lengths: torch.Tensor
    cached_prefix: bool = False


@contextmanager
def _evaluating(*roots: nn.Module) -> Iterator[None]:
    # Disable dropout in both rollout and replay without disabling replay gradients.
    modes = {module: module.training for root in roots for module in root.modules()}
    try:
        for root in roots:
            root.eval()
        yield
    finally:
        for module, training in modes.items():
            module.training = training


class SequentialReasoner(nn.Module):
    """Generate repeated-token or residual reasoning with fixed or adaptive depth.

    With residual connections, z[r+1] = z[r] + P(T(context, Norm(z[r].detach()))).
    Otherwise every reasoning position receives Norm(the dedicated trainable start embedding)
    and its representation is the projected transformer output at that position.
    P is one shared, learned H-to-H linear map, initialized to identity.

    Adaptive depth selects among environment actions and one final think action.
    Before min_steps only think is available; at max_steps only environment
    actions are available.
    Fixed depth evaluates only environment actions after exactly max_steps updates.
    Backbone RoPE supplies positions. The trainable input RMSNorm scale starts at
    0.02. Keep the backbone in eval mode through checkpointed backward so stochastic
    recomputation matches generation.
    """

    def __init__(self, *, hidden_dim: int, min_steps: int, max_steps: int, rms_norm_eps: float,
                 residual_connections: bool, adaptive_depth: bool) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(min_steps) is not int or not 1 <= min_steps <= max_steps:
            raise ValueError("min_steps must be an integer in [1, max_steps]")
        if type(residual_connections) is not bool or type(adaptive_depth) is not bool:
            raise ValueError("residual_connections and adaptive_depth must be booleans")
        if type(rms_norm_eps) not in (int, float) or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.residual_connections = residual_connections
        self.adaptive_depth = adaptive_depth
        self.start_embedding = nn.Embedding(1, hidden_dim, dtype=torch.float32)
        nn.init.normal_(self.start_embedding.weight, std=0.02)
        self.norm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps, elementwise_affine=True, dtype=torch.float32)
        nn.init.constant_(self.norm.weight, 0.02)
        self.update_norm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps, elementwise_affine=False, dtype=torch.float32)
        self.update_projection = nn.Linear(hidden_dim, hidden_dim, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.update_projection.weight)
        nn.init.zeros_(self.update_projection.bias)

    def project_updates(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mix hidden coordinates identically in generation and differentiable replay."""
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.update_projection(self.update_norm(hidden.float())).float()

    @staticmethod
    def _embeds(tokens: TensorDict, embedding: MultiEmbedding) -> torch.Tensor:
        tensors = tokens.to(next(embedding.parameters()).device)
        return embedding(
            tensors["token_ids"], tokenizer_ids=tensors["tokenizer_ids"],
            tokenizer_names=tokens["tokenizer_names"],
            numeric_values=tensors["numeric_values"], numeric_mask=tensors["numeric_mask"],
            numeric_field_ids=tensors["numeric_field_ids"], numeric_field_names=tokens["numeric_field_names"],
            position_values=tensors["position_values"], position_mask=tensors["position_mask"],
        )
    @staticmethod
    def _heads(tokens: TensorDict) -> tuple[int, ...]:
        mask = tokens.get("autoregressive_mask", None)
        if mask is None or not torch.equal(mask, tokens["head_mask"]):
            raise ValueError("Sequential reasoning requires autoregressive=True on every head-output token")
        return tuple(mask.nonzero(as_tuple=True)[0].tolist())

    @torch.no_grad()
    def _generate_trace(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        action_values: Callable[[torch.Tensor], torch.Tensor],
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        select_actions: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
        cache: ContinuousDecodeSession | None = None,
    ) -> _ReasoningTrace:
        """Consume episode events and retain every generated reasoning token.

        Adaptive depth treats the final action-value column as think, masked at
        the cap, and selects after every update. Fixed depth evaluates environment
        values and selects only at the final update. The optional selector receives
        available values, decision indices and depths. Residual feedback retains
        the stopping state in history; repeated-token reasoning has no extra final
        input. Only episode completion clears a caller-owned cache.
        """
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if backend not in ("uncached", "continuous"):
            raise ValueError("backend must be uncached or continuous")
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError("max_active_requests must be a positive integer")
        if cache is not None and backend != "continuous":
            raise ValueError("Persistent episode generation requires the continuous backend")
        heads = self._heads(tokens)
        if not heads and cache is None:
            device = next(embedding.parameters()).device
            lengths = torch.empty(0, device=device, dtype=torch.long)
            return _ReasoningTrace(heads, torch.empty((0, self.max_steps + 1, self.hidden_dim),
                                                   device=device), lengths)
        with _evaluating(embedding, backbone, self):
            embeds = self._embeds(tokens, embedding)
            initial = self.start_embedding.weight.float().expand(len(heads), -1)
            states = initial[:, None, :].expand(-1, self.max_steps + 1, -1).clone()
            counts = [0] * len(heads)
            groups = tokens["sequence_ids"].to(embeds.device)
            episode_heads: list[list[int]] = []
            episode_ends: list[int] = []
            rows, prompts = [], []
            for group in range(tokens["num_sequences"]):
                start = int(torch.searchsorted(groups, group))
                end = int(torch.searchsorted(groups, group, right=True))
                if start == end:
                    continue
                indices = [i for i, head in enumerate(heads) if int(groups[head]) == group]
                rows.append(group)
                episode_heads.append(indices)
                episode_ends.append(end)
                prompts.append(torch.cat((embeds[start:heads[indices[0]]], self.norm(initial[indices[0]:indices[0] + 1])))
                               if indices else embeds[start:end])
            cursors = [0] * len(rows)
            def advance(requests: list[int], hidden: torch.Tensor) -> list[torch.Tensor | None]:
                chunks: list[torch.Tensor | None] = [None] * len(requests)
                live = [(j, request, episode_heads[request][cursors[request]])
                        for j, request in enumerate(requests) if cursors[request] < len(episode_heads[request])]
                if not live:
                    return chunks
                updates = self.project_updates(hidden)
                for j, request, i in live:
                    counts[i] += 1
                    step = counts[i]
                    states[i, step] = (states[i, step - 1] + updates[j]
                                       if self.residual_connections else updates[j])
                decisions = [i for _, _, i in live
                             if self.adaptive_depth or counts[i] == self.max_steps]
                thinking = {i: True for _, _, i in live}
                if decisions:
                    indices = torch.tensor(decisions, device=states.device)
                    depths = torch.tensor([counts[i] for i in decisions], device=states.device)
                    values = action_values(states[indices, depths])
                    minimum = 2 if self.adaptive_depth else 1
                    if values.ndim != 2 or values.shape[0] != len(decisions) or values.shape[1] < minimum:
                        raise ValueError("action_values must return environment actions, followed by think only for adaptive depth")
                    values = values.clone()
                    if self.adaptive_depth:
                        values[depths < self.min_steps, :-1] = -torch.inf
                        values[depths == self.max_steps, -1] = -torch.inf
                    actions = (values.argmax(-1) if select_actions is None
                               else select_actions(values, indices, depths))
                    if (actions.shape != depths.shape or actions.dtype != torch.long
                            or bool(((actions < 0) | (actions >= values.shape[1])).any())
                            or not bool(torch.isfinite(values.gather(1, actions[:, None])).all())):
                        raise ValueError("select_actions must select one available action per reasoning state")
                    for i, action in zip(decisions, actions.tolist(), strict=True):
                        thinking[i] = self.adaptive_depth and action == values.shape[1] - 1
                for j, request, i in live:
                    step = counts[i]
                    state = states[i, step:step + 1]
                    if thinking[i]:
                        chunks[j] = self.norm(state if self.residual_connections else initial[i:i + 1])
                        continue
                    states[i, step + 1:] = state
                    cursors[request] += 1
                    next_index = cursors[request]
                    prefix = self.norm(state) if self.residual_connections else state[:0]
                    if next_index < len(episode_heads[request]):
                        following = episode_heads[request][next_index]
                        chunks[j] = torch.cat((prefix, embeds[heads[i] + 1:heads[following]],
                                               self.norm(initial[following:following + 1])))
                    else:
                        tail = torch.cat((prefix, embeds[heads[i] + 1:episode_ends[request]]))
                        if cache is not None and len(tail):
                            cache.defer(rows[request], tail)
                        # No readout consumes this suffix. A persistent episode
                        # processes it with the next observation; an owned cache
                        # is about to be discarded.
                        chunks[j] = None
                return chunks
            generate(backbone=backbone, prompts=prompts, advance=advance, backend=backend,
                     max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
                     cache_capacity_tokens=cache_capacity_tokens, cache=cache,
                     cache_rows=rows if cache is not None else None)
            lengths = torch.tensor(counts, device=embeds.device, dtype=torch.long)
            return _ReasoningTrace(heads, states, lengths)

    def generate(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        action_values: Callable[[torch.Tensor], torch.Tensor],
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        select_actions: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
        cache: ContinuousDecodeSession | None = None,
    ) -> ReasoningBatch:
        """Generate once and return complete context/latent inputs for forward.

        A cached call returns the new chunk for inference. Its earlier context
        lives in the KV cache, so it cannot be used as a standalone training batch.
        """
        cached_prefix = cache is not None and any(cache.lengths)
        trace = self._generate_trace(
            tokens, embedding=embedding, backbone=backbone, action_values=action_values,
            select_actions=select_actions, backend=backend,
            max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
            cache_capacity_tokens=cache_capacity_tokens, cache=cache,
        )
        return self._materialize(tokens, trace, cached_prefix=cached_prefix)

    def _materialize(self, tokens: TensorDict, trace: _ReasoningTrace, *,
                     cached_prefix: bool) -> ReasoningBatch:
        """Insert every detached latent into its complete episode sequence."""
        device = trace.states.device
        source = tokens.to(device)
        groups = source["sequence_ids"].tolist()
        by_head = {head: i for i, head in enumerate(trace.head_token_ids)}
        # Each entry is (source token index, decision index, latent step).
        rows: list[list[tuple[int, int, int]]] = []
        initial = [0] * len(by_head)
        readouts = [0] * len(by_head)
        positions = torch.zeros((len(by_head), self.max_steps), device=device, dtype=torch.long)
        offset = 0
        cursor = 0
        for group in range(tokens["num_sequences"]):
            row: list[tuple[int, int, int]] = []
            while cursor < len(groups) and groups[cursor] == group:
                if cursor in by_head:
                    i = by_head[cursor]
                    start = offset + len(row)
                    initial[i] = start
                    length = int(trace.lengths[i])
                    readouts[i] = start + length - (not self.residual_connections)
                    positions[i, :length] = torch.arange(start, start + length, device=device)
                    token_count = length + int(self.residual_connections)
                    row.extend((cursor, i, step) for step in range(token_count))
                else:
                    row.append((cursor, -1, -1))
                cursor += 1
            rows.append(row)
            offset += len(row)
        entries = [entry for row in rows for entry in row]
        indices = torch.tensor([entry[0] for entry in entries], device=device, dtype=torch.long)
        packed = source.clone()
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                packed[key] = value[indices].detach().clone()
        packed["sequence_ids"] = torch.repeat_interleave(
            torch.arange(len(rows), device=device), torch.tensor([len(row) for row in rows], device=device))
        packed.set_non_tensor("num_sequences", len(rows))
        latent_mask = torch.tensor([entry[1] >= 0 for entry in entries], device=device, dtype=torch.bool)
        latents = trace.states.new_zeros((len(entries), self.hidden_dim))
        if latent_mask.any():
            decision = torch.tensor([i for _, i, _ in entries if i >= 0], device=device)
            step = torch.tensor([step for _, i, step in entries if i >= 0], device=device)
            latents[latent_mask] = trace.states[decision, step if self.residual_connections else 0].detach()
        packed["latent_mask"] = latent_mask
        packed["latent_values"] = latents.detach()
        packed["head_mask"] = torch.zeros(len(entries), device=device, dtype=torch.bool)
        packed["autoregressive_mask"] = torch.zeros_like(packed["head_mask"])
        starts = torch.tensor(initial, device=device, dtype=torch.long)
        ends = torch.tensor(readouts, device=device, dtype=torch.long)
        packed["head_mask"][ends] = True
        # Generation is complete: starts are metadata, not pending markers.
        return ReasoningBatch(packed, starts, positions, ends, trace.head_token_ids,
                             trace.states, trace.lengths, cached_prefix)

    def forward(
        self, generated: ReasoningBatch, *, embedding: MultiEmbedding, backbone: BaseBackbone,
    ) -> torch.Tensor:
        """Evaluate a generated batch once, returning all depth representations.

        Context is embedded by this network. Residual feedback detaches before
        normalization and the additive path is rebuilt from this forward's updates.
        Repeated-token inputs use this network's live dedicated start embedding.
        No generation or sequence insertion happens here.
        """
        if generated.cached_prefix:
            raise ValueError("Forward requires complete episodes, not a chunk with a cached prefix")
        count = len(generated.head_token_ids)
        if (generated.update_positions.shape != (count, self.max_steps)
                or generated.initial_positions.shape != (count,)
                or generated.readout_positions.shape != (count,)):
            raise ValueError("Generated positions must match readouts and reasoning budget")
        if (generated.lengths.shape != (count,) or generated.lengths.dtype != torch.long
                or bool(((generated.lengths < (self.min_steps if self.adaptive_depth else 1))
                         | (generated.lengths > self.max_steps)).any())):
            raise ValueError("Generated lengths must be int64 counts within the reasoning budget")
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if not count:
            states = generated.states[:, 1:].float()
            return states
        with _evaluating(embedding, backbone, self):
            tokens = generated.tokens.to(next(embedding.parameters()).device)
            embeds = self._embeds(tokens, embedding)
            initial = self.start_embedding.weight.float().expand(count, -1)
            mask = tokens["latent_mask"]
            inputs = embeds.clone()
            latent_inputs = (tokens["latent_values"][mask].detach().float()
                             if self.residual_connections else self.start_embedding.weight.float().expand_as(embeds[mask]))
            inputs[mask] = self.norm(latent_inputs).to(embeds.dtype)
            hidden = cast(torch.Tensor, backbone(inputs, sequence_ids=tokens["sequence_ids"]))
            updates = self.project_updates(hidden[generated.update_positions.to(hidden.device)])
            valid = torch.arange(self.max_steps, device=hidden.device)[None] < generated.lengths.to(hidden.device)[:, None]
            updates = updates.masked_fill(~valid[..., None], 0)
            if self.residual_connections:
                return initial[:, None] + updates.cumsum(1)
            final = updates[torch.arange(count, device=hidden.device), generated.lengths.to(hidden.device) - 1]
            return torch.where(valid[..., None], updates, final[:, None])
