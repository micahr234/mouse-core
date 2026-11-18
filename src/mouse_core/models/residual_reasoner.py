"""Residual generation returns complete token batches for ordinary forward calls."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Literal, cast

import torch
from tensordict import TensorDict
from torch import nn

from mouse_core.models.backbone.base import BaseBackbone
from mouse_core.models.backbone.continuous_decode import ContinuousDecodeSession
from mouse_core.models.embedding import MultiEmbedding
from mouse_core.models.generation import generate


@dataclass(frozen=True)
class _ResidualTrace:
    """Internal rollout states before materializing full token sequences.

    lengths includes optional training continuations; history_lengths records the
    hard stopping point that becomes context for subsequent environment events.
    """

    head_token_ids: tuple[int, ...]
    states: torch.Tensor
    lengths: torch.Tensor
    history_lengths: torch.Tensor


@dataclass(frozen=True)
class ResidualBatch:
    """Complete generated token sequences ready for either network's forward.

    ``tokens`` contains ordinary token data plus detached ``latent_values`` and
    ``latent_mask``. Episode rows retain every stopped reasoning token. Extra
    training continuations occupy independent rows, never future event context.
    Positions identify the residual updates and their tokenizer-defined starts;
    readout_positions mark the final reasoning tokens in the episode rows.
    ``states`` records generated values for incremental inference/diagnostics only;
    forward recomputes all residual outputs from transformer updates.
    """

    tokens: TensorDict
    initial_positions: torch.Tensor
    update_positions: torch.Tensor
    readout_positions: torch.Tensor
    head_token_ids: tuple[int, ...]
    states: torch.Tensor
    lengths: torch.Tensor
    history_lengths: torch.Tensor
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


class ResidualReasoner(nn.Module):
    """z[r+1] = z[r] + T(context, Norm(stopgrad(z[0:r+1])))[-1].

    The caller owns the embedding, backbone and final projection. Context
    embeddings remain differentiable during replay; only residual-state feedback
    is detached before normalization.
    Each reasoning state occupies the next causal position, using backbone RoPE;
    no explicit time embedding or vocabulary head is added. States accumulate
    in float32, starting from the tokenizer-marked generation token's embedding.
    The input RMSNorm has a trainable per-element scale initialized to 0.02.

    With gradient checkpointing, keep the backbone in eval mode through backward
    as well, so recomputed stochastic layers match the detached rollout.
    """

    def __init__(self, *, hidden_dim: int, max_steps: int) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.norm = nn.RMSNorm(hidden_dim, eps=1e-6, elementwise_affine=True, dtype=torch.float32)
        nn.init.constant_(self.norm.weight, 0.02)
        self.stop_head = nn.Linear(hidden_dim, 1, dtype=torch.float32)
        nn.init.zeros_(self.stop_head.weight)
        nn.init.constant_(self.stop_head.bias, -2.0)

    def stop_logits(self, states: torch.Tensor) -> torch.Tensor:
        """Nonnegative logits deterministically stop after the current update."""
        normalized = torch.nn.functional.rms_norm(states.float(), (self.hidden_dim,), eps=1e-6)
        return self.stop_head(normalized).squeeze(-1)

    @staticmethod
    def halting_weights(logits: torch.Tensor) -> torch.Tensor:
        """Differentiable first-stop probabilities, with remaining mass at the cap.

        This relaxation trains the stop head. Generation uses a hard zero-logit
        threshold, never samples these probabilities.
        """
        hazards = logits[:, :-1].sigmoid()
        survival = torch.cat((torch.ones_like(logits[:, :1]), (1 - hazards).cumprod(-1)), -1)
        return survival * torch.cat((hazards, torch.ones_like(logits[:, :1])), -1)

    @staticmethod
    def stopping_lengths(logits: torch.Tensor) -> torch.Tensor:
        stop = logits >= 0
        first = stop.long().argmax(-1) + 1
        return torch.where(stop.any(-1), first, logits.shape[-1])

    def halting_loss(self, task_losses: torch.Tensor, logits: torch.Tensor, *,
                     row_weights: torch.Tensor, step_cost: float) -> dict[str, torch.Tensor]:
        """Expected task error plus a cost per extra update, with valid-row weighting.

        Training evaluates every possible stopping point up to max_steps in one
        replay. Generation instead takes the first nonnegative stop logit.
        """
        if not math.isfinite(step_cost) or step_cost < 0:
            raise ValueError("step_cost must be finite and nonnegative")
        if task_losses.shape != logits.shape or row_weights.shape != logits.shape[:1]:
            raise ValueError("Expected task losses/logits [P, R] and row_weights [P]")
        probabilities = self.halting_weights(logits)
        extra = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
        denominator = row_weights.sum().clamp_min(1)
        task = ((probabilities * task_losses).sum(-1) * row_weights).sum() / denominator
        extra_steps = ((probabilities * extra).sum(-1) * row_weights).sum() / denominator
        steps = ((probabilities * (extra + 1)).sum(-1) * row_weights).sum() / denominator
        cost = step_cost * extra_steps
        return dict(loss=task + cost, task_loss=task, reasoning_cost=cost, expected_steps=steps)

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
            raise ValueError("Residual reasoning requires autoregressive=True on every head-output token")
        return tuple(mask.nonzero(as_tuple=True)[0].tolist())

    def _history_prefixes(self, tokens: TensorDict, embeds: torch.Tensor,
                          trace: _ResidualTrace) -> list[torch.Tensor]:
        """Reconstruct each decision's causal prefix, including earlier latents."""
        groups = tokens["sequence_ids"].to(embeds.device)
        prefixes = []
        parts: list[torch.Tensor] = []
        previous_group = -1
        cursor = 0
        for i, head in enumerate(trace.head_token_ids):
            group = int(groups[head])
            if group != previous_group:
                parts = []
                cursor = int(torch.searchsorted(groups, group))
                previous_group = group
            parts.append(embeds[cursor:head])
            prefixes.append(torch.cat(parts))
            length = int(trace.history_lengths[i])
            # Retain z0 through the final stopping state. Generated inputs are
            # detached before trainable normalization, including across actions.
            parts.append(self.norm(trace.states[i, :length + 1].detach().float()))
            cursor = head + 1
        return prefixes

    @torch.no_grad()
    def _generate_trace(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
        stop_early: bool = True, cache: ContinuousDecodeSession | None = None,
    ) -> _ResidualTrace:
        """Consume episode events and retain every generated reasoning token.

        Each marker starts residual updates. The deterministic stopping state
        supplies the value readout and is appended before subsequent event tokens.
        Only episode completion clears a caller-owned cache. Training's
        stop_early=False evaluates candidate continuations against the SAME
        hard-stopped episode history; extra candidates never enter that history.
        """
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if backend not in ("uncached", "continuous"):
            raise ValueError("backend must be uncached or continuous")
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError("max_active_requests must be a positive integer")
        if cache is not None and (backend != "continuous" or not stop_early):
            raise ValueError("Persistent episode generation requires continuous hard stopping")
        heads = self._heads(tokens)
        if not heads and cache is None:
            device = next(embedding.parameters()).device
            lengths = torch.empty(0, device=device, dtype=torch.long)
            return _ResidualTrace(heads, torch.empty((0, self.max_steps + 1, self.hidden_dim),
                                                   device=device), lengths, lengths.clone())
        with _evaluating(embedding, backbone, self):
            if not stop_early:
                history = self._generate_trace(tokens, embedding=embedding, backbone=backbone,
                    backend=backend, max_active_requests=max_active_requests,
                    max_batch_tokens=max_batch_tokens, cache_capacity_tokens=cache_capacity_tokens)
                if not heads:
                    return history
                return self._candidates(tokens, history, embedding=embedding, backbone=backbone,
                    backend=backend, max_active_requests=max_active_requests,
                    max_batch_tokens=max_batch_tokens, cache_capacity_tokens=cache_capacity_tokens)
            embeds = self._embeds(tokens, embedding)
            initial = embeds[list(heads)].float()
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
                current = []
                for j, request, i in live:
                    counts[i] += 1
                    step = counts[i]
                    states[i, step] = states[i, step - 1] + hidden[j].float()
                    current.append(states[i, step])
                stops = (self.stop_logits(torch.stack(current)) >= 0).tolist()
                for (j, request, i), stop in zip(live, stops, strict=True):
                    step = counts[i]
                    state = states[i, step:step + 1]
                    if not stop and step < self.max_steps:
                        chunks[j] = self.norm(state)
                        continue
                    states[i, step + 1:] = state
                    cursors[request] += 1
                    next_index = cursors[request]
                    if next_index < len(episode_heads[request]):
                        following = episode_heads[request][next_index]
                        chunks[j] = torch.cat((self.norm(state), embeds[heads[i] + 1:heads[following]],
                                               self.norm(initial[following:following + 1])))
                    else:
                        # Consume the final generated token even when this input
                        # ends at the marker; future environment events follow it.
                        chunks[j] = torch.cat((self.norm(state), embeds[heads[i] + 1:episode_ends[request]]))
                return chunks
            generate(backbone=backbone, prompts=prompts, advance=advance, backend=backend,
                     max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
                     cache_capacity_tokens=cache_capacity_tokens, cache=cache,
                     cache_rows=rows if cache is not None else None)
            lengths = torch.tensor(counts, device=embeds.device, dtype=torch.long)
            return _ResidualTrace(heads, states, lengths, lengths.clone())

    def _candidates(self, tokens: TensorDict, history: _ResidualTrace, *, embedding: MultiEmbedding,
                    backbone: BaseBackbone, backend: Literal["uncached", "continuous"],
                    max_active_requests: int, max_batch_tokens: int,
                    cache_capacity_tokens: int | None) -> _ResidualTrace:
        embeds = self._embeds(tokens, embedding)
        prefixes = self._history_prefixes(tokens, embeds, history)
        initial = embeds[list(history.head_token_ids)].float()
        states = initial[:, None, :].expand(-1, self.max_steps + 1, -1).clone()
        counts = [0] * len(prefixes)
        def advance(requests: list[int], hidden: torch.Tensor) -> list[torch.Tensor | None]:
            chunks = []
            for j, i in enumerate(requests):
                counts[i] += 1
                step = counts[i]
                states[i, step] = states[i, step - 1] + hidden[j].float()
                chunks.append(None if step == self.max_steps else self.norm(states[i, step:step + 1]))
            return chunks
        generate(backbone=backbone,
                 prompts=[torch.cat((prefix, self.norm(initial[i:i + 1]))) for i, prefix in enumerate(prefixes)],
                 advance=advance, backend=backend, max_active_requests=max_active_requests,
                 max_batch_tokens=max_batch_tokens, cache_capacity_tokens=cache_capacity_tokens)
        lengths = torch.full_like(history.lengths, self.max_steps)
        return _ResidualTrace(history.head_token_ids, states, lengths, history.history_lengths)

    @torch.no_grad()
    def generate(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
        stop_early: bool = True, cache: ContinuousDecodeSession | None = None,
    ) -> ResidualBatch:
        """Generate once and return complete context/latent inputs for forward.

        A cached call returns the new chunk for inference. Its earlier context
        lives in the KV cache, so it cannot be used as a standalone training batch.
        """
        cached_prefix = cache is not None and any(cache.lengths)
        trace = self._generate_trace(
            tokens, embedding=embedding, backbone=backbone, backend=backend,
            max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
            cache_capacity_tokens=cache_capacity_tokens, stop_early=stop_early, cache=cache,
        )
        return self._materialize(tokens, trace, cached_prefix=cached_prefix)

    def _materialize(self, tokens: TensorDict, trace: _ResidualTrace, *,
                     cached_prefix: bool) -> ResidualBatch:
        """Insert latents once, including isolated optional continuation branches."""
        device = trace.states.device
        source = tokens.to(device)
        groups = source["sequence_ids"].tolist()
        by_head = {head: i for i, head in enumerate(trace.head_token_ids)}
        # Each entry is (source token index, decision index, latent step).
        rows: list[list[tuple[int, int, int]]] = []
        prefixes: dict[int, list[tuple[int, int, int]]] = {}
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
                    length = int(trace.history_lengths[i])
                    if int(trace.lengths[i]) > length:
                        prefixes[i] = list(row)
                    readouts[i] = start + length
                    positions[i, :length] = torch.arange(start, start + length, device=device)
                    row.extend((cursor, i, step) for step in range(length + 1))
                else:
                    row.append((cursor, -1, -1))
                cursor += 1
            rows.append(row)
            offset += len(row)
        # Only post-stop exploration needs another causal row. Its prefix ends
        # at this decision and contains only actual earlier episode history.
        for i, head in enumerate(trace.head_token_ids):
            length, stopped = int(trace.lengths[i]), int(trace.history_lengths[i])
            if length > stopped:
                row = prefixes[i] + [(head, i, step) for step in range(length + 1)]
                start = offset + len(prefixes[i])
                positions[i, stopped:length] = torch.arange(start + stopped, start + length, device=device)
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
            latents[latent_mask] = trace.states[decision, step].detach()
        packed["latent_mask"] = latent_mask
        packed["latent_values"] = latents.detach()
        packed["head_mask"] = torch.zeros(len(entries), device=device, dtype=torch.bool)
        packed["autoregressive_mask"] = torch.zeros_like(packed["head_mask"])
        starts = torch.tensor(initial, device=device, dtype=torch.long)
        ends = torch.tensor(readouts, device=device, dtype=torch.long)
        packed["head_mask"][ends] = True
        # Generation is complete: starts are metadata, not pending markers.
        return ResidualBatch(packed, starts, positions, ends, trace.head_token_ids,
                             trace.states, trace.lengths, trace.history_lengths, cached_prefix)

    def forward(
        self, generated: ResidualBatch, *, embedding: MultiEmbedding, backbone: BaseBackbone,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a generated batch once, returning residual states/stop logits.

        Context is embedded by this network. Latents detach before normalization;
        the solid residual path is rebuilt from z0 and this forward's updates.
        No generation, sequence insertion, or stopping decisions happen here.
        """
        if generated.cached_prefix:
            raise ValueError("Forward requires complete episodes, not a chunk with a cached prefix")
        count = len(generated.head_token_ids)
        if (generated.update_positions.shape != (count, self.max_steps)
                or generated.initial_positions.shape != (count,)
                or generated.readout_positions.shape != (count,)):
            raise ValueError("Generated positions must match readouts and reasoning budget")
        if (generated.lengths.shape != (count,) or generated.lengths.dtype != torch.long
                or bool(((generated.lengths < 1) | (generated.lengths > self.max_steps)).any())):
            raise ValueError("Generated lengths must be int64 counts within the reasoning budget")
        if (generated.history_lengths.shape != (count,)
                or generated.history_lengths.dtype != torch.long
                or bool(((generated.history_lengths < 1) | (generated.history_lengths > generated.lengths)).any())):
            raise ValueError("History lengths must be int64 counts within evaluated lengths")
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if not count:
            states = generated.states[:, 1:].float()
            return states, self.stop_logits(states)
        with _evaluating(embedding, backbone, self):
            tokens = generated.tokens.to(next(embedding.parameters()).device)
            embeds = self._embeds(tokens, embedding)
            initial = embeds[generated.initial_positions.to(embeds.device)].float()
            mask = tokens["latent_mask"]
            inputs = embeds.clone()
            inputs[mask] = self.norm(tokens["latent_values"][mask].detach().float()).to(embeds.dtype)
            hidden = cast(torch.Tensor, backbone(inputs, sequence_ids=tokens["sequence_ids"]))
            updates = hidden[generated.update_positions.to(hidden.device)].float()
            valid = torch.arange(self.max_steps, device=hidden.device)[None] < generated.lengths.to(hidden.device)[:, None]
            updates = updates.masked_fill(~valid[..., None], 0)
            residuals = initial[:, None] + updates.cumsum(1)
            return residuals, self.stop_logits(residuals)
