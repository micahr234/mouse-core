"""Sequential reasoning generates complete token batches for ordinary forward calls."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import Literal, cast

import math
from mouse_core.models.reasoning_action import ReasoningActionValues

import torch
from tensordict import TensorDict
from torch import nn

from mouse_core.data.token_batch import head_output_segments, align_decision_objectives
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
    ``states`` includes an initial-latent slot, followed by generated outputs,
    for incremental inference/diagnostics only. Forward recomputes the outputs.
    Each decision retains its existing first input and inserts one continuation
    input per THINK transition. An environment action inserts no trailing token.
    """

    tokens: TensorDict
    initial_positions: torch.Tensor
    update_positions: torch.Tensor
    readout_positions: torch.Tensor
    head_token_ids: tuple[int, ...]
    states: torch.Tensor
    lengths: torch.Tensor
    cached_prefix: bool = False

    @property
    def readouts(self) -> TensorDict:
        """Actual transformer outputs by decision/depth."""
        depths = torch.arange(self.update_positions.shape[1], device=self.lengths.device)
        valid = depths[None] < self.lengths[:, None]
        decisions = torch.arange(len(self.lengths), device=self.lengths.device)[:, None].expand_as(valid)
        return TensorDict({
            "token_positions": self.update_positions[valid],
            "decision_ids": decisions[valid],
            "depths": depths.expand_as(valid)[valid] + 1,
            "is_final": (depths[None] == self.lengths[:, None] - 1)[valid],
        }, batch_size=[int(valid.sum())])

    def to(self, device: torch.device | str) -> ReasoningBatch:
        """Move the complete trace without changing precision or detaching gradients."""
        return replace(self, tokens=self.tokens.to(device), **{
            key: getattr(self, key).to(device) for key in
            ('initial_positions', 'update_positions', 'readout_positions', 'states', 'lengths')})

    def align_objective_data(self, source_tokens: TensorDict, objective_data: TensorDict) -> TensorDict:
        """Select each decision's original final head row without shifting step labels.

        Ordinary DQN may predict at many tokens per step. Reasoning predicts at
        only the last one, while actions/rewards still retain all environment steps.
        Call this once on the original source data, before splitting replay episodes.
        """
        positions = source_tokens["step_readout_mask"].nonzero().flatten()
        if tuple(positions.tolist()) != self.head_token_ids:
            raise ValueError("Reasoning starts must identify the source decision positions")
        return align_decision_objectives(source_tokens, objective_data)



@contextmanager
def _evaluating(*roots: nn.Module) -> Iterator[None]:
    # Disable dropout in both rollout and replay without disabling replay gradients.
    modes = {module: module.training for root in roots for module in root.modules()}
    try:
        if any(modes.values()):
            for root in roots:
                root.eval()
        yield
    finally:
        for module, training in modes.items():
            if module.training != training:
                module.training = training


class _ReasoningInputMLP(nn.Module):
    """Qwen3's bias-free SiLU-gated MLP, without a residual bypass.

    Fresh projections use standard nn.Linear initialization. The surrounding
    reasoner owns input RMS normalization and the explicit static output scale.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch.float32)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch.float32)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs))


class SequentialReasoner(nn.Module):
    """Generate repeated-token or residual reasoning with fixed or adaptive depth.

    latent_feedback_scope="decision" feeds back within each environment decision; "sequence"
    also carries the final latent into the next decision, resetting only for a new
    sequence. "none" repeats the tokenizer-selected embedding at every token.
    The first input at each wrapping boundary is its embedding directly.
    Continuations use only projected detached feedback, without an embedding offset.
    At each scope boundary the residual feedback latent is zero. Residual
    connections add that latent (before normalization) to the raw transformer output. Input RMSNorm precedes a shared SiLU-gated MLP with standard
    nn.Linear initialization and an explicit static output scale; there is no output-side norm or projection. Only
    wrapping plus residual addition accumulates updates.

    Adaptive depth first selects EXECUTE or THINK using a separate binary head.
    EXECUTE selects an environment action using the action head. Before min_depth
    only THINK is available; at max_depth only EXECUTE is available.
    Fixed depth evaluates only environment actions after exactly max_depth updates.
    Backbone RoPE supplies positions. The trainable input RMSNorm scale starts at
    0.02. Keep the backbone in eval mode through checkpointed backward so stochastic
    recomputation matches generation.
    """

    def __init__(self, *, hidden_dim: int, input_mlp_intermediate_dim: int, input_mlp_output_scale: float,
                 min_depth: int, max_depth: int, rms_norm_eps: float,
                 latent_residual_enabled: bool, latent_feedback_scope: Literal["none", "decision", "sequence"], adaptive_depth: bool) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if type(max_depth) is not int or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if type(min_depth) is not int or not 1 <= min_depth <= max_depth:
            raise ValueError("min_depth must be an integer in [1, max_depth]")
        if any(type(value) is not bool for value in (latent_residual_enabled, adaptive_depth)):
            raise ValueError("latent_residual_enabled and adaptive_depth must be booleans")
        if type(rms_norm_eps) not in (int, float) or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if latent_feedback_scope not in ("none", "decision", "sequence"):
            raise ValueError("latent_feedback_scope must be none, decision or sequence")
        if type(input_mlp_intermediate_dim) is not int or input_mlp_intermediate_dim < 1:
            raise ValueError("input_mlp_intermediate_dim must be a positive integer")
        if type(input_mlp_output_scale) not in (int, float) or not math.isfinite(input_mlp_output_scale) or input_mlp_output_scale <= 0:
            raise ValueError("input_mlp_output_scale must be finite and positive")
        self.input_mlp_intermediate_dim = input_mlp_intermediate_dim
        self.input_mlp_output_scale = float(input_mlp_output_scale)
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.latent_residual_enabled = latent_residual_enabled
        self.latent_feedback_scope = latent_feedback_scope
        self.adaptive_depth = adaptive_depth
        self.norm = nn.RMSNorm(hidden_dim, eps=rms_norm_eps, elementwise_affine=True, dtype=torch.float32)
        nn.init.constant_(self.norm.weight, 0.02)
        self.input_projection = _ReasoningInputMLP(hidden_dim, input_mlp_intermediate_dim)

    def project_inputs(self, latent: torch.Tensor) -> torch.Tensor:
        """Project a continuation latent; the first input bypasses this block."""
        with torch.autocast(device_type=latent.device.type, enabled=False):
            return self.input_projection(self.norm(latent.float())).float() * self.input_mlp_output_scale

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

    def _select_actions(self, states, indices, depths, action_values, select_actions):
        values = action_values(states)
        if self.adaptive_depth:
            if not isinstance(values, ReasoningActionValues) or values.action_values.ndim != 2 or len(values) != len(indices):
                raise ValueError('Adaptive action_values must return ReasoningActionValues with separate action and EXECUTE/THINK heads')
            values = values.masked(depths, min_depth=self.min_depth, max_depth=self.max_depth)
            actions = values.greedy() if select_actions is None else select_actions(values, indices, depths)
            available = values.available()
        else:
            if values.ndim != 2 or values.shape[0] != len(indices) or values.shape[1] < 1:
                raise ValueError('Fixed-depth action_values must return environment-action values')
            actions = values.argmax(-1) if select_actions is None else select_actions(values, indices, depths)
            available = torch.isfinite(values)
        if (actions.shape != depths.shape or actions.dtype != torch.long
                or bool(((actions < 0) | (actions >= available.shape[1])).any())
                or not bool(available.gather(1, actions[:, None]).all())):
            raise ValueError('select_actions must select one available action per reasoning state')
        return actions, available.shape[1]

    @torch.no_grad()
    def _generate_trace(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        action_values: Callable[[torch.Tensor], torch.Tensor | ReasoningActionValues],
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        select_actions: Callable[[torch.Tensor | ReasoningActionValues, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
        cache: ContinuousDecodeSession | None = None,
    ) -> _ReasoningTrace:
        """Consume episode events and retain every generated reasoning token.

        Adaptive depth uses separate EXECUTE/THINK values, masking THINK at
        the cap, and selects after every update. Fixed depth evaluates environment
        values and selects only at the final update. The optional selector receives
        available values, decision indices and depths. Only THINK inserts a
        continuation input. Only episode completion clears a caller-owned cache.
        """
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if backend not in ("uncached", "continuous"):
            raise ValueError("backend must be uncached or continuous")
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError("max_active_requests must be a positive integer")
        if cache is not None and backend != "continuous":
            raise ValueError("Persistent episode generation requires the continuous backend")
        segments = head_output_segments(tokens)
        heads = tuple(head for episode in segments for head in episode.head_token_ids)
        if not heads and cache is None:
            device = next(embedding.parameters()).device
            lengths = torch.empty(0, device=device, dtype=torch.long)
            return _ReasoningTrace(heads, torch.empty((0, self.max_depth + 1, self.hidden_dim),
                                                   device=device), lengths)
        with _evaluating(embedding, backbone, self):
            embeds = self._embeds(tokens, embedding)
            initial = embeds.new_zeros((len(heads), self.hidden_dim), dtype=torch.float32)
            query_embeddings = embeds[list(heads)]
            states = initial[:, None, :].expand(-1, self.max_depth + 1, -1).clone()
            if self.max_depth == 1 and cache is None and self.latent_feedback_scope != "sequence":
                # All inputs already exist: evaluate every decision in one packed
                # causal forward. There is no generated feedback or decode session.
                positions = torch.tensor(heads, device=embeds.device, dtype=torch.long)
                inputs = embeds
                hidden = cast(torch.Tensor, backbone(inputs, sequence_ids=tokens['sequence_ids'].to(inputs.device)))
                updates = hidden[positions].float()
                states[:, 1] = initial + updates if self.latent_residual_enabled else updates
                lengths = torch.ones(len(heads), device=embeds.device, dtype=torch.long)
                self._select_actions(states[:, 1], torch.arange(len(heads), device=embeds.device),
                                     lengths, action_values, select_actions)
                return _ReasoningTrace(heads, states, lengths)
            start_inputs = query_embeddings
            counts = [0] * len(heads)
            episode_heads: list[list[int]] = []
            contexts: list[tuple[slice, ...]] = []
            rows, prompts = [], []
            head_index = 0
            for episode in segments:
                if not episode.head_token_ids and episode.context_slices[0].start == episode.context_slices[0].stop:
                    continue
                indices = list(range(head_index, head_index + len(episode.head_token_ids)))
                head_index += len(indices)
                rows.append(episode.sequence_id)
                episode_heads.append(indices)
                contexts.append(episode.context_slices)
                prefix = embeds[episode.context_slices[0]]
                first_input = start_inputs[indices[:1]]
                if indices and self.latent_feedback_scope == "sequence" and cache is not None:
                    carried = cache.generation_state[episode.sequence_id].get("reasoning_latent")
                    if carried is not None:
                        states[indices[0], 0] = carried
                        first_input = self.project_inputs(carried[None]).to(embeds.dtype)
                prompts.append(torch.cat((prefix, first_input)) if indices else prefix)
            cursors = [0] * len(rows)
            def advance(requests: list[int], hidden: torch.Tensor) -> list[torch.Tensor | None]:
                chunks: list[torch.Tensor | None] = [None] * len(requests)
                deferred: dict[int, torch.Tensor] = {}
                live = [(j, request, episode_heads[request][cursors[request]])
                        for j, request in enumerate(requests) if cursors[request] < len(episode_heads[request])]
                if not live:
                    return chunks
                live_indices = torch.tensor([i for _, _, i in live], device=states.device)
                for _, _, i in live:
                    counts[i] += 1
                live_depths = torch.tensor([counts[i] for _, _, i in live], device=states.device)
                updates = hidden[[j for j, _, _ in live]].float()
                base = (states[live_indices, live_depths - 1] if self.latent_feedback_scope != "none" else initial[live_indices])
                current = base + updates if self.latent_residual_enabled else updates
                states[live_indices, live_depths] = current
                decisions = [i for _, _, i in live
                             if self.adaptive_depth or counts[i] == self.max_depth]
                thinking = {i: True for _, _, i in live}
                if decisions:
                    indices = torch.tensor(decisions, device=states.device)
                    depths = torch.tensor([counts[i] for i in decisions], device=states.device)
                    actions, action_count = self._select_actions(
                        states[indices, depths], indices, depths, action_values, select_actions)
                    for i, action in zip(decisions, actions.tolist(), strict=True):
                        thinking[i] = self.adaptive_depth and action == action_count - 1
                # Only THINK inserts a continuation. Sequence wrapping also
                # projects a carry into the next decision's existing first slot.
                projection_rows = [position for position, (_, request, i) in enumerate(live)
                                   if self.latent_feedback_scope != "none" and (thinking[i] or
                                       (self.latent_feedback_scope == "sequence" and
                                        cursors[request] + 1 < len(episode_heads[request])))]
                projected = (self.project_inputs(current[projection_rows]).to(embeds.dtype)
                             if projection_rows else embeds[:0])
                projected_inputs = {position: projected[row:row + 1]
                                    for row, position in enumerate(projection_rows)}
                for position, (j, request, i) in enumerate(live):
                    projected_input = projected_inputs.get(position, embeds[:0])
                    step = counts[i]
                    state = states[i, step:step + 1]
                    if thinking[i]:
                        chunks[j] = projected_input if self.latent_feedback_scope != "none" else start_inputs[i:i + 1]
                        continue
                    states[i, step + 1:] = state
                    cursors[request] += 1
                    next_index = cursors[request]
                    if self.latent_feedback_scope == "sequence" and cache is not None:
                        cache.generation_state[rows[request]]["reasoning_latent"] = state[0].detach().clone()
                    if next_index < len(episode_heads[request]):
                        next_decision = episode_heads[request][next_index]
                        next_input = start_inputs[next_decision:next_decision + 1]
                        if self.latent_feedback_scope == "sequence":
                            states[episode_heads[request][next_index], 0] = state[0]
                            next_input = projected_input
                        chunks[j] = torch.cat((embeds[contexts[request][next_index]],
                                               next_input))
                    else:
                        tail = embeds[contexts[request][-1]]
                        if cache is not None and len(tail):
                            deferred[rows[request]] = tail
                        # No readout consumes this suffix. A persistent episode
                        # processes it with the next observation; an owned cache
                        # is about to be discarded.
                        chunks[j] = None
                if cache is not None and deferred:
                    cache.defer_chunks(deferred)
                return chunks
            generate(backbone=backbone, prompts=prompts, advance=advance, backend=backend,
                     max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
                     cache_capacity_tokens=cache_capacity_tokens, cache=cache,
                     cache_rows=rows if cache is not None else None)
            lengths = torch.tensor(counts, device=embeds.device, dtype=torch.long)
            return _ReasoningTrace(heads, states, lengths)

    def generate(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        action_values: Callable[[torch.Tensor], torch.Tensor | ReasoningActionValues],
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        select_actions: Callable[[torch.Tensor | ReasoningActionValues, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
        cache: ContinuousDecodeSession | None = None,
    ) -> ReasoningBatch:
        """Generate once and return complete context/latent inputs for forward.

        With max_depth=1 and no sequence wrapping, episodes need one packed forward, with no
        sequential generation or inserted feedback. Replay collection can skip
        this forward entirely using start_only_reasoning_episodes.
        A cached call returns the new chunk for inference. Its earlier context
        lives in the KV cache, so it cannot be used as a standalone training batch.
        """
        cached_prefix = cache is not None and any(cache.lengths)
        # Owned sessions never escape generation, so their temporary tensors need
        # no autograd view/version tracking. A caller-owned session may also be
        # used outside inference mode; keep its tensor lifetime contract intact.
        with torch.inference_mode() if cache is None else nullcontext():
            trace = self._generate_trace(
                tokens, embedding=embedding, backbone=backbone, action_values=action_values,
                select_actions=select_actions, backend=backend,
                max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
                cache_capacity_tokens=cache_capacity_tokens, cache=cache,
            )
        if cache is None and not torch.is_inference_mode_enabled():
            # Replay must contain ordinary detached tensors: backward may save
            # these latents/indices while training the input block and embeddings.
            trace = replace(trace, states=trace.states.clone(), lengths=trace.lengths.clone())
        return self._materialize(tokens, trace, cached_prefix=cached_prefix)

    @torch.no_grad()
    def prepare_replay(
        self, tokens: TensorDict, *, lengths: torch.Tensor,
        embedding: MultiEmbedding, backbone: BaseBackbone,
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        max_batch_tokens: int = 1024, cache_capacity_tokens: int | None = None,
    ) -> ReasoningBatch:
        """Prepare recorded episodes with policy-independent, already selected depths.

        With step wrapping, combine all known inputs through the next decision
        needing a continuation. Only outputs needed to construct latent inputs
        are evaluated; the final input joins the next context chunk. Entire
        one-token episodes and repeated-embedding reasoning need no forward.
        Sequence wrapping still needs each output to construct the next input.
        Replay recomputes all readouts during training. Returned ``states`` are
        zero placeholders, as with replay loaded from ReasoningDataset; use
        ``generate`` when inference outputs or policy-selected depths are needed.
        """
        segments = head_output_segments(tokens)
        heads = tuple(head for segment in segments for head in segment.head_token_ids)
        if (lengths.dtype != torch.long or lengths.shape != (len(heads),)
                or bool(((lengths < self.min_depth) | (lengths > self.max_depth)).any())):
            raise ValueError("Replay lengths must be one int64 depth per decision within min/max steps")
        if not self.adaptive_depth and bool((lengths != self.max_depth).any()):
            raise ValueError("Fixed-depth replay requires max_depth at every decision")
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if backend not in ("uncached", "continuous"):
            raise ValueError("backend must be uncached or continuous")
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError("max_active_requests must be a positive integer")
        if 'latent_values' in tokens.keys() or 'latent_mask' in tokens.keys():
            raise ValueError("Replay preparation requires clean source tokens")
        desired = lengths.tolist()
        if self.latent_feedback_scope == "sequence":
            # Every decision depends on the preceding output in this mode.
            def select(values, indices, depths):
                target = lengths.to(depths.device)[indices]
                return torch.where(depths < target, 1, 0)  # Synthetic one-action head: index 1 is THINK.
            trace = self.generate(
                tokens, embedding=embedding, backbone=backbone,
                action_values=lambda states: (ReasoningActionValues(states.new_zeros((len(states), 1)), states.new_zeros((len(states), 2)))
                                             if self.adaptive_depth else states.new_zeros((len(states), 1))),
                select_actions=select, backend=backend, max_active_requests=max_active_requests,
                max_batch_tokens=max_batch_tokens, cache_capacity_tokens=cache_capacity_tokens)
            trace.states.zero_()
            return trace
        needs_generation = self.latent_feedback_scope == "decision" and any(depth > 1 for depth in desired)
        device = next(embedding.parameters()).device if needs_generation else tokens['token_ids'].device
        states = torch.zeros((len(heads), self.max_depth + 1, self.hidden_dim), device=device)
        if needs_generation:
            with _evaluating(embedding, backbone, self), torch.inference_mode():
                embeds = self._embeds(tokens, embedding)
                prompts, requests = [], []
                offset = 0
                for segment in segments:
                    indices = [offset + i for i in range(len(segment.head_token_ids))
                               if desired[offset + i] > 1]
                    offset += len(segment.head_token_ids)
                    if not indices:
                        continue
                    prompts.append(embeds[segment.context_slices[0].start:heads[indices[0]] + 1])
                    requests.append(indices)
                cursors = [0] * len(requests)
                depths = [0] * len(heads)

                def advance(active: list[int], hidden: torch.Tensor) -> list[torch.Tensor | None]:
                    decisions = [requests[request][cursors[request]] for request in active]
                    previous = [depths[i] for i in decisions]
                    updates = hidden.float()
                    current = states[decisions, previous] + updates if self.latent_residual_enabled else updates
                    projected = self.project_inputs(current).to(embeds.dtype)
                    chunks: list[torch.Tensor | None] = []
                    for row, (request, decision) in enumerate(zip(active, decisions, strict=True)):
                        depths[decision] += 1
                        depth = depths[decision]
                        states[decision, depth] = current[row]
                        continuation = projected[row:row + 1]
                        if depth < desired[decision] - 1:
                            chunks.append(continuation)
                        else:
                            cursors[request] += 1
                            if cursors[request] < len(requests[request]):
                                next_decision = requests[request][cursors[request]]
                                # Final reasoning input, recorded environment steps,
                                # and all one-token decisions are one prefill span.
                                context = embeds[heads[decision] + 1:heads[next_decision] + 1]
                                chunks.append(torch.cat((continuation, context)))
                            else:
                                # Remaining recorded context needs no generated latents.
                                chunks.append(None)
                    return chunks

                generate(backbone=backbone, prompts=prompts, advance=advance, backend=backend,
                         max_active_requests=max_active_requests, max_batch_tokens=max_batch_tokens,
                         cache_capacity_tokens=cache_capacity_tokens)
        trace = _ReasoningTrace(heads, states, lengths.to(device).clone())
        prepared = self._materialize(tokens, trace, cached_prefix=False)
        # Materialization copied the needed feedback into latent_values. Reuse
        # this allocation for replay's unused diagnostic output slots.
        states.zero_()
        return prepared

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
        positions = [[0] * self.max_depth for _ in by_head]
        lengths = trace.lengths.tolist()
        offset = 0
        cursor = 0
        for group in range(tokens["num_sequences"]):
            row: list[tuple[int, int, int]] = []
            while cursor < len(groups) and groups[cursor] == group:
                if cursor in by_head:
                    i = by_head[cursor]
                    start = offset + len(row)
                    initial[i] = start
                    length = lengths[i]
                    readouts[i] = start + length - 1
                    positions[i][:length] = range(start, start + length)
                    row.extend((cursor, i, step) for step in range(length))
                else:
                    row.append((cursor, -1, -1))
                cursor += 1
            rows.append(row)
            offset += len(row)
        entries = [entry for row in rows for entry in row]
        indices = torch.tensor([entry[0] for entry in entries], device=device, dtype=torch.long)
        packed = source.clone(recurse=False)
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                packed[key] = value[indices].detach().clone()
        packed["sequence_ids"] = torch.repeat_interleave(
            torch.arange(len(rows), device=device), torch.tensor([len(row) for row in rows], device=device))
        packed.set_non_tensor("num_sequences", len(rows))
        latent_mask = torch.tensor([entry[1] >= 0 for entry in entries], device=device, dtype=torch.bool)
        latents = trace.states.new_zeros((len(entries), self.hidden_dim))
        latent_rows = [row for row, (_, i, _) in enumerate(entries) if i >= 0]
        if latent_rows:
            decision = torch.tensor([i for _, i, _ in entries if i >= 0], device=device)
            step = torch.tensor([step for _, i, step in entries if i >= 0], device=device)
            latents[torch.tensor(latent_rows, device=device)] = trace.states[decision, step if self.latent_feedback_scope != "none" else 0].detach()
        inserted = torch.tensor([step > 0 for _, _, step in entries], device=device)
        packed["field_start_mask"][inserted] = True
        packed["field_end_mask"][latent_mask] = True
        packed["latent_mask"] = latent_mask
        packed["latent_values"] = latents.detach()
        packed["head_mask"] = torch.zeros(len(entries), device=device, dtype=torch.bool)
        starts = torch.tensor(initial, device=device, dtype=torch.long)
        ends = torch.tensor(readouts, device=device, dtype=torch.long)
        packed["head_mask"][ends] = True
        packed["step_readout_mask"] = packed["head_mask"].clone()
        # Generation is complete: starts are metadata, not pending markers.
        update_positions = torch.tensor(positions, device=device, dtype=torch.long).reshape(-1, self.max_depth)
        return ReasoningBatch(packed, starts, update_positions, ends, trace.head_token_ids,
                             trace.states, trace.lengths, cached_prefix)

    def forward(
        self, generated: ReasoningBatch, *, embedding: MultiEmbedding, backbone: BaseBackbone,
    ) -> torch.Tensor:
        """Evaluate a generated batch once, returning all depth representations.

        Context is embedded by this network. Wrapped inputs retain detached saved
        feedback; scope starts use the live token embedding directly. Continuations
        pass only detached feedback through the input block, without an embedding
        offset. With residuals and wrapping, the additive path is rebuilt from this forward's updates.
        No generation or sequence insertion happens here.
        """
        if generated.cached_prefix:
            raise ValueError("Forward requires complete episodes, not a chunk with a cached prefix")
        count = len(generated.head_token_ids)
        if (generated.update_positions.shape != (count, self.max_depth)
                or generated.initial_positions.shape != (count,)
                or generated.readout_positions.shape != (count,)):
            raise ValueError("Generated positions must match readouts and reasoning budget")
        if (generated.lengths.shape != (count,) or generated.lengths.dtype != torch.long
                or bool(((generated.lengths < (self.min_depth if self.adaptive_depth else 1))
                         | (generated.lengths > self.max_depth)).any())):
            raise ValueError("Generated lengths must be int64 counts within the reasoning budget")
        if backbone.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim must match the backbone")
        if not count:
            states = generated.states[:, 1:].float()
            return states
        with _evaluating(embedding, backbone, self):
            tokens = generated.tokens.to(next(embedding.parameters()).device)
            embeds = self._embeds(tokens, embedding)
            initial = embeds.new_zeros((count, self.hidden_dim), dtype=torch.float32)
            inputs = embeds.clone()
            starts = generated.initial_positions.to(embeds.device)
            first = torch.ones(count, device=embeds.device, dtype=torch.bool)
            if self.latent_feedback_scope == "sequence":
                groups = tokens["sequence_ids"][starts]
                first[1:] = groups[1:] != groups[:-1]
            if self.latent_feedback_scope != "none":
                # Scope starts use their configured embedding directly. All
                # continuation inputs are projected detached feedback only.
                continuation = tokens["latent_mask"].clone()
                continuation[starts[first]] = False
                if bool(continuation.any()):
                    feedback = tokens["latent_values"][continuation].detach()
                    inputs[continuation] = self.project_inputs(feedback).to(embeds.dtype)
            hidden = cast(torch.Tensor, backbone(inputs, sequence_ids=tokens["sequence_ids"]))
            updates = hidden[generated.update_positions.to(hidden.device)].float()
            valid = torch.arange(self.max_depth, device=hidden.device)[None] < generated.lengths.to(hidden.device)[:, None]
            updates = updates.masked_fill(~valid[..., None], 0)
            if self.latent_residual_enabled and self.latent_feedback_scope != "none":
                if self.latent_feedback_scope == "sequence":
                    # Rebuild the live additive path across all decisions in each
                    # sequence; generated projection inputs remain detached.
                    first_indices = first.nonzero().flatten()
                    counts = torch.diff(torch.cat((first_indices, first_indices.new_tensor([count]))))
                    return torch.cat([
                        piece.flatten(0, 1).cumsum(0).view_as(piece)
                        for piece in updates.split(counts.tolist())
                    ])
                return initial[:, None] + updates.cumsum(1)
            states = initial[:, None] + updates if self.latent_residual_enabled else updates
            final = states[torch.arange(count, device=hidden.device), generated.lengths.to(hidden.device) - 1]
            return torch.where(valid[..., None], states, final[:, None])
