"""Autoregressive reasoning using the caller's existing Mouse embedding/backbone."""
from __future__ import annotations
from tensordict import TensorDict

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


from mouse_core.models.backbone.base import BaseBackbone
from mouse_core.models.backbone.continuous_decode import ContinuousDecodeSession
from mouse_core.models.embedding import MultiEmbedding
from mouse_core.models.generation import generate
from mouse_core.data.token_batch import decision_positions, head_output_segments


@dataclass(frozen=True)
class ReasoningTrace:
    """A readout position in the original TensorDict and its sampled continuation."""
    head_token_id: int
    generated_ids: tuple[int, ...]
    end_token_id: int
    final_hidden: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.head_token_id < 0:
            raise ValueError("head_token_id must be nonnegative")
        if (not self.generated_ids or self.generated_ids[-1] != self.end_token_id
                or self.end_token_id in self.generated_ids[:-1]):
            raise ValueError("A reasoning trace must end at its first generated end token")


@dataclass
class ReasoningReadout:
    final_hidden: torch.Tensor
    value_hidden: torch.Tensor
    log_prob_sums: torch.Tensor | None


@contextmanager
def _evaluating(*modules: nn.Module) -> Iterator[None]:
    """Match sampling and teacher-forcing distributions without disabling gradients."""
    modes = [module.training for module in modules]
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, mode in zip(modules, modes, strict=True):
            module.train(mode)


class AutoregressiveReasoner(nn.Module):
    """Add a vocabulary head to the SAME embedding/backbone used for values.

    The caller passes its actual modules; this object never creates or copies a
    transformer or embedding. Only the vocabulary head is registered here.
    Generation uses Mouse's paged ContinuousDecodeSession with row recycling, or an
    uncached FlexAttention forward. This is not Transformers' generate_batch API.
    """
    def __init__(self, token_head: nn.Module) -> None:
        super().__init__()
        self.token_head = token_head

    def end_token_ids(self, *, end_token_id: int | None, vocab_size: int) -> tuple[int, ...]:
        """Resolve an explicit stop ID or the checkpoint's generation defaults.

        A model may specify multiple default end tokens; any of them stops
        generation. An explicit ID replaces those defaults for this call.
        """
        values = ((end_token_id,) if end_token_id is not None
                  else getattr(self.token_head, "default_end_token_ids", ()))
        if not values:
            raise ValueError("The model has no default end token; specify end_token_id")
        if any(type(value) is not int or not 0 <= value < vocab_size for value in values):
            raise ValueError("Every end token ID must be an integer inside the selected vocabulary")
        return tuple(values)

    @staticmethod
    def _embeds(tokens: TensorDict, embedding: MultiEmbedding) -> torch.Tensor:
        tensors = tokens.to(next(embedding.parameters()).device)
        embeds = embedding(
            tensors["token_ids"], tokenizer_ids=tensors["tokenizer_ids"],
            tokenizer_names=tokens['tokenizer_names'],
            numeric_values=tensors["numeric_values"], numeric_mask=tensors["numeric_mask"],
            numeric_field_ids=tensors["numeric_field_ids"], numeric_field_names=tokens['numeric_field_names'],
            position_values=tensors["position_values"], position_mask=tensors["position_mask"],
        )
        return embeds

    @staticmethod
    def _embed_generated(ids: Sequence[int], embedding: MultiEmbedding, tokenizer_name: str) -> torch.Tensor:
        index = embedding.tokenizer_names.index(tokenizer_name)
        token_ids = torch.tensor(ids, dtype=torch.long, device=next(embedding.parameters()).device)
        return embedding(token_ids, tokenizer_ids=torch.full_like(token_ids, index),
                         tokenizer_names=embedding.tokenizer_names)

    @staticmethod
    def _packed(backbone: BaseBackbone, rows: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        lengths = [len(row) for row in rows]
        embeds = torch.cat(rows)
        ids = torch.repeat_interleave(torch.arange(len(rows), device=embeds.device),
                                      torch.tensor(lengths, device=embeds.device))
        return cast(torch.Tensor, backbone(embeds, sequence_ids=ids)), lengths

    @torch.no_grad()
    def generate(
        self, tokens: TensorDict, *, embedding: MultiEmbedding, backbone: BaseBackbone,
        tokenizer_name: str, max_new_tokens: int, end_token_id: int | None = None,
        backend: Literal["uncached", "continuous"], max_active_requests: int,
        do_sample: bool, temperature: float,
        cache: ContinuousDecodeSession | None = None,
    ) -> list[ReasoningTrace]:
        if backend not in ("uncached", "continuous"):
            raise ValueError("backend must be uncached or continuous")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError("max_active_requests must be a positive integer")
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be finite and positive")
        if tokenizer_name not in embedding.tokenizer_names:
            raise ValueError("Reasoning tokenizer must be one of the existing embedding tables")
        table = embedding.embeddings[tokenizer_name]
        end_token_ids = self.end_token_ids(end_token_id=end_token_id, vocab_size=cast(int, table.vocab_size))
        if not len(decision_positions(tokens)) and cache is None:
            return []
        with _evaluating(embedding, backbone, self.token_head):
            embeds = self._embeds(tokens, embedding)
            segments = [segment for segment in head_output_segments(tokens)
                        if segment.head_token_ids or segment.context_slices[0].start != segment.context_slices[0].stop]
            heads = decision_positions(tokens).tolist()
            head_rows = {position: i for i, position in enumerate(heads)}
            generated: list[list[int]] = [[] for _ in heads]
            cursors = [0] * len(segments)
            awaiting_end = [False] * len(segments)
            final_hidden = [None] * len(heads)
            prompts = [embeds[segment.context_slices[0].start:segment.head_token_ids[0] + 1]
                       if segment.head_token_ids else embeds[segment.context_slices[0]] for segment in segments]
            def advance(requests: list[int], hidden: torch.Tensor) -> list[torch.Tensor | None]:
                chunks: list[torch.Tensor | None] = [None] * len(requests)
                active = []
                for j, request in enumerate(requests):
                    segment = segments[request]
                    cursor = cursors[request]
                    if cursor >= len(segment.head_token_ids):
                        continue
                    if awaiting_end[request]:
                        decision = head_rows[segment.head_token_ids[cursor]]
                        final_hidden[decision] = hidden[j].detach().clone()
                        cursors[request] += 1
                        awaiting_end[request] = False
                        if cursor + 1 < len(segment.head_token_ids):
                            context = segment.context_slices[cursor + 1]
                            chunks[j] = embeds[context.start:segment.head_token_ids[cursor + 1] + 1]
                        else:
                            tail = embeds[segment.context_slices[-1]]
                            if len(tail):
                                chunks[j] = tail
                        continue
                    active.append((j, request, head_rows[segment.head_token_ids[cursor]]))
                if not active:
                    return chunks
                logits = self.token_head(hidden[[j for j, _, _ in active]].float()) / temperature
                if logits.shape[-1] != table.vocab_size:
                    raise ValueError("Vocabulary head must match the selected embedding table")
                draws = (torch.multinomial(logits.softmax(-1), 1).squeeze(-1) if do_sample
                         else logits.argmax(-1)).tolist()
                for (j, request, decision), token in zip(active, draws, strict=True):
                    generated[decision].append(token)
                    if token in end_token_ids:
                        awaiting_end[request] = True
                    elif len(generated[decision]) == max_new_tokens:
                        raise RuntimeError(f"Reasoning request {decision} did not emit an end token within {max_new_tokens} tokens")
                    chunks[j] = self._embed_generated([token], embedding, tokenizer_name)
                return chunks
            generate(backbone=backbone, prompts=prompts, advance=advance, backend=backend,
                     max_active_requests=max_active_requests, cache=cache,
                     cache_rows=[segment.sequence_id for segment in segments] if cache is not None else None)
        return [ReasoningTrace(int(head), tuple(ids), ids[-1], feature)
                for head, ids, feature in zip(heads, generated, final_hidden, strict=True)]

    def forward(
        self, tokens: TensorDict, traces: Sequence[ReasoningTrace], *,
        embedding: MultiEmbedding, backbone: BaseBackbone, tokenizer_name: str,
        temperature: float, score_tokens: bool,
    ) -> ReasoningReadout:
        """Recompute end token Q features AND log probabilities with the same modules.

        Log probabilities use the position BEFORE each sampled token, including
        end token. Value features use the position AFTER consuming end token. The baseline
        uses the decision input before its own samples, while retaining earlier traces.
        Prefix numeric/position features and tokenizer routing are preserved.
        """
        if not traces or [t.head_token_id for t in traces] != decision_positions(tokens).tolist():
            raise ValueError("Traces must match decision positions in order")
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be finite and positive")
        with _evaluating(embedding, backbone, self.token_head):
            embeds = self._embeds(tokens, embedding)
            by_head = {trace.head_token_id: trace for trace in traces}
            rows, value_positions, final_positions, generated_positions = [], [], [], []
            offset = 0
            for segment in head_output_segments(tokens):
                pieces = []
                length = 0
                for index, head in enumerate(segment.head_token_ids):
                    context = embeds[segment.context_slices[index].start:head + 1]
                    pieces.append(context)
                    length += len(context)
                    value_positions.append(offset + length - 1)
                    trace = by_head[head]
                    generated_positions.append((offset + length - 1, trace.generated_ids))
                    continuation = self._embed_generated(trace.generated_ids, embedding, tokenizer_name)
                    pieces.append(continuation)
                    length += len(continuation)
                    final_positions.append(offset + length - 1)
                pieces.append(embeds[segment.context_slices[-1]])
                row = torch.cat(pieces)
                if len(row):
                    rows.append(row)
                    offset += len(row)
            hidden, _ = self._packed(backbone, rows)
            log_probs = []
            if score_tokens:
                def sequence_log_prob(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                    with _evaluating(self.token_head):
                        return -F.cross_entropy(self.token_head(features.float()) / temperature, labels, reduction="sum")
                for start, ids in generated_positions:
                    positions = hidden[start:start + len(ids)]
                    labels = torch.tensor(ids, dtype=torch.long, device=hidden.device)
                    log_probs.append(checkpoint(sequence_log_prob, positions, labels, use_reentrant=False))
            return ReasoningReadout(hidden[final_positions], hidden[value_positions],
                                    torch.stack(log_probs) if score_tokens else None)
