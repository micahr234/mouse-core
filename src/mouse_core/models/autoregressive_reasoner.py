"""Autoregressive reasoning using the caller's existing Mouse embedding/backbone."""
from __future__ import annotations
from tensordict import TensorDict

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


from mouse_core.models.backbone.base import BaseBackbone
from mouse_core.models.embedding import MultiEmbedding


@dataclass(frozen=True)
class ReasoningTrace:
    """A readout position in the original TensorDict and its sampled continuation."""
    head_token_id: int
    generated_ids: tuple[int, ...]
    end_token_id: int

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
    Generation uses Mouse's paged FlexDecodeSession with row recycling, or an
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
    def _prefixes(tokens: TensorDict, embedding: MultiEmbedding) -> list[torch.Tensor]:
        tensors = tokens.to(next(embedding.parameters()).device)
        embeds = embedding(
            tensors["token_ids"], tokenizer_ids=tensors["tokenizer_ids"],
            tokenizer_names=tokens['tokenizer_names'],
            numeric_values=tensors["numeric_values"], numeric_mask=tensors["numeric_mask"],
            numeric_field_ids=tensors["numeric_field_ids"], numeric_field_names=tokens['numeric_field_names'],
            position_values=tensors["position_values"], position_mask=tensors["position_mask"],
        )
        starts = torch.searchsorted(tokens['sequence_ids'], tokens['sequence_ids'][tokens["head_token_mask"].nonzero(as_tuple=True)[0]])
        return [embeds[int(start):int(end) + 1]
                for start, end in zip(starts, tokens["head_token_mask"].nonzero(as_tuple=True)[0], strict=True)]

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
        if not len(tokens["head_token_mask"].nonzero(as_tuple=True)[0]):
            return []
        with _evaluating(embedding, backbone, self.token_head):
            prefixes = self._prefixes(tokens, embedding)
            count = len(prefixes)
            slots = min(count, max_active_requests)
            active: list[int | None] = list(range(slots))
            pending = slots
            generated: list[list[int]] = [[] for _ in prefixes]
            chunks = prefixes[:slots]
            cache = backbone.decode_session(batch_size=slots) if backend == "continuous" else None
            try:
                while any(request is not None for request in active):
                    if cache is not None:
                        lengths = [len(chunk) for chunk in chunks]
                        width = max(lengths)
                        padded = prefixes[0].new_zeros((slots, width, backbone.hidden_dim))
                        for slot, chunk in enumerate(chunks):
                            if len(chunk):
                                padded[slot, -len(chunk):] = chunk
                        hidden = cast(torch.Tensor, backbone(padded, cache=cache, lengths=lengths))[:, -1]
                    else:
                        rows = [torch.cat((prefixes[request], self._embed_generated(
                            generated[request], embedding, tokenizer_name)))
                            for request in active if request is not None]
                        features, lengths = self._packed(backbone, rows)
                        ends = torch.tensor(lengths, device=features.device).cumsum(0) - 1
                        hidden = features.new_zeros((slots, backbone.hidden_dim))
                        hidden[[slot for slot, request in enumerate(active) if request is not None]] = features[ends]
                    live_slots = [slot for slot, request in enumerate(active) if request is not None]
                    logits = self.token_head(hidden[live_slots].float()) / temperature
                    if logits.shape[-1] != table.vocab_size:
                        raise ValueError("Vocabulary head must match the selected embedding table")
                    draws = (torch.multinomial(logits.softmax(-1), 1).squeeze(-1) if do_sample
                             else logits.argmax(-1)).tolist()
                    reset = []
                    for slot, token in zip(live_slots, draws, strict=True):
                        request = active[slot]
                        assert request is not None
                        generated[request].append(token)
                        if token in end_token_ids:
                            reset.append(slot)
                            if pending < count:
                                active[slot] = pending
                                chunks[slot] = prefixes[pending]
                                pending += 1
                            else:
                                active[slot] = None
                                chunks[slot] = prefixes[0][:0]
                        elif len(generated[request]) == max_new_tokens:
                            raise RuntimeError(f"Reasoning request {request} did not emit an end token within {max_new_tokens} tokens")
                        else:
                            chunks[slot] = self._embed_generated([token], embedding, tokenizer_name)
                    if cache is not None:
                        cache.reset_rows(reset)
            finally:
                if cache is not None:
                    cache.close()
        return [ReasoningTrace(int(head), tuple(ids), ids[-1])
                for head, ids in zip(tokens["head_token_mask"].nonzero(as_tuple=True)[0], generated, strict=True)]

    def forward(
        self, tokens: TensorDict, traces: Sequence[ReasoningTrace], *,
        embedding: MultiEmbedding, backbone: BaseBackbone, tokenizer_name: str,
        temperature: float, score_tokens: bool,
    ) -> ReasoningReadout:
        """Recompute end token Q features AND log probabilities with the same modules.

        Log probabilities use the position BEFORE each sampled token, including
        end token. Value features use the position AFTER consuming end token. The baseline
        uses the value token before any samples, so it is independent of them.
        Prefix numeric/position features and tokenizer routing are preserved.
        """
        if not traces or [t.head_token_id for t in traces] != tokens["head_token_mask"].nonzero(as_tuple=True)[0].tolist():
            raise ValueError("Traces must match all head-token positions in order")
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be finite and positive")
        with _evaluating(embedding, backbone, self.token_head):
            prefixes = self._prefixes(tokens, embedding)
            rows = [torch.cat((prefix, self._embed_generated(trace.generated_ids, embedding, tokenizer_name)))
                    for prefix, trace in zip(prefixes, traces, strict=True)]
            hidden, lengths = self._packed(backbone, rows)
            ends = torch.tensor(lengths, device=hidden.device).cumsum(0)
            starts = ends - torch.tensor(lengths, device=hidden.device)
            value_positions = starts + torch.tensor([len(p) - 1 for p in prefixes], device=hidden.device)
            log_probs = []
            if score_tokens:
                def sequence_log_prob(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                    with _evaluating(self.token_head):
                        return -F.cross_entropy(self.token_head(features.float()) / temperature, labels, reduction="sum")
                for start, prefix, trace in zip(starts.tolist(), prefixes, traces, strict=True):
                    positions = hidden[start + len(prefix) - 1:start + len(prefix) + len(trace.generated_ids) - 1]
                    labels = torch.tensor(trace.generated_ids, dtype=torch.long, device=hidden.device)
                    # Recompute vocabulary logits on backward instead of retaining
                    # [all generated tokens, vocabulary size] activations.
                    log_probs.append(checkpoint(sequence_log_prob, positions, labels, use_reentrant=False))
            return ReasoningReadout(hidden[ends - 1], hidden[value_positions],
                                    torch.stack(log_probs) if score_tokens else None)
