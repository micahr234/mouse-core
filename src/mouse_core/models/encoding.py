"""Token embedding, packed backbones, reasoning, and cached decoding."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import nn

from mouse_core.models.backbone.base import BaseBackbone
from mouse_core.models.backbone.flex_decode import FlexDecodeSession, packed_rope_positions
from mouse_core.models.reasoner import LatentReasoner, _InsertionPlan, _plan_insertions

if TYPE_CHECKING:
    from mouse_core.data.token_batch import TokenBatch

def _last_hidden(
    session_out: torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    """Last-layer hidden states (``[L, D]`` train, ``[B, S, D]`` decode)."""
    if isinstance(session_out, tuple):
        return session_out[0]
    return session_out


@dataclass
class DecodeCache:
    """KV state carried between ``use_cache=True`` calls.

    Holds one session per backbone stage: :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession`.
    Pass ``out.cache`` back as ``cache=``. Call :meth:`close` when the
    rollout is finished so VMM pages unmap before the next train step
    (``__del__`` is too late if a CUDA graph still holds views).
    """

    sessions: tuple[FlexDecodeSession, ...]

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``).

        The rows' next tokens start at position 0 and their KV pages return
        to the shared pool.
        """
        for session in self.sessions:
            session.reset_rows(rows)

    def close(self) -> None:
        """Unmap every session's VMM pool and drop captured CUDA graphs."""
        for session in self.sessions:
            session.close()


@dataclass
class EncodedTokens:
    """BaseBackbone states and readout positions for packed training or cached decode."""

    last_hidden_state: torch.Tensor
    head_output_indices: torch.Tensor
    cache: DecodeCache | None = None
    head_output_valid: torch.Tensor | None = None

    def pool(self) -> torch.Tensor:
        """Select head tokens before running any projection or prediction."""
        h, indices = self.last_hidden_state, self.head_output_indices
        if h.ndim == 2:
            return h[indices.reshape(-1)]
        return h.gather(1, indices.unsqueeze(-1).expand(*indices.shape, h.shape[-1]))

    def with_predictions(self, predictions: dict[str, torch.Tensor]) -> ModelOutput:
        return ModelOutput(
            predictions=predictions, last_hidden_state=self.last_hidden_state,
            head_output_indices=self.head_output_indices, cache=self.cache,
            head_output_valid=self.head_output_valid,
        )


@dataclass(kw_only=True)
class ModelOutput(EncodedTokens):
    """Named predictions alongside the backbone states and optional decode cache."""

    predictions: dict[str, torch.Tensor]


class TokenEncoder(nn.Module):
    """Reusable token processing; model subclasses decide how to use its output.

    Backbones run in order, with token embedding on the first only. Each
    transformer returns its final residual stream without an output norm.
    """

    def __init__(self, backbones: Sequence[BaseBackbone], *, reasoner: LatentReasoner | None = None) -> None:
        super().__init__()
        if not backbones or any(not isinstance(backbone, BaseBackbone) for backbone in backbones):
            raise ValueError("backbones must be a non-empty sequence of BaseBackbone modules.")
        self.hidden_dim = backbones[0].hidden_dim
        if any(backbone.hidden_dim != self.hidden_dim for backbone in backbones):
            raise ValueError("stacked backbones must have matching hidden_dim.")
        if reasoner is not None and reasoner.hidden_dim != self.hidden_dim:
            raise ValueError("hidden_dim mismatch between reasoner and encoder.")
        self.backbones = nn.ModuleList(backbones)
        self.reasoner = reasoner

    def forward(
        self, batch: TokenBatch, *, cache: DecodeCache | None = None,
        use_cache: bool = False, reasoning: Sequence[int] | np.ndarray | None = None,
    ) -> EncodedTokens:
        """Encode all tokens, retaining the indices needed for head-only readout.

        Training returns flat packed states. Cached decoding returns batched
        states and uses one decode session per backbone; pass ``out.cache``
        into the next call and close it after the rollout. ``reasoning``
        inserts latent thoughts before selected readouts during training.
        """
        from mouse_core.data.token_batch import TokenBatch as _TokenBatch

        if cache is not None and not use_cache:
            raise ValueError("Passing cache= requires use_cache=True.")
        if not isinstance(batch, _TokenBatch):
            raise TypeError(
                f"TokenEncoder.forward expects a TokenBatch, got {type(batch).__name__}. "
                "Use pack_token_batch(steps=[transform(step)], ..., continuing=None) "
                "or DataLoader(transform=...)."
            )

        token_batch = batch
        B = token_batch.B
        step_counts_np = token_batch.step_counts()
        N = token_batch.N
        S_max = token_batch.S

        if use_cache and B > 0 and N == 0:
            raise ValueError("TokenEncoder.forward requires at least one non-empty row in batch.")

        plan: _InsertionPlan | None = None
        if reasoning is not None:
            if use_cache:
                raise ValueError("reasoning= is not supported with use_cache=True.")
            if self.reasoner is None:
                raise ValueError(
                    "TokenEncoder.forward(reasoning=...) requires a reasoner; construct "
                    "TokenEncoder(..., reasoner=LatentReasoner(...))."
                )
            plan = _plan_insertions(
                token_batch,
                np.asarray(reasoning, dtype=np.int64).reshape(-1),
                self.reasoner.num_thoughts,
            )

        embeds, resolved_indices = cast(BaseBackbone, self.backbones[0]).embed(token_batch)
        # embeds: [L, D]; resolved_indices: [P]
        t = token_batch.to_tensors(embeds.device)
        group_ids = t["group_ids"]

        new_cache: DecodeCache | None

        if use_cache:
            from mouse_core.models.embedding.packing import left_align_content

            # Decode pools one position per step: the step's *last* head-output
            # token (the most informed one when a step has several).
            psteps = t["head_output_steps"]
            last_of_step = torch.ones(
                psteps.shape[0], dtype=torch.bool, device=psteps.device
            )
            last_of_step[:-1] = psteps[1:] != psteps[:-1]
            batched_embeds, token_lengths, local_indices = _flat_to_batched_left_pad(
                embeds,
                group_ids,
                resolved_indices[last_of_step],
                B,
                S_max,
                step_counts_np.tolist(),
            )
            sessions = cache.sessions if cache is not None else self._decode_sessions(B)
            if len(sessions) != len(self.backbones):
                raise ValueError("cache must contain one session per backbone stage.")
            flex_embeds, resolved_indices = left_align_content(
                batched_embeds, local_indices, token_lengths
            )
            # ``token_lengths`` already counts only real tokens (tokenize is ragged;
            # empty rows contribute 0). Do not re-derive from left-padded step indices.
            session_out = flex_embeds
            for session in sessions:
                session_out = session.forward(
                    output_hidden_states=False,
                    embeds=_last_hidden(session_out),
                    lengths=token_lengths,
                )
            new_cache = DecodeCache(sessions=sessions)
            counts = torch.as_tensor(
                step_counts_np.tolist(), device=embeds.device, dtype=torch.long
            )
            # Left-padded steps: trailing ``n`` columns of row ``b`` are real.
            head_output_valid = (
                torch.arange(S_max, device=embeds.device).unsqueeze(0)
                >= (S_max - counts).clamp(min=0).unsqueeze(1)
            )
            head_output_valid = head_output_valid & counts.unsqueeze(1).gt(0)
        else:
            if plan is not None:
                # Gradient-taped latent generation; swaps in the extended stream.
                embeds, group_ids, resolved_indices, _ = self._generate_latents(
                    embeds=embeds,
                    group_ids=group_ids,
                    plan=plan,
                )
            session_out = self._train_backbone_forward(
                embeds,
                group_ids,
            )
            new_cache = None
            head_output_valid = None

        last_h = _last_hidden(session_out)
        return EncodedTokens(
            last_hidden_state=last_h,
            head_output_indices=resolved_indices,
            cache=new_cache,
            head_output_valid=head_output_valid,
        )

    def _decode_sessions(self, batch_size: int) -> tuple[FlexDecodeSession, ...]:
        sessions: list[FlexDecodeSession] = []
        try:
            for backbone in self.backbones:
                sessions.append(cast(BaseBackbone, backbone).decode_session(batch_size=batch_size))
        except Exception:
            for session in sessions:
                session.close()
            raise
        return tuple(sessions)

    def _train_backbone_forward(
        self, embeds: torch.Tensor, group_ids: torch.Tensor,
    ) -> torch.Tensor:
        for backbone in self.backbones:
            embeds = self._run_backbone(cast(BaseBackbone, backbone), embeds, group_ids)
        return embeds

    @staticmethod
    def _run_backbone(
        backbone: BaseBackbone,
        embeds: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Uncached backbone pass over the flat packed stream.

        Transformer backbones run :func:`packed_forward` with the backbone's
        ``train_kernel`` (``"varlen"``: flash varlen, CUDA bf16/fp16 q/k/v
        only — raises otherwise; ``"padded"``: dense causal SDPA on segments
        padded to ``max_seqlen``; ``"flex"``: FlexAttention; ``"reference"``:
        masked SDPA, O(L^2)) and ``train_autocast_dtype`` (bf16/fp16 mixed
        precision over fp32 weights, ``None`` for the base dtype). Backbones
        without packed kernels (``IdentityBackbone``, generic HuggingFace
        stacks) take the rectangular route with a dense per-sequence
        causal mask. ``embeds`` come from ``backbone.embed`` (fp32 tables /
        adapters) and are cast to the backbone's base dtype here.
        """
        embeds = embeds.to(dtype=backbone.dtype)
        if getattr(backbone, "uses_packed", False):
            from mouse_core.models.backbone.packed_train import packed_forward

            transformer = getattr(backbone, "model", None)
            if transformer is None:
                raise TypeError(f"{type(backbone).__name__}.uses_packed is True but it has no .model.")
            return packed_forward(
                model=cast(nn.Module, transformer),
                embeds=embeds,
                group_ids=group_ids,
                output_hidden_states=False,
                checkpoint=backbone.gradient_checkpointing,
                train_kernel=backbone.train_kernel,
                autocast_dtype=backbone.train_autocast_dtype,
            )
        attention_mask = _flat_sequence_causal_mask(
            dtype=embeds.dtype,
            group_ids=group_ids,
        )
        position_ids = _flat_sequence_position_ids(group_ids=group_ids)
        session_out = backbone(
            embeds.unsqueeze(0),
            output_hidden_states=False,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        if isinstance(session_out, torch.Tensor) and session_out.ndim == 3:
            return session_out.squeeze(0)
        return cast(torch.Tensor, session_out)

    def _generate_latents(
        self,
        *,
        embeds: torch.Tensor,
        group_ids: torch.Tensor,
        plan: _InsertionPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate latent thoughts on the autograd tape and assemble the extended stream.

        Runs ``R`` extra backbone passes over the growing per-burst prefixes
        (all bursts in lockstep; attention isolation makes dropping the other
        sequences exact). Thought ``r``'s input embedding is the reasoner
        adapter applied to the backbone output at the previous position, so
        gradients flow through the whole latent chain. Returns
        ``(ext_embeds, ext_group_ids, ext_head_output_indices,
        token_indices)`` where ``token_indices`` maps
        each original token to its extended-stream position.
        """
        reasoner = self.reasoner
        assert reasoner is not None
        R = plan.num_thoughts
        device = embeds.device
        nb = int(plan.burst_rows.size)

        prefix_bounds = list(
            zip(plan.prefix_starts.tolist(), plan.anchors.tolist())
        )
        thoughts: list[torch.Tensor] = []  # thought r → [nb, D]
        for r in range(R):
            parts: list[torch.Tensor] = []
            seq_parts: list[torch.Tensor] = []
            last_positions: list[int] = []
            offset = 0
            for j, (start, anchor) in enumerate(prefix_bounds):
                parts.append(embeds[start:anchor])
                seq_parts.append(group_ids[start:anchor])
                if r > 0:
                    parts.append(torch.stack([thoughts[q][j] for q in range(r)]))
                    seq_parts.append(
                        group_ids.new_full((r,), int(plan.burst_rows[j]))
                    )
                block = (anchor - start) + r
                last_positions.append(offset + block - 1)
                offset += block
            gen_out = self._train_backbone_forward(
                torch.cat(parts),
                torch.cat(seq_parts),
            )
            h_last = cast(torch.Tensor, gen_out)[
                torch.as_tensor(last_positions, device=device)
            ]
            thoughts.append(reasoner(h_last).to(dtype=embeds.dtype))

        latent_embeds = torch.stack(thoughts, dim=1).reshape(nb * R, embeds.shape[-1])
        token_indices = torch.as_tensor(plan.token_positions, device=device)
        latent_indices = torch.as_tensor(plan.latent_positions, device=device)
        ext_embeds = (
            embeds.new_zeros(plan.ext_length, embeds.shape[-1])
            .index_copy(0, token_indices, embeds)
            .index_copy(0, latent_indices, latent_embeds)
        )
        ext_group_ids = torch.as_tensor(plan.ext_group_ids, device=device)
        ext_head_output_indices = torch.as_tensor(
            plan.ext_head_output_indices, device=device
        )
        return (
            ext_embeds,
            ext_group_ids,
            ext_head_output_indices,
            token_indices,
        )

def _flat_to_batched_left_pad(
    embeds: torch.Tensor,
    group_ids: torch.Tensor,
    head_output_indices: torch.Tensor,
    B: int,
    S: int,
    step_counts: list[int],
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Scatter flat ``[L, D]`` embeds into a rectangular ``[B, Lmax, D]`` layout.

    Content is packed from index 0 within each row (right-padded).
    ``head_output_indices`` is flat ``[N]`` (one index per step — the caller
    passes each step's last head-output token). Returns local rectangular
    ``head_output_indices`` ``[B, S]`` with real steps in trailing columns
    (left-padded in the step dimension for decode).
    """
    L, D = embeds.shape
    token_lengths = [int((group_ids == b).sum().item()) for b in range(B)]
    Lmax = max(token_lengths) if token_lengths else 0
    out = embeds.new_zeros(B, Lmax, D)
    local_indices = torch.zeros(B, S, device=embeds.device, dtype=torch.long)

    # Map absolute token index → local index within its sequence.
    local_of_abs = torch.full((L,), -1, device=embeds.device, dtype=torch.long)
    for b in range(B):
        mask = group_ids == b
        toks = embeds[mask]
        out[b, : toks.shape[0]] = toks
        abs_idx = torch.where(mask)[0]
        local_of_abs[abs_idx] = torch.arange(toks.shape[0], device=embeds.device)

    flat_offset = 0
    for b in range(B):
        n = int(step_counts[b]) if b < len(step_counts) else S
        for s_local in range(n):
            abs_i = int(head_output_indices[flat_offset + s_local].item())
            # Place into trailing step columns.
            local_indices[b, S - n + s_local] = int(local_of_abs[abs_i].item())
        flat_offset += n
    return out, token_lengths, local_indices


def _flat_sequence_causal_mask(
    *,
    group_ids: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Additive attention mask ``[1, 1, L, L]`` for a packed flat sequence."""
    L = group_ids.shape[0]
    device = group_ids.device
    q = torch.arange(L, device=device)
    kv = torch.arange(L, device=device)
    causal = kv.unsqueeze(0) <= q.unsqueeze(1)
    same_seq = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)
    allow = causal & same_seq
    neg = torch.finfo(dtype).min
    mask = torch.where(
        allow,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), neg, device=device, dtype=dtype),
    )
    return mask.view(1, 1, L, L)


def _flat_sequence_position_ids(*, group_ids: torch.Tensor) -> torch.Tensor:
    """RoPE positions ``[1, L]``: count of earlier same-sequence tokens.

    Same rule as the packed train forward and cached decode.
    """
    return packed_rope_positions(group_ids=group_ids).unsqueeze(0)
