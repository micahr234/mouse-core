"""Composable model execution, head readout, and delayed model copies."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mouse_core.models.backbone.base import Backbone, _reject_dtype_cast
from mouse_core.models.backbone.flex_decode import FlexDecodeSession, packed_rope_positions
from mouse_core.models.heads.base import BaseHead, _bind_prediction_key
from mouse_core.models.heads.classification import ClassificationHead
from mouse_core.models.heads.regression import RegressionHead
from mouse_core.models.projection import ProjectionHead
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


def _pool_head_outputs(
    h: torch.Tensor, head_output_indices: torch.Tensor
) -> torch.Tensor:
    """Gather head-output tokens from backbone states.

    ``h`` is ``[L, D]`` (flat packed) or ``[B, L, D]`` (decode).
    Train: ``head_output_indices`` is ``[P]`` into ``0 .. L-1``.
    Decode: ``head_output_indices`` is ``[B, S]`` into the token axis of ``h``.
    """
    if h.ndim == 2:
        return h[head_output_indices.reshape(-1)]
    B, S = head_output_indices.shape
    D = h.shape[-1]
    idx = head_output_indices.unsqueeze(-1).expand(B, S, D)
    return h.gather(1, idx)


def _run_heads(
    heads: dict[str, BaseHead],
    h: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run ``heads`` on pooled last-layer ``h`` (``[N, D]`` or ``[B, S, D]``).

    Each head is called as a module. ``head.forward`` skips forward and
    backward hooks, so a hook on the head would never see the gradient
    into this pooled state.
    """
    h = h.float()
    return {name: head(h) for name, head in heads.items()}


@dataclass
class DecodeCache:
    """KV state carried between ``use_cache=True`` calls.

    Holds one :class:`~mouse_core.models.backbone.flex_decode.FlexDecodeSession`.
    Pass ``out.cache`` back as ``cache=``. Call :meth:`close` when the
    rollout is finished so VMM pages unmap before the next train step
    (``__del__`` is too late if a CUDA graph still holds views).
    """

    session: FlexDecodeSession

    def reset_rows(self, rows: Sequence[int] | None = None) -> None:
        """Restart the given batch rows (all rows when ``None``).

        The rows' next tokens start at position 0 and their KV pages return
        to the shared pool.
        """
        self.session.reset_rows(rows)

    def close(self) -> None:
        """Unmap the session's VMM pool and drop captured CUDA graphs."""
        self.session.close()


@dataclass
class ModelOutput:
    """Head predictions plus the token states they were read from.

    ``predictions`` and ``last_hidden_state`` come from the single
    backbone forward.

    ``head_output_indices`` maps token states to the rows heads read. A
    reasoning forward extends the stream, so those indices and the states
    describe the stream *with* latents inserted. Cached decode also sets
    ``head_output_valid`` (``[B, S]``): True where a step slot is a real
    last-head-output readout, False on left-padded idle columns.
    Incremental decode carries ``cache`` — pass ``out.cache`` back as
    ``cache=`` with ``use_cache=True``.
    """

    predictions: dict[str, torch.Tensor]
    last_hidden_state: torch.Tensor
    head_output_indices: torch.Tensor
    cache: DecodeCache | None = None
    head_output_valid: torch.Tensor | None = None


class Model(nn.Module):
    """Composable MOUSE model with an optional shared projection before heads.

    The model is assembled from pluggable parts:

    - ``backbone``: a :class:`~mouse_core.models.backbone.Backbone` that
      embeds a :class:`~mouse_core.data.token_batch.TokenBatch` and maps
      those tokens to last-layer hidden states ``[L, D]``. Token
      embeddings live on the backbone: ``TransformerBackbone`` loads
      native ``embed_tokens``;
      :class:`~mouse_core.models.backbone.IdentityBackbone` owns a
      ``vocab_size`` × ``hidden_dim`` table. ``num_frequencies`` adds
      a Fourier projection onto ``__numeric__`` token embeddings.
    - ``heads``: heads can be provided in several ergonomic ways:
        - a single :class:`~mouse_core.models.heads.base.BaseHead` (e.g. ``RegressionHead(...)``):
          it becomes the only enabled head;
        - a list of head instances (e.g. ``[RegressionHead(...), ClassificationHead(...)]``):
          names are inferred from type;
        - a dict mapping caller-chosen names to head instances or ``None``.
      When a plain head is passed without a name the key is inferred from
      type (``action_value`` / ``action``);
      use the dict form to pick the key.

    ``action_source`` is the ``heads`` dict key ``get_action`` consults.
    Required.
    ``reasoner`` is required (pass ``None`` when unused).
    ``projection`` optionally applies a :class:`ProjectionHead` to pooled
    backbone states before every prediction head. It has its own optimizer
    parameter group and Polyak rate; omit it to keep the direct readout.

    Full construction::

        backbone = TransformerBackbone(pretrained=..., ...)
        head = RegressionHead(...)

        model = Model(
            backbone=backbone,
            heads={"action_value": head},
            action_source="action_value",
            reasoner=None,
        )

    ``forward`` returns a :class:`ModelOutput` with ``predictions`` and
    ``last_hidden_state``.
    The delayed DQN model comes from :meth:`copy`, which names every
    section (``heads``, ``backbone``, ``reasoner``, and ``projection`` when
    present). Copied sections are
    frozen; uncopied sections stay this model's modules.
    :func:`~mouse_core.polyak.model_polyak` interpolates only copied sections.
    """

    def __init__(
        self,
        *,
        backbone: Backbone,
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None],
        action_source: str,
        reasoner: LatentReasoner | None,
        projection: ProjectionHead | None = None,
    ):
        """Construct a Model from backbone and heads.

        All arguments except ``projection`` are required. ``reasoner`` enables
        Coconut-style latent reasoning via ``forward(batch, reasoning=...)``. Pass ``None`` when
        unused.
        """
        super().__init__()

        if not isinstance(backbone, Backbone):
            raise TypeError("backbone must be a Backbone (from mouse_core.models.backbone).")

        bb_dim = backbone.hidden_dim
        if bb_dim is None:
            raise ValueError("backbone.hidden_dim is required.")
        hidden_dim = int(bb_dim)

        self.backbone: Backbone = backbone

        if projection is not None:
            if not isinstance(projection, ProjectionHead):
                raise TypeError(
                    f"projection must be a ProjectionHead, got {type(projection).__name__}."
                )
            if projection.in_features != hidden_dim:
                raise ValueError(
                    f"hidden_dim mismatch between projection ({projection.in_features}) "
                    f"and model ({hidden_dim})."
                )
        self.projection: ProjectionHead | None = projection

        if reasoner is not None:
            if not isinstance(reasoner, LatentReasoner):
                raise TypeError(
                    f"reasoner must be a LatentReasoner, got {type(reasoner).__name__}."
                )
            if reasoner.hidden_dim != hidden_dim:
                raise ValueError(
                    f"hidden_dim mismatch between reasoner ({reasoner.hidden_dim}) "
                    f"and model ({hidden_dim})."
                )
        self.reasoner: LatentReasoner | None = reasoner

        self._heads = Model._normalize_heads(heads)
        self.heads = nn.ModuleDict(self._heads)
        for name, head in self._heads.items():
            _bind_prediction_key(head, name)

        self.action_source = Model._require_action_source(self._heads, action_source)

        self.hidden_dim = hidden_dim
        # Best-effort inference of action cardinality for introspection only.
        self.max_num_actions: int = 0
        for head in self._heads.values():
            out = getattr(head, "out_features", None)
            if isinstance(out, int) and out > 0:
                self.max_num_actions = out
                break

    def forward(
        self,
        batch: TokenBatch,
        *,
        cache: DecodeCache | None = None,
        use_cache: bool = False,
        reasoning: Sequence[int] | np.ndarray | None = None,
    ) -> ModelOutput:
        """Run a forward pass over a :class:`TokenBatch`.

        Training: ``inputs, objective_data, group_id = loader.next_batch()`` then
        ``out = model(inputs)``. Delayed DQN: ``delayed_model =
        model.copy(heads=True, backbone=True, reasoner=False)`` then
        ``delayed_model(inputs)`` under ``torch.no_grad()`` (same
        ``TokenBatch`` and ``reasoning=`` as the online forward). When
        ``backbone=False`` (and ``reasoner=False``), skip the second
        backbone pass: ``delayed_model.head(h=delayed_model.pool(output=out))``
        under ``torch.no_grad()``. Interpolate with
        a bound ``update = model_polyak(online=model, delayed=delayed_model)``
        and then ``update(tau_heads=..., tau_backbone=...)`` after each
        optimizer step. Use ``None`` for a shared section.
        If a projection is present, also choose ``projection=True`` or
        ``False`` in ``copy`` and pass ``tau_projection=...`` when copied.
        Online / inference: ``inputs, _, _ = pack_token_batch(steps=[eval_transform(step)],
        group_ids=[0], continuing=None)`` then ``model(inputs, use_cache=True)``
        (optionally ragged; empty-only batches raise). Pass ``out.cache``
        back as ``cache=``.

        ``reasoning`` (training only, requires ``Model(reasoner=...)``) is a
        ``[B]`` array of local step indices from
        :func:`~mouse_core.models.reasoner.sample_reasoning_splits`
        (``-1`` skips a row). For each selected step the model generates
        ``reasoner.num_thoughts`` latent thought embeddings on the autograd
        tape — each thought's input is the reasoner adapter applied to the
        backbone output at the previous position — and inserts them
        immediately before that step's first head-output token, so the Q
        readout and all later same-run tokens attend to them. Predictions
        keep the flat ``[P, ...]`` contract; ``last_hidden_state`` /
        ``head_output_indices`` describe the extended stream.

        Training attention runs the packed stream forward over the flat
        concatenated token stream (causal within one ``group_id``;
        kernel chosen by ``backbone.train_kernel``). Cached
        decode keeps one ``FlexDecodeSession``, a paged KV pool in which each
        sequence owns only the pages its own history needs. On CUDA the pool grows by mapping more physical
        pages (no copy of existing K/V). Call ``out.cache.close()`` when the
        rollout ends.

        Training predictions are flat over head-output tokens (``[P, ...]``,
        one row per head-output token; ``objective_data["head_output_count"]``
        maps rows to steps). Cached decode returns rectangular ``[B, S]``
        tensors pooled at each step's last head-output token.
        """
        from mouse_core.data.token_batch import TokenBatch as _TokenBatch

        if cache is not None and not use_cache:
            raise ValueError("Passing cache= requires use_cache=True.")
        if not isinstance(batch, _TokenBatch):
            raise TypeError(
                f"Model.forward expects a TokenBatch, got {type(batch).__name__}. "
                "Use pack_token_batch(steps=[transform(step)], ..., continuing=None) "
                "or DataLoader(transform=...)."
            )

        token_batch = batch
        B = token_batch.B
        step_counts_np = token_batch.step_counts()
        N = token_batch.N
        S_max = token_batch.S

        if use_cache and B > 0 and N == 0:
            raise ValueError("Model.forward requires at least one non-empty row in batch.")

        plan: _InsertionPlan | None = None
        if reasoning is not None:
            if use_cache:
                raise ValueError("reasoning= is not supported with use_cache=True.")
            if self.reasoner is None:
                raise ValueError(
                    "Model.forward(reasoning=...) requires a reasoner; construct "
                    "Model(..., reasoner=LatentReasoner(...))."
                )
            plan = _plan_insertions(
                token_batch,
                np.asarray(reasoning, dtype=np.int64).reshape(-1),
                self.reasoner.num_thoughts,
            )

        embeds, resolved_indices = self.backbone.embed(token_batch)
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
            session = (
                cache.session
                if cache is not None
                else self.backbone.decode_session(batch_size=B)
            )
            flex_embeds, resolved_indices = left_align_content(
                batched_embeds, local_indices, token_lengths
            )
            # ``token_lengths`` already counts only real tokens (tokenize is ragged;
            # empty rows contribute 0). Do not re-derive from left-padded step indices.
            session_out = session.forward(
                output_hidden_states=False,
                embeds=flex_embeds,
                lengths=token_lengths,
            )
            new_cache = DecodeCache(session=session)
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
                self.backbone,
                embeds,
                group_ids,
            )
            new_cache = None
            head_output_valid = None

        last_h = _last_hidden(session_out)
        return ModelOutput(
            predictions=self.head(
                h=_pool_head_outputs(last_h, resolved_indices),
            ),
            last_hidden_state=last_h,
            head_output_indices=resolved_indices,
            cache=new_cache,
            head_output_valid=head_output_valid,
        )

    def head(
        self,
        *,
        h: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the shared projection, then prediction heads, to backbone ``h``.

        Heads take ``[N, D]`` (train) or ``[B, S, D]`` (decode).
        Each head is called as a module, so its forward and backward
        hooks run.
        """
        if self.projection is not None:
            h = self.projection(h)
        return _run_heads(self._heads, h)

    def get_action(
        self,
        *,
        out: dict[str, torch.Tensor] | ModelOutput,
        temperature: float,
        num_actions: int | None = None,
    ) -> torch.Tensor:
        """Select an action from the value head at the last valid head-output.

        Cached decode writes Q at each step's last head-output token
        (left-padded ``[B, S, A]``). Pass the
        :class:`ModelOutput` from ``forward(..., use_cache=True)`` so this
        reads the last-layer residual stream at the last *valid* (non-pad)
        head-output of each row — not a padded step column and not a token
        after the head-output field. A scores dict is treated as
        already aligned: the last step axis is used.

        Flat training outputs ``[N, A]`` are rejected unless ``N == 1``.

        Scores come from the head named by ``action_source``.
        """
        if isinstance(out, ModelOutput):
            preds = out.predictions
            valid = out.head_output_valid
        else:
            preds = out
            valid = None
        scores = _last_action_scores(
            cast(torch.Tensor, preds[self.action_source]),
            name=self.action_source,
            valid=valid,
        )
        if num_actions is not None:
            scores = scores[:, :num_actions]
        if temperature == 0.0:
            return scores.argmax(dim=-1)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = F.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def pool(self, *, output: ModelOutput) -> torch.Tensor:
        """Gather last-layer states at the rows heads read.

        ``output.last_hidden_state`` is ``[L, D]`` (train) or
        ``[B, T, D]`` (decode). ``output.head_output_indices`` selects
        the head-output tokens: ``[P]`` into the packed stream, or
        ``[B, S]`` into the decode token axis. The result is the tensor
        :meth:`head` consumes.
        """
        if not isinstance(output, ModelOutput):
            raise TypeError(
                f"pool output must be a ModelOutput, got {type(output).__name__}."
            )
        return _pool_head_outputs(output.last_hidden_state, output.head_output_indices)

    def copy(
        self,
        *,
        heads: bool | Sequence[BaseHead],
        backbone: bool,
        reasoner: bool,
        projection: bool | None = None,
    ) -> "Model":
        """Return a model that copies the named sections and shares the rest.

        ``heads``, ``backbone``, and ``reasoner`` are required.
        ``heads=True`` copies every head. ``heads=False`` attaches this
        model's heads. A sequence of head instances copies only those —
        use that to leave out a policy or behavior head whose delayed
        values nothing reads. Every instance must be one of this model's
        heads and the list must not be empty. At least one section must
        be copied. The copy's ``action_source`` is this model's when it
        is among the copied or shared heads, else the first head listed
        (the copy does not pick actions).

        ``backbone=True`` deep-copies trainable backbone weights (token
        embeddings included; a frozen LoRA base stays shared by
        reference). ``False`` attaches this model's backbone module —
        no second embed / backbone pass; run the copy's heads on
        :meth:`pool` of the online :class:`ModelOutput`.

        ``reasoner=True`` deep-copies the reasoner. Illegal when this
        model has no reasoner. ``False`` is ``None`` when unused,
        otherwise this model's reasoner module.

        When this model has a projection, ``projection=True`` copies it
        and ``projection=False`` shares it; an explicit choice is required.
        Omit ``projection`` only when the model has none. The projection
        is applied by :meth:`head`, including when reusing online states.

        Copied modules are frozen and left in ``train()``. Shared
        modules are not frozen (they belong to this model).
        :func:`~mouse_core.polyak.model_polyak` interpolates only copied
        sections. Construct after ``model.to(...)``. Do not call
        ``requires_grad_`` / ``to`` on the copy: shared frozen
        parameters (and a shared backbone / reasoner) belong to this
        model too.
        """
        if type(backbone) is not bool:
            raise TypeError(f"backbone must be a bool, got {type(backbone).__name__}.")
        if type(reasoner) is not bool:
            raise TypeError(f"reasoner must be a bool, got {type(reasoner).__name__}.")
        if projection is None:
            if self.projection is not None:
                raise ValueError("copy requires projection=True or False when the model has a projection.")
            projection = False
        if type(projection) is not bool:
            raise TypeError(f"projection must be a bool, got {type(projection).__name__}.")
        if projection and self.projection is None:
            raise ValueError("copy projection=True requires a projection on the model.")
        if reasoner and self.reasoner is None:
            raise ValueError(
                "copy reasoner=True requires a reasoner; construct "
                "Model(..., reasoner=LatentReasoner(...)) or pass reasoner=False."
            )
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError(
                "copy needs a trainable online model (no parameter requires grad)."
            )
        share_heads = False
        if heads is True:
            names = tuple(self._heads)
            if not names:
                raise ValueError("copy heads=True requires the model to have at least one head.")
        elif heads is False:
            names = tuple(self._heads)
            if not names:
                raise ValueError("copy heads=False requires the model to have at least one head.")
            share_heads = True
        elif isinstance(heads, (str, BaseHead)) or not isinstance(heads, Sequence):
            raise TypeError(
                f"copy heads must be True, False, or a sequence of head instances, "
                f"got {type(heads).__name__}."
            )
        else:
            head_list = tuple(heads)
            if not head_list:
                raise ValueError("copy heads must include at least one head.")
            names = tuple(
                Model._head_name(self._heads, head, what="copy heads")
                for head in head_list
            )
            if len(set(names)) != len(names):
                raise ValueError(f"copy heads has duplicate heads: {names}.")
        if share_heads and not backbone and not reasoner and not projection:
            raise ValueError(
                "copy must copy at least one section; pass heads=True, "
                "backbone=True, reasoner=True, or projection=True."
            )

        def _copy(module: nn.Module) -> nn.Module:
            shared = {id(p): p for p in module.parameters() if not p.requires_grad}
            delayed = copy.deepcopy(module, memo=shared)
            delayed.requires_grad_(False)
            delayed.train()
            return delayed

        if share_heads:
            copied_heads = {name: self._heads[name] for name in names}
            action_name = self.action_source
        else:
            copied_heads = {name: cast(BaseHead, _copy(self._heads[name])) for name in names}
            action_name = self.action_source if self.action_source in names else names[0]
        copied_backbone = (
            cast(Backbone, _copy(self.backbone)) if backbone else self.backbone
        )
        if self.reasoner is None:
            copied_reasoner: LatentReasoner | None = None
        elif reasoner:
            copied_reasoner = cast(LatentReasoner, _copy(self.reasoner))
        else:
            copied_reasoner = self.reasoner
        copied_projection = (
            cast(ProjectionHead, _copy(self.projection))
            if projection and self.projection is not None
            else self.projection
        )
        return Model(
            backbone=copied_backbone,
            heads=copied_heads,
            action_source=action_name,
            reasoner=copied_reasoner,
            projection=copied_projection,
        )

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move the model to a device; never casts.

        Dtypes are fixed when the pieces are built: the transformer backbone
        takes ``dtype=`` (``torch.float32`` to fine-tune the base weights,
        ``preferred_dtype(device=device)`` for a frozen LoRA base or inference) and
        every other section — LoRA adapters, Identity ``embed_tokens``,
        reasoner, projection, heads — is float32, which ``AdamW`` and
        ``model_polyak`` require of every
        trainable parameter. Inputs are cast to the backbone dtype at the
        backbone boundary and its output back to fp32 for the projection
        and heads. Passing a dtype here raises ``TypeError``.
        """
        _reject_dtype_cast("Model", *args, **kwargs)
        return super().to(*args, **kwargs)

    def half(self) -> "Model":
        raise TypeError("Model.half() is not supported; set the backbone dtype when building it.")

    def bfloat16(self) -> "Model":
        raise TypeError("Model.bfloat16() is not supported; set the backbone dtype when building it.")

    def double(self) -> "Model":
        raise TypeError("Model.double() is not supported; set the backbone dtype when building it.")

    def float(self) -> "Model":
        raise TypeError("Model.float() is not supported; set the backbone dtype when building it.")

    @staticmethod
    def _normalize_heads(
        heads: BaseHead | list[BaseHead] | Mapping[str, BaseHead | None] | None,
    ) -> dict[str, BaseHead]:
        """Convert the flexible ``heads=`` argument into the internal ``name -> head`` dict.

        Supported inputs:
          - dict (caller-chosen names to head or None): passed through.
          - single BaseHead instance: becomes the only head; name is inferred
            from type (``action_value`` / ``action``).
          - list/tuple of BaseHead: each gets an inferred name.
        """
        if heads is None:
            return {}

        # Dict form gives full control over names.
        if isinstance(heads, Mapping):
            filtered: dict[str, BaseHead] = {}
            for name, head in heads.items():
                if head is None:
                    continue
                if not isinstance(name, str) or not name:
                    raise ValueError(f"head name must be a non-empty string, got {name!r}")
                if not isinstance(head, BaseHead):
                    raise TypeError(f"head {name!r} must be a BaseHead or None, got {type(head)}")
                filtered[name] = head
            return filtered

        # Single head instance gives an implicit single-head model.
        if isinstance(heads, BaseHead):
            return {Model._infer_head_name(heads): heads}

        # List of heads → names inferred from type.
        if isinstance(heads, (list, tuple)):
            result: dict[str, BaseHead] = {}
            for head in heads:
                if not isinstance(head, BaseHead):
                    raise TypeError(f"items in heads list must be BaseHead instances, got {type(head)}")
                name = Model._infer_head_name(head)
                if name in result:
                    raise ValueError(
                        f"Multiple heads would map to the same inferred name {name!r}. "
                        "Use a dict form to provide distinct names, e.g. "
                        "heads={'action_value': h1, 'action': h2}."
                    )
                result[name] = head
            return result

        raise TypeError(
            f"heads must be a BaseHead, list[BaseHead], or dict[str, BaseHead|None], "
            f"got {type(heads)}"
        )

    @staticmethod
    def _infer_head_name(head: BaseHead) -> str:
        """Infer a default storage key from the head type when no dict key is given.

        ``RegressionHead`` → ``action_value``, ``ClassificationHead``
        → ``action``.
        """
        if isinstance(head, ClassificationHead):
            return "action"
        if isinstance(head, RegressionHead):
            return "action_value"
        raise TypeError(
            f"Cannot infer a name for head of type {type(head).__name__}. "
            "Use the dict form with an explicit key."
        )

    @staticmethod
    def _head_name(heads: Mapping[str, BaseHead], head: BaseHead, *, what: str) -> str:
        """Resolve a head instance to the storage key of that same object."""
        if not isinstance(head, BaseHead):
            raise TypeError(f"{what} must be a BaseHead instance, got {type(head).__name__}.")
        for name, existing in heads.items():
            if existing is head:
                return name
        raise ValueError(
            f"{what} is not one of the heads passed to heads=; pass the same instance."
        )

    @staticmethod
    def _require_action_source(heads: Mapping[str, BaseHead], action_source: str) -> str:
        """Require ``action_source`` to be a key in ``heads``."""
        if not isinstance(action_source, str) or not action_source:
            raise TypeError(
                f"action_source must be a non-empty string, got {action_source!r}."
            )
        if action_source not in heads:
            raise ValueError(
                f"action_source {action_source!r} is not among heads {tuple(heads)}."
            )
        return action_source

    def _train_backbone_forward(
        self,
        backbone: Backbone,
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
                self.backbone,
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


def _last_valid_step(valid: torch.Tensor) -> torch.Tensor:
    """Last True index along dim=-1. Every row must have a valid head-output."""
    if valid.ndim != 2:
        raise ValueError(
            f"head_output_valid must have shape [B, S], got {tuple(valid.shape)}"
        )
    has = valid.any(dim=-1)
    if not bool(has.all()):
        empty = (~has).nonzero(as_tuple=True)[0]
        raise ValueError(
            "get_action needs a valid head-output token in every row; "
            f"empty rows: {empty.tolist()}."
        )
    idx = valid.long() * torch.arange(valid.shape[1], device=valid.device)
    return idx.max(dim=-1).values


def _last_action_scores(
    raw: torch.Tensor,
    *,
    name: str,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Action scores at the last valid head-output of each row.

    Q / logit heads: ``[B, S, A]`` / ``[N, A]``. ``valid`` is the decode
    ``[B, S]`` mask; omitted, the last step column is used (left-padded
    decode).
    """
    if valid is not None and raw.ndim >= 3 and valid.shape != raw.shape[:2]:
        raise ValueError(
            f"{name} scores {tuple(raw.shape)} do not match "
            f"head_output_valid {tuple(valid.shape)}"
        )
    step = None if valid is None else _last_valid_step(valid)
    batch = None if step is None else torch.arange(raw.shape[0], device=raw.device)
    if raw.ndim == 3:
        if step is None:
            return raw[:, -1]
        return raw[batch, step]
    if raw.ndim == 2:
        if raw.shape[0] != 1:
            raise ValueError(
                f"{name} has shape {tuple(raw.shape)}; get_action on flat "
                "[N, A] training outputs needs N=1 (one step). Use "
                "cached-decode [B, S, A] outputs for a batch."
            )
        return raw[-1].unsqueeze(0)
    raise ValueError(f"{name} expects [B, S, A] or [N, A], got {tuple(raw.shape)}")


def preferred_dtype(*, device: torch.device | str | None = None) -> torch.dtype:
    """Dtype for a frozen backbone base: ``bfloat16`` on CUDA, else ``float32``.

    Pass as the backbone ``dtype`` (``TransformerBackbone(dtype=preferred_dtype(device=device),
    ...)`` or ``load_model(..., dtype=preferred_dtype(device=device))``) for a LoRA
    backbone (frozen base) or for inference; the CUDA flash varlen kernel
    needs bf16/fp16. Every trainable section is float32 regardless. To
    fine-tune the whole backbone, build it with ``dtype=torch.float32``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


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
