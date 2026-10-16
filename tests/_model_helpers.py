"""Standard model fixtures and explicit module groups for component tests."""
import copy

import torch
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from mouse_core.models import Model, TokenBatch, Embedding, DecodeCache, prepare_decode, select_discrete_action
from mouse_core.models.backbone import BaseBackbone
from mouse_core.models.reasoner import LatentReasoner


class ExampleModel(Model):
    """Concrete implementation owned by these tests.

    The library does not supply or reconstruct this class.
    """

    def __init__(
        self, *, embedding: Embedding, backbone: BaseBackbone, heads: Mapping[str, nn.Module],
        reasoner: LatentReasoner | None = None,
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len({id(head) for head in heads.values()}) != len(heads):
            raise ValueError("each prediction head must be a distinct module instance.")
        self.embedding = embedding
        self.backbone = backbone
        self.reasoner = reasoner
        self._backbone_has_parameters = next(backbone.parameters(), None) is not None
        self.projection = projection if projection is not None else nn.Identity()
        self.heads = nn.ModuleDict(heads)

    def get_action(self, *, out, temperature, num_actions=None):
        # This fixture explicitly acts from its first prediction. Real agents
        # choose their own tensors and action representation in this method.
        predictions = out["predictions"]
        valid = out.get("head_valid_mask")
        return select_discrete_action(
            scores=next(iter(predictions.values())), valid=valid,
            temperature=temperature, num_actions=num_actions,
        )

    def forward(
        self, batch, *, cache=None, use_cache=False, reasoning=None,
    ) -> dict[str, Any]:
        if not isinstance(batch, TokenBatch):
            raise TypeError("ExampleModel.forward expects a TokenBatch.")
        if cache is not None and not use_cache:
            raise ValueError("Passing cache= requires use_cache=True.")
        tensors = batch.to_tensors(next(self.parameters()).device)
        token_ids = tensors["token_ids"]
        numeric_values = tensors["numeric_values"]
        numeric_mask = tensors["numeric_mask"]
        sequence_ids = tensors["sequence_ids"]
        head_token_ids = tensors["head_token_ids"]
        head_step_ids = tensors["head_step_ids"]
        embed = self.embedding(
            token_ids, numeric_values=numeric_values, numeric_mask=numeric_mask,
        )
        head_valid_mask = None
        if reasoning is not None:
            if use_cache:
                raise ValueError("reasoning= is not supported with use_cache=True.")
            if self.reasoner is None:
                raise ValueError("reasoning= requires a reasoner.")
            embed, sequence_ids, head_token_ids = self.reasoner.insert(
                embeds=embed, tokens=batch, sequence_ids=sequence_ids,
                head_token_ids=head_token_ids,
                splits=reasoning, backbone=self.backbone,
            )
        if use_cache:
            embed, lengths, head_token_ids, head_valid_mask = prepare_decode(
                embeds=embed, tokens=batch, sequence_ids=sequence_ids,
                head_token_ids=head_token_ids, head_step_ids=head_step_ids,
            )
            if cache is None:
                cache = DecodeCache(sessions=(self.backbone.decode_session(batch_size=batch.B),))
            if len(cache.sessions) != 1:
                raise ValueError("cache must contain one session per backbone stage.")
            features = self.backbone(embed, cache=cache.sessions[0], lengths=lengths)
            h = features.gather(1, head_token_ids.unsqueeze(-1).expand(*head_token_ids.shape, features.shape[-1]))
        else:
            features = self.backbone(embed, sequence_ids=sequence_ids)
            h = features[head_token_ids]
        return {
            "predictions": self.head(h=h),
            "last_hidden_state": features,
            "head_token_ids": head_token_ids,
            "head_valid_mask": head_valid_mask,
            "cache": cache,
        }

    def head(self, *, h) -> dict[str, Any]:
        """Read already-selected tokens, also useful for delayed head reuse."""
        h = self.projection(h.float())
        return {name: head(h) for name, head in self.heads.items()}


def simple_model(
    *, backbone: BaseBackbone, heads: Mapping[str, nn.Module],
    reasoner: LatentReasoner | None, projection: nn.Module | None = None,
    embedding: Embedding | None = None,
) -> ExampleModel:
    return ExampleModel(
        embedding=embedding if embedding is not None else Embedding(hidden_dim=backbone.hidden_dim, vocab_size=32), backbone=backbone, reasoner=reasoner,
        heads=heads, projection=projection,
    )


def copy_model(
    model: ExampleModel, *, backbone: bool, heads: bool | Sequence[nn.Module],
    reasoner: bool, projection: bool = False,
) -> ExampleModel:
    selected = list(model.heads) if isinstance(heads, bool) else [
        next(name for name, existing in model.heads.items() if existing is head) for head in heads
    ]
    modules = (["embedding"] if backbone else []) + (["backbone"] if backbone and model._backbone_has_parameters else []) + (["projection"] if projection else [])
    if reasoner:
        modules.append("reasoner")
    if heads is not False:
        modules.extend(f"heads.{name}" for name in selected)
    delayed = model.copy(modules=modules)
    # Output selection is explicit Python assembly, never inferred from forward.
    delayed.heads = nn.ModuleDict({name: delayed.heads[name] for name in selected})
    return delayed


def fresh_model[T: Model](model: T) -> T:
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in fresh.parameters():
            parameter.zero_()
    return fresh
