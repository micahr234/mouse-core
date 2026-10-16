"""Standard model fixtures and explicit module groups for component tests."""
import copy

import torch
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from mouse_core.models import Model, ModelOutput, TokenEncoder, select_discrete_action
from mouse_core.models.backbone import BaseBackbone
from mouse_core.models.reasoner import LatentReasoner
from mouse_core.polyak import model_polyak as bind_polyak


class ExampleModel(Model):
    """Concrete implementation owned by these tests.

    The library does not supply or reconstruct this class.
    """

    def __init__(
        self, *, encoder: TokenEncoder, heads: Mapping[str, nn.Module],
        projection: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len({id(head) for head in heads.values()}) != len(heads):
            raise ValueError("each prediction head must be a distinct module instance.")
        self.encoder = encoder
        self.projection = projection if projection is not None else nn.Identity()
        self.heads = nn.ModuleDict(heads)

    def get_action(self, *, out, temperature, num_actions=None):
        # This fixture explicitly acts from its first prediction. Real agents
        # choose their own tensors and action representation in this method.
        predictions = out.predictions if isinstance(out, ModelOutput) else out
        valid = out.head_output_valid if isinstance(out, ModelOutput) else None
        return select_discrete_action(
            scores=next(iter(predictions.values())), valid=valid,
            temperature=temperature, num_actions=num_actions,
        )

    def forward(
        self, batch, *, cache=None, use_cache=False, reasoning=None,
    ):
        encoded = self.encoder(batch, cache=cache, use_cache=use_cache, reasoning=reasoning)
        predictions = self.head(h=encoded.pool())
        return encoded.with_predictions(predictions)

    def head(self, *, h) -> dict[str, Any]:
        """Read already-selected tokens, also useful for delayed head reuse."""
        h = self.projection(h.float())
        return {name: head(h) for name, head in self.heads.items()}


def simple_model(
    *, backbone: BaseBackbone, heads: Mapping[str, nn.Module],
    reasoner: LatentReasoner | None, projection: nn.Module | None = None,
) -> ExampleModel:
    return ExampleModel(
        encoder=TokenEncoder([backbone], reasoner=reasoner),
        heads=heads, projection=projection,
    )


def copy_model(
    model: ExampleModel, *, backbone: bool, heads: bool | Sequence[nn.Module],
    reasoner: bool, projection: bool = False,
) -> ExampleModel:
    selected = list(model.heads) if isinstance(heads, bool) else [
        next(name for name, existing in model.heads.items() if existing is head) for head in heads
    ]
    modules = (["encoder.backbones.0"] if backbone else []) + (["projection"] if projection else [])
    if reasoner:
        modules.append("encoder.reasoner")
    if heads is not False:
        modules.extend(f"heads.{name}" for name in selected)
    delayed = model.copy(modules=modules)
    # Output selection is explicit Python assembly, never inferred from forward.
    delayed.heads = nn.ModuleDict({name: delayed.heads[name] for name in selected})
    return delayed


def node_rates(
    model: ExampleModel, *, backbone: float | None,
    heads: float | Mapping[str, float | None] | None, projection: float | None = None,
    reasoner: float | None = None,
) -> dict[str, Any]:
    rates: dict[str, Any] = {"encoder.backbones.0": backbone}
    if not isinstance(model.projection, nn.Identity):
        rates["projection"] = projection
    if isinstance(heads, Mapping):
        rates.update({f"heads.{name}": rate for name, rate in heads.items()})
    else:
        rates.update({f"heads.{name}": heads for name in model.heads})
    if model.encoder.reasoner is not None:
        rates["encoder.reasoner"] = reasoner
    return rates


def model_polyak(*, online: ExampleModel, delayed: ExampleModel):
    if not isinstance(delayed, ExampleModel):
        return bind_polyak(online=online, delayed=delayed)
    return bind_polyak(online=online, delayed=delayed,
                       modules=tuple(node_rates(delayed, backbone=None, heads=None)))


def fresh_model[T: Model](model: T) -> T:
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in fresh.parameters():
            parameter.zero_()
    return fresh
