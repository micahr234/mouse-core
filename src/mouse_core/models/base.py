"""A PyTorch model base with optimizer grouping and delayed copies."""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Self

import torch
from torch import nn

from mouse_core.models.backbone.base import _reject_dtype_cast


class Model(nn.Module, ABC):
    """Subclass and define the computation in ``forward`` using ordinary PyTorch.

    Register modules as attributes, ModuleLists, or Sequentials. There is no
    graph specification: Python defines the flow. LR groups use the
    same dotted module paths as ``get_submodule`` and ``named_parameters``.
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define the model's computation in the implementation subclass."""
        ...

    def parameter_groups(self, *, lr: Mapping[str, float]) -> list[dict[str, Any]]:
        """Give each trainable parameter exactly one module's learning rate."""
        modules = _module_groups(self, tuple(lr), cover_trainable=True)
        return [{"params": [p for p in module.parameters() if p.requires_grad], "lr": lr[name]}
                for name, module in modules.items()]

    def copy(self, *, modules: Sequence[str] | None = None) -> Self:
        """Preserve this subclass, copying selected modules and sharing the rest.

        Omit ``modules`` to copy the whole model. Copied trainable parameters
        become frozen; already frozen parameters (such as LoRA bases) stay
        shared. Python forward code is preserved, so outputs are not pruned.
        """
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError("copy needs a trainable online model (no parameter requires grad).")
        selected = _module_groups(self, modules)
        copied_ids = {id(child) for module in selected.values() for child in module.modules()}
        memo: dict[int, Any] = {id(p): p for p in self.parameters() if not p.requires_grad}
        if modules is not None:
            ancestors = {id(self)}
            for name in selected:
                parts = name.split('.')
                ancestors.update(id(self.get_submodule('.'.join(parts[:i]))) for i in range(1, len(parts)))
            memo.update({id(module): module for module in self.modules()
                         if id(module) not in copied_ids | ancestors})
            # Parameters directly on an ancestor also belong to the shared part.
            copied_params = {id(p) for module in selected.values() for p in module.parameters()}
            memo.update({id(p): p for p in self.parameters() if id(p) not in copied_params})
        delayed = copy.deepcopy(self, memo)
        for name, module in selected.items():
            target = delayed.get_submodule(name)
            source_ids = {id(param) for param in module.parameters()}
            for param in target.parameters():
                if id(param) not in source_ids:
                    param.requires_grad_(False)
        return delayed

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move devices; configure precision when constructing the backbones."""
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

def _module_groups(
    model: nn.Module, names: Sequence[str] | None, *, cover_trainable: bool = False,
) -> dict[str, nn.Module]:
    """Resolve disjoint parameter groups through standard PyTorch module paths."""
    if names is None:
        return {"": model}
    if isinstance(names, str) or not names or len(names) != len(set(names)):
        raise ValueError("modules must be a non-empty sequence of distinct module paths.")
    groups: dict[str, nn.Module] = {}
    seen: set[int] = set()
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError("module paths must be non-empty strings.")
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError(f"unknown module path {name!r}.") from exc
        ids = {id(p) for p in module.parameters()}
        if not ids:
            raise ValueError(f"module {name!r} has no parameters.")
        if seen & ids:
            raise ValueError("module groups must not overlap or share parameters.")
        seen.update(ids)
        groups[name] = module
    if cover_trainable and not {id(p) for p in model.parameters() if p.requires_grad} <= seen:
        raise ValueError("module groups must cover every trainable parameter.")
    return groups


def preferred_dtype(*, device: torch.device | str | None = None) -> torch.dtype:
    """Dtype for a frozen backbone base: ``bfloat16`` on CUDA, else ``float32``.

    Pass as the backbone ``dtype`` (``Backbone(dtype=preferred_dtype(device=device),
    ...)``) for a LoRA
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
