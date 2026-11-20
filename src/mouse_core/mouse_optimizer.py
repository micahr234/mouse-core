"""One optimizer step for online parameters and their delayed counterparts."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, sqrt
from numbers import Integral, Real
from typing import Any

import torch
from torch import nn

from mouse_core.models.lora import backbone_parameter_view
from mouse_core.optim import AdamW
from mouse_core.rounding import polyak_parameter_
from mouse_core.precision_optim import PrecisionOptimizer


def _rate(value: Any, *, name: str, maximum: float = float("inf")) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= float(value) <= maximum:
        raise ValueError(f"{name} must be finite and in [0, {maximum}].")
    return float(value)


def _make_optimizer(params: list[dict[str, Any]], specification: Any, *, dtype=torch.float32, stochastic_rounding=False):
    if not isinstance(specification, (str, Mapping)):
        raise TypeError("optimizer must be a name or an options dictionary.")
    options: dict[str, Any] = {"name": specification} if isinstance(specification, str) else dict(specification)
    name = options.pop("name", None)
    if name in ("adam_fused", "adamw_fused", "sgd_fused"):
        name = name.removesuffix("_fused")
        if "fused" in options and options["fused"] is not True:
            raise ValueError("A fused optimizer name requires fused=True.")
        options["fused"] = True
    allowed = {
        "adam": {"betas", "eps", "fused"},
        "adamw": {"betas", "eps", "fused"},
        "sgd": {"momentum", "fused"},
        "rmsprop": {"alpha", "eps", "momentum", "centered"},
    }
    if name not in allowed:
        raise ValueError(f"Unknown optimizer {name!r}; choose adam, adamw, sgd, or rmsprop.")
    allowed[name].add("weight_decay")
    if options.keys() - allowed[name]:
        raise ValueError(f"Unsupported {name} optimizer options: {sorted(options.keys() - allowed[name])}")
    for key, value in options.items():
        if key == "betas":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("betas must contain two rates in [0, 1).")
            for beta in value:
                if _rate(beta, name="beta", maximum=1) == 1:
                    raise ValueError("betas must be in [0, 1).")
        elif key == "fused":
            if value is not None and type(value) is not bool:
                raise ValueError("fused must be a boolean or None.")
        elif key == "centered":
            if type(value) is not bool:
                raise ValueError("centered must be a boolean.")
        else:
            number = _rate(value, name=key, maximum=1 if key == "alpha" else float("inf"))
            if (key == "eps" and number == 0) or (key == "alpha" and number == 1):
                raise ValueError("eps must be positive and alpha must be in [0, 1).")
    if name in ("adam", "adamw"):
        options.setdefault("betas", (0.9, 0.95))
        options.setdefault("eps", 1e-8)
    if stochastic_rounding or dtype != torch.float32:
        return PrecisionOptimizer(params, name=name, options=options, dtype=dtype,
                                  stochastic_rounding=stochastic_rounding)
    if name == "adamw":
        return AdamW(params=params, **options)
    if name == "adam":
        return torch.optim.Adam(params, **options)
    if name == "sgd":
        return torch.optim.SGD(params, **options)
    return torch.optim.RMSprop(params, **options)


class MouseOptimizer:
    """One selectable optimizer and explicit online/target groups.

    ``groups`` maps labels to settings: an ``online`` module, ``lr`` for trainable
    groups, and optional ``target`` plus ``polyak_tau`` (provided together).
    ``parameters="base"`` selects non-LoRA weights; ``parameters="lora"`` selects
    LoRA factors on both online and target modules. Omit it to select all weights.
    Empty and fully frozen groups are discarded during construction, so callers
    need no filtering loop. Known frozen groups can simply be omitted.
    ``dtype`` selects both optimizer arithmetic and moment storage precision.
    ``stochastic_rounding`` controls rounding when writing online weights and
    delayed Polyak targets, without master weights.
    ``langevin_noise_beta`` is a nonnegative number or batch-index schedule.
    When positive, ``step`` requires ``online_transition_count`` and adds noise
    to raw gradients with standard deviation ``beta * sqrt(2 / (lr * count))``
    before moment updates. Noise uses optimizer arithmetic precision. Zero-rate
    groups receive no noise. With SGD without momentum this has SGLD scaling;
    with adaptive optimizers it is gradient noise, not preconditioned SGLD.
    Online and target modules live in the groups; labels are not module paths.
    Online groups must be disjoint. Trainability follows ``requires_grad``.

    ``optimizer`` selects the single inner optimizer for all groups: AdamW,
    Adam, SGD or RMSprop. A dictionary supplies ``name`` and global options
    such as weight_decay, betas, eps or momentum. These settings cannot be
    overridden per group; only learning rates and Polyak rates vary by group.

    Learning rates and taus may be numbers or zero-based training-batch schedules.
    A tau may map parameter names to scalar/broadcast tensor rates. All rates
    and pairs are validated before the online update. Targets follow the final
    online weights. Frozen source weights, shared parameters and buffers stay put.
    """

    def __init__(
        self, *, groups: Mapping[str, Mapping[str, Any]], optimizer: Any = "adamw",
        dtype: torch.dtype = torch.float32, stochastic_rounding: bool = False,
        langevin_noise_beta: Any = 0.0,
    ) -> None:
        if type(stochastic_rounding) is not bool:
            raise TypeError("Stochastic rounding options must be booleans")
        if dtype not in (torch.float64, torch.float32, torch.bfloat16):
            raise ValueError("Optimizer dtype must be float64, float32 or bfloat16")
        if not callable(langevin_noise_beta):
            _rate(langevin_noise_beta, name="langevin_noise_beta")
        self.langevin_noise_beta = langevin_noise_beta
        self.dtype = dtype
        self.stochastic_rounding = stochastic_rounding
        if not groups:
            raise ValueError("MouseOptimizer requires at least one named group.")
        self.groups: dict[str, dict[str, Any]] = {}
        self._sources: dict[str, nn.Module] = {}
        self._parameters: dict[str, dict[str, nn.Parameter]] = {}
        self._ignored_sources: list[nn.Module] = []
        online_ids: set[int] = set()
        for name, settings in groups.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Optimizer group names must be nonempty strings.")
            if (not isinstance(settings, Mapping) or "online" not in settings
                    or settings.keys() - {"online", "lr", "target", "polyak_tau", "parameters"}):
                raise ValueError(f"{name}: groups require online; supported settings are lr, target, polyak_tau and parameters.")
            settings = dict(settings)
            source = settings["online"]
            if not isinstance(source, nn.Module):
                raise TypeError(f"{name}: online must be a module.")
            selection = settings.pop("parameters", "all")
            if selection not in ("all", "base", "lora"):
                raise ValueError(f"{name}: parameters must be all, base or lora.")
            if selection != "all":
                source = backbone_parameter_view(source, adapters=selection == "lora")
                settings["online"] = source
            parameters = dict(source.named_parameters())
            if not any(p.requires_grad for p in parameters.values()):
                self._ignored_sources.append(source)
                continue
            if "lr" not in settings or settings["lr"] is None:
                raise ValueError(f"{name}: a learning rate is required for a trainable group.")
            target = settings.get("target")
            if target is not None and selection != "all":
                if not isinstance(target, nn.Module):
                    raise TypeError(f"{name}: target must be a module.")
                settings["target"] = backbone_parameter_view(target, adapters=selection == "lora")
            ids = {id(p) for p in parameters.values()}
            if ids & online_ids:
                raise ValueError("Online groups must not overlap or share parameters.")
            online_ids.update(ids)
            self._sources[name], self._parameters[name] = source, parameters
            self.groups[name] = settings
        if not self.groups:
            raise ValueError("MouseOptimizer requires at least one trainable parameter.")
        self._trainable = {id(p) for parameters in self._parameters.values() for p in parameters.values() if p.requires_grad}
        targets_seen: set[int] = set()
        for name, settings in self.groups.items():
            target = settings.get("target")
            if (target is None) != (settings.get("polyak_tau") is None):
                raise ValueError(f"{name}: target and polyak_tau must be provided together.")
            if target is not None:
                if not isinstance(target, nn.Module):
                    raise TypeError(f"{name}: target must be a module.")
                for key, p in target.named_parameters():
                    source = self._parameters[name].get(key)
                    if p is source:
                        continue
                    if id(p) in targets_seen or id(p) in online_ids:
                        raise ValueError("Target groups must not overlap each other or other online parameters.")
                    if p.requires_grad:
                        raise ValueError(f"{name}.{key}: independent target parameters must be frozen.")
                    targets_seen.add(id(p))
        rates, _ = self._prepare_step(0)
        params = [dict(params=[p for p in self._parameters[name].values() if p.requires_grad], lr=rates[name])
                  for name in self.groups]
        allowed_dtypes = (torch.float32, torch.bfloat16) if stochastic_rounding else (torch.float32,)
        if any(p.dtype not in allowed_dtypes for group in params for p in group["params"]):
            raise TypeError("MouseOptimizer trains fp32 parameters unless stochastic_rounding enables BF16.")
        self.inner = self._build_optimizer(params, optimizer)
        # Implementations may split groups (e.g. packed-head rates), but must retain
        # every trainable online parameter exactly once and no target weights.
        registered = [id(p) for group in self.inner.param_groups for p in group["params"]]
        if len(registered) != len(set(registered)) or set(registered) != self._trainable:
            raise ValueError("Inner optimizer must cover every trainable online parameter exactly once.")
        owners = {id(p): name for name, parameters in self._parameters.items() for p in parameters.values()}
        self._inner_groups: dict[str, list[int]] = {name: [] for name in self.groups}
        self._lr_factors: dict[str, list[float]] = {name: [] for name in self.groups}
        for index, group in enumerate(self.inner.param_groups):
            names = {owners[id(p)] for p in group["params"]}
            if len(names) > 1:
                raise ValueError("Inner optimizer must not merge distinct named groups.")
            if names:
                name = names.pop()
                actual_lr = _rate(group["lr"], name=f"{name} inner lr")
                if rates[name] == 0 and actual_lr != 0:
                    raise ValueError("An inner optimizer cannot override a zero group learning rate.")
                group["name"] = name
                self._inner_groups[name].append(index)
                self._lr_factors[name].append(actual_lr / rates[name] if rates[name] else 1.0)
        for name, indices in self._inner_groups.items():
            if rates[name] == 0 and callable(self.groups[name]["lr"]) and len(indices) > 1:
                raise ValueError("Split groups need a nonzero initial lr to determine their schedule multipliers.")

    def _build_optimizer(self, params, optimizer):
        """Construct the single optimizer; subclasses may split packed readouts."""
        return _make_optimizer(params, optimizer, dtype=self.dtype, stochastic_rounding=self.stochastic_rounding)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.inner.param_groups

    def _prepare_step(self, step: int):
        rates, pairs = {}, []
        for name, settings in self.groups.items():
            lr = settings["lr"]
            rates[name] = _rate(lr(step) if callable(lr) else lr, name=f"{name} lr")
            source = dict(self._sources[name].named_parameters())
            if source.keys() != self._parameters[name].keys() or any(
                p is not self._parameters[name][key] for key, p in source.items()
            ):
                raise ValueError("Online parameters changed after optimizer construction.")
            target = settings.get("target")
            if target is None:
                continue
            targets = dict(target.named_parameters())
            if source.keys() != targets.keys():
                raise ValueError(f"{name}: Polyak modules must have matching parameter names.")
            tau = settings["polyak_tau"]
            tau = tau(step) if callable(tau) else tau
            if isinstance(tau, Mapping) and tau.keys() != source.keys():
                raise ValueError(f"{name}: Polyak rate mapping must cover every parameter.")
            for key, p in source.items():
                delayed = targets[key]
                if (p.shape != delayed.shape or p.dtype != delayed.dtype or p.device != delayed.device
                        or not p.is_floating_point() or p.layout != torch.strided
                        or delayed.layout != torch.strided):
                    raise ValueError(f"{name}.{key}: Polyak pairs need matching shape, dtype, device and dense floating tensors.")
                value = tau[key] if isinstance(tau, Mapping) else tau
                if isinstance(value, torch.Tensor):
                    if (value.requires_grad or not value.is_floating_point()
                            or not bool(torch.isfinite(value).all()) or not bool(((value >= 0) & (value <= 1)).all())):
                        raise ValueError("Polyak rate tensor must be finite and in [0, 1], without gradients.")
                    if torch.broadcast_shapes(p.shape, value.shape) != p.shape:
                        raise ValueError("Polyak rate tensor must broadcast to its parameter shape.")
                    value = value.to(device=p.device, dtype=self.dtype)
                else:
                    value = _rate(value, name="Polyak rate", maximum=1.0)
                if p.requires_grad and delayed is not p:
                    if p.dtype == torch.bfloat16 and not self.stochastic_rounding:
                        raise ValueError("BF16 delayed parameters require stochastic_rounding")
                    if delayed.requires_grad:
                        raise ValueError("Independent target parameters must be frozen.")
                    pairs.append((p, delayed, value))
        return rates, pairs

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def _needs_update_transform(self):
        """Whether the subclass needs a pre-rounding update transform."""
        return False

    def _transform_update(self, parameter, update):
        """Hook for a subclass to transform an update before precision rounding."""

    def _transform_noise(self, parameter, noise):
        """Hook for subclass-specific effective learning rates; mutate noise."""

    def _step_parameters(self):
        """Apply the native optimizer update using the prepared gradients."""
        self.inner.step()

    def _step_online(self, step: int, *, online_transition_count: int | None = None, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"Unsupported optimizer step arguments: {sorted(kwargs)}")
        beta = self.langevin_noise_beta
        beta = _rate(beta(step) if callable(beta) else beta, name="langevin_noise_beta")
        scales = {}
        if beta:
            if (isinstance(online_transition_count, bool)
                    or not isinstance(online_transition_count, Integral)
                    or online_transition_count <= 0):
                raise ValueError("Langevin noise requires a positive online_transition_count")
            for group in self.param_groups:
                lr = group["lr"]
                scale = beta * sqrt((2 / lr) / online_transition_count) if lr else 0.
                if not isfinite(scale):
                    raise ValueError("Langevin gradient noise standard deviation must be finite")
                scales.update((id(p), scale) for p in group["params"])
        noisy = any(scales.values())
        if noisy and any(p.grad is not None and p.grad.is_sparse
                         for group in self.param_groups for p in group["params"]):
            raise RuntimeError("Langevin noise requires dense gradients")

        def add_noise(parameter, gradient):
            scale = scales.get(id(parameter), 0.)
            if scale:
                noise = torch.randn_like(gradient)
                self._transform_noise(parameter, noise)
                gradient.add_(noise, alpha=scale)

        if isinstance(self.inner, PrecisionOptimizer):
            # Inject after casting to arithmetic dtype, before moments or updates.
            # The final model write remains the only stochastic rounding boundary.
            self.inner.gradient_transform = add_noise if noisy else None
            self.inner.update_transform = self._transform_update if self._needs_update_transform() else None
            try:
                self.inner.step()
            finally:
                self.inner.gradient_transform = None
                self.inner.update_transform = None
        else:
            previous = []
            try:
                if noisy:
                    for group in self.param_groups:
                        for parameter in group["params"]:
                            if (parameter.requires_grad and parameter.grad is not None
                                    and scales.get(id(parameter), 0.)):
                                gradient = parameter.grad.clone()
                                add_noise(parameter, gradient)
                                previous.append((parameter, parameter.grad))
                                parameter.grad = gradient
                self._step_parameters()
            finally:
                # Do not accumulate random noise into retained batch gradients.
                for parameter, gradient in previous:
                    parameter.grad = gradient

    @torch.no_grad()
    def step(self, step: int, **kwargs: Any) -> None:
        """Apply one update at the explicit zero-based training-batch index.

        With gradient accumulation, pass the index of the last batch in the
        group. The caller owns batch counting, evaluation and accumulation;
        schedules never infer their position from the number of updates.
        """
        if type(step) is not int or step < 0:
            raise ValueError("training step must be a nonnegative integer")
        if ({id(p) for module in self._sources.values() for p in module.parameters() if p.requires_grad} != self._trainable
                or any(p.requires_grad for module in self._ignored_sources for p in module.parameters())):
            raise ValueError("Trainability changed; rebuild the optimizer after changing requires_grad.")
        rates, pairs = self._prepare_step(step)
        resolved = [(index, _rate(rates[name] * factor, name=f"{name} effective lr"))
                    for name, entries in self._inner_groups.items()
                    for index, factor in zip(entries, self._lr_factors[name], strict=True)]
        for index, rate in resolved:
            self.inner.param_groups[index]["lr"] = rate
        self._step_online(step, **kwargs)
        self._step_targets(pairs)

    def _step_targets(self, pairs) -> None:
        for source, target, tau in pairs:
            polyak_parameter_(target, source, tau, stochastic_rounding=self.stochastic_rounding,
                              arithmetic_dtype=self.dtype)

    def state_dict(self) -> dict[str, Any]:
        native = isinstance(self.inner, torch.optim.Optimizer)
        state = ({"algorithm": type(self.inner).__name__.lower(), "inner": self.inner.state_dict()}
                 if native else self.inner.state_dict())
        return {**state, "groups": self._group_signature(),
                "precision": (str(self.dtype), self.stochastic_rounding)}

    def _group_signature(self):
        return [(name, list(parameters), self.groups[name].get("target") is not None)
                for name, parameters in self._parameters.items()]

    def load_state_dict(self, state: dict[str, Any]) -> None:
        native = isinstance(self.inner, torch.optim.Optimizer)
        algorithm = type(self.inner).__name__.lower() if native else None
        if state.get("algorithm") != algorithm:
            raise ValueError("Different optimizer states are not interchangeable")
        if state.get("precision") != (str(self.dtype), self.stochastic_rounding):
            raise ValueError("Optimizer checkpoint precision settings do not match")
        expected = {"groups", "inner", "algorithm", "precision"} if native else {"groups", "inner", "precision"}
        if state.keys() != expected:
            raise ValueError("Optimizer checkpoint fields do not match the current format.")
        if state["groups"] != self._group_signature():
            raise ValueError("Optimizer checkpoint parameter groups do not match.")
        self.inner.load_state_dict(state["inner"] if native else {"inner": state["inner"]})

    def state_to(self, device: torch.device | str) -> None:
        """Move optimizer moments without moving online or target parameters."""
        optimizer = self.inner if isinstance(self.inner, torch.optim.Optimizer) else self.inner._inner
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    destination = "cpu" if isinstance(optimizer, PrecisionOptimizer) and key == "step" else device
                    state[key] = value.to(destination)
