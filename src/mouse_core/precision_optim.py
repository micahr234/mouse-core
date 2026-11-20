"""Explicit optimizer arithmetic/state precision and rounding."""

from collections.abc import Callable
from typing import Any

import torch

from mouse_core.rounding import copy_update_, scalar_tensor


def _rmsprop_update_(parameter, gradient, square_avg, grad_avg, momentum_buffer,
                     *, lr, alpha, eps, weight_decay, momentum, stochastic_rounding=True):
    """Native single-tensor RMSprop math, with just one parameter writeback.

    CUDA compilation fuses the selected-precision temporaries, moment updates, and stochastic
    rounding. No full-size FP32 parameter or gradient buffers need to survive
    the kernel. CPU runs the same expression eagerly as a numerical reference.
    """
    value = parameter.to(square_avg.dtype)
    grad = gradient.to(square_avg.dtype)
    if weight_decay != 0:
        grad = grad.add(value, alpha=weight_decay)
    square_value = square_avg
    square_value.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
    if grad_avg is not None:
        grad_value = grad_avg
        grad_value.lerp_(grad, 1 - alpha)
        average = square_value.addcmul(grad_value, grad_value, value=-1).sqrt_()
    else:
        average = square_value.sqrt()
    average.add_(eps)
    if momentum_buffer is not None:
        momentum_value = momentum_buffer
        momentum_value.mul_(momentum).addcdiv_(grad, average)
        value.add_(momentum_value, alpha=-lr)
    else:
        value.addcdiv_(grad, average, value=-lr)
    copy_update_(parameter, value, stochastic_rounding=stochastic_rounding)


_compiled_rmsprop_update = None


def _rmsprop_kernel(device):
    global _compiled_rmsprop_update
    if device.type != "cuda":
        return _rmsprop_update_
    if _compiled_rmsprop_update is None:
        # Stochastic rounding requires actual narrowed neighbors, not elided casts.
        _compiled_rmsprop_update = torch.compile(
            _rmsprop_update_, fullgraph=True, dynamic=True,
            options={"triton.cudagraphs": False, "emulate_precision_casts": True},
        )
    return _compiled_rmsprop_update


def _rmsprop_step_(updates, *, stochastic_rounding):
    """One graph for heterogeneous tensors and groups, without master weights."""
    for parameter, gradient, square_avg, grad_avg, momentum_buffer, options in updates:
        _rmsprop_update_(parameter, gradient, square_avg, grad_avg, momentum_buffer,
                         stochastic_rounding=stochastic_rounding, **options)


def _native_step_(updates, algorithm, dtype, stochastic_rounding, update_transform):
    """Compile native optimizer equations and model writeback in one graph.

    Arithmetic-precision parameter values are temporaries, never cached masters.
    Native functional updates preserve the optimizer's state layout and rules.
    """
    functional = torch.optim._functional
    for parameter, gradient, state, options in updates:
        value = parameter.to(dtype, copy=True)
        grad = gradient.to(dtype)
        if algorithm in ("adam", "adamw"):
            # Native Adam normally extracts this CPU counter as a Python float.
            # Under compile it stays a tensor: preserve FP64 bias-correction math.
            step = state["step"].to(torch.float64) if dtype == torch.float64 else state["step"]
            functional.adam(
                [value], [grad], [state["exp_avg"]], [state["exp_avg_sq"]],
                [state["max_exp_avg_sq"]] if options["amsgrad"] else [], [step],
                foreach=False, capturable=False, differentiable=False, fused=False,
                amsgrad=options["amsgrad"], beta1=options["betas"][0], beta2=options["betas"][1],
                lr=options["lr"], weight_decay=options["weight_decay"], eps=options["eps"],
                maximize=options["maximize"], decoupled_weight_decay=algorithm == "adamw",
            )
            if dtype == torch.float64:
                state["step"].copy_(step)
        elif algorithm == "sgd":
            buffers = [state.get("momentum_buffer")]
            functional.sgd(
                [value], [grad], buffers, foreach=False, fused=False, has_sparse_grad=False,
                weight_decay=options["weight_decay"], momentum=options["momentum"],
                lr=options["lr"], dampening=options["dampening"], nesterov=options["nesterov"],
                maximize=options["maximize"],
            )
            if options["momentum"]:
                state["momentum_buffer"] = buffers[0]
        else:
            functional.rmsprop(
                [value], [grad], [state["square_avg"]],
                [state["grad_avg"]] if options["centered"] else [],
                [state["momentum_buffer"]] if options["momentum"] else [], [state["step"]],
                foreach=False, capturable=False, differentiable=False,
                lr=options["lr"], alpha=options["alpha"], eps=options["eps"],
                weight_decay=options["weight_decay"], momentum=options["momentum"],
                centered=options["centered"], maximize=options["maximize"],
            )
        if update_transform is not None:
            update_transform(parameter, value)
        copy_update_(parameter, value, stochastic_rounding=stochastic_rounding)


class PrecisionOptimizer(torch.optim.Optimizer):
    """Selectable FP64/FP32/BF16 arithmetic and state, without master weights.

    Groups with ``compile=True`` compile FP32/FP64 RMSprop CUDA updates
    together when no transforms are active. False uses eager updates.
    Transformed gradients retain per-tensor dispatch to bound temporary storage.
    Other algorithms use native PyTorch functional math inside a compiled graph
    when enabled; disabled groups delegate to the eager native optimizer.
    Moment storage follows dtype; compiled kernels cache code, never weights.
    ``update_transform`` optionally modifies the result in optimizer dtype before its single
    writeback (for packed-head rate multipliers). ``gradient_transform`` modifies
    a temporary gradient in arithmetic dtype before optimizer moments and updates.
    """

    def __init__(self, params, *, name, options, dtype=torch.float32, stochastic_rounding=True):
        if options.get("fused"):
            raise ValueError("Explicit optimizer precision or stochastic rounding does not support fused optimizer variants")
        classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                   "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
        if dtype not in (torch.float64, torch.float32, torch.bfloat16):
            raise ValueError("Optimizer dtype must be float64, float32 or bfloat16")
        if type(stochastic_rounding) is not bool:
            raise TypeError("stochastic_rounding must be a boolean")
        self.dtype = dtype
        self.stochastic_rounding = stochastic_rounding
        self.algorithm = name
        self._compiled_rmsprop_step = None
        self._compiled_native_step = None
        self._kernel = classes[name]([{"params": [], "lr": 0.}], **options, foreach=False)
        super().__init__(params, {**self._kernel.defaults, "compile": False})
        self._kernel.param_groups = []
        self.gradient_transform: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        self.update_transform: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        for group in self.param_groups:
            if type(group["compile"]) is not bool:
                raise TypeError("compile must be a boolean")
            for p in group["params"]:
                if p.dtype not in (torch.float64, torch.float32, torch.bfloat16):
                    raise TypeError("Precision optimizer supports FP64, FP32 and BF16 parameters only")

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate before mutating any parameter or optimizer state.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and p.grad.is_sparse:
                    raise RuntimeError("Stochastic optimizer requires dense gradients")
        cuda_updates = []
        native_updates = []
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                gradient = p.grad
                if self.gradient_transform is not None:
                    gradient = gradient.detach().to(self.dtype, copy=True)
                    self.gradient_transform(p, gradient)
                if self.dtype in (torch.float32, torch.float64) and self.algorithm == "rmsprop" and self.update_transform is None:
                    state = self.state[p]
                    if not state:
                        state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                        state["square_avg"] = torch.zeros_like(p, dtype=self.dtype)
                        if group["momentum"] > 0:
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=self.dtype)
                        if group["centered"]:
                            state["grad_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["step"].add_(1)
                    # Detached aliases share storage and leave live parameters and
                    # their backward graphs intact. Only scalar metadata is cached.
                    options = dict(
                        lr=scalar_tensor(group["lr"], self.dtype) if p.is_cuda else group["lr"],
                        alpha=group["alpha"], eps=group["eps"],
                        weight_decay=group["weight_decay"], momentum=group["momentum"],
                    )
                    update_args = (p.detach(), gradient, state["square_avg"],
                                   state.get("grad_avg"), state.get("momentum_buffer"))
                    if group["compile"] and p.is_cuda and self.gradient_transform is None:
                        cuda_updates.append((*update_args, options))
                    else:
                        kernel = _rmsprop_kernel(p.device) if group["compile"] else _rmsprop_update_
                        kernel(
                            *update_args, **options, stochastic_rounding=self.stochastic_rounding,
                        )
                    continue
                if group["compile"] and p.is_cuda:
                    state = self.state[p]
                    if not state:
                        if self.algorithm in ("adam", "adamw"):
                            state.update(step=torch.zeros((), dtype=torch.float32),
                                         exp_avg=torch.zeros_like(p, dtype=self.dtype),
                                         exp_avg_sq=torch.zeros_like(p, dtype=self.dtype))
                            if group["amsgrad"]:
                                state["max_exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)
                        elif self.algorithm == "rmsprop":
                            state.update(step=torch.zeros((), dtype=torch.float32),
                                         square_avg=torch.zeros_like(p, dtype=self.dtype))
                            if group["centered"]:
                                state["grad_avg"] = torch.zeros_like(p, dtype=self.dtype)
                            if group["momentum"]:
                                state["momentum_buffer"] = torch.zeros_like(p, dtype=self.dtype)
                    options = {k: v for k, v in group.items() if k not in ("params", "name", "compile")}
                    options["lr"] = scalar_tensor(group["lr"], self.dtype)
                    native_updates.append((p, gradient, state, options))
                    continue
                update = torch.nn.Parameter(p.detach().to(self.dtype, copy=True))
                update.grad = gradient.detach().to(self.dtype)
                kernel_group = {**group, "params": [update]}
                self._kernel.param_groups = [kernel_group]
                self._kernel.state[update] = {
                    key: value.to(self.dtype) if key != "step" and isinstance(value, torch.Tensor) else value
                    for key, value in self.state[p].items()
                }
                native_parameter = update
                try:
                    self._kernel.step()
                    if self.update_transform is not None:
                        self.update_transform(p, update)
                    copy_update_(p, update, stochastic_rounding=self.stochastic_rounding)
                    self.state[p].update(self._kernel.state[native_parameter])
                finally:
                    self._kernel.state.clear()
                    self._kernel.param_groups = []
                del update, native_parameter, kernel_group
        if cuda_updates:
            self._apply_rmsprop_step(cuda_updates)
        if native_updates:
            if self._compiled_native_step is None:
                self._compiled_native_step = torch.compile(
                    _native_step_, dynamic=False, isolate_recompiles=True,
                    options={"triton.cudagraphs": False, "emulate_precision_casts": True},
                )
            self._compiled_native_step(native_updates, self.algorithm, self.dtype,
                                       self.stochastic_rounding, self.update_transform)
        return loss

    def _apply_rmsprop_step(self, updates):
        if self._compiled_rmsprop_step is None:
            # Preserve narrowing casts used to find stochastic-rounding neighbors.
            # Model shapes are fixed, unlike the old shared per-parameter graph.
            # Isolate each optimizer's budget. Unusual changing missing-gradient
            # patterns can fall back to eager after the standard recompile limit.
            self._compiled_rmsprop_step = torch.compile(
                _rmsprop_step_, dynamic=False, isolate_recompiles=True,
                options={"triton.cudagraphs": False, "emulate_precision_casts": True},
            )
        self._compiled_rmsprop_step(updates, stochastic_rounding=self.stochastic_rounding)

    def state_dict(self):
        return {**super().state_dict(), "algorithm": self.algorithm,
                "precision": (str(self.dtype), self.stochastic_rounding)}

    def load_state_dict(self, state_dict):
        if state_dict.get("algorithm") != self.algorithm:
            raise ValueError("Different stochastic optimizer algorithms are not interchangeable")
        if state_dict.get("precision") != (str(self.dtype), self.stochastic_rounding):
            raise ValueError("Optimizer checkpoint precision settings do not match")
        compilation = [group["compile"] for group in self.param_groups]
        super().load_state_dict(state_dict)
        for group, enabled in zip(self.param_groups, compilation, strict=True):
            group["compile"] = enabled
        # Restore from the original state, avoiding PyTorch's intermediate
        # cast to parameter dtype (which can differ from optimizer state dtype).
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for key, p in zip(saved_group["params"], group["params"], strict=True):
                if key not in state_dict["state"]:
                    continue
                self.state[p] = {
                    name: (value.to(device="cpu" if name == "step" else p.device,
                                    dtype=torch.float32 if name == "step" else self.dtype, copy=True)
                           if isinstance(value, torch.Tensor) else value)
                    for name, value in state_dict["state"][key].items()
                }
