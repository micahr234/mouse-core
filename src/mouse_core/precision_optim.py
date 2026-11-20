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
        _compiled_rmsprop_update = torch.compile(
            _rmsprop_update_, fullgraph=True, dynamic=True,
            options={"triton.cudagraphs": False},
        )
    return _compiled_rmsprop_update


class PrecisionOptimizer(torch.optim.Optimizer):
    """Selectable FP64/FP32/BF16 arithmetic and state, without master weights.

    FP32/FP64 RMSprop without an update transform uses a fused CUDA pointwise update.
    Other algorithms and transformed updates delegate to native PyTorch math.
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
        self._kernel = classes[name]([{"params": [], "lr": 0.}], **options, foreach=False)
        super().__init__(params, self._kernel.defaults)
        self._kernel.param_groups = []
        self.gradient_transform: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        self.update_transform: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        for group in self.param_groups:
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
                    _rmsprop_kernel(p.device)(
                        p, gradient, state["square_avg"], state.get("grad_avg"),
                        state.get("momentum_buffer"),
                        lr=scalar_tensor(group["lr"], self.dtype) if p.is_cuda else group["lr"],
                        alpha=group["alpha"], eps=group["eps"],
                        weight_decay=group["weight_decay"], momentum=group["momentum"],
                        stochastic_rounding=self.stochastic_rounding,
                    )
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
        return loss

    def state_dict(self):
        return {**super().state_dict(), "algorithm": self.algorithm,
                "precision": (str(self.dtype), self.stochastic_rounding)}

    def load_state_dict(self, state_dict):
        if state_dict.get("algorithm") != self.algorithm:
            raise ValueError("Different stochastic optimizer algorithms are not interchangeable")
        if state_dict.get("precision") != (str(self.dtype), self.stochastic_rounding):
            raise ValueError("Optimizer checkpoint precision settings do not match")
        super().load_state_dict(state_dict)
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
