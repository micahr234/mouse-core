"""Native optimizer math with BF16 stochastic writeback and FP32 moments."""

from collections.abc import Callable
from typing import Any

import torch

from mouse_core.rounding import copy_update_, scalar_tensor


def _rmsprop_update_(parameter, gradient, square_avg, grad_avg, momentum_buffer,
                     *, lr, alpha, eps, weight_decay, momentum):
    """Native single-tensor RMSprop math, with just one parameter writeback.

    CUDA compilation fuses the FP32 temporaries, moment updates, and stochastic
    rounding. No full-size FP32 parameter or gradient buffers need to survive
    the kernel. CPU runs the same expression eagerly as a numerical reference.
    """
    value = parameter.float()
    grad = gradient.float()
    if weight_decay != 0:
        grad = grad.add(value, alpha=weight_decay)
    square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
    if grad_avg is not None:
        grad_avg.lerp_(grad, 1 - alpha)
        average = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_()
    else:
        average = square_avg.sqrt()
    average.add_(eps)
    if momentum_buffer is not None:
        momentum_buffer.mul_(momentum).addcdiv_(grad, average)
        value.add_(momentum_buffer, alpha=-lr)
    else:
        value.addcdiv_(grad, average, value=-lr)
    copy_update_(parameter, value)


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


class StochasticOptimizer(torch.optim.Optimizer):
    """FP32 update math with BF16 storage, without persistent master weights.

    RMSprop without an update transform uses a fused CUDA pointwise update.
    Other algorithms and transformed updates delegate to native PyTorch math.
    Only moments persist in FP32; compiled kernels cache code, never weights.
    ``update_transform`` optionally modifies the FP32 result before its single
    writeback (for packed-head rate multipliers and Langevin noise).
    """

    def __init__(self, params, *, name, options):
        if options.get("fused"):
            raise ValueError("Stochastic rounding does not support fused optimizer variants")
        classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
                   "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
        self.algorithm = name
        self._kernel = classes[name]([{"params": [], "lr": 0.}], **options, foreach=False)
        super().__init__(params, self._kernel.defaults)
        self._kernel.param_groups = []
        self.update_transform: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.dtype not in (torch.float32, torch.bfloat16):
                    raise TypeError("Stochastic optimizer supports FP32 and BF16 parameters only")

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
                if self.algorithm == "rmsprop" and self.update_transform is None:
                    state = self.state[p]
                    if not state:
                        state["step"] = torch.zeros((), dtype=torch.float32, device="cpu")
                        state["square_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        if group["momentum"] > 0:
                            state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                        if group["centered"]:
                            state["grad_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"].add_(1)
                    _rmsprop_kernel(p.device)(
                        p, p.grad, state["square_avg"], state.get("grad_avg"),
                        state.get("momentum_buffer"),
                        lr=scalar_tensor(group["lr"]) if p.is_cuda else group["lr"],
                        alpha=group["alpha"], eps=group["eps"],
                        weight_decay=group["weight_decay"], momentum=group["momentum"],
                    )
                    continue
                update = torch.nn.Parameter(p.detach().to(torch.float32, copy=True))
                update.grad = p.grad.detach().float()
                kernel_group = {**group, "params": [update]}
                self._kernel.param_groups = [kernel_group]
                self._kernel.state[update] = self.state[p]
                try:
                    self._kernel.step()
                    if self.update_transform is not None:
                        self.update_transform(p, update)
                    copy_update_(p, update)
                finally:
                    self._kernel.state.clear()
                    self._kernel.param_groups = []
                del update, kernel_group
        return loss

    def state_dict(self):
        return {**super().state_dict(), "algorithm": self.algorithm}

    def load_state_dict(self, state_dict):
        if state_dict.get("algorithm") != self.algorithm:
            raise ValueError("Different stochastic optimizer algorithms are not interchangeable")
        super().load_state_dict(state_dict)
        # PyTorch's loader casts moments to parameter dtype. Restore from the
        # original serialized FP32 tensors, avoiding a lossy BF16 round trip.
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups, strict=True):
            for key, p in zip(saved_group["params"], group["params"], strict=True):
                if key not in state_dict["state"]:
                    continue
                self.state[p] = {
                    name: (value.to(device="cpu" if name == "step" else p.device,
                                    dtype=torch.float32, copy=True)
                           if isinstance(value, torch.Tensor) else value)
                    for name, value in state_dict["state"][key].items()
                }
