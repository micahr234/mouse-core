"""Native optimizer math with BF16 stochastic writeback and FP32 moments."""

import torch

from mouse_core.rounding import copy_update_


class StochasticOptimizer(torch.optim.Optimizer):
    """Step one parameter at a time in FP32, without persistent master weights.

    Delegating arithmetic to native PyTorch preserves Adam/AdamW/SGD/RMSprop
    semantics. Only moments persist in FP32. A temporary parameter and gradient
    exist for the current tensor, and are released before the next tensor.
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
        self.update_transform = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.dtype not in (torch.float32, torch.bfloat16):
                    raise TypeError("Stochastic optimizer supports FP32 and BF16 parameters only")

    @torch.no_grad()
    def step(self, closure=None):
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
