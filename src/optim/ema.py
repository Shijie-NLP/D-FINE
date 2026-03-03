"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.

Refactored for optimal memory bandwidth utilization (strict in-place ops),
type safety, and removal of silent typos.
"""

import math
from copy import deepcopy
from typing import Any, Callable

import torch
import torch.nn as nn

from ..core import register
from ..misc import dist_utils


__all__ = ["ModelEMA", "ExponentialMovingAverage"]


@register()
class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmups: int = 1000,
        start: int = 0,
    ) -> None:
        super().__init__()

        # Deepcopy the base model to initialize EMA parameters
        self.module = deepcopy(dist_utils.de_parallel(model)).eval()

        self.decay = decay
        self.warmups = warmups
        self.before_start = 0
        self.start = start
        self.updates = 0  # number of EMA updates

        if warmups == 0:
            self.decay_fn: Callable[[int], float] = lambda x: decay
        else:
            self.decay_fn: Callable[[int], float] = lambda x: decay * (1.0 - math.exp(-x / warmups))

        # Strictly freeze the EMA model parameters
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        if self.before_start < self.start:
            self.before_start += 1
            return

        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = dist_utils.de_parallel(model).state_dict()

            # [SAC Note]: Replaced sequential allocations with fused in-place operations.
            # This minimizes memory bandwidth overhead during high-frequency EMA updates.
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    # Mathematically: v = d * v + (1 - d) * msd[k]
                    # Optimized in-place execution:
                    v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    def to(self, *args: Any, **kwargs: Any) -> "ModelEMA":
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(self) -> dict[str, Any]:
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        self.module.load_state_dict(state["module"], strict=strict)
        if "updates" in state:
            self.updates = state["updates"]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Prevent calling EMA model directly via forward to avoid confusion.
        Always use ema.module(x) for inference.
        """
        raise RuntimeError("EMA model should not be called directly via forward(). Use ema.module(x) instead.")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """
    Maintains moving averages of model parameters using an exponential decay.
    """

    def __init__(self, model: nn.Module, decay: float, device: str = "cpu", use_buffers: bool = True):
        self.decay_fn = lambda x: decay * (1.0 - math.exp(-x / 2000.0))

        def ema_avg(avg_model_param: torch.Tensor, model_param: torch.Tensor, num_averaged: int) -> torch.Tensor:
            d = self.decay_fn(num_averaged)
            return d * avg_model_param + (1.0 - d) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)
