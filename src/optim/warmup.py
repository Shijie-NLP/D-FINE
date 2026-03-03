"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Refactored for strict type safety, defensive division-by-zero prevention,
and robust state_dict serialization.
"""

from typing import Any

from torch.optim.lr_scheduler import LRScheduler

from ..core import register


class Warmup:
    """
    Base class for learning rate warmup strategies.
    Safely manipulates optimizer parameter groups before the main LR scheduler takes over.
    """

    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        self.lr_scheduler = lr_scheduler
        # Cache the initial learning rates to properly scale them during warmup
        self.warmup_end_values = [pg["lr"] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = max(0, warmup_duration)  # Defensive enforcement

        # Initialize the first step
        self.step()

    def state_dict(self) -> dict[str, Any]:
        """
        Safely serialize the warmup state, excluding the external LRScheduler reference.
        """
        return {
            "warmup_end_values": self.warmup_end_values,
            "last_step": self.last_step,
            "warmup_duration": self.warmup_duration,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Safely deserialize the warmup state with explicit key assignments
        instead of dangerous __dict__.update() blind injection.
        """
        if "warmup_end_values" in state_dict:
            self.warmup_end_values = state_dict["warmup_end_values"]
        if "last_step" in state_dict:
            self.last_step = state_dict["last_step"]
        if "warmup_duration" in state_dict:
            self.warmup_duration = state_dict["warmup_duration"]

    def get_warmup_factor(self, step: int, **kwargs: Any) -> float:
        raise NotImplementedError("get_warmup_factor must be implemented by subclasses.")

    def step(self) -> None:
        self.last_step += 1

        if self.last_step >= self.warmup_duration:
            return

        factor = self.get_warmup_factor(self.last_step)

        # Apply the computed factor to all parameter groups
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            pg["lr"] = factor * self.warmup_end_values[i]

    def finished(self) -> bool:
        """
        Returns True if the warmup phase has completed.
        """
        return self.last_step >= self.warmup_duration


@register()
class LinearWarmup(Warmup):
    """
    Linear learning rate warmup strategy.
    Linearly scales the learning rate from near-zero to the initial scheduled LR.
    """

    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        super().__init__(lr_scheduler, warmup_duration, last_step)

    def get_warmup_factor(self, step: int) -> float:
        # Defensive check to prevent ZeroDivisionError if warmup_duration is somehow 0
        if self.warmup_duration == 0:
            return 1.0
        return min(1.0, (step + 1) / self.warmup_duration)
