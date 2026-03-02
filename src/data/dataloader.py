"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for memory-efficient tensor collation, strict typing, and robust
multi-scale training logic.
"""

import random
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch import Tensor

from ..core import register


__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    """
    Custom DataLoader wrapper that enables dynamic dependency injection and
    epoch-state propagation down to the dataset and collate functions.
    """

    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "(\n"
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += f"    {n}: {getattr(self, n, 'N/A')}\n"
        format_string += ")"
        return format_string

    def set_epoch(self, epoch: int) -> None:
        """Propagates the current training epoch to underlying components."""
        self._epoch = epoch

        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)

    @property
    def shuffle(self) -> bool:
        return getattr(self, "_shuffle", False)

    @shuffle.setter
    def shuffle(self, shuffle_flag: bool) -> None:
        assert isinstance(shuffle_flag, bool), "Shuffle attribute must be a boolean."
        self._shuffle = shuffle_flag


@register()
def batch_image_collate_fn(items: list[tuple[Tensor, Any]]) -> tuple[Tensor, list[Any]]:
    """
    Standard collation function for image batches.

    Optimization: Replaced memory-fragmenting list comprehension & torch.cat
    with C-optimized torch.stack for strictly faster execution.
    """
    # torch.stack directly concatenates a sequence of tensors along a new dimension (dim=0)
    images = torch.stack([x[0] for x in items], dim=0)
    targets = [x[1] for x in items]
    return images, targets


class BaseCollateFunction:
    """Interface for customized epoch-aware collate functions."""

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        return getattr(self, "_epoch", -1)

    def __call__(self, items: list[Any]) -> Any:
        raise NotImplementedError("Subclasses must implement the __call__ method.")


def generate_scales(base_size: int, base_size_repeat: int) -> list[int]:
    """
    Generates a list of scaling factors for Multi-Scale Training.
    Produces an array of dimensions surrounding the base_size to ensure
    scale-invariance during dense prediction tasks.
    """
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales: list[int] = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register()
class BatchImageCollateFunction(BaseCollateFunction):
    """
    Advanced collate function supporting dynamic multi-scale resizing.
    Crucial for scale invariance in modern detection architectures.
    """

    def __init__(
        self,
        stop_epoch: Optional[int] = None,
        ema_restart_decay: float = 0.9999,
        base_size: int = 640,
        base_size_repeat: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales: Optional[list[int]] = (
            generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        )
        # Fallback to an arbitrarily large number if no stop epoch is provided
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100_000_000
        self.ema_restart_decay = ema_restart_decay

    def __call__(self, items: list[tuple[Tensor, dict[str, Tensor]]]) -> tuple[Tensor, list[dict[str, Tensor]]]:
        # Optimization: C-level continuous memory stack
        images = torch.stack([x[0] for x in items], dim=0)
        targets = [x[1] for x in items]

        # Multi-scale dynamic interpolation (Early exit curriculum learning)
        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)

            # Note: Explicitly passing spatial dimensions as tuples ensures tighter
            # compatibility with newer PyTorch F.interpolate implementations.
            images = F.interpolate(images, size=(sz, sz))

            if "masks" in targets[0]:
                # SAC Warning: 'nearest' mode will erase extremely small instance masks.
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=(sz, sz), mode="nearest")
                raise NotImplementedError(
                    "Mask interpolation via nearest-neighbor can degrade performance. Custom handling is required."
                )

        return images, targets
