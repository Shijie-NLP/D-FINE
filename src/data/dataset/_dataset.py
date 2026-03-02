"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict static typing, explicit attribute initialization,
and safe object-oriented interface contracts.
"""

from typing import Any, Callable, Optional

import torch.utils.data as data


class DetDataset(data.Dataset):
    """
    Abstract Base Class for Detection Datasets.
    Establishes a strict contract for dynamic epoch tracking and transform injection,
    vital for curriculum learning and multi-scale augmentation strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        # Optimization: Explicitly initialize attributes to prevent AttributeError
        # in subclasses that fail to construct these variables properly.
        self.transforms: Optional[Callable] = None
        self._epoch: int = -1

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        """
        Fetches the raw item and applies the transformation pipeline.
        """
        img, target = self.load_item(index)

        if self.transforms is not None:
            # The transform pipeline expects (image, target, dataset_instance)
            # as defined in the advanced Compose orchestrator.
            img, target, _ = self.transforms(img, target, self)

        return img, target

    def load_item(self, index: int) -> tuple[Any, dict[str, Any]]:
        """
        Abstract method to load raw data from disk or memory.
        Must be strictly overridden by dataset implementations (e.g., CocoDetection).
        """
        raise NotImplementedError(
            "Subclasses must implement `load_item` to return raw (image, target) "
            "tuples prior to applying the `transforms` pipeline."
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Injects the current training epoch state into the dataset.
        Critical for epoch-aware augmentations (e.g., stop_epoch policies).
        """
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        """Safely retrieves the current epoch state."""
        return getattr(self, "_epoch", -1)
