"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for hot-loop execution speed, immutable configuration handling,
and strict routing typings.
"""

from typing import Any, Callable, Optional, Union

import torch.nn as nn
import torchvision.transforms.v2 as T

from ...core import GLOBAL_CONFIG, register
from ._transforms import EmptyTransform


@register()
class Compose(T.Compose):
    """
    Advanced composition of transforms supporting curriculum learning policies
    (e.g., stopping specific augmentations after a certain epoch/sample count).
    """

    def __init__(
        self,
        ops: Optional[list[Union[dict[str, Any], nn.Module]]],
        policy: Optional[dict[str, Any]] = None,
    ) -> None:
        transforms: list[nn.Module] = []

        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    # Optimization: Use a shallow copy to prevent destructive
                    # in-place mutation of the shared configuration dictionary.
                    op_kwargs = op.copy()
                    name = op_kwargs.pop("type")

                    target_module = GLOBAL_CONFIG[name]["_pymodule"]
                    target_class_name = GLOBAL_CONFIG[name]["_name"]

                    transform = getattr(target_module, target_class_name)(**op_kwargs)
                    transforms.append(transform)

                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError(f"Unsupported operation type in Compose: {type(op)}")
        else:
            transforms = [EmptyTransform()]

        super().__init__(transforms=transforms)

        self.policy = policy if policy is not None else {"name": "default"}
        self.global_samples = 0

        # Optimization: Pre-bind the forward routing function during initialization.
        # This completely eliminates dictionary instantiations and string lookups
        # from the critical path (the forward hot-loop).
        self._forward_router: Callable = self._get_forward_fn(self.policy["name"])

    def forward(self, *inputs: Any) -> Any:
        """Routes the input directly to the pre-bound policy function."""
        return self._forward_router(*inputs)

    def _get_forward_fn(self, name: str) -> Callable:
        """Resolves the policy routing exclusively during initialization."""
        forwards = {
            "default": self.default_forward,
            "stop_epoch": self.stop_epoch_forward,
            "stop_sample": self.stop_sample_forward,
        }
        if name not in forwards:
            raise KeyError(f"Unknown transform policy name: '{name}'.")
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        # Standard tuple extraction logic preserved
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        # SAC Warning: Hardcoding the dataset extraction to sample[-1]
        # makes the pipeline heavily coupled with the specific Dataloader output format.
        dataset = sample[-1]
        cur_epoch = getattr(dataset, "epoch", -1)

        policy_ops = self.policy.get("ops", [])
        policy_epoch = self.policy.get("epoch", float("inf"))

        for transform in self.transforms:
            # Bypass specific transforms if the curriculum epoch threshold is met
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                continue
            sample = transform(sample)

        return sample

    def stop_sample_forward(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]

        policy_ops = self.policy.get("ops", [])
        policy_sample = self.policy.get("sample", float("inf"))

        for transform in self.transforms:
            # Bypass specific transforms if the curriculum sample threshold is met
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                continue
            sample = transform(sample)

        self.global_samples += 1

        return sample
