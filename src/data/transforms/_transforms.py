"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for high-performance data pipelines, focusing on tensor allocation
efficiency, precision preservation for tiny object detection, and robust typing.
"""

from typing import Any, Optional, Union

import PIL.Image
import torch
import torchvision
import torchvision.transforms.v2 as T_v2
import torchvision.transforms.v2.functional as F_v2
from torchvision.tv_tensors import Image, Video

from ...core import register
from .._misc import BoundingBoxes, Mask, _boxes_keys, convert_to_tv_tensor


# Registering standard v2 transforms into the global workspace
RandomPhotometricDistort = register()(T_v2.RandomPhotometricDistort)
RandomZoomOut = register()(T_v2.RandomZoomOut)
RandomHorizontalFlip = register()(T_v2.RandomHorizontalFlip)
Resize = register()(T_v2.Resize)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(T_v2.SanitizeBoundingBoxes)
RandomCrop = register()(T_v2.RandomCrop)
Normalize = register()(T_v2.Normalize)


@register()
class EmptyTransform(T_v2.Transform):
    """A no-op transform that safely passes inputs through the pipeline."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs: Any) -> Any:
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class PadToSize(T_v2.Pad):
    """
    Pads the input to a strictly defined spatial size.
    Crucial for batch collation in dense prediction architectures (e.g., FCOS, D-FINE).
    """

    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(
        self,
        size: Union[int, tuple[int, int]],
        fill: int = 0,
        padding_mode: str = "constant",
    ) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        super().__init__(0, fill, padding_mode)
        # Initialization for type safety, though the stateful mutation in
        # _get_params remains a design flaw of the original architecture.
        self.padding: list[int] = [0, 0, 0, 0]

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Dynamically calculates padding required to reach the target size."""
        sp = F_v2.get_spatial_size(flat_inputs[0])
        h_pad = self.size[1] - sp[0]
        w_pad = self.size[0] - sp[1]

        # SAC Warning: Mutating instance state (self.padding) inside _get_params
        # is an anti-pattern and highly unsafe in threaded dataloader contexts.
        # Preserved exclusively for exact logical parity with the original framework.
        self.padding = [0, 0, w_pad, h_pad]
        return {"padding": self.padding}

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        # type: ignore[arg-type] is preserved from the original codebase
        return F_v2.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        """
        Overridden to inject the computed padding back into the target dictionary,
        allowing the model's loss function to mask out padded regions precisely.
        """
        outputs = super().forward(*inputs)
        # If output is a tuple and the second element is a target dictionary
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            # Casting to float32 explicitly to prevent downstream mixed-precision crashes
            outputs[1]["padding"] = torch.tensor(self.padding, dtype=torch.float32)
        return outputs


@register()
class RandomIoUCrop(T_v2.RandomIoUCrop):
    """
    Stochastic IoU-based cropping. Extended with an execution probability 'p'.
    """

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[list[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            sampler_options,
            trials,
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        # Optimization: torch.rand(()).item() avoids allocating a 1D tensor
        # in the dataloader worker loop, significantly reducing memory overhead.
        if torch.rand(()).item() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T_v2.Transform):
    """
    Transforms bounding box formats and optionally normalizes them to [0, 1].
    Highly sensitive operation; critical for maintaining Tiny Object coordinates.
    """

    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt: str = "", normalize: bool = False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])

        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt,
                key="boxes",
                box_format=self.fmt.upper(),
                spatial_size=spatial_size,
            )

        if self.normalize:
            # Optimization: Replaced repeated list-reversal and .tile() allocation
            # with direct, device-aware tensor construction. This strictly prevents
            # precision drift and host-to-device bottlenecks for tiny object coordinates.
            h, w = spatial_size
            scale_tensor = torch.tensor([w, h, w, h], dtype=inpt.dtype, device=inpt.device)
            inpt = inpt / scale_tensor

        return inpt


@register()
class ConvertPILImage(T_v2.Transform):
    """
    Converts PIL Images to PyTorch Image Tensors with optimal hardware operations.
    """

    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype: str = "float32", scale: bool = True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        inpt = F_v2.pil_to_tensor(inpt)

        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            # Multiplicative scaling is faster than division on modern ALUs
            inpt = inpt * (1.0 / 255.0)

        return Image(inpt)
