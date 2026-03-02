"""
Optimized for modern PyTorch (>= 2.0) and TorchVision (>= 0.17.0).
Legacy bug-fixes for empty tensors have been eradicated.
Strict device and dtype propagation implemented to prevent Host-to-Device sync bottlenecks.
"""

from typing import Any, Optional

import torch
import torchvision.transforms.functional as F


def interpolate(
    input: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> torch.Tensor:
    """
    Direct routing to native PyTorch interpolate.
    Legacy empty batch size workarounds for torchvision < 0.7 are completely removed.
    """
    return torch.nn.functional.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


def crop(
    image: Any, target: dict[str, Any], region: tuple[int, int, int, int]
) -> tuple[Any, dict[str, Any]]:
    """Crops the image and adjusts bounding boxes/masks safely."""
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    target["size"] = torch.tensor([h, w])
    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        device = boxes.device
        dtype = boxes.dtype

        # Optimization: Pre-allocate tensors on the exact device to prevent
        # implicit CPU-to-GPU synchronization blocks during arithmetic ops.
        max_size = torch.tensor([w, h], dtype=dtype, device=device)
        offset = torch.tensor([j, i, j, i], dtype=dtype, device=device)

        cropped_boxes = boxes - offset
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)

        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            # Retain valid boxes where width and height > 0
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(dim=1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image: Any, target: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Horizontally flips the image and inverts bounding box coordinates."""
    flipped_image = F.hflip(image)
    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        device = boxes.device
        dtype = boxes.dtype

        # Optimization: Strict device placement for geometric inversion vectors
        flip_vector = torch.tensor([-1, 1, -1, 1], dtype=dtype, device=device)
        offset_vector = torch.tensor([w, 0, w, 0], dtype=dtype, device=device)

        boxes = boxes[:, [2, 1, 0, 3]] * flip_vector + offset_vector
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(
    image: Any,
    target: Optional[dict[str, Any]],
    size: Any,
    max_size: Optional[int] = None,
) -> tuple[Any, Optional[dict[str, Any]]]:
    """
    Resizes the image and precisely scales bounding box coordinates.
    """

    def get_size_with_aspect_ratio(
        image_size: tuple[int, int], size: int, max_size: Optional[int] = None
    ) -> tuple[int, int]:
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def get_size(
        image_size: tuple[int, int], size: Any, max_size: Optional[int] = None
    ) -> tuple[int, int]:
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        device = boxes.device
        dtype = boxes.dtype

        # Optimization: Prevent host-to-device scaling bottleneck
        scale_tensor = torch.tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height],
            dtype=dtype,
            device=device,
        )
        scaled_boxes = boxes * scale_tensor
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        # SAC Warning: "nearest" mode here will permanently delete masks of
        # objects smaller than the interpolation stride.
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    return rescaled_image, target


def pad(
    image: Any, target: Optional[dict[str, Any]], padding: tuple[int, int]
) -> tuple[Any, Optional[dict[str, Any]]]:
    """Pads the image purely on the bottom-right corners."""
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None

    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])

    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(
            target["masks"], (0, padding[0], 0, padding[1])
        )
    return padded_image, target
