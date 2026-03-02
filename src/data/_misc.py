"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for modern TorchVision API (>= 0.17.0).
Legacy support has been stripped for architectural purity and maintainability.
"""

import importlib.metadata
from typing import Any, Optional

import torchvision
from torch import Tensor


# ---------------------------------------------------------------------------
# Strict Version Enforcement
# ---------------------------------------------------------------------------
def _parse_version(version_str: str) -> tuple[int, ...]:
    """Extracts numeric version tuple from strings like '0.17.0+cu118'."""
    clean_str = version_str.split("+")[0].split("a")[0].split("b")[0]
    return tuple(map(int, (clean_str.split(".") + ["0", "0"])[:3]))


_tv_version_str = importlib.metadata.version("torchvision")
_tv_version = _parse_version(_tv_version_str)

# SAC Insight: Maintaining legacy datapoints API creates technical debt.
# We strictly enforce >= 0.17.0 for stable tv_tensors support.
if _tv_version < (0, 17, 0):
    raise RuntimeError(
        f"TorchVision version too low ({_tv_version_str}). "
        f"Please upgrade to >= 0.17.0 for stable tv_tensors API support. "
        f"Run: `pip install --upgrade torchvision`"
    )

# Suppress beta warnings as tv_tensors are utilized heavily
torchvision.disable_beta_transforms_warning()

from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Mask  # noqa

_boxes_keys = ["format", "canvas_size"]


def convert_to_tv_tensor(
    tensor: Tensor,
    key: str,
    box_format: str = "xyxy",
    spatial_size: Optional[tuple[int, int]] = None,
) -> Tensor:
    """
    Converts raw PyTorch tensors into specialized TorchVision TV_Tensors.

    Args:
        tensor (Tensor): Input tensor containing coordinate or mask data.
        key (str): Target modality, must be 'boxes' or 'masks'.
        box_format (str): Bounding box format, defaults to 'xyxy'.
        spatial_size (Tuple[int, int], optional): Canvas size for the bounding boxes.

    Returns:
        Tensor: A specialized TV_Tensor subclass (BoundingBoxes or Mask).
    """
    assert key in ("boxes", "masks"), f"Unsupported key: '{key}'. Only 'boxes' and 'masks' are supported."

    if key == "boxes":
        # Dynamically fetch the format enum safely
        fmt_enum = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs: dict[str, Any] = dict(zip(_boxes_keys, [fmt_enum, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "masks":
        return Mask(tensor)
