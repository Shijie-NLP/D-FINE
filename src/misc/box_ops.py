"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for optimal memory efficiency, PyTorch 2.x compilation friendliness,
and asynchronous CUDA execution compatibility.
"""

import torch
import torchvision
from torch import Tensor


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Computes Generalized IoU (GIoU) between two sets of boxes.
    """
    # [SAC Note]: Explicit assert with `.all()` triggers a Device-to-Host (D2H) sync,
    # breaking CUDA stream asynchrony. Kept here for strict I/O compatibility,
    # but in production, consider removing these or using torch._assert() for torch.compile.
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 must be in [x1, y1, x2, y2] format"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 must be in [x1, y1, x2, y2] format"
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
    """
    Computes element-wise IoU and union area.

    Args:
        boxes1: Tensor of shape [N, 4] in (x1, y1, x2, y2)
        boxes2: Tensor of shape [N, 4] in (x1, y1, x2, y2)
    Returns:
        iou: Tensor of shape [N, ]
        union: Tensor of shape [N, ]
    """
    area1 = torchvision.ops.box_area(boxes1)  # [N, ]
    area2 = torchvision.ops.box_area(boxes2)  # [N, ]

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]

    # In-place clamp_ is faster and saves memory
    wh = (rb - lt).clamp_(min=0)  # [N, 2]

    inter = wh[:, 0] * wh[:, 1]  # [N, ]
    union = area1 + area2 - inter
    iou = inter / union

    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Computes element-wise Generalized IoU (GIoU).

    Args:
        boxes1: Tensor of shape [N, 4] with [x1, y1, x2, y2]
        boxes2: Tensor of shape [N, 4] with [x1, y1, x2, y2]
    Returns:
        giou: Tensor of shape [N, ]
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 must be in [x1, y1, x2, y2] format"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 must be in [x1, y1, x2, y2] format"

    iou, union = elementwise_box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]

    # In-place clamp_
    wh = (rb - lt).clamp_(min=0)  # [N, 2]
    area = wh[:, 0] * wh[:, 1]

    # Fallback to mathematically correct GIoU formula
    return iou - (area - union) / area


def check_point_inside_box(points: Tensor, boxes: Tensor, eps: float = 1e-9) -> Tensor:
    """
    Checks if given points reside inside the bounding boxes.

    Args:
        points: Tensor of shape [K, 2], (x, y)
        boxes: Tensor of shape [N, 4], (x1, y1, x2, y2)
        eps: float, margin for numerical stability
    Returns:
        mask: Boolean Tensor of shape [K, N]
    """
    # [SHIJIE Note]: Optimized by avoiding the massive [K, N, 4] intermediate tensor stacking.
    # Mathematical equivalence: min(l, t, r, b) > eps <=> (l > eps) & (t > eps) & (r > eps) & (b > eps)

    pt = points.unsqueeze(1)  # [K, 1, 2]
    box_lt, box_rb = boxes.unsqueeze(0).chunk(2, dim=-1)  # [1, N, 2] each

    lt_dist = pt - box_lt  # [K, N, 2]
    rb_dist = box_rb - pt  # [K, N, 2]

    # Apply logical AND across dimensions instead of computing minimums
    mask = (lt_dist > eps).all(dim=-1) & (rb_dist > eps).all(dim=-1)  # [K, N]

    return mask


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    """
    Computes distances from points to the boundaries of bounding boxes.

    Args:
        points: Tensor of shape [N, 2], (x, y)
        boxes: Tensor of shape [N, 4], (x1, y1, x2, y2)
    Returns:
        distances: Tensor of shape [N, 4], (l, t, r, b)
    """
    # chunk is slightly faster and more idiomatic than split for exact halving
    box_lt, box_rb = boxes.chunk(2, dim=-1)

    lt = points - box_lt
    rb = box_rb - points

    return torch.cat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    """
    Decodes bounding boxes given points and their distances to box boundaries.

    Args:
        points: Tensor of shape [N, 2], (x, y)
        distances: Tensor of shape [N, 4], (l, t, r, b)
    Returns:
        boxes: Tensor of shape [N, 4], (x1, y1, x2, y2)
    """
    lt, rb = distances.chunk(2, dim=-1)

    box_lt = points - lt
    box_rb = points + rb

    return torch.cat([box_lt, box_rb], dim=-1)
