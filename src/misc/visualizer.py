"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Refactored for vectorized tensor operations, optimized PIL rendering,
and cleaner dependency management.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision
from PIL import ImageDraw, ImageFont
from torchvision.ops import box_convert
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import functional as F_v2
from torchvision.utils import draw_bounding_boxes


torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]


# Predefined standard colors
# fmt: off
BOX_COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
    "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "navy",
    "olive", "coral", "turquoise", "gold",
]
LABEL_TEXT_COLOR = "white"
# fmt: on


@torch.no_grad()
def save_samples(
    samples: torch.Tensor,
    targets: list[dict[str, Any]],
    output_dir: str,
    split: str,
    normalized: bool,
    box_fmt: str,
) -> None:
    """
    Saves visualization of image samples overlaid with target bounding boxes.

    Args:
        normalized: whether the boxes are normalized to [0, 1]
        box_fmt: 'xyxy', 'xywh', 'cxcywh'. (D-FINE uses 'cxcywh' for training)
    """
    out_path = Path(output_dir) / f"{split}_samples"
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.load_default()
        # Note: Default font might not support size scaling on all OS.
        # Kept size attribute for structural consistency.
        font.size = 32
    except Exception:
        font = None

    for sample, target in zip(samples, targets):
        # Detach to avoid retaining computation graphs, move to CPU
        sample_cpu = sample.detach().cpu()
        target_boxes = target["boxes"].detach().cpu()
        target_labels = target["labels"].detach().cpu()

        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem

        sample_pil = to_pil_image(sample_cpu)
        w, h = sample_pil.size

        if normalized:
            # Vectorized denormalization: [w, h, w, h] broadcast multiplier
            scale_tensor = torch.tensor([w, h, w, h], dtype=torch.float32)
            target_boxes = target_boxes * scale_tensor

        target_boxes = box_convert(target_boxes, in_fmt=box_fmt, out_fmt="xyxy")

        # Vectorized clipping to image boundaries
        target_boxes[:, 0::2] = target_boxes[:, 0::2].clamp(min=0, max=w)
        target_boxes[:, 1::2] = target_boxes[:, 1::2].clamp(min=0, max=h)

        boxes_np = target_boxes.numpy().astype(np.int32)
        labels_np = target_labels.numpy().astype(np.int32)

        draw = ImageDraw.Draw(sample_pil)

        for box, label in zip(boxes_np, labels_np):
            x1, y1, x2, y2 = box
            box_color = BOX_COLORS[int(label) % len(BOX_COLORS)]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            label_text = str(label)

            # Measure text size robustly
            if hasattr(draw, "textbbox"):
                left, top, right, bottom = draw.textbbox((0, 0), label_text, font=font)
                text_width = right - left
                text_height = bottom - top
            else:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(label_text, font=font)

            padding = 2
            draw.rectangle(
                [x1, y1 - text_height - padding * 2, x1 + text_width + padding * 2, y1],
                fill=box_color,
            )

            draw.text(
                (x1 + padding, y1 - text_height - padding),
                label_text,
                fill=LABEL_TEXT_COLOR,
                font=font,
            )

        save_path = out_path / f"{target_image_id}_{target_image_path_stem}.webp"
        sample_pil.save(save_path)


@torch.no_grad()
def show_sample(sample: tuple[Any, dict[str, Any]]) -> None:
    """Helper for COCO dataset/dataloader visualization."""
    image, target = sample

    if isinstance(image, PIL.Image.Image):
        image = F_v2.to_image_tensor(image)

    image = F_v2.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    # Matplotlib expects HxWxC
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    fig.tight_layout()
    plt.show()
