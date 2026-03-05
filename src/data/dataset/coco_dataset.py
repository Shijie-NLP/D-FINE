"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Optimized for strict typing, robust empty-annotation handling, and
minimized CPU overhead during multi-processing data loading.
"""

import os
from typing import Any, Callable, Optional

import faster_coco_eval.core.mask as coco_mask
import numpy as np
import torch
from faster_coco_eval.utils.pytorch import FasterCocoDetection
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from ._dataset import DetDataset


# Disable PIL decompression bomb limits for extreme resolution images
Image.MAX_IMAGE_PIXELS = None

__all__ = ["CocoDetection"]


@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    """
    Highly optimized COCO Dataset wrapper tailored for dense prediction architectures.
    Supports dynamic category remapping and hardware-accelerated TV_Tensors.
    """

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable],
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
        num_samples: Optional[int] = None,
    ) -> None:
        img_folder = os.path.expanduser(img_folder)
        ann_file = os.path.expanduser(ann_file)

        super().__init__(img_folder, ann_file)

        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        if num_samples is not None:
            self.ids = self.ids[:num_samples]

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any]]:
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx: int) -> tuple[Any, dict[str, Any]]:
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        # Avoid unnecessary index lookups if possible; caching this in production
        # for million-scale datasets is recommended.
        image_path = os.path.join(self.img_folder, self.coco.loadImgs(image_id)[0]["file_name"])
        target = {"image_id": image_id, "image_path": image_path, "annotations": target}

        if self.remap_mscoco_category:
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            image, target = self.prepare(image, target)

        target["idx"] = torch.tensor([idx], dtype=torch.int64)

        if "boxes" in target:
            # Safely promote raw tensors to TV_Tensors for the transform pipeline
            target["boxes"] = convert_to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])

        if "masks" in target:
            target["masks"] = convert_to_tv_tensor(target["masks"], key="masks")

        return image, target

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        if hasattr(self, "_preset") and self._preset is not None:
            s += f" preset:\n   {repr(self._preset)}"
        return s

    @property
    def categories(self) -> list[dict[str, Any]]:
        return self.coco.dataset["categories"]

    @property
    def category2name(self) -> dict[int, str]:
        return {cat["id"]: cat["name"] for cat in self.categories}

    @property
    def category2label(self) -> dict[int, int]:
        return {cat["id"]: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self) -> dict[int, int]:
        return {i: cat["id"] for i, cat in enumerate(self.categories)}


@register()
class StratifiedCocoSubset(CocoDetection):
    """
    A stratified 20% subset of CocoDetection for rapid idea validation.

    Compared to naive num_samples truncation, this ensures:
      1. Every category is represented proportionally.
      2. Multi-label images (containing multiple categories) are preserved naturally.
      3. The subset is reproducible across runs via `seed`.

    Typical speedup vs full COCO: ~5x on training time per epoch.

    Args:
        img_folder:            Path to COCO image directory.
        ann_file:              Path to COCO annotation JSON.
        transforms:            Transform pipeline (same as CocoDetection).
        return_masks:          Whether to return segmentation masks.
        remap_mscoco_category: Whether to remap to contiguous label space.
        subset_ratio:          Fraction of images to keep (default 0.2).
        seed:                  Random seed for reproducibility (default 42).
    """

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
        subset_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        # Initialize the full dataset first (populates self.coco and self.ids)
        super().__init__(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
            num_samples=None,  # Load everything first, then we filter
        )

        assert 0.0 < subset_ratio <= 1.0, f"subset_ratio must be in (0, 1], got {subset_ratio}"
        self.subset_ratio = subset_ratio
        self.seed = seed

        # Replace self.ids with the stratified subset
        self.ids = self.get_balanced_subset(subset_ratio, seed)

    def get_balanced_subset(self, ratio=0.2, seed=42):
        rng = np.random.default_rng(seed)

        # 1. 计算目标总数
        total_target = int(len(self.ids) * ratio)
        sampled_ids = set()

        # 2. 统计每个类别的稀有程度（包含图片越少的类越优先）
        # cat_to_imgs: {cat_id: [img_id1, img_id2, ...]}
        cat_counts = {cat_id: len(img_ids) for cat_id, img_ids in self.coco.catToImgs.items()}
        sorted_cats = sorted(cat_counts.keys(), key=lambda x: cat_counts[x])

        # 3. 预估每个类别应有的配额 (目标总数 * 该类在全集中的占比)
        cat_targets = {cat_id: max(1, int(len(img_ids) * ratio)) for cat_id, img_ids in self.coco.catToImgs.items()}
        current_cat_counts = dict.fromkeys(self.coco.catToImgs.keys(), 0)

        # 4. 第一轮：贪心处理稀有类别
        for cat_id in sorted_cats:
            img_ids = list(self.coco.catToImgs[cat_id])
            rng.shuffle(img_ids)

            for img_id in img_ids:
                if len(sampled_ids) >= total_target:
                    break

                # 如果这张图还没被选中，且该类别还没达标
                if img_id not in sampled_ids and current_cat_counts[cat_id] < cat_targets[cat_id]:
                    sampled_ids.add(img_id)
                    # 关键：选中一张图后，要更新它包含的所有类别的计数
                    for c in self.coco.imgToAnns[img_id]:
                        current_cat_counts[c["category_id"]] += 1

        # 5. 第二轮：如果还没凑满 20%（因为重叠导致提前达标），随机补齐
        if len(sampled_ids) < total_target:
            remaining_ids = list(set(self.ids) - sampled_ids)
            rng.shuffle(remaining_ids)
            needed = total_target - len(sampled_ids)
            sampled_ids.update(remaining_ids[:needed])

        total = len(self.ids)
        kept = len(sampled_ids)
        actual_ratio = kept / total if total > 0 else 0.0
        print(
            f"[StratifiedCocoSubset] "
            f"total={total}, kept={kept}, "
            f"target_ratio={ratio:.1%}, actual_ratio={actual_ratio:.1%}, "
            f"seed={seed}"
        )

        return sorted(sampled_ids)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        base += f"\n subset_ratio: {self.subset_ratio}"
        base += f"\n seed: {self.seed}"
        return base


def convert_coco_poly_to_mask(segmentations: list[Any], height: int, width: int) -> torch.Tensor:
    """Decodes COCO polygon formats into binary dense masks."""
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)

        if len(mask.shape) < 3:
            mask = mask[..., None]

        # Optimization: Ensures memory is shared seamlessly from NumPy array to PyTorch Tensor
        mask_tensor = torch.as_tensor(mask, dtype=torch.uint8)
        mask_tensor = mask_tensor.any(dim=2)
        masks.append(mask_tensor)

    if masks:
        return torch.stack(masks, dim=0)
    else:
        return torch.zeros((0, height, width), dtype=torch.uint8)


class ConvertCocoPolysToMask:
    """
    Core parsing engine bridging raw COCO annotations into continuous tensor blocks
    suitable for dense object detection evaluation and loss calculation.
    """

    def __init__(self, return_masks: bool = False) -> None:
        self.return_masks = return_masks

    def __call__(
        self, image: Image.Image, target: dict[str, Any], **kwargs: Any
    ) -> tuple[Image.Image, dict[str, Any]]:
        w, h = image.size

        image_id = torch.tensor([target["image_id"]], dtype=torch.int64)
        image_path = target["image_path"]

        # Filter out crowd annotations to maintain clean anchor assignments
        anno = [obj for obj in target["annotations"] if obj.get("iscrowd", 0) == 0]

        # Early exit for background images (no objects)
        if not anno:
            empty_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": image_id,
                "image_path": image_path,
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.tensor([int(w), int(h)], dtype=torch.int64),
            }
            if self.return_masks:
                empty_target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            return image, empty_target

        # Process Bounding Boxes (XYWH to XYXY)
        boxes = torch.tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Process Labels dynamically
        category2label = kwargs.get("category2label")
        if category2label is not None:
            labels = torch.tensor([category2label[obj["category_id"]] for obj in anno], dtype=torch.int64)
        else:
            labels = torch.tensor([obj["category_id"] for obj in anno], dtype=torch.int64)

        # Process Masks
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # Process Keypoints
        keypoints = None
        if "keypoints" in anno[0]:
            keypoints_list = [obj["keypoints"] for obj in anno]
            keypoints_tensor = torch.tensor(keypoints_list, dtype=torch.float32)
            num_keypoints = keypoints_tensor.shape[0]
            if num_keypoints:
                keypoints = keypoints_tensor.view(num_keypoints, -1, 3)

        # Spatial Filtering: Drop degenerate boxes (w <= 0 or h <= 0)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        boxes = boxes[keep]
        labels = labels[keep]

        target_dict = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "image_path": image_path,
        }

        if self.return_masks:
            target_dict["masks"] = masks[keep]

        if keypoints is not None:
            target_dict["keypoints"] = keypoints[keep]

        # Extract auxiliary fields for COCO evaluation
        area = torch.tensor([obj["area"] for obj in anno], dtype=torch.float32)
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)

        target_dict["area"] = area[keep]
        target_dict["iscrowd"] = iscrowd[keep]
        target_dict["orig_size"] = torch.tensor([int(w), int(h)], dtype=torch.int64)

        return image, target_dict


# Static MS-COCO mapping dictionaries preserved for seamless evaluation metrics
# fmt: off
mscoco_category2name = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}
# fmt: on

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}
