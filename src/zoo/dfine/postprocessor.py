"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.

Refactored for zero-allocation tensor operations, vectorized mapping,
and ONNX/TensorRT modern compliance.
"""

import torch
import torch.nn as nn
import torchvision

from ...core import register


__all__ = ["DFINEPostProcessor"]


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = [
        "num_classes",
        "use_focal_loss",
        "num_top_queries",
        "remap_mscoco_category",
    ]

    def __init__(
        self,
        num_classes: int = 80,
        use_focal_loss: bool = True,
        num_top_queries: int = 300,
        remap_mscoco_category: bool = False,
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

        # [SAC Note]: Pre-allocate the MSCOCO mapping tensor during initialization
        # to avoid Host-Device syncs and python-loop bottlenecks during the forward pass.
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            # Determine maximum label index to size the buffer appropriately
            if isinstance(mscoco_label2category, dict):
                max_label = max(mscoco_label2category.keys())
            else:
                max_label = len(mscoco_label2category) - 1

            mapping = torch.zeros(max_label + 1, dtype=torch.long)

            iterator = (
                mscoco_label2category.items()
                if isinstance(mscoco_label2category, dict)
                else enumerate(mscoco_label2category)
            )
            for k, v in iterator:
                mapping[k] = v

            self.register_buffer("mscoco_mapping", mapping)

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs: dict[str, torch.Tensor], orig_target_sizes: torch.Tensor):
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # In-place multiplication for memory efficiency
        scale_tensor = orig_target_sizes.repeat(1, 2).unsqueeze(1)
        bbox_pred.mul_(scale_tensor)

        if self.use_focal_loss:
            scores = logits.sigmoid()
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)

            # Modern ONNX/TRT natively supports modulo and integer division.
            labels = index % self.num_classes
            index = torch.div(index, self.num_classes, rounding_mode="floor")

            # Use expand instead of repeat to achieve zero-memory-allocation broadcasting
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]))

        else:
            scores = logits.softmax(dim=-1)[..., :-1]
            scores, labels = scores.max(dim=-1)

            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)

                # Use expand instead of tile
                boxes = torch.gather(bbox_pred, dim=1, index=index.unsqueeze(-1).expand(-1, -1, bbox_pred.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            # Vectorized O(1) mapping entirely on GPU, completely eliminating D2H syncs
            labels = self.mscoco_mapping[labels]

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            results.append(dict(labels=lab, boxes=box, scores=sco))

        return results

    def deploy(self):
        self.eval()
        self.deploy_mode = True
        return self
